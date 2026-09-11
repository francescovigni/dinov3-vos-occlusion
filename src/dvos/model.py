"""Gated memory head on frozen DINOv3 features.

Frame 0 (key, value) is permanent. Later frames are written only when the visibility head
says the object is visible. Readout is attention from the current frame's keys to the
memory keys; the decoder upsamples the readout to 1/4 resolution mask logits.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeyValueEncoder(nn.Module):
    def __init__(self, c_in: int, c_key: int = 64, c_value: int = 256):
        super().__init__()
        self.key = nn.Conv2d(c_in, c_key, 1)
        self.value = nn.Sequential(
            nn.Conv2d(c_in + 2, c_value, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_value, c_value, 3, padding=1),
        )

    def encode_key(self, feat: torch.Tensor) -> torch.Tensor:
        return self.key(feat)

    def encode_value(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``mask``: (B,1,h,w) in [0,1] at feature resolution."""
        return self.value(torch.cat([feat, mask, 1.0 - mask], dim=1))


class MemoryBank:
    """Keys (B,Ck,N) and values (B,Cv,N). Entry 0 is never evicted."""

    def __init__(self, max_size: int = 8, keep_first: bool = True):
        self.max_size = max(2, max_size)
        self.keep_first = keep_first
        self.keys: list[torch.Tensor] = []
        self.vals: list[torch.Tensor] = []

    def add(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self.keys.append(key.flatten(2))
        self.vals.append(value.flatten(2))
        if len(self.keys) > self.max_size:
            evict = 1 if self.keep_first else 0
            del self.keys[evict], self.vals[evict]

    def read(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.cat(self.keys, dim=2), torch.cat(self.vals, dim=2)

    def __len__(self) -> int:
        return len(self.keys)


def readout(
    qk: torch.Tensor, mk: torch.Tensor, mv: torch.Tensor, topk: int | None = None
) -> torch.Tensor:
    """qk (B,Ck,Q), mk (B,Ck,N), mv (B,Cv,N) -> (B,Cv,Q)."""
    logits = torch.einsum("bcq,bcn->bqn", qk, mk) / math.sqrt(qk.shape[1])
    n = logits.shape[-1]
    if topk is not None and topk < n:
        vals, idx = logits.topk(topk, dim=-1)
        wts = torch.zeros_like(logits).scatter_(-1, idx, torch.softmax(vals, dim=-1))
    else:
        wts = torch.softmax(logits, dim=-1)
    return torch.einsum("bqn,bcn->bcq", wts, mv)


class Decoder(nn.Module):
    """(B, Cv + C_in, h, w) -> (B, 1, 4h, 4w) logits."""

    def __init__(self, c_in: int, hidden: int = 256):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(c_in, hidden, 3, padding=1), nn.GELU())
        self.block2 = nn.Sequential(nn.Conv2d(hidden, hidden // 2, 3, padding=1), nn.GELU())
        self.block3 = nn.Sequential(nn.Conv2d(hidden // 2, hidden // 4, 3, padding=1), nn.GELU())
        self.out = nn.Conv2d(hidden // 4, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block3(x)
        return self.out(x)


class VisibilityHead(nn.Module):
    def __init__(self, c_value: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * c_value, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """r (B,Cv,Q) -> (B,1) logit."""
        pooled = torch.cat([r.mean(-1), r.amax(-1)], dim=1)
        return self.mlp(pooled)


class MemoryVOS(nn.Module):
    def __init__(
        self,
        c_in: int = 384,
        c_key: int = 64,
        c_value: int = 256,
        hidden: int = 256,
        readout_topk: int | None = 32,
    ):
        super().__init__()
        self.kv = KeyValueEncoder(c_in, c_key, c_value)
        self.decoder = Decoder(c_value + c_in, hidden)
        self.vis = VisibilityHead(c_value)
        self.readout_topk = readout_topk

    def encode(
        self, feat: torch.Tensor, mask_lr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kv.encode_key(feat), self.kv.encode_value(feat, mask_lr)

    def forward(self, feat: torch.Tensor, mk: torch.Tensor, mv: torch.Tensor) -> dict:
        B, C, h, w = feat.shape
        qk = self.kv.encode_key(feat).flatten(2)
        r = readout(qk, mk, mv, self.readout_topk)  # (B,Cv,hw)
        dec_in = torch.cat([r.view(B, -1, h, w), feat], dim=1)
        return dict(mask_logits=self.decoder(dec_in), vis_logit=self.vis(r), readout=r)


def to_feature_res(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """(B,1,H,W) in [0,1] -> (B,1,h,w) by area averaging."""
    return F.interpolate(mask.float(), size=size, mode="area")


def bce_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    dice = 1.0 - (2.0 * inter + 1.0) / (p.sum() + target.sum() + 1.0)
    return bce + dice


@torch.no_grad()
def track(
    model: MemoryVOS,
    feats: torch.Tensor,
    first_mask: torch.Tensor,
    *,
    memory_max: int = 8,
    vis_gate: float = 0.5,
    keep_first: bool = True,
) -> tuple[list[torch.Tensor], list[float]]:
    """``feats`` (T,C,h,w), ``first_mask`` (1,1,4h,4w). Returns mask probs (1,1,4h,4w) per frame, vis probs."""
    T, _, h, w = feats.shape
    mem = MemoryBank(memory_max, keep_first=keep_first)
    k0, v0 = model.encode(feats[0:1], to_feature_res(first_mask, (h, w)))
    mem.add(k0, v0)
    probs, vis = [first_mask.float()], [1.0]
    for t in range(1, T):
        mk, mv = mem.read()
        out = model(feats[t : t + 1], mk, mv)
        p = torch.sigmoid(out["mask_logits"])
        v = float(torch.sigmoid(out["vis_logit"]).item())
        probs.append(p)
        vis.append(v)
        if v > vis_gate:
            mem.add(*model.encode(feats[t : t + 1], to_feature_res(p, (h, w))))
    return probs, vis
