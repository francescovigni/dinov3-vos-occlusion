"""Gated memory head on frozen DINOv3 features.

Frame 0 (key, value) is permanent. Later frames are written only when the visibility head
says the object is visible, and always as hard masks. Readout is attention from the current
frame's keys to the memory keys; keys carry 2-D position channels and reads from later
memory frames are restricted to a spatial neighbourhood (frame 0 stays global, so a
re-appearing object can be matched anywhere). Optionally the zero-shot k-NN label
propagation from the same memory is computed and handed to the decoder and the visibility
head as a prior: the head then corrects the zero-shot answer instead of replacing it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sincos_2d(h: int, w: int, dim: int, device=None) -> torch.Tensor:
    """(1, dim, h, w) fixed 2-D sinusoidal position channels, ``dim`` divisible by 4."""
    if dim % 4:
        raise ValueError("pos_dim must be divisible by 4")
    q = dim // 4
    freqs = torch.exp(-math.log(10000.0) * torch.arange(q, device=device) / q)  # (q,)
    ys = torch.arange(h, device=device).float()[:, None] * freqs[None]  # (h, q)
    xs = torch.arange(w, device=device).float()[:, None] * freqs[None]  # (w, q)
    py = torch.cat([ys.sin(), ys.cos()], dim=1).t()[:, :, None].expand(-1, h, w)  # (2q, h, w)
    px = torch.cat([xs.sin(), xs.cos()], dim=1).t()[:, None, :].expand(-1, h, w)  # (2q, h, w)
    return torch.cat([py, px], dim=0)[None]


class KeyValueEncoder(nn.Module):
    def __init__(self, c_in: int, c_key: int = 64, c_value: int = 256, pos_dim: int = 0):
        super().__init__()
        self.pos_dim = pos_dim
        self.key = nn.Conv2d(c_in + pos_dim, c_key, 1)
        self.value = nn.Sequential(
            nn.Conv2d(c_in + 2, c_value, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_value, c_value, 3, padding=1),
        )

    def encode_key(self, feat: torch.Tensor) -> torch.Tensor:
        if self.pos_dim:
            pos = sincos_2d(feat.shape[-2], feat.shape[-1], self.pos_dim, feat.device)
            feat = torch.cat([feat, pos.expand(feat.shape[0], -1, -1, -1)], dim=1)
        return self.key(feat)

    def encode_value(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``mask``: (B,1,h,w) in [0,1] at feature resolution."""
        return self.value(torch.cat([feat, mask, 1.0 - mask], dim=1))


class MemoryBank:
    """Keys (B,Ck,N) and values (B,Cv,N), plus features and labels for the label prior.

    Entry 0 is never evicted unless ``keep_first`` is off.
    """

    def __init__(self, max_size: int = 8, keep_first: bool = True):
        self.max_size = max(2, max_size)
        self.keep_first = keep_first
        self.keys: list[torch.Tensor] = []
        self.vals: list[torch.Tensor] = []
        self.feats: list[torch.Tensor | None] = []  # L2-normalised features
        self.labels: list[torch.Tensor | None] = []  # masks at feature resolution
        self.is_first: list[bool] = []

    def add(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        feat: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
    ) -> None:
        self.is_first.append(not self.keys)
        self.keys.append(key.flatten(2))
        self.vals.append(value.flatten(2))
        self.feats.append(F.normalize(feat.flatten(2), dim=1) if feat is not None else None)
        self.labels.append(label.flatten(2) if label is not None else None)
        if len(self.keys) > self.max_size:
            evict = 1 if self.keep_first else 0
            for lst in (self.keys, self.vals, self.feats, self.labels, self.is_first):
                del lst[evict]

    def read(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.cat(self.keys, dim=2), torch.cat(self.vals, dim=2)

    def read_prior(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalised features (B,C,N) and labels (B,1,N) of every entry."""
        return torch.cat(self.feats, dim=2), torch.cat(self.labels, dim=2)

    def locality_mask(self, h: int, w: int, radius: int, device=None) -> torch.Tensor | None:
        """(hw, N) bool: frame-0 entries readable everywhere, later entries within ``radius``."""
        if radius <= 0:
            return None
        from dvos.propagate import neighborhood_mask

        nb = neighborhood_mask(h, w, radius, device)
        full = torch.ones_like(nb)
        return torch.cat([full if f else nb for f in self.is_first], dim=1)

    def __len__(self) -> int:
        return len(self.keys)


def readout(
    qk: torch.Tensor,
    mk: torch.Tensor,
    mv: torch.Tensor,
    topk: int | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """qk (B,Ck,Q), mk (B,Ck,N), mv (B,Cv,N) -> (B,Cv,Q). ``mask`` (Q,N) bool restricts reads."""
    logits = torch.einsum("bcq,bcn->bqn", qk, mk) / math.sqrt(qk.shape[1])
    if mask is not None:
        logits = logits.masked_fill(~mask[None], float("-inf"))
    n = logits.shape[-1]
    if topk is not None and topk < n:
        vals, idx = logits.topk(topk, dim=-1)
        wts = torch.zeros_like(logits).scatter_(-1, idx, torch.softmax(vals, dim=-1))
    else:
        wts = torch.softmax(logits, dim=-1)
    return torch.einsum("bqn,bcn->bcq", wts, mv)


def label_prior(
    qfeat: torch.Tensor,
    mfeat: torch.Tensor,
    mlabel: torch.Tensor,
    mask: torch.Tensor | None = None,
    topk: int = 5,
    temperature: float = 0.07,
) -> torch.Tensor:
    """k-NN label propagation from memory: qfeat (B,C,Q), mfeat (B,C,N), mlabel (B,1,N) -> (B,1,Q).

    Same rule as the zero-shot baseline, restricted to the memory frames and their hard masks.
    """
    q = F.normalize(qfeat, dim=1)
    aff = torch.einsum("bcq,bcn->bqn", q, mfeat) / temperature
    if mask is not None:
        aff = aff.masked_fill(~mask[None], float("-inf"))
    k = min(topk, aff.shape[-1])
    vals, idx = aff.topk(k, dim=-1)
    wts = torch.softmax(vals, dim=-1)  # (B,Q,k)
    gathered = torch.gather(mlabel.expand(-1, aff.shape[1], -1), 2, idx)  # (B,Q,k)
    return (gathered * wts).sum(-1, keepdim=True).transpose(1, 2)  # (B,1,Q)


class Decoder(nn.Module):
    """(B, C, h, w) -> (B, 1, 4h, 4w) logits."""

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
    def __init__(self, c_value: int, hidden: int = 128, extra: int = 0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * c_value + extra, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(self, r: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        """r (B,Cv,Q) [+ extra (B,E)] -> (B,1) logit."""
        pooled = torch.cat([r.mean(-1), r.amax(-1)], dim=1)
        if extra is not None:
            pooled = torch.cat([pooled, extra], dim=1)
        return self.mlp(pooled)


class MemoryVOS(nn.Module):
    def __init__(
        self,
        c_in: int = 384,
        c_key: int = 64,
        c_value: int = 256,
        hidden: int = 256,
        readout_topk: int | None = 32,
        pos_dim: int = 0,
        locality_radius: int = 0,
        use_prior: bool = False,
        prior_topk: int = 5,
        prior_temperature: float = 0.07,
    ):
        super().__init__()
        self.kv = KeyValueEncoder(c_in, c_key, c_value, pos_dim)
        self.decoder = Decoder(c_value + c_in + (1 if use_prior else 0), hidden)
        self.vis = VisibilityHead(c_value, extra=2 if use_prior else 0)
        self.readout_topk = readout_topk
        self.locality_radius = locality_radius
        self.use_prior = use_prior
        self.prior_topk = prior_topk
        self.prior_temperature = prior_temperature

    def encode(
        self, feat: torch.Tensor, mask_lr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kv.encode_key(feat), self.kv.encode_value(feat, mask_lr)

    def forward(
        self,
        feat: torch.Tensor,
        mk: torch.Tensor,
        mv: torch.Tensor,
        mask: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
    ) -> dict:
        """``prior``: (B,1,hw) propagated soft label from the memory, required when ``use_prior``."""
        B, C, h, w = feat.shape
        qk = self.kv.encode_key(feat).flatten(2)
        r = readout(qk, mk, mv, self.readout_topk, mask)  # (B,Cv,hw)
        parts = [r.view(B, -1, h, w), feat]
        extra = None
        if self.use_prior:
            if prior is None:
                raise ValueError("model built with use_prior=True needs a prior map")
            parts.append(prior.view(B, 1, h, w))
            extra = torch.cat([prior.mean(-1), prior.amax(-1)], dim=1)  # (B,2)
        dec_in = torch.cat(parts, dim=1)
        return dict(
            mask_logits=self.decoder(dec_in), vis_logit=self.vis(r, extra), readout=r, prior=prior
        )

    def step(self, feat: torch.Tensor, mem: MemoryBank) -> dict:
        """One tracking step: locality mask + prior from the bank, then forward."""
        h, w = feat.shape[2:]
        mk, mv = mem.read()
        mask = mem.locality_mask(h, w, self.locality_radius, feat.device)
        prior = None
        if self.use_prior:
            mf, ml = mem.read_prior()
            prior = label_prior(
                feat.flatten(2), mf, ml, mask, self.prior_topk, self.prior_temperature
            )
        return self.forward(feat, mk, mv, mask, prior)

    def write(self, mem: MemoryBank, feat: torch.Tensor, mask_lr: torch.Tensor) -> None:
        """Encode a frame with its mask at feature resolution and append it to the bank."""
        key, value = self.encode(feat, mask_lr)
        mem.add(key, value, feat, mask_lr)


def build_model(c_in: int, mc) -> MemoryVOS:
    """Construct from a config namespace or dict; missing keys keep the v1 defaults."""
    get = (lambda k, d: mc.get(k, d)) if isinstance(mc, dict) else (lambda k, d: getattr(mc, k, d))
    return MemoryVOS(
        c_in,
        get("c_key", 64),
        get("c_value", 256),
        get("hidden", 256),
        get("readout_topk", 32),
        get("pos_dim", 0),
        get("locality_radius", 0),
        get("use_prior", False),
        get("prior_topk", 5),
        get("prior_temperature", 0.07),
    )


def to_feature_res(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """(B,1,H,W) in [0,1] -> (B,1,h,w) by area averaging.

    MPS has no adaptive pooling for non-divisible sizes; fall back to CPU for that case.
    """
    divisible = mask.shape[-2] % size[0] == 0 and mask.shape[-1] % size[1] == 0
    if mask.device.type == "mps" and not divisible:
        return F.interpolate(mask.float().cpu(), size=size, mode="area").to(mask.device)
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
    model.write(mem, feats[0:1], (to_feature_res(first_mask, (h, w)) > 0.5).float())
    probs, vis = [first_mask.float()], [1.0]
    for t in range(1, T):
        out = model.step(feats[t : t + 1], mem)
        p = torch.sigmoid(out["mask_logits"])
        v = float(torch.sigmoid(out["vis_logit"]).item())
        probs.append(p)
        vis.append(v)
        if v > vis_gate:
            # hard mask: the value encoder is trained on binary ground truth, and soft masks
            # fed back frame after frame eroded the object (v1 failure)
            model.write(
                mem, feats[t : t + 1], (to_feature_res((p > 0.5).float(), (h, w)) > 0.5).float()
            )
    return probs, vis
