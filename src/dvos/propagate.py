"""Zero-shot label propagation on frozen features (DINO video-segmentation protocol).

Each query patch at frame t looks at frame 0 plus the last ``n_last`` frames, restricted to
a spatial neighbourhood of ``radius`` patches, keeps the ``topk`` most similar patches and
copies their (soft) labels with softmax weights at ``temperature``. Nothing is trained.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def neighborhood_mask(
    h: int, w: int, radius: int, device: torch.device | None = None
) -> torch.Tensor:
    """(h*w, h*w) bool: True where |dy| <= radius and |dx| <= radius."""
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    ys, xs = ys.flatten().to(device), xs.flatten().to(device)
    dy = (ys[:, None] - ys[None, :]).abs()
    dx = (xs[:, None] - xs[None, :]).abs()
    return (dy <= radius) & (dx <= radius)


@torch.no_grad()
def propagate(
    feats: list[torch.Tensor],
    first_labels: torch.Tensor,
    *,
    n_last: int = 7,
    topk: int = 5,
    radius: int = 12,
    temperature: float = 0.07,
    hidden: list[bool] | None = None,
) -> list[torch.Tensor]:
    """``feats``: T tensors (C,h,w). ``first_labels``: (K,h,w) soft/one-hot. Returns T tensors (K,h,w).

    ``hidden[t]`` marks frames where the object is declared hidden: their label is all
    background (channel 0) and they are never used as context. Frame 0 is never hidden.
    """
    if not feats:
        return []
    device = feats[0].device
    C, h, w = feats[0].shape
    K = first_labels.shape[0]
    fn = [F.normalize(f.reshape(C, -1), dim=0) for f in feats]  # (C, hw)
    labels = [first_labels.reshape(K, -1).float().to(device)]
    nb = neighborhood_mask(h, w, radius, device)
    hidden = [False] * len(feats) if hidden is None else list(hidden)
    hidden[0] = False
    empty = torch.zeros(K, h * w, device=device)
    empty[0] = 1.0
    for t in range(1, len(feats)):
        if hidden[t]:
            labels.append(empty.clone())
            continue
        recent = [i for i in range(max(1, t - n_last), t) if not hidden[i]]
        ctx = [0] + recent
        keys = torch.cat([fn[i] for i in ctx], dim=1)  # (C, N*hw)
        labs = torch.cat([labels[i] for i in ctx], dim=1)  # (K, N*hw)
        aff = (fn[t].t() @ keys) / temperature  # (hw, N*hw)
        aff = aff.masked_fill(~nb.repeat(1, len(ctx)), float("-inf"))
        k = min(topk, aff.shape[1])
        vals, idx = aff.topk(k, dim=1)  # (hw, k)
        wts = torch.softmax(vals, dim=1)
        gathered = labs[:, idx]  # (K, hw, k)
        labels.append((gathered * wts[None]).sum(-1))
    return [lab.reshape(K, h, w) for lab in labels]
