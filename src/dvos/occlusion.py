"""Synthetic occlusions with exact ground truth.

An occluder is an object cut out of another sequence (RGB crop + alpha). It is pasted over
the target for a contiguous episode of frames, centred on the target's centroid with some
jitter, sized relative to the target's bounding box. Per frame we keep the visible mask,
the full (unoccluded) mask, the occluder mask and the occluded fraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def convex_fill(alpha: np.ndarray) -> np.ndarray:
    """Fill a silhouette with its convex hull, so a thin occluder still covers what it is placed on."""
    ys, xs = np.where(alpha)
    if len(ys) < 3:
        return alpha.astype(bool)
    hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
    out = np.zeros(alpha.shape, np.uint8)
    cv2.fillConvexPoly(out, hull, 1)
    return out.astype(bool)


@dataclass
class OccluderBank:
    items: list[tuple[np.ndarray, np.ndarray]] = field(
        default_factory=list
    )  # (rgb HxWx3, alpha HxW bool)

    @classmethod
    def from_masks(
        cls,
        images: list[np.ndarray],
        masks: list[np.ndarray],
        min_side: int = 24,
        fill_hull: bool = True,
    ) -> OccluderBank:
        """Cut every object out of every (image, id-mask) pair given.

        With ``fill_hull`` the alpha is the object's convex hull: the pasted patch then shows the
        object plus a little of its original surroundings, and reliably hides whatever it covers.
        """
        bank = cls()
        for img, msk in zip(images, masks, strict=True):
            for oid in np.unique(msk):
                if oid == 0:
                    continue
                alpha = msk == oid
                ys, xs = np.where(alpha)
                if len(ys) == 0:
                    continue
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                if (y1 - y0) < min_side or (x1 - x0) < min_side:
                    continue
                a = alpha[y0:y1, x0:x1].copy()
                bank.items.append((img[y0:y1, x0:x1].copy(), convex_fill(a) if fill_hull else a))
        return bank

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if not self.items:
            raise ValueError("empty occluder bank")
        return self.items[int(rng.integers(len(self.items)))]


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def occlusion_schedule(
    n_frames: int, rng: np.random.Generator, min_len: int = 4, max_len: int = 12, start_min: int = 2
) -> tuple[int, int]:
    """``(start, end)`` with end exclusive; the episode never touches frame 0."""
    if n_frames <= start_min + 1:
        raise ValueError(f"sequence too short for an occlusion: {n_frames} frames")
    length = int(rng.integers(min_len, max_len + 1))
    length = min(length, n_frames - start_min)
    start = int(rng.integers(start_min, n_frames - length + 1))
    return start, start + length


def paste(
    image: np.ndarray,
    occ_rgb: np.ndarray,
    occ_alpha: np.ndarray,
    center: tuple[float, float],
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Paste the occluder resized to ``size`` (h, w) centred at ``center`` (y, x).

    Returns the composited image and the occluder mask in image coordinates.
    """
    H, W = image.shape[:2]
    h, w = max(1, int(size[0])), max(1, int(size[1]))
    rgb = cv2.resize(occ_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.resize(occ_alpha.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(
        bool
    )
    y0 = int(round(center[0] - h / 2))
    x0 = int(round(center[1] - w / 2))
    ys, ye = max(0, y0), min(H, y0 + h)
    xs, xe = max(0, x0), min(W, x0 + w)
    out = image.copy()
    occ_mask = np.zeros((H, W), dtype=bool)
    if ys >= ye or xs >= xe:
        return out, occ_mask
    a = alpha[ys - y0 : ye - y0, xs - x0 : xe - x0]
    out[ys:ye, xs:xe][a] = rgb[ys - y0 : ye - y0, xs - x0 : xe - x0][a]
    occ_mask[ys:ye, xs:xe] = a
    return out, occ_mask


def occlude_sequence(
    images: list[np.ndarray],
    full_masks: list[np.ndarray],
    bank: OccluderBank,
    rng: np.random.Generator,
    *,
    min_len: int = 4,
    max_len: int = 12,
    start_min: int = 2,
    scale: tuple[float, float] = (0.8, 1.4),
    jitter: float = 0.15,
) -> dict:
    """Apply one occlusion episode to a binary-mask sequence of a single target.

    Frames where the target is already absent keep the occluder where the last known
    bbox was, so an episode can overlap a real disappearance without crashing.
    """
    n = len(images)
    start, end = occlusion_schedule(n, rng, min_len, max_len, start_min)
    occ_rgb, occ_alpha = bank.sample(rng)
    s = float(rng.uniform(*scale))
    out_imgs, visible, occluders, fraction = [], [], [], []
    last_box = bbox(full_masks[0]) or (0, images[0].shape[0], 0, images[0].shape[1])
    for t in range(n):
        m = full_masks[t].astype(bool)
        if not (start <= t < end):
            out_imgs.append(images[t])
            visible.append(m)
            occluders.append(np.zeros_like(m))
            fraction.append(0.0)
            continue
        box = bbox(m) or last_box
        last_box = box
        y0, y1, x0, x1 = box
        bh, bw = y1 - y0, x1 - x0
        cy = (y0 + y1) / 2 + rng.uniform(-jitter, jitter) * bh
        cx = (x0 + x1) / 2 + rng.uniform(-jitter, jitter) * bw
        img, occ = paste(images[t], occ_rgb, occ_alpha, (cy, cx), (s * bh, s * bw))
        vis = m & ~occ
        out_imgs.append(img)
        visible.append(vis)
        occluders.append(occ)
        area = float(m.sum())
        fraction.append(0.0 if area == 0 else float((m & occ).sum()) / area)
    return dict(
        images=out_imgs,
        visible=visible,
        full=[m.astype(bool) for m in full_masks],
        occluder=occluders,
        fraction=np.array(fraction, dtype=np.float32),
        episode=(start, end),
    )
