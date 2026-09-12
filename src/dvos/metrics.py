"""DAVIS-style J and F, plus occlusion-specific numbers."""

from __future__ import annotations

import math

import cv2
import numpy as np


def jaccard(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def _boundary_map(mask: np.ndarray) -> np.ndarray:
    """1-pixel boundary map offset half a pixel towards the origin (Martin et al. 2003 convention)."""
    seg = mask.astype(bool)
    e, s_, se = np.zeros_like(seg), np.zeros_like(seg), np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]
    s_[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]
    b = (seg ^ e) | (seg ^ s_) | (seg ^ se)
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s_[:, -1]
    b[-1, -1] = False
    return b


def _disk(radius: int) -> np.ndarray:
    r = int(radius)
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y <= r * r).astype(np.uint8)


def boundary_f(pred: np.ndarray, gt: np.ndarray, bound_th: float = 0.008) -> float:
    """Contour F-measure, DAVIS protocol: boundary pixels matched within a Euclidean disk of
    radius ceil(bound_th × ||image shape||). Re-implemented from the algorithm description;
    agrees with the reference implementation to machine precision on DAVIS val (see tests).
    """
    pred, gt = pred.astype(bool), gt.astype(bool)
    radius = int(math.ceil(bound_th * float(np.linalg.norm(pred.shape))))
    bp, bg = _boundary_map(pred), _boundary_map(gt)
    n_p, n_g = int(bp.sum()), int(bg.sum())
    if n_p == 0 and n_g == 0:
        return 1.0
    if n_p == 0 or n_g == 0:
        return 0.0
    k = _disk(radius)
    dp = cv2.dilate(bp.astype(np.uint8), k).astype(bool)
    dg = cv2.dilate(bg.astype(np.uint8), k).astype(bool)
    precision = float((bp & dg).sum()) / n_p
    recall = float((bg & dp).sum()) / n_g
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def sequence_jf(preds: list[np.ndarray], gts: list[np.ndarray]) -> dict:
    """Mean J and F over frames 1..T-2, the DAVIS convention (first frame given, last frame
    excluded). ``J_per_frame`` still covers every frame after the first, for episode metrics."""
    js = [jaccard(p, g) for p, g in zip(preds[1:], gts[1:], strict=True)]
    fs = [boundary_f(p, g) for p, g in zip(preds[1:-1], gts[1:-1], strict=True)]
    js_scored = js[:-1]
    j = float(np.mean(js_scored)) if js_scored else float("nan")
    f = float(np.mean(fs)) if fs else float("nan")
    return dict(J=j, F=f, JF=(j + f) / 2, J_per_frame=[float("nan")] + js)


def recovery_delay(
    j_per_frame: list[float], episodes: list[tuple[int, int]], thr: float = 0.5
) -> list[int | None]:
    """For each episode ``(start, end)``: frames after ``end`` until J >= thr, or None if never."""
    out: list[int | None] = []
    for _, end in episodes:
        delay = None
        for t in range(end, len(j_per_frame)):
            if j_per_frame[t] >= thr:
                delay = t - end
                break
        out.append(delay)
    return out


def leak_ratio(pred: np.ndarray, occluder: np.ndarray) -> float:
    """Fraction of the predicted mask lying on the occluder. 0 if nothing predicted."""
    pred, occluder = pred.astype(bool), occluder.astype(bool)
    n = pred.sum()
    return 0.0 if n == 0 else float((pred & occluder).sum() / n)


def visibility_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC by rank statistic; NaN if one class is missing."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def hidden_frames(visible: np.ndarray, fraction, thr: float = 0.9) -> np.ndarray:
    """(T,) bool: object effectively hidden — empty visible mask or occluded fraction >= thr."""
    empty = ~visible.reshape(len(visible), -1).any(axis=1)
    frac = np.asarray(fraction, dtype=float) if fraction is not None else np.zeros(len(visible))
    return empty | (frac >= thr)


def mask_to_box(mask: np.ndarray) -> np.ndarray:
    """Filled bounding box of a binary mask (empty stays empty). For scoring against box GT."""
    m = mask.astype(bool)
    if not m.any():
        return np.zeros_like(m)
    ys, xs = np.where(m)
    out = np.zeros_like(m)
    out[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = True
    return out
