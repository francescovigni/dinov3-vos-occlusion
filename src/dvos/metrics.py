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


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    er = cv2.erode(m, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (m - er).astype(bool)


def boundary_f(pred: np.ndarray, gt: np.ndarray, bound_th: float = 0.008) -> float:
    """Contour F-measure with a tolerance of ``bound_th`` × image diagonal (DAVIS default)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    h, w = gt.shape
    r = max(1, int(math.ceil(bound_th * math.sqrt(h * h + w * w))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    bp, bg = _boundary(pred), _boundary(gt)
    dp = cv2.dilate(bp.astype(np.uint8), k).astype(bool)
    dg = cv2.dilate(bg.astype(np.uint8), k).astype(bool)
    precision = float((bp & dg).sum()) / max(1, bp.sum())
    recall = float((bg & dp).sum()) / max(1, bg.sum())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def sequence_jf(preds: list[np.ndarray], gts: list[np.ndarray]) -> dict:
    """Mean J and F over frames 1..T-1 (frame 0 is given), DAVIS convention."""
    js = [jaccard(p, g) for p, g in zip(preds[1:], gts[1:], strict=True)]
    fs = [boundary_f(p, g) for p, g in zip(preds[1:], gts[1:], strict=True)]
    j, f = float(np.mean(js)) if js else float("nan"), float(np.mean(fs)) if fs else float("nan")
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
