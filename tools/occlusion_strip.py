"""The occlusion, frame by frame, with each run's mask drawn on it.

One row per run, one column per frame, so the two resolutions can be compared at the
moment that separates them: the hand arrives, both masks collapse, and only one comes back.
Masks are read from the npz `scene_eval --save-masks` writes, not from the overlay video,
so the figure is derived from the measurements rather than from a compressed picture of them.

    python tools/occlusion_strip.py \
        --runs runs/scene_occ_480_baseline runs/scene_occ_384_fp32 \
        --labels "480x864" "384x672" \
        --scene <scene dir> --frames 40 47 50 54 60 70 \
        --out docs/figures/occlusion_strip.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

TILE_W = 300
MASK_RGB = (0, 0, 255)  # BGR: prediction
MISS_RGB = (0, 200, 0)  # target the prediction missed


def load_masks(run: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(run / "masks.npz")
    h, w = z["shape"]
    pred = np.unpackbits(z["pred"], axis=-1)[..., :w].astype(bool)
    gt = np.unpackbits(z["gt"], axis=-1)[..., :w].astype(bool)
    return pred, gt, z["hidden"]


def tile(frame: np.ndarray, pred: np.ndarray, gt: np.ndarray, hidden: bool, label: str):
    vis = frame.copy()
    vis[pred] = (0.4 * vis[pred] + 0.6 * np.array(MASK_RGB)).astype(np.uint8)
    edge = gt & ~pred
    vis[edge] = (0.5 * vis[edge] + 0.5 * np.array(MISS_RGB)).astype(np.uint8)
    h = int(TILE_W * vis.shape[0] / vis.shape[1])
    vis = cv2.resize(vis, (TILE_W, h))
    if hidden:
        cv2.rectangle(vis, (0, 0), (TILE_W - 1, h - 1), (0, 0, 255), 3)
    if label:
        cv2.putText(
            vis, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
        )
    return vis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--frames", nargs="+", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    labels = args.labels or [r.name for r in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("--labels must match --runs")

    paths = sorted(args.scene.glob("frame_*.png"))
    frames = {t: cv2.imread(str(paths[t]), cv2.IMREAD_COLOR) for t in args.frames}

    rows = []
    for run, label in zip(args.runs, labels, strict=True):
        pred, gt, hidden = load_masks(run)
        # scene_eval and silhouette_track report different headline numbers; show whichever
        # the run actually produced rather than demanding one shape of metrics file.
        mfile = next(
            (f for f in ("scene_metrics.json", "silhouette_metrics.json") if (run / f).exists()),
            None,
        )
        if mfile is None:
            raise SystemExit(f"no metrics file in {run}")
        summary = json.loads((run / mfile).read_text())["summary"]
        if "J_after_last_episode" in summary:
            tag = f"{label}  J_after={summary['J_after_last_episode']:.3f}"
        else:
            tag = (
                f"{label}  J={summary['J_visible_mean']:.3f}"
                f"  boundaryF={summary['boundaryF_visible_mean']:.3f}"
            )
        tiles = []
        for i, t in enumerate(args.frames):
            cap = f"t={t}" + ("  HIDDEN" if hidden[t] else "")
            tiles.append(tile(frames[t], pred[t], gt[t], bool(hidden[t]), cap))
        strip = np.hstack(tiles)
        bar = np.full((26, strip.shape[1], 3), 32, np.uint8)
        cv2.putText(
            bar, tag, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )
        rows.append(np.vstack([bar, strip]))

    sheet = np.vstack(rows)
    legend = np.full((30, sheet.shape[1], 3), 32, np.uint8)
    cv2.putText(
        legend,
        "red = predicted mask   green = target the mask missed   red border = fully occluded",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    sheet = np.vstack([sheet, legend])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), sheet)
    print(f"wrote {args.out}  ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
