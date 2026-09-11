"""Qualitative figure: best / median / worst sequence of a run, three frames each.

Frames: just before the occlusion episode, inside it, and a few frames after it ends
(for clean runs: first quarter, middle, last quarter). Overlays: prediction (red fill),
ground-truth visible mask (green contour), occluder (yellow contour). cv2 only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dvos.config import feature_dir, load_config
from dvos.davis import Davis
from dvos.extract import build_bank, seq_seed
from dvos.occlusion import occlude_sequence

RED, GREEN, YELLOW, WHITE = (220, 40, 40), (40, 220, 40), (240, 220, 40), (255, 255, 255)


def overlay(img: np.ndarray, pred: np.ndarray, gt: np.ndarray, occ: np.ndarray) -> np.ndarray:
    out = img.copy()
    fill = out[pred.astype(bool)]
    out[pred.astype(bool)] = (0.55 * fill + 0.45 * np.array(RED)).astype(np.uint8)
    for mask, colour in ((gt, GREEN), (occ, YELLOW)):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, colour, 2)
    return out


def pick_frames(n: int, episode: list[int] | None) -> list[int]:
    if episode:
        s, e = episode
        return [max(1, s - 1), (s + e) // 2, min(n - 1, e + 2)]
    return [max(1, n // 4), n // 2, max(1, 3 * n // 4)]


def sequence_images(cfg, split: str, seq: str, variant: str, bank, target: int) -> list[np.ndarray]:
    """Re-create the exact frames the features were extracted from (same seed, same bank)."""
    davis = Davis(cfg.data.davis_root, split, cfg.data.year, cfg.data.resolution)
    imgs, msks = davis.load(seq)
    if variant == "clean":
        return imgs
    rng = np.random.default_rng(seq_seed(seq, int(variant[3:])))
    o = cfg.occlusion
    d = occlude_sequence(
        imgs,
        [(m == target) for m in msks],
        bank,
        rng,
        min_len=o.min_len,
        max_len=o.max_len,
        start_min=o.start_min,
        scale=tuple(o.scale),
        jitter=o.jitter,
    )
    return d["images"]


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", required=True, help="evaluate output dir with metrics.json and masks/")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=427, help="per-tile width")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Path(args.run)
    metrics = json.loads((run / "metrics.json").read_text())
    variant = metrics["summary"]["variant"]
    rows = sorted(metrics["sequences"], key=lambda r: r["J"])
    chosen = [("worst", rows[0]), ("median", rows[len(rows) // 2]), ("best", rows[-1])]
    bank = build_bank(cfg) if variant != "clean" else None
    tiles = []
    for tag, row in chosen:
        seq = row["seq"]
        fdir = feature_dir(cfg, args.split, seq, variant)
        meta = json.loads((fdir / "meta.json").read_text())
        masks = np.load(fdir / "masks.npz")
        preds = np.load(run / "masks" / f"{seq}.npz")["pred"]
        imgs = sequence_images(cfg, args.split, seq, variant, bank, meta["target_id"])
        frames = pick_frames(len(imgs), meta["episode"])
        strip = []
        for t in frames:
            tile = overlay(imgs[t], preds[t], masks["visible"][t], masks["occluder"][t])
            scale = args.width / tile.shape[1]
            tile = cv2.resize(tile, (args.width, int(tile.shape[0] * scale)))
            j = row["J"] if t == frames[0] else float("nan")
            txt = f"{tag}: {seq}  t={t}" + (f"  J={j:.2f}" if not np.isnan(j) else "")
            strip.append(label(tile, txt))
        tiles.append(np.concatenate(strip, axis=1))
    grid = np.concatenate(tiles, axis=0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"wrote {out} {grid.shape[1]}x{grid.shape[0]}")


if __name__ == "__main__":
    main()
