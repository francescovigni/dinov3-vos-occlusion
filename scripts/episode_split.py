#!/usr/bin/env python3
"""Where does a run lose accuracy: inside occlusion episodes, or outside them?

Reads ``<run>/masks/<seq>.npz`` (written by ``dvos.evaluate --save-masks``) and the cached
ground truth, and reports mean J on frames inside the episodes, in the 10 frames after each
episode, and everywhere else. Usage: ``episode_split.py runs/baseline_occ0 [more runs...]``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dvos.config import feature_dir, load_config  # noqa: E402
from dvos.metrics import jaccard  # noqa: E402


def split_j(run: Path, cfg, after: int = 10) -> dict:
    metrics = json.loads((run / "metrics.json").read_text())
    s = metrics["summary"]
    inside, post, outside = [], [], []
    for row in metrics["sequences"]:
        seq = row["seq"]
        fdir = feature_dir(cfg, s["split"], seq, s["variant"])
        gt = np.load(fdir / "masks.npz")["visible"].astype(bool)
        pred = np.load(run / "masks" / f"{seq}.npz")["pred"].astype(bool)
        episodes = [tuple(e) for e in row["episodes"]]
        for t in range(1, len(gt)):
            j = jaccard(pred[t], gt[t])
            if any(a <= t < b for a, b in episodes):
                inside.append(j)
            elif any(b <= t < b + after for _, b in episodes):
                post.append(j)
            else:
                outside.append(j)
    mean = lambda v: float(np.mean(v)) if v else float("nan")  # noqa: E731
    return dict(
        run=run.name,
        inside=mean(inside),
        post=mean(post),
        outside=mean(outside),
        n_inside=len(inside),
        n_post=len(post),
        n_outside=len(outside),
    )


def main() -> None:
    cfg = load_config("configs/default.yaml")
    runs = [Path(r) for r in sys.argv[1:]] or sorted(p.parent for p in Path("runs").glob("*/masks"))
    print(
        "| run | J inside episode | J 10 frames after | J elsewhere | frames (in / after / else) |"
    )
    print("|---|---|---|---|---|")
    for run in runs:
        if not (run / "masks").exists():
            continue
        r = split_j(run, cfg)
        print(
            f"| {r['run']} | {r['inside']:.3f} | {r['post']:.3f} | {r['outside']:.3f} | {r['n_inside']} / {r['n_post']} / {r['n_outside']} |"
        )


if __name__ == "__main__":
    main()
