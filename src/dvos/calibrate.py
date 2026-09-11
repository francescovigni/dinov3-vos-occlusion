"""Pick the visibility gate on held-out training sequences (never on the validation set).

Runs the tracker on the ``occ0`` variant of the sequences listed in ``<run>/split.json``
and chooses the threshold that maximises balanced accuracy of "hidden" detection.
Writes ``<run>/gate.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dvos.backbone import pick_device
from dvos.config import feature_dir, load_config
from dvos.metrics import hidden_frames
from dvos.model import build_model, track


def collect_scores(
    model, dirs: list[Path], mc, device, hidden_thr: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
    scores, hidden = [], []
    for d in dirs:
        feats = torch.from_numpy(np.load(d / "feats.npy").astype(np.float32)).to(device)
        gt = np.load(d / "masks.npz")["visible"].astype(bool)
        fraction = json.loads((d / "meta.json").read_text()).get("fraction")
        hid = hidden_frames(gt, fraction, hidden_thr)
        h, w = feats.shape[2:]
        first = torch.from_numpy(gt[0].astype(np.float32))[None, None]
        first = (F.interpolate(first, size=(4 * h, 4 * w), mode="area") > 0.5).float().to(device)
        _, vis = track(model, feats, first, memory_max=mc.memory_max, vis_gate=mc.vis_gate)
        for t in range(1, len(gt)):
            scores.append(vis[t])
            hidden.append(bool(hid[t]))
    return np.array(scores), np.array(hidden)


def best_threshold(scores: np.ndarray, hidden: np.ndarray, grid=None) -> tuple[float, float]:
    """Threshold maximising balanced accuracy: mean(hidden caught, visible kept)."""
    grid = np.linspace(0.05, 0.95, 19) if grid is None else grid
    if hidden.sum() == 0 or (~hidden).sum() == 0:
        return 0.5, float("nan")
    best_thr, best_ba = 0.5, -1.0
    for thr in grid:
        caught = (scores[hidden] < thr).mean()
        kept = (scores[~hidden] >= thr).mean()
        ba = 0.5 * (caught + kept)
        if ba > best_ba:
            best_thr, best_ba = float(thr), float(ba)
    return best_thr, best_ba


def gate_by_j(scores_per_seq, dirs: list[Path], bcfg, device, grid) -> tuple[float, float, dict]:
    """Gate maximising mean J of zero-shot propagation gated by the head's visibility scores."""
    from dvos.metrics import jaccard
    from dvos.propagate import propagate

    cache = []
    for d in dirs:
        feats = torch.from_numpy(np.load(d / "feats.npy").astype(np.float32)).to(device)
        gt = np.load(d / "masks.npz")["visible"].astype(bool)
        cache.append((feats, gt))
    curve = {}
    for thr in grid:
        js = []
        for (feats, gt), vis in zip(cache, scores_per_seq, strict=True):
            T, _, h, w = feats.shape
            first = torch.from_numpy(gt[0].astype(np.float32))[None, None]
            fg = F.interpolate(first, size=(h, w), mode="area")[0, 0]
            labels = torch.stack([1 - fg, fg]).to(device)
            soft = propagate(
                list(feats),
                labels,
                n_last=bcfg.n_last,
                topk=bcfg.topk,
                radius=bcfg.radius,
                temperature=bcfg.temperature,
                hidden=[v < thr for v in vis],
            )
            for t in range(1, T):
                up = F.interpolate(
                    soft[t][None], size=gt.shape[1:], mode="bilinear", align_corners=False
                )[0]
                js.append(jaccard((up.argmax(0) == 1).cpu().numpy(), gt[t]))
        curve[float(thr)] = float(np.mean(js))
    best = max(curve, key=curve.get)
    return best, curve[best], curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", default="runs/head")
    ap.add_argument("--split", default="train")
    ap.add_argument("--variant", default="occ0")
    ap.add_argument("--objective", choices=["balanced", "j"], default="balanced")
    ap.add_argument(
        "--out", default=None, help="output json (default <run>/gate.json or gate_j.json)"
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Path(args.run)
    holdout = json.loads((run / "split.json").read_text())["holdout"]
    dirs = [feature_dir(cfg, args.split, s, args.variant) for s in holdout]
    dirs = [d for d in dirs if (d / "feats.npy").exists()]
    if not dirs:
        raise SystemExit("no held-out sequences with the requested variant in the feature cache")
    device = pick_device(cfg.backbone.device)
    ck = torch.load(run / "model.pt", map_location="cpu")
    model = build_model(ck["c_in"], ck.get("config", {}).get("model", None) or cfg.model)
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    scores, hidden = collect_scores(model, dirs, cfg.model, device, cfg.occlusion.hidden_fraction)
    thr, ba = best_threshold(scores, hidden)
    out = dict(
        vis_gate=thr,
        balanced_accuracy=ba,
        n_frames=int(len(scores)),
        n_hidden=int(hidden.sum()),
        sequences=holdout,
        variant=args.variant,
        objective=args.objective,
    )
    if args.objective == "j":
        per_seq, k = [], 0
        for d in dirs:
            n = np.load(d / "feats.npy", mmap_mode="r").shape[0] - 1
            per_seq.append([1.0] + list(scores[k : k + n]))
            k += n
        grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        thr_j, j, curve = gate_by_j(per_seq, dirs, cfg.baseline, device, grid)
        out.update(vis_gate=thr_j, mean_J=j, curve=curve)
    default_name = "gate.json" if args.objective == "balanced" else "gate_j.json"
    target = Path(args.out) if args.out else run / default_name
    target.write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
