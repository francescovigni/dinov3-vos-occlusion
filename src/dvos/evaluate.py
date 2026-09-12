"""Evaluate the zero-shot baseline or a trained head on cached features.

Single target per sequence (the largest object on frame 0). Writes ``metrics.json`` and ``summary.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dvos.backbone import pick_device
from dvos.config import load_config
from dvos.features import has_feats, load_feats
from dvos.metrics import (
    hidden_frames,
    leak_ratio,
    mask_to_box,
    recovery_delay,
    sequence_jf,
    visibility_auc,
)
from dvos.model import MemoryVOS, build_model, track
from dvos.propagate import propagate


def upsample(prob: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    return F.interpolate(prob, size=size, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def run_baseline(
    feats: torch.Tensor, first: np.ndarray, size, bcfg, hidden: list[bool] | None = None
) -> tuple[list[np.ndarray], list[float]]:
    T, _, h, w = feats.shape
    fg = F.interpolate(torch.from_numpy(first).float()[None, None], size=(h, w), mode="area")[0, 0]
    labels = torch.stack([1 - fg, fg]).to(feats.device)
    soft = propagate(
        list(feats),
        labels,
        n_last=bcfg.n_last,
        topk=bcfg.topk,
        radius=bcfg.radius,
        temperature=bcfg.temperature,
        hidden=hidden,
    )
    preds, vis = [], []
    for lab in soft:
        up = F.interpolate(lab[None], size=size, mode="bilinear", align_corners=False)[0]
        preds.append((up.argmax(0) == 1).cpu().numpy())
        vis.append(float(lab[1].max().item()))
    return preds, vis


def run_model(
    model: MemoryVOS,
    feats: torch.Tensor,
    first: np.ndarray,
    size,
    mcfg,
    *,
    vis_gate: float,
    keep_first: bool,
) -> tuple[list[np.ndarray], list[float]]:
    _, _, h, w = feats.shape
    first_t = F.interpolate(
        torch.from_numpy(first).float()[None, None], size=(4 * h, 4 * w), mode="area"
    ).to(feats.device)
    probs, vis = track(
        model,
        feats,
        (first_t > 0.5).float(),
        memory_max=mcfg.memory_max,
        vis_gate=vis_gate,
        keep_first=keep_first,
    )
    preds = [upsample(p, size) > 0.5 for p in probs]
    return preds, vis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--method", choices=["baseline", "model", "gated_baseline"], default="baseline")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--variant", default="clean")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--vis-gate", type=float, default=None, help="override model.vis_gate (0 = ungated)"
    )
    ap.add_argument("--no-keep-first", action="store_true", help="pure FIFO memory (ablation)")
    ap.add_argument(
        "--save-masks", action="store_true", help="store predictions under <out>/masks/"
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = pick_device(cfg.backbone.device)
    root = Path(cfg.data.features_root) / args.split
    dirs = sorted(d for d in root.glob(f"*/{args.variant}") if has_feats(d))
    if args.seqs:
        dirs = [d for d in dirs if d.parent.name in args.seqs]
    if not dirs:
        raise SystemExit(f"no cached features under {root} for variant {args.variant}")
    model = None
    if args.method in ("model", "gated_baseline"):
        ck = torch.load(args.checkpoint, map_location="cpu")
        c_in = int(ck.get("c_in") or ck["model"]["kv.key.weight"].shape[1])
        model_cfg = ck.get("config", {}).get("model", None) or cfg.model
        model = build_model(c_in, model_cfg)
        model.load_state_dict(ck["model"])
        model.eval().to(device)
    gate = cfg.model.vis_gate if args.vis_gate is None else args.vis_gate
    keep_first = not args.no_keep_first
    rows = []
    for d in dirs:
        meta = json.loads((d / "meta.json").read_text())
        feats = torch.from_numpy(load_feats(d)).to(device)
        masks = np.load(d / "masks.npz")
        visible, occluder = masks["visible"].astype(bool), masks["occluder"].astype(bool)
        size = tuple(meta["image_size"])
        if args.method == "baseline":
            preds, vis = run_baseline(feats, visible[0], size, cfg.baseline)
        elif args.method == "gated_baseline":
            # the head decides when the object is hidden; zero-shot propagation does the masks
            _, vis = run_model(
                model, feats, visible[0], size, cfg.model, vis_gate=gate, keep_first=keep_first
            )
            preds, _ = run_baseline(
                feats, visible[0], size, cfg.baseline, hidden=[v < gate for v in vis]
            )
        else:
            preds, vis = run_model(
                model, feats, visible[0], size, cfg.model, vis_gate=gate, keep_first=keep_first
            )
        if args.save_masks:
            mdir = Path(args.out) / "masks"
            mdir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                mdir / f"{meta['seq']}.npz",
                pred=np.stack(preds).astype(np.uint8),
                vis=np.array(vis),
            )
        if meta.get("gt_kind") == "box":
            # box-level ground truth: score the predicted mask's bounding box (box IoU)
            preds = [mask_to_box(p) for p in preds]
        jf = sequence_jf(preds, list(visible))
        episodes = [tuple(meta["episode"])] if meta["episode"] else []
        episodes += [tuple(e) for e in meta["real_episodes"]]
        delays = recovery_delay(jf["J_per_frame"], episodes)
        occ_frames = [t for s, e in episodes for t in range(s, e)]
        leak = (
            float(np.mean([leak_ratio(preds[t], occluder[t]) for t in occ_frames]))
            if occ_frames
            else float("nan")
        )
        hidden = hidden_frames(visible, meta.get("fraction"), cfg.occlusion.hidden_fraction)
        auc = visibility_auc(np.array(vis[1:]), ~hidden[1:])
        rows.append(
            dict(
                seq=meta["seq"],
                J=jf["J"],
                F=jf["F"],
                JF=jf["JF"],
                leak=leak,
                vis_auc=auc,
                recovery=[(-1 if x is None else x) for x in delays],
                episodes=episodes,
            )
        )

    def nanmean(k):
        v = [r[k] for r in rows if not (isinstance(r[k], float) and np.isnan(r[k]))]
        return float(np.mean(v)) if v else float("nan")

    rec = [x for r in rows for x in r["recovery"]]
    summary = dict(
        method=args.method,
        variant=args.variant,
        split=args.split,
        checkpoint=args.checkpoint,
        vis_gate=None if args.method == "baseline" else gate,
        keep_first=None if args.method in ("baseline", "gated_baseline") else keep_first,
        n_seq=len(rows),
        J=nanmean("J"),
        F=nanmean("F"),
        JF=nanmean("JF"),
        leak=nanmean("leak"),
        vis_auc=nanmean("vis_auc"),
        recovery_median=(
            float(np.median([x for x in rec if x >= 0]))
            if any(x >= 0 for x in rec)
            else float("nan")
        ),
        never_recovered=sum(1 for x in rec if x < 0),
        n_episodes=len(rec),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(dict(summary=summary, sequences=rows), indent=1))
    lines = [
        f"# {args.method} · {args.variant} · {args.split} ({len(rows)} sequences)",
        "",
        "| J | F | J&F | leak | vis AUC | recovery median | never recovered / episodes |",
        "|---|---|---|---|---|---|---|",
        f"| {summary['J']:.3f} | {summary['F']:.3f} | {summary['JF']:.3f} | {summary['leak']:.3f} | {summary['vis_auc']:.3f} | {summary['recovery_median']} | {summary['never_recovered']} / {summary['n_episodes']} |",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
