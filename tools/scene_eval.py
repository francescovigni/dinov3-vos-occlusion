"""Score a real camera scene against the occlusion metrics, with no hand labelling.

Real frames have no ground truth and labelling a hundred masks by hand is not worth it.
The way out is in the scene design: make the **target** the thing that is trivially
segmentable — a saturated ball — and occlude it with something that is not. Then

    GT[t]       = largest connected blob in the target's hue window, every frame
    hidden[t]   = that blob falls below `hidden_frac` of its median visible area
    J[t]        = IoU of the propagated mask against GT[t], exact
    ghost[t]    = predicted area while hidden, over the target's median area

all come out automatically, on real optics with real auto-exposure, **and the target is
allowed to move** — which a static-rig design would have had to forbid. The occluder is
whatever was to hand, literally; it is never thresholded, so nothing depends on it.

`ghost` replaces DAVIS's `leak_ratio` here. Leak needs an occluder mask; while the target
is hidden its GT is empty, so every predicted pixel is already a false positive and the
area of it says the same thing without inventing a skin detector.

    python tools/scene_eval.py --feats feats/occlusion_384x672 --scene scenes/occlusion \
        --out runs/scene_occlusion_384x672 --video
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dvos.metrics import jaccard, recovery_delay, visibility_auc
from dvos.model import build_model, track
from dvos.propagate import propagate
from dvos.video import write_mp4

# Saturated orange ball under indoor light. The wider window [0,110,90]-[30,255,255] also
# catches skin and pine, which is why this one is tight; both are reported in the write-up
# with the frame counts they produce, because a threshold nobody can see is unfalsifiable.
TARGET_LO = np.array([5, 120, 100], dtype=np.uint8)
TARGET_HI = np.array([25, 255, 255], dtype=np.uint8)


def target_mask(bgr: np.ndarray) -> np.ndarray:
    """Largest blob inside the hue window. Empty when the target is covered."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, TARGET_LO, TARGET_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    if n <= 1:
        return np.zeros(m.shape, dtype=bool)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == i


def contiguous_runs(flags: list[bool]) -> list[tuple[int, int]]:
    """Hidden frames -> occlusion episodes as inclusive (start, end) index pairs."""
    runs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def redetect(feats_t: torch.Tensor, exemplar: torch.Tensor, abs_thr: float, rel_thr: float):
    """Largest blob of patches matching the exemplar, or None if nothing matches well enough.

    The propagation vote cannot re-acquire a target smaller than its own ``topk``: frame 0 is
    pinned in memory but contributes a handful of positive patches against thousands of
    context patches, so it never wins. This bypasses the vote entirely — a global match
    against the frame-0 descriptors, which the features support long after the mask is gone.
    """
    c, h, w = feats_t.shape
    f = F.normalize(feats_t.reshape(c, -1), dim=0)
    sim = (exemplar.T @ f)[0]  # cosine against one mean prototype
    peak = float(sim.max())
    if peak < abs_thr:
        return None, peak
    cand = (sim >= rel_thr * peak).reshape(h, w).numpy().astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    if n <= 1:
        return None, peak
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return torch.from_numpy(lab == i), peak


def load_feats(d: Path) -> list[torch.Tensor]:
    paths = sorted(d.glob("feat_*.npy"))
    if not paths:
        raise SystemExit(f"no feat_*.npy in {d}")
    return [torch.from_numpy(np.load(p).astype(np.float32)) for p in paths]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", type=Path, required=True)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--hidden-frac",
        type=float,
        default=0.2,
        help="target area below this share of its median counts as hidden",
    )
    ap.add_argument("--n-last", type=int, default=7)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--radius", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--video", action="store_true")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="trained memory head; without it the zero-shot baseline runs",
    )
    ap.add_argument("--vis-gate", type=float, default=0.5)
    ap.add_argument(
        "--redetect",
        action="store_true",
        help="re-seed from a global exemplar match when the mask is lost",
    )
    ap.add_argument("--redetect-abs", type=float, default=0.75)
    ap.add_argument("--redetect-rel", type=float, default=0.90)
    ap.add_argument(
        "--redetect-gate",
        choices=["none", "oracle"],
        default="none",
        help="'oracle' suppresses re-detection on frames the ground truth calls hidden. "
        "Not deployable — it isolates whether a visibility signal is what the branch needs.",
    )
    ap.add_argument(
        "--save-masks",
        action="store_true",
        help="write predictions as a packed npz so figures come from data, not video",
    )
    args = ap.parse_args()

    feats = load_feats(args.feats)
    frame_paths = sorted(args.scene.glob("frame_*.png"))[: len(feats)]
    if len(frame_paths) < len(feats):
        raise SystemExit(f"{args.scene} has fewer frames than {args.feats}")
    frames = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in frame_paths]
    H, W = frames[0].shape[:2]
    _, h, w = feats[0].shape
    args.out.mkdir(parents=True, exist_ok=True)

    gts = [target_mask(f) for f in frames]
    areas = np.array([g.sum() for g in gts], dtype=float)
    median_area = float(np.median(areas[areas > 0])) if (areas > 0).any() else 1.0
    hidden = [bool(a < args.hidden_frac * median_area) for a in areas]
    if hidden[0]:
        raise SystemExit("frame 0 has no visible target; the first mask is the only input there is")

    # Frame-0 GT becomes the initial label, exactly as DAVIS gives the first mask.
    small = cv2.resize(gts[0].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(
        bool
    )
    first = torch.zeros(2, h, w)
    first[1][torch.from_numpy(small)] = 1.0
    first[0] = 1.0 - first[1]

    vis_scores = None
    n_redetect = 0
    if args.checkpoint is None:
        kw = dict(
            n_last=args.n_last,
            topk=args.topk,
            radius=args.radius,
            temperature=args.temperature,
        )
        preds_small = propagate(feats, first, **kw)
        if args.redetect:
            # Segment-wise: run the tested propagator, and wherever the mask dies, re-seed
            # from the exemplar and run it again on the remainder. The propagator itself is
            # untouched, so nothing here can silently change the baseline.
            c = feats[0].shape[0]
            ex = F.normalize(feats[0].reshape(c, -1)[:, torch.from_numpy(small.reshape(-1))], dim=0)
            t = 1
            while t < len(feats):
                # The mask dies in the argmax, not in the soft scores: after the vote
                # collapses, channel 1 stays small but nonzero everywhere.
                if int((preds_small[t].argmax(0) == 1).sum()) > 0:
                    t += 1
                    continue
                if args.redetect_gate == "oracle" and hidden[t]:
                    # Re-seeding while the object is genuinely behind something grabs the
                    # occluder. A deployed system needs a visibility signal here; this
                    # oracle only tests whether that is the missing component.
                    t += 1
                    continue
                seed, _peak = redetect(feats[t], ex, args.redetect_abs, args.redetect_rel)
                if seed is None:
                    t += 1
                    continue
                lab = torch.zeros(2, *seed.shape)
                lab[1][seed] = 1.0
                lab[0] = 1.0 - lab[1]
                tail = propagate(feats[t:], lab, **kw)
                preds_small[t:] = tail
                n_redetect += 1
                t += 1
        preds = []
        for p in preds_small:
            up = F.interpolate(p[None], size=(H, W), mode="bilinear", align_corners=False)[0]
            preds.append((up.argmax(0) == 1).numpy())
    else:
        # The trained memory head, driven exactly as dvos.evaluate drives it, so the only
        # thing differing from the DAVIS numbers is the video it is being shown.
        ck = torch.load(args.checkpoint, map_location="cpu")
        c_in = int(ck.get("c_in") or ck["model"]["kv.key.weight"].shape[1])
        model = build_model(c_in, ck.get("config", {}).get("model", None))
        model.load_state_dict(ck["model"])
        model.eval()
        stack = torch.stack(feats)
        first_mask = torch.from_numpy(
            cv2.resize(gts[0].astype(np.uint8), (4 * w, 4 * h), interpolation=cv2.INTER_NEAREST)
        ).float()[None, None]
        probs, vis_scores = track(model, stack, first_mask, vis_gate=args.vis_gate)
        preds = []
        for p in probs:
            up = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            preds.append((up > 0.5).numpy())

    rows, j_all, ghosts = [], [], []
    for t, (pred, gt) in enumerate(zip(preds, gts, strict=True)):
        j = float(jaccard(pred, gt)) if not hidden[t] else float("nan")
        ghost = float(pred.sum() / median_area) if hidden[t] else float("nan")
        j_all.append(0.0 if np.isnan(j) else j)
        if hidden[t]:
            ghosts.append(ghost)
        rows.append(
            {
                "frame": t,
                "J": None if np.isnan(j) else round(j, 4),
                "ghost_area": None if np.isnan(ghost) else round(ghost, 4),
                "gt_area_frac": round(float(areas[t] / median_area), 4),
                "hidden": hidden[t],
                "pred_px": int(pred.sum()),
            }
        )

    episodes = contiguous_runs(hidden)
    delays = recovery_delay(j_all, episodes) if episodes else []
    visible_j = [r["J"] for r in rows if r["J"] is not None]
    summary = {
        "scene": args.scene.name,
        "feats": args.feats.name,
        "frames": len(frames),
        "patch_grid": [h, w],
        "frame_size": [H, W],
        "target_hsv_window": [TARGET_LO.tolist(), TARGET_HI.tolist()],
        "hidden_frames": int(sum(hidden)),
        "episodes": episodes,
        "J_visible_mean": round(float(np.mean(visible_j)), 4) if visible_j else None,
        "J_before_first_episode": (
            round(float(np.mean([r["J"] for r in rows[: episodes[0][0]] if r["J"] is not None])), 4)
            if episodes
            else None
        ),
        "J_after_last_episode": (
            round(
                float(np.mean([r["J"] for r in rows[episodes[-1][1] + 1 :] if r["J"] is not None])),
                4,
            )
            if episodes and episodes[-1][1] + 1 < len(rows)
            else None
        ),
        "ghost_area_while_hidden": round(float(np.mean(ghosts)), 4) if ghosts else None,
        "method": ("baseline" if args.checkpoint is None else "memory_head")
        + ("+redetect" if args.redetect else "")
        + ("+oraclegate" if args.redetect and args.redetect_gate == "oracle" else ""),
        "redetections": n_redetect,
        "recovery_delay": delays,
        "never_recovered": sum(1 for d in delays if d is None),
    }
    if vis_scores is not None:
        # Does the head know it cannot see the ball? It scored 0.91 on synthetic DAVIS
        # occlusions; this asks the same question of a real hand.
        summary["visibility_auc"] = round(
            float(visibility_auc(1.0 - np.array(vis_scores), np.array(hidden))), 4
        )
    (args.out / "scene_metrics.json").write_text(
        json.dumps({"summary": summary, "frames": rows}, indent=1)
    )
    print(json.dumps(summary, indent=1))

    if args.save_masks:
        np.savez_compressed(
            args.out / "masks.npz",
            pred=np.packbits(np.stack(preds), axis=-1),
            gt=np.packbits(np.stack(gts), axis=-1),
            hidden=np.array(hidden),
            shape=np.array([H, W]),
        )

    if args.video:
        # dvos.video.write_mp4 takes RGB and encodes H.264; OpenCV's own mp4v stream plays
        # as a green screen on macOS, which is how the first version of this looked.
        overlay = []
        for frame, pred, gt, hid in zip(frames, preds, gts, hidden, strict=True):
            vis = frame.copy()
            vis[pred] = (0.45 * vis[pred] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
            vis[gt & ~pred] = (0, 255, 0)  # target the mask missed
            vis[pred & ~gt] = (0, 255, 255)  # mask with no target under it
            if hid:
                cv2.putText(vis, "hidden", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            overlay.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        path = args.out / "overlay.mp4"
        write_mp4(overlay, path, 8)
        print("wrote " + str(path))


if __name__ == "__main__":
    main()
