"""Track by fitting a known silhouette, so the occluded part is measured against the real object.

The encoder works on a 1/16 patch grid, so a propagated mask can never have a boundary better
than ±24 px, and GrabCut costs ~300 ms on a desktop core to buy it back. Neither is useful on a
Jetson Nano that already spends 666 ms on the backbone.

The way out is to stop segmenting per frame. A reference image of the object — the first frame
here, exactly what the VOS protocol hands you — is segmented **once**, where cost does not
matter, and that silhouette is pixel-accurate by construction. At runtime the object is only
*located*, by the same exemplar match the re-detection branch uses, and the silhouette is placed
there. Occlusion is then the share of the placed silhouette whose patches no longer look like
the object.

    borders   come from the reference, at full resolution, computed once
    position  comes from a patch-grid match, one matvec per frame
    occlusion comes from comparing patches inside the known shape against the exemplar

Limits, stated up front: this assumes a roughly rigid object whose appearance does not change
with pose. A deformable or self-occluding target breaks the fit and sends you back to per-frame
segmentation and its latency wall.

    python tools/silhouette_track.py --feats feats/occ_384_fp32 --scene scenes/occlusion \\
        --out runs/sil_384 --video
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dvos.metrics import boundary_f, jaccard
from dvos.video import write_mp4

TARGET_LO = np.array([5, 120, 100], dtype=np.uint8)
TARGET_HI = np.array([25, 255, 255], dtype=np.uint8)


def target_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, TARGET_LO, TARGET_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    if n <= 1:
        return np.zeros(m.shape, dtype=bool)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == i


def place(
    silhouette: np.ndarray, cy: float, cx: float, scale: float, ref_cy: float, ref_cx: float
) -> np.ndarray:
    """Reference silhouette scaled about its own centroid and moved so that centroid lands on (cy, cx)."""
    h, w = silhouette.shape
    m = np.array(
        [[scale, 0, cx - scale * ref_cx], [0, scale, cy - scale * ref_cy]], dtype=np.float32
    )
    out = cv2.warpAffine(
        silhouette.astype(np.uint8), m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
    )
    return out.astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", type=Path, required=True)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ref", type=int, default=0, help="frame of the scene used as reference")
    ap.add_argument(
        "--ref-image",
        type=Path,
        default=None,
        help="a separate photograph of the object, instead of a frame of the scene. Needs "
        "--ref-feat: the same encoder at the same input size must produce its descriptors, or "
        "they are not comparable.",
    )
    ap.add_argument("--ref-feat", type=Path, default=None, help="features for --ref-image")
    ap.add_argument(
        "--ref-mask",
        type=Path,
        default=None,
        help="PNG mask of the object in --ref-image; without it the hue threshold is used",
    )
    ap.add_argument(
        "--match-rel",
        type=float,
        default=0.86,
        help="a pixel counts as the object when its similarity reaches this fraction of the "
        "object's own similarity scale, measured once over --calib-frames at the start. Not "
        "per-frame: a threshold tracking each frame's peak falls as the object is covered, so "
        "the survivors still pass and the occlusion is measured as zero — which is exactly "
        "what the first partial-occlusion clip showed, bias -0.61 across every coverage band. "
        "Not absolute either: an external reference photo sits at ~0.66 similarity against "
        "~0.94 for an in-scene one, so a fixed number tuned in-scene matches nothing.",
    )
    ap.add_argument(
        "--locate-rel",
        type=float,
        default=0.95,
        help="patches within this fraction of the peak vote on the position",
    )
    ap.add_argument("--hidden-frac", type=float, default=0.2)
    ap.add_argument(
        "--full-area-pct",
        type=float,
        default=90.0,
        help="percentile of visible area taken as the object's unoccluded size",
    )
    ap.add_argument(
        "--scale-search",
        action="store_true",
        help="search the scale once on the first frame and freeze it, by placing the shape at "
        "each candidate and scoring the match inside it. Corrects a reference photographed at "
        "a different distance. Per-frame re-estimation is deliberately not offered: it injects "
        "more noise than it removes when the object's apparent size is nearly constant.",
    )
    ap.add_argument(
        "--ref-scale",
        type=float,
        default=1.0,
        help="resize the reference silhouette before use; simulates a reference image shot "
        "at a different distance",
    )
    ap.add_argument(
        "--calib-frames",
        type=int,
        default=10,
        help="frames at the start, assumed to show the object unoccluded, used to fix the "
        "similarity scale for the whole clip",
    )
    ap.add_argument("--scale-min", type=float, default=0.5)
    ap.add_argument("--scale-max", type=float, default=2.0)
    ap.add_argument("--scale-step", type=float, default=0.05)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--save-masks", action="store_true")
    args = ap.parse_args()

    feat_paths = sorted(args.feats.glob("feat_*.npy"))
    frame_paths = sorted(args.scene.glob("frame_*.png"))[: len(feat_paths)]
    frames = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in frame_paths]
    feats = [torch.from_numpy(np.load(p).astype(np.float32)) for p in feat_paths]
    H, W = frames[0].shape[:2]
    C, h, w = feats[0].shape
    args.out.mkdir(parents=True, exist_ok=True)

    # Ground truth for scoring only. The tracker never sees it after the reference frame.
    gts = [target_mask(f) for f in frames]
    areas = np.array([g.sum() for g in gts], float)
    # The object's unoccluded size, used to turn visible area into an occluded fraction.
    # The median is only right when most frames are unoccluded; in a clip shot to show
    # partial occlusion it is dragged down and every true_occluded value comes out too small.
    # A high percentile survives both cases.
    ref_area = float(np.percentile(areas[areas > 0], args.full_area_pct))
    true_occ = 1.0 - np.clip(areas / ref_area, 0, 1)
    hidden = [bool(a < args.hidden_frac * ref_area) for a in areas]

    # --- offline: the reference silhouette, at full pixel resolution, computed once ---
    if args.ref_image is not None:
        if args.ref_feat is None:
            raise SystemExit("--ref-image needs --ref-feat from the same encoder and input size")
        ref_img = cv2.imread(str(args.ref_image), cv2.IMREAD_COLOR)
        if ref_img is None:
            raise SystemExit(f"cannot read {args.ref_image}")
        ref_feat = torch.from_numpy(np.load(args.ref_feat).astype(np.float32))
        if ref_feat.shape[0] != C:
            raise SystemExit("reference features have a different channel count than the scene")
        if ref_img.shape[:2] != (H, W):
            ref_img = cv2.resize(ref_img, (W, H), interpolation=cv2.INTER_AREA)
        silhouette = (
            cv2.imread(str(args.ref_mask), cv2.IMREAD_GRAYSCALE) > 127
            if args.ref_mask
            else target_mask(ref_img)
        )
        if silhouette.shape != (H, W):
            silhouette = cv2.resize(
                silhouette.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        ref_source, ref_h, ref_w = str(args.ref_image), ref_feat.shape[1], ref_feat.shape[2]
    else:
        silhouette = gts[args.ref]
        ref_feat = feats[args.ref]
        ref_source, ref_h, ref_w = f"scene frame {args.ref}", h, w
    if silhouette.sum() == 0:
        raise SystemExit(f"the reference ({ref_source}) has no visible target")
    if (ref_h, ref_w) != (h, w):
        raise SystemExit(
            f"reference features are {ref_h}x{ref_w} but the scene is {h}x{w}; "
            "run the reference through the same engine"
        )

    ys, xs = np.nonzero(silhouette)
    ref_cy, ref_cx = ys.mean(), xs.mean()
    if args.ref_scale != 1.0:
        silhouette = place(silhouette, ref_cy, ref_cx, args.ref_scale, ref_cy, ref_cx)
    ref_small = cv2.resize(
        silhouette.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    f_ref = F.normalize(ref_feat.reshape(C, -1), dim=0)
    patches = f_ref[:, torch.from_numpy(ref_small.reshape(-1))]
    # One mean prototype, not a max over every exemplar patch. The max grows more permissive
    # as the exemplar gains patches, so at 480x864 (8 patches) it flagged twice the object's
    # area as a match and dragged the centroid 28 px off; the prototype is 4.1 px, and does
    # not depend on how many patches the reference happens to contain.
    exemplar = F.normalize(patches.mean(1, keepdim=True), dim=0)

    # The reference's own high-similarity extent is the yardstick for scale: the object may
    # simply be nearer or further in the video than it was in the reference image. Frame 0 of
    # this clip is 1.34x its own median size, and using it unscaled costs 0.05 J and four
    # times the occlusion error.
    sim_ref = (exemplar.T @ f_ref)[0].reshape(h, w).numpy()
    sim_ref_up = cv2.resize(sim_ref, (W, H), interpolation=cv2.INTER_LINEAR)
    ref_blob = float((sim_ref_up >= args.locate_rel * sim_ref_up.max()).sum())
    ref_patches = int(ref_small.sum())

    # --- runtime ---
    # The object's similarity scale, measured once while it is visible. Everything after is
    # judged against this, so covering the object actually lowers the match count.
    calib = []
    for feat in feats[: max(args.calib_frames, 1)]:
        fc = F.normalize(feat.reshape(C, -1), dim=0)
        calib.append(float((exemplar.T @ fc)[0].max()))
    calib_peak = float(np.median(calib))
    match_thr = args.match_rel * calib_peak

    rows, preds, placed_only, scale, t_total = [], [], [], 1.0, 0.0
    scale_fixed = False
    for t, feat in enumerate(feats):
        t0 = time.time()
        f = F.normalize(feat.reshape(C, -1), dim=0)
        sim = (exemplar.T @ f)[0].reshape(h, w).numpy()

        # The silhouette is pixel-accurate; intersecting it with a patch-resolution match
        # map would chop it back into 30 px blocks and throw that away. Upsample the
        # similarity first, so every comparison below happens at pixel resolution.
        sim_up = cv2.resize(sim, (W, H), interpolation=cv2.INTER_LINEAR)
        peak = float(sim_up.max())

        # A global weighted centroid is dragged off-target by scattered background matches,
        # and background similarity climbs with resolution: p99 is 0.67 at 384x672 but
        # 0.83-0.92 at 480x864. Taking the largest *connected* blob first imposes the spatial
        # coherence the raw similarity map lacks.
        vote = (sim_up >= args.locate_rel * peak).astype(np.uint8)
        if not vote.any():
            vote = (sim_up >= np.percentile(sim_up, 99.99)).astype(np.uint8)
        n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(vote, 8)
        if n_cc > 1:
            vote = lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
        else:
            vote = vote.astype(bool)
        vy, vx = np.nonzero(vote)
        wts = sim_up[vy, vx]
        cy = float((vy * wts).sum() / wts.sum())
        cx = float((vx * wts).sum() / wts.sum())

        if args.scale_search and not scale_fixed:
            # Search scale instead of regressing it. Every cue tried (blob area, soft area,
            # radial spread) correlated only r~0.6 with true scale and wobbled twice as much
            # as the scale itself, so a direct search on match quality is the honest method:
            # place the shape at each candidate and keep the one whose interior matches best.
            best = (-1.0, scale)
            for cand in np.arange(args.scale_min, args.scale_max + 1e-6, args.scale_step):
                trial = place(silhouette, cy, cx, float(cand), ref_cy, ref_cx)
                if trial.sum() < 50:
                    continue
                # Mean similarity inside the shape is degenerate: shrinking onto the peak
                # always wins. Summing (similarity - threshold) is a matched filter — it pays
                # for covering object pixels and is charged for covering background, so the
                # optimum sits at the true extent.
                score = float((sim_up[trial] - match_thr).sum())
                if score > best[0]:
                    best = (score, float(cand))
            scale = best[1]
            scale_fixed = True
        placed = place(silhouette, cy, cx, scale, ref_cy, ref_cx)

        # Inside the known shape, which pixels still look like the object?
        match = sim_up >= match_thr
        inside = int(placed.sum())
        matched = int((placed & match).sum())
        est_occ = 1.0 - matched / max(inside, 1)
        pred = placed & match
        placed_only.append(placed)
        t_total += time.time() - t0

        preds.append(pred)
        rows.append(
            {
                "frame": t,
                "peak_sim": round(peak, 4),
                "est_occluded": round(float(est_occ), 4),
                "true_occluded": round(float(true_occ[t]), 4),
                "hidden": hidden[t],
                "J": None if gts[t].sum() == 0 else round(float(jaccard(pred, gts[t])), 4),
                "scale": round(scale, 3),
            }
        )

    vis = [t for t in range(len(frames)) if not hidden[t] and gts[t].sum() > 0]
    js = [jaccard(preds[t], gts[t]) for t in vis]
    fs = [boundary_f(preds[t], gts[t]) for t in vis]
    scored = areas > 0
    est = np.array([r["est_occluded"] for r in rows])
    summary = {
        "scene": args.scene.name,
        "feats": args.feats.name,
        "method": "silhouette_fit",
        "frames": len(frames),
        "reference": ref_source,
        "ref_scale": args.ref_scale,
        "scale_used": round(scale, 3),
        "reference_patches": ref_patches,
        "match_rel": args.match_rel,
        "calib_peak": round(calib_peak, 4),
        "match_threshold": round(match_thr, 4),
        "J_visible_mean": round(float(np.mean(js)), 4),
        # Localisation on its own: the known shape placed where the match says, with no
        # occlusion masking. Separates "it is in the wrong place" from "the match is noisy".
        "J_placed_only": round(float(np.mean([jaccard(placed_only[t], gts[t]) for t in vis])), 4),
        "boundaryF_visible_mean": round(float(np.mean(fs)), 4),
        "occlusion_pearson_r": round(float(np.corrcoef(est[scored], true_occ[scored])[0, 1]), 4),
        "occlusion_mae": round(float(np.abs(est[scored] - true_occ[scored]).mean()), 4),
        "runtime_ms_per_frame": round(1000.0 * t_total / len(feats), 2),
    }
    (args.out / "silhouette_metrics.json").write_text(
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
        overlay = []
        for frame, pred, gt, row in zip(frames, preds, gts, rows, strict=True):
            v = frame.copy()
            v[pred] = (0.45 * v[pred] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
            v[gt & ~pred] = (0, 255, 0)
            cv2.putText(
                v,
                f"occluded est {row['est_occluded']:.2f}  true {row['true_occluded']:.2f}",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            overlay.append(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
        write_mp4(overlay, args.out / "overlay.mp4", 8)
        print("wrote " + str(args.out / "overlay.mp4"))


if __name__ == "__main__":
    main()
