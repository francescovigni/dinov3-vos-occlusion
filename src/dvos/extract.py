"""Cache DINOv3 features and occlusion variants for a DAVIS split.

Per ``<features_root>/<split>/<seq>/<variant>/``:
  feats.npy   (T, C, h, w) float16
  masks.npz   visible, full, occluder: (T, H, W) uint8 at original resolution
  meta.json   target_id, episode, fraction, real_episodes, image size, feature size
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dvos.backbone import extract_features, load_dinov3, preprocess
from dvos.config import feature_dir, load_config
from dvos.datasets import annotation_gaps, make_dataset, prepare_sequence
from dvos.davis import Davis
from dvos.features import has_feats, save_feats
from dvos.occlusion import OccluderBank, occlude_sequence


def build_bank(cfg) -> OccluderBank:
    """Occluders cut from the first frames of the *training* split, or from a directory of
    image/mask pairs when ``cfg.occlusion.bank`` names one (e.g. Kvasir-Instrument)."""
    fill = getattr(cfg.occlusion, "fill_hull", True)
    bank_dir = getattr(cfg.occlusion, "bank", None)
    if bank_dir:
        from dvos.polyp import image_mask_pairs

        images, masks = image_mask_pairs(Path(bank_dir))
        return OccluderBank.from_masks(images, masks, fill_hull=fill)
    train = make_dataset(cfg, "train")
    images, masks = [], []
    for seq in train.sequences:
        imgs, msks = train.load(seq)
        images.append(imgs[0])
        masks.append(msks[0])
    return OccluderBank.from_masks(images, masks, fill_hull=fill)


def seq_seed(seq: str, variant_seed: int) -> int:
    return variant_seed * 100_003 + zlib.crc32(seq.encode()) % 100_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--seqs", nargs="*", default=None, help="subset of sequences")
    ap.add_argument("--max-frames", type=int, default=None, help="debug: truncate sequences")
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="keep every k-th frame (train-time augmentation, saves disk)",
    )
    ap.add_argument("--variants", nargs="*", default=None, help="override data.variants")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dataset = make_dataset(cfg, args.split)
    model = load_dinov3(
        cfg.backbone.name, cfg.backbone.repo, cfg.backbone.weights, cfg.backbone.device
    )
    size = tuple(cfg.data.size)
    variants = args.variants or list(cfg.data.variants)
    bank = build_bank(cfg) if any(v.startswith("occ") for v in variants) else None
    seqs = args.seqs or dataset.sequences
    for seq in tqdm(seqs, desc=f"extract {args.split}"):
        imgs, msks, recipe = prepare_sequence(dataset, seq, args.stride, args.max_frames)
        if not imgs:
            continue
        ids = Davis.object_ids(msks[0])
        target = Davis.primary_object(msks[0])
        if target is None:
            continue
        full = [(m == target) for m in msks]
        real = annotation_gaps([bool((m == target).any()) for m in msks])
        for variant in variants:
            out = feature_dir(cfg, args.split, seq, variant)
            if has_feats(out):
                continue
            if variant == "clean":
                images, visible = imgs, full
                occl = [np.zeros_like(m) for m in full]
                fraction, episode = [0.0] * len(imgs), None
            else:
                rng = np.random.default_rng(seq_seed(seq, int(variant[3:])))
                o = cfg.occlusion
                d = occlude_sequence(
                    imgs,
                    full,
                    bank,
                    rng,
                    min_len=o.min_len,
                    max_len=o.max_len,
                    start_min=o.start_min,
                    scale=tuple(o.scale),
                    jitter=o.jitter,
                )
                images, visible, occl = d["images"], d["visible"], d["occluder"]
                fraction, episode = d["fraction"].tolist(), list(d["episode"])
            feats = []
            for img in images:
                feats.append(extract_features(model, preprocess(img, size)).cpu().half().numpy()[0])
            feats = np.stack(feats)
            out.mkdir(parents=True, exist_ok=True)
            save_feats(out, feats, bool(getattr(cfg.data, "quantize", False)))
            np.savez_compressed(
                out / "masks.npz",
                visible=np.stack(visible).astype(np.uint8),
                full=np.stack(full).astype(np.uint8),
                occluder=np.stack(occl).astype(np.uint8),
            )
            meta = dict(
                seq=seq,
                split=args.split,
                variant=variant,
                target_id=target,
                object_ids=ids,
                episode=episode,
                fraction=fraction,
                real_episodes=real,
                image_size=list(imgs[0].shape[:2]),
                feature_size=list(feats.shape[2:]),
                backbone=cfg.backbone.name,
                dataset=dataset.name,
                gt_kind=dataset.gt_kind,
                stride=args.stride,
                max_frames=args.max_frames,
                start_offset=recipe["start"],
                input_size=list(size),
                n_frames=len(imgs),
            )
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
