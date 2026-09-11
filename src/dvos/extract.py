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

import numpy as np
import torch
from tqdm import tqdm

from dvos.backbone import extract_features, load_dinov3, preprocess
from dvos.config import feature_dir, load_config
from dvos.davis import Davis, real_occlusion_episodes
from dvos.occlusion import OccluderBank, occlude_sequence


def build_bank(cfg) -> OccluderBank:
    train = Davis(cfg.data.davis_root, "train", cfg.data.year, cfg.data.resolution)
    images, masks = [], []
    for seq in train.sequences:
        images.append(Davis.read_image(train.frames(seq)[0]))
        masks.append(Davis.read_mask(train.masks(seq)[0]))
    return OccluderBank.from_masks(images, masks)


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
    davis = Davis(cfg.data.davis_root, args.split, cfg.data.year, cfg.data.resolution)
    model = load_dinov3(
        cfg.backbone.name, cfg.backbone.repo, cfg.backbone.weights, cfg.backbone.device
    )
    size = tuple(cfg.data.size)
    variants = args.variants or list(cfg.data.variants)
    bank = build_bank(cfg) if any(v.startswith("occ") for v in variants) else None
    seqs = args.seqs or davis.sequences
    for seq in tqdm(seqs, desc=f"extract {args.split}"):
        imgs, msks = davis.load(seq)
        if args.stride > 1:
            imgs, msks = imgs[:: args.stride], msks[:: args.stride]
        if args.max_frames:
            imgs, msks = imgs[: args.max_frames], msks[: args.max_frames]
        ids = Davis.object_ids(msks[0])
        if not ids:
            continue
        target = ids[0]
        full = [(m == target) for m in msks]
        real = real_occlusion_episodes(msks, target)
        for variant in variants:
            out = feature_dir(cfg, args.split, seq, variant)
            if (out / "feats.npy").exists():
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
            np.save(out / "feats.npy", feats)
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
                stride=args.stride,
                input_size=list(size),
                n_frames=len(imgs),
            )
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
