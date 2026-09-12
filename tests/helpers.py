"""Shared helpers: a synthetic mini-DAVIS tree and a weight-free stand-in for DINOv3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def write_mini_davis(root: Path, n_frames: int = 8, size: tuple[int, int] = (96, 128)) -> Path:
    """Two train sequences, one val sequence; a red square moving right, a green static box."""
    h, w = size
    splits = {"train": ["seqa", "seqb"], "val": ["seqc"]}
    (root / "ImageSets" / "2017").mkdir(parents=True, exist_ok=True)
    for split, seqs in splits.items():
        (root / "ImageSets" / "2017" / f"{split}.txt").write_text("\n".join(seqs) + "\n")
        for si, seq in enumerate(seqs):
            (root / "JPEGImages" / "480p" / seq).mkdir(parents=True, exist_ok=True)
            (root / "Annotations" / "480p" / seq).mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(si)
            for t in range(n_frames):
                img = (rng.random((h, w, 3)) * 60 + 30).astype(np.uint8)
                m = np.zeros((h, w), np.uint8)
                x0 = 20 + 5 * t
                img[30:60, x0 : x0 + 30] = [220, 40, 40]
                m[30:60, x0 : x0 + 30] = 1
                img[70:90, 90:110] = [40, 220, 40]
                m[70:90, 90:110] = 2
                Image.fromarray(img).save(
                    root / "JPEGImages" / "480p" / seq / f"{t:05d}.jpg", quality=95
                )
                pm = Image.fromarray(m, mode="P")
                pm.putpalette([0, 0, 0, 128, 0, 0, 0, 128, 0] + [0] * (768 - 9))
                pm.save(root / "Annotations" / "480p" / seq / f"{t:05d}.png")
    return root


class FakeBackbone(nn.Module):
    """Patch-16 conv with the same ``get_intermediate_layers`` contract as DINOv3."""

    def __init__(self, channels: int = 16):
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(3, channels, kernel_size=16, stride=16)

    def get_intermediate_layers(self, x, *, n=1, reshape=True, norm=True):
        return [self.conv(x)]


def write_config(path: Path, davis_root: Path, features_root: Path) -> Path:
    path.write_text(
        f"""
data: {{davis_root: {davis_root}, features_root: {features_root}, year: "2017", resolution: 480p, size: [96, 128], variants: [clean, occ0], quantize: true}}
backbone: {{name: dinov3_vits16, repo: null, weights: null, device: cpu}}
occlusion: {{min_len: 2, max_len: 3, start_min: 2, scale: [1.2, 1.5], jitter: 0.05, hidden_fraction: 0.9}}
baseline: {{n_last: 3, topk: 5, radius: 4, temperature: 0.07}}
model: {{c_key: 16, c_value: 32, hidden: 32, memory_max: 4, vis_gate: 0.5, readout_topk: 16, pos_dim: 8, locality_radius: 2, use_prior: true, prior_topk: 3, prior_temperature: 0.1}}
train: {{epochs: 1, lr: 1.0e-3, clip_len: 5, gap_max: 2, clips_per_epoch: 3, holdout: 1, teacher_forcing_start: 1.0, teacher_forcing_end: 0.5, w_mask: 1.0, w_vis: 0.5, seed: 0}}
"""
    )
    return path
