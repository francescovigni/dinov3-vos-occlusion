"""Feature cache I/O: fp16 ``feats.npy`` or uint8 ``feats_u8.npz`` (per-tensor affine quantisation).

Quantisation halves the cache; DINOv3 patch tokens are layer-normed, so an 8-bit grid over
the tensor's own range keeps the relative error around 1e-2, well below what k-NN
propagation or the head notice (checked in tests).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_feats(dir_: Path, feats: np.ndarray, quantize: bool) -> Path:
    """``feats`` (T,C,h,w) float -> file on disk; returns the written path."""
    dir_ = Path(dir_)
    if not quantize:
        out = dir_ / "feats.npy"
        np.save(out, feats.astype(np.float16))
        return out
    f = feats.astype(np.float32)
    lo, hi = float(f.min()), float(f.max())
    scale = (hi - lo) / 255.0 if hi > lo else 1.0
    q = np.round((f - lo) / scale).clip(0, 255).astype(np.uint8)
    out = dir_ / "feats_u8.npz"
    np.savez(out, q=q, lo=np.float32(lo), scale=np.float32(scale))
    return out


def feats_path(dir_: Path) -> Path:
    dir_ = Path(dir_)
    for name in ("feats.npy", "feats_u8.npz"):
        if (dir_ / name).exists():
            return dir_ / name
    raise FileNotFoundError(f"no feature file under {dir_}")


def has_feats(dir_: Path) -> bool:
    dir_ = Path(dir_)
    return (dir_ / "feats.npy").exists() or (dir_ / "feats_u8.npz").exists()


def load_feats(dir_: Path, idx=None) -> np.ndarray:
    """(T,C,h,w) float32; ``idx`` selects frames (list/array/slice) before dequantising."""
    p = feats_path(dir_)
    if p.suffix == ".npy":
        a = np.load(p, mmap_mode="r")
        a = a[idx] if idx is not None else a[:]
        return np.array(a, dtype=np.float32, copy=True)
    z = np.load(p)
    q = z["q"][idx] if idx is not None else z["q"]
    return q.astype(np.float32) * float(z["scale"]) + float(z["lo"])


def n_frames(dir_: Path) -> int:
    p = feats_path(dir_)
    if p.suffix == ".npy":
        return int(np.load(p, mmap_mode="r").shape[0])
    with np.load(p) as z:
        return int(z["q"].shape[0])


def n_channels(dir_: Path) -> int:
    p = feats_path(dir_)
    if p.suffix == ".npy":
        return int(np.load(p, mmap_mode="r").shape[1])
    with np.load(p) as z:
        return int(z["q"].shape[1])
