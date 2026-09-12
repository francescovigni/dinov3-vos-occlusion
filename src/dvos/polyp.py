"""Colonoscopy adapters: Kvasir-Instrument occluder bank, LDPolypVideo (boxes) and PolypGen
video sequences (masks). Formats are read from disk as distributed; nothing is redistributed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from dvos.datasets import read_image


def image_mask_pairs(
    root: Path, max_items: int | None = None
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Kvasir-style folder: ``images/<name>.jpg`` + ``masks/<name>.png`` (binary). Masks come
    back as uint8 id masks with 1 = foreground, ready for ``OccluderBank.from_masks``."""
    root = Path(root)
    images_dir = next(p for p in [root / "images", *root.glob("*/images")] if p.is_dir())
    masks_dir = images_dir.parent / "masks"
    images, masks = [], []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        cands = list(masks_dir.glob(img_path.stem + ".*"))
        if not cands:
            continue
        m = np.asarray(Image.open(cands[0]).convert("L"))
        images.append(read_image(img_path))
        masks.append((m > 127).astype(np.uint8))
        if max_items and len(images) >= max_items:
            break
    if not images:
        raise FileNotFoundError(f"no image/mask pairs under {root}")
    return images, masks


class LDPolypVideo:
    """LDPolypVideo (Ma et al., MICCAI 2021): frames per video with one box file per frame."""

    name = "ldpolypvideo"
    gt_kind = "box"

    def __init__(self, root: str | Path, split: str):
        raise NotImplementedError("adapter written once the archive layout is on disk")


class PolypGenSequences:
    """PolypGen positive video sequences (Ali et al., Sci Data 2023) with per-frame masks."""

    name = "polypgen"
    gt_kind = "mask"

    def __init__(self, root: str | Path, split: str):
        raise NotImplementedError("adapter written once the archive layout is on disk")
