"""DAVIS 2017 reader and real-occlusion episode detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class Davis:
    def __init__(
        self, root: str | Path, split: str = "val", year: str = "2017", resolution: str = "480p"
    ):
        self.root = Path(root)
        self.split = split
        self.year = year
        self.resolution = resolution
        list_file = self.root / "ImageSets" / year / f"{split}.txt"
        if not list_file.exists():
            raise FileNotFoundError(f"{list_file} missing; run scripts/download_davis.sh")
        self.sequences = [s.strip() for s in list_file.read_text().splitlines() if s.strip()]

    def frames(self, seq: str) -> list[Path]:
        return sorted((self.root / "JPEGImages" / self.resolution / seq).glob("*.jpg"))

    def masks(self, seq: str) -> list[Path]:
        return sorted((self.root / "Annotations" / self.resolution / seq).glob("*.png"))

    @staticmethod
    def read_image(path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"))

    @staticmethod
    def read_mask(path: Path) -> np.ndarray:
        """Palette PNG -> HxW uint8 of object ids, 0 = background."""
        return np.asarray(Image.open(path)).astype(np.uint8)

    def load(self, seq: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
        imgs = [self.read_image(p) for p in self.frames(seq)]
        msks = [self.read_mask(p) for p in self.masks(seq)]
        if len(imgs) != len(msks):
            raise ValueError(f"{seq}: {len(imgs)} frames vs {len(msks)} masks")
        return imgs, msks

    @staticmethod
    def object_ids(mask0: np.ndarray) -> list[int]:
        return [int(i) for i in np.unique(mask0) if i != 0]


def real_occlusion_episodes(masks: list[np.ndarray], obj_id: int) -> list[tuple[int, int]]:
    """Frames where ``obj_id`` has zero area between two non-empty frames.

    Returns ``[(start, end)]`` with ``end`` exclusive. Leading/trailing absence is not an
    occlusion (the object never appeared, or left the scene and did not return).
    """
    present = np.array([(m == obj_id).any() for m in masks], dtype=bool)
    if not present.any():
        return []
    first, last = int(np.argmax(present)), len(present) - 1 - int(np.argmax(present[::-1]))
    episodes: list[tuple[int, int]] = []
    t = first
    while t <= last:
        if not present[t]:
            s = t
            while t <= last and not present[t]:
                t += 1
            episodes.append((s, t))
        else:
            t += 1
    return episodes
