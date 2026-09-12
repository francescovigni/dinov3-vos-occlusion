"""Dataset adapters behind one interface, so extraction and evaluation do not care whether the
ground truth is a DAVIS palette mask, a binary polyp mask or a bounding box.

Every adapter exposes ``sequences`` and ``load(seq) -> (images, id_masks)`` where ``id_masks``
are uint8 HxW arrays with 0 = background and small integers for object ids. Box-level
datasets return the box filled as a mask; ``gt_kind`` records which it was, so J on a box
dataset is read as box IoU on filled rectangles, not as segmentation quality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from dvos.davis import Davis


class SequenceDataset(Protocol):
    name: str
    gt_kind: str  # "mask" | "box"
    sequences: list[str]

    def load(self, seq: str) -> tuple[list[np.ndarray], list[np.ndarray]]: ...


class DavisAdapter:
    name = "davis"
    gt_kind = "mask"

    def __init__(self, root: str | Path, split: str, year: str = "2017", resolution: str = "480p"):
        self.davis = Davis(root, split, year, resolution)
        self.sequences = self.davis.sequences

    def load(self, seq: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return self.davis.load(seq)


def read_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def box_mask(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """uint8 HxW with box i (x0, y0, x1, y1, exclusive) filled with id i+1; later boxes win."""
    m = np.zeros(shape, np.uint8)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(shape[1], int(x1)), min(shape[0], int(y1))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = i + 1
    return m


def annotation_gaps(present: list[bool]) -> list[tuple[int, int]]:
    """Episodes ``(start, end)`` where the object is absent between two frames where it is present."""
    p = np.asarray(present, bool)
    if not p.any():
        return []
    first, last = int(np.argmax(p)), len(p) - 1 - int(np.argmax(p[::-1]))
    out, t = [], first
    while t <= last:
        if not p[t]:
            s = t
            while t <= last and not p[t]:
                t += 1
            out.append((s, t))
        else:
            t += 1
    return out


def make_dataset(cfg, split: str) -> SequenceDataset:
    """Build the adapter named by ``cfg.data.dataset`` (default ``davis``)."""
    kind = getattr(cfg.data, "dataset", "davis")
    if kind == "davis":
        return DavisAdapter(cfg.data.davis_root, split, cfg.data.year, cfg.data.resolution)
    if kind == "ldpolypvideo":
        from dvos.polyp import LDPolypVideo

        return LDPolypVideo(cfg.data.root, split)
    if kind == "polypgen":
        from dvos.polyp import PolypGenSequences

        return PolypGenSequences(cfg.data.root, split)
    raise ValueError(f"unknown dataset {kind!r}")


def prepare_sequence(
    dataset: SequenceDataset,
    seq: str,
    stride: int = 1,
    max_frames: int | None = None,
    trim: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    """Load a sequence and apply the cache protocol: start at the first frame where any object
    is annotated (colonoscopy clips often begin before the polyp is in view), then keep every
    ``stride``-th frame, then cap at ``max_frames``. Returns images, id masks and the recipe
    (``start``, ``stride``, ``max_frames``) so figures and videos can reproduce the frames.
    """
    images, masks = dataset.load(seq)
    start = 0
    if trim:
        start = next((i for i, m in enumerate(masks) if (m > 0).any()), len(masks))
    images, masks = images[start:], masks[start:]
    if stride > 1:
        images, masks = images[::stride], masks[::stride]
    if max_frames:
        images, masks = images[:max_frames], masks[:max_frames]
    return images, masks, dict(start=start, stride=stride, max_frames=max_frames)
