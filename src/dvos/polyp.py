"""Colonoscopy adapters: Kvasir-Instrument occluder bank, LDPolypVideo (boxes) and PolypGen
video sequences (masks). Formats are read from disk as distributed; nothing is redistributed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from dvos.datasets import box_mask, read_image


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


def read_boxes(path: Path) -> list[tuple[int, int, int, int]]:
    """LDPolypVideo annotation: first token = number of boxes, then ``x0 y0 x1 y1`` per box."""
    tok = path.read_text().split()
    n = int(tok[0]) if tok else 0
    vals = [int(float(t)) for t in tok[1 : 1 + 4 * n]]
    return [tuple(vals[i : i + 4]) for i in range(0, 4 * n, 4)]


def _iou(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])  # noqa: E731
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def associate_target(boxes_per_frame: list[list[tuple[int, int, int, int]]]) -> list[tuple | None]:
    """One target track through frames of unlabelled boxes.

    Frame 0: the largest box. Later frames: the box with the best IoU to the last known
    target box, falling back to the nearest centre when nothing overlaps (after a gap).
    Frames without boxes give ``None``. Boxes carry no identity in LDPolypVideo, so this
    is a heuristic; most videos contain a single polyp.
    """
    track: list[tuple | None] = []
    last = None
    for boxes in boxes_per_frame:
        if not boxes:
            track.append(None)
            continue
        if last is None:
            best = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        else:
            best = max(boxes, key=lambda b: _iou(b, last))
            if _iou(best, last) == 0.0:
                cx, cy = (last[0] + last[2]) / 2, (last[1] + last[3]) / 2
                best = min(
                    boxes,
                    key=lambda b: ((b[0] + b[2]) / 2 - cx) ** 2 + ((b[1] + b[3]) / 2 - cy) ** 2,
                )
        track.append(tuple(best))
        last = tuple(best)
    return track


class LDPolypVideo:
    """LDPolypVideo (Ma et al., MICCAI 2021): ``<root>/<Split>/Images/<video>/NNNN.jpg`` with
    ``Annotations/<video>/NNNN.txt``. Split names are ``TrainValid`` and ``Test``; ``train``
    and ``val`` map onto them. Target = largest box on frame 0, followed by IoU association."""

    name = "ldpolypvideo"
    gt_kind = "box"
    SPLITS = {"train": "TrainValid", "trainvalid": "TrainValid", "val": "Test", "test": "Test"}

    def __init__(self, root: str | Path, split: str):
        self.root = Path(root) / self.SPLITS.get(split.lower(), split)
        if not (self.root / "Images").is_dir():
            raise FileNotFoundError(
                f"{self.root}/Images missing; run scripts/download_polyp_drive.sh"
            )
        self.sequences = sorted(
            (p.name for p in (self.root / "Images").iterdir() if p.is_dir()), key=int
        )

    def frames(self, seq: str) -> list[Path]:
        return sorted((self.root / "Images" / seq).glob("*.jpg"))

    def boxes(self, seq: str) -> list[list[tuple[int, int, int, int]]]:
        return [
            read_boxes(self.root / "Annotations" / seq / (f.stem + ".txt"))
            for f in self.frames(seq)
        ]

    def load(self, seq: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
        frames = self.frames(seq)
        images = [read_image(f) for f in frames]
        track = associate_target(self.boxes(seq))
        shape = images[0].shape[:2]
        masks = [box_mask(shape, [b] if b is not None else []) for b in track]
        return images, masks


class PolypGenSequences:
    """PolypGen positive video sequences (Ali et al., Sci Data 2023):
    ``<root>/positive_cropped/seqN/{images,masks}/<frame>.jpg`` with binary JPEG masks.

    Small (23 sequences), so the split is by sequence id: ``val`` = every third sequence
    (seq3, seq6, ...), ``train`` = the rest, ``all`` = everything.
    """

    name = "polypgen"
    gt_kind = "mask"

    def __init__(self, root: str | Path, split: str):
        base = Path(root)
        base = base / "positive_cropped" if (base / "positive_cropped").is_dir() else base
        if not base.is_dir():
            raise FileNotFoundError(f"{base} missing; run scripts/download_polyp_drive.sh")
        self.root = base
        all_seqs = sorted(
            (p.name for p in base.glob("seq*") if p.is_dir()), key=lambda n: int(n[3:])
        )
        val = [s for s in all_seqs if int(s[3:]) % 3 == 0]
        self.sequences = {
            "val": val,
            "test": val,
            "train": [s for s in all_seqs if s not in val],
            "all": all_seqs,
        }[split.lower()]

    def frames(self, seq: str) -> list[Path]:
        return sorted((self.root / seq / "images").glob("*.jpg"), key=lambda p: int(p.stem))

    def load(self, seq: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
        images, masks = [], []
        for f in self.frames(seq):
            images.append(read_image(f))
            m = np.asarray(Image.open(self.root / seq / "masks" / f.name).convert("L"))
            masks.append((m > 127).astype(np.uint8))
        return images, masks
