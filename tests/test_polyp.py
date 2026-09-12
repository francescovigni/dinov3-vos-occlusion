import numpy as np
from PIL import Image

from dvos.occlusion import OccluderBank
from dvos.polyp import LDPolypVideo, associate_target, image_mask_pairs, read_boxes


def test_image_mask_pairs_and_bank(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    for i in range(3):
        img = np.full((40, 60, 3), 90, np.uint8)
        m = np.zeros((40, 60), np.uint8)
        m[10:30, 15 + 5 * i : 40 + 5 * i] = 255
        Image.fromarray(img).save(tmp_path / "images" / f"f{i}.jpg")
        Image.fromarray(m).save(tmp_path / "masks" / f"f{i}.png")
    Image.fromarray(np.zeros((40, 60, 3), np.uint8)).save(tmp_path / "images" / "orphan.jpg")
    images, masks = image_mask_pairs(tmp_path)
    assert len(images) == 3 and masks[0].dtype == np.uint8 and masks[0].max() == 1
    bank = OccluderBank.from_masks(images, masks, min_side=8)
    assert len(bank.items) == 3


def test_read_boxes_and_association(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("2\n10 10 30 30\n50 50 90 90\n")
    assert read_boxes(f) == [(10, 10, 30, 30), (50, 50, 90, 90)]
    (tmp_path / "e.txt").write_text("0\n")
    assert read_boxes(tmp_path / "e.txt") == []
    frames = [
        [(10, 10, 30, 30), (50, 50, 90, 90)],  # largest wins on frame 0
        [(12, 12, 32, 32)],  # nothing overlaps the big one -> nearest centre
        [],  # gap
        [(52, 52, 92, 92), (200, 200, 210, 210)],  # re-acquire by IoU with last known
    ]
    assert associate_target(frames) == [(50, 50, 90, 90), (12, 12, 32, 32), None, (52, 52, 92, 92)]


def test_ldpolypvideo_adapter(tmp_path):
    root = tmp_path / "TrainValid"
    for t in range(4):
        (root / "Images" / "7").mkdir(parents=True, exist_ok=True)
        (root / "Annotations" / "7").mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((48, 56, 3), np.uint8)).save(
            root / "Images" / "7" / f"{t:04d}.jpg"
        )
        (root / "Annotations" / "7" / f"{t:04d}.txt").write_text(
            "1\n10 10 30 30\n" if t != 2 else "0\n"
        )
    ds = LDPolypVideo(tmp_path, "train")
    assert ds.sequences == ["7"] and ds.gt_kind == "box"
    images, masks = ds.load("7")
    assert len(images) == 4 and masks[0][10:30, 10:30].min() == 1 and masks[2].max() == 0
