import numpy as np
from PIL import Image

from dvos.occlusion import OccluderBank
from dvos.polyp import image_mask_pairs


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
