import numpy as np

from dvos.occlusion import OccluderBank, occlude_sequence, occlusion_schedule, paste


def scene(n=10, h=96, w=128):
    images, masks = [], []
    for t in range(n):
        img = np.full((h, w, 3), 40, np.uint8)
        m = np.zeros((h, w), np.uint8)
        x0 = 20 + 4 * t
        img[30:60, x0 : x0 + 30] = 200
        m[30:60, x0 : x0 + 30] = 1
        images.append(img)
        masks.append(m)
    return images, masks


def test_schedule_never_touches_frame_zero(rng):
    for _ in range(50):
        s, e = occlusion_schedule(20, rng, min_len=4, max_len=12, start_min=2)
        assert 2 <= s < e <= 20 and 4 <= e - s <= 12


def test_paste_clips_to_image():
    img = np.zeros((32, 32, 3), np.uint8)
    occ = np.full((10, 10, 3), 255, np.uint8)
    alpha = np.ones((10, 10), bool)
    out, m = paste(img, occ, alpha, center=(2.0, 2.0), size=(10, 10))
    assert m.shape == (32, 32) and m.sum() == 7 * 7
    assert out[m].min() == 255 and out[~m].max() == 0


def test_occlude_sequence_ground_truth(rng):
    images, masks = scene()
    bank = OccluderBank.from_masks([images[0]], [masks[0]], min_side=8)
    d = occlude_sequence(
        images, [m == 1 for m in masks], bank, rng, min_len=3, max_len=5, scale=(1.0, 1.0)
    )
    s, e = d["episode"]
    assert s >= 2
    for t in range(len(images)):
        vis, full, occ = d["visible"][t], d["full"][t], d["occluder"][t]
        assert not (vis & ~full).any(), "visible must be a subset of full"
        assert not (vis & occ).any(), "visible must not overlap the occluder"
        assert 0.0 <= d["fraction"][t] <= 1.0
        if not (s <= t < e):
            assert np.array_equal(d["images"][t], images[t]) and not occ.any()
    assert any(d["fraction"][s:e] > 0.5), (
        "an occluder the size of the target should hide most of it"
    )
