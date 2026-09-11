import numpy as np

from dvos.occlusion import OccluderBank, convex_fill, occlude_sequence, occlusion_schedule, paste


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


def test_convex_fill_covers_silhouette():
    alpha = np.zeros((20, 20), bool)
    alpha[2:18, 9:11] = True  # thin vertical bar
    alpha[9:11, 2:18] = True  # thin horizontal bar -> a cross
    hull = convex_fill(alpha)
    assert hull[alpha].all() and hull.sum() > 2 * alpha.sum()
    assert convex_fill(np.zeros((5, 5), bool)).sum() == 0


def test_hull_bank_alpha_contains_silhouette_and_hides_target():
    images, masks = scene()
    cross = np.zeros_like(masks[0])
    cross[20:80, 60:64] = 1
    cross[48:52, 40:84] = 1
    img = images[0].copy()
    img[cross == 1] = 255
    raw = OccluderBank.from_masks([img], [cross], min_side=4, fill_hull=False).items[0][1]
    hull = OccluderBank.from_masks([img], [cross], min_side=4, fill_hull=True).items[0][1]
    assert hull[raw].all() and hull.sum() > 2 * raw.sum()
    bank = OccluderBank.from_masks([img], [cross], min_side=4, fill_hull=True)
    d = occlude_sequence(
        images,
        [m == 1 for m in masks],
        bank,
        np.random.default_rng(1),
        min_len=3,
        max_len=3,
        scale=(1.5, 1.5),
        jitter=0.0,
    )
    s, e = d["episode"]
    assert min(d["fraction"][s:e]) >= 0.9
