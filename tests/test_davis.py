import numpy as np

from dvos.davis import Davis, real_occlusion_episodes


def test_real_occlusion_episodes():
    m1 = np.ones((4, 4), np.uint8)
    m0 = np.zeros((4, 4), np.uint8)
    masks = [m0, m1, m1, m0, m0, m1, m0, m1, m0]
    assert real_occlusion_episodes(masks, 1) == [(3, 5), (6, 7)]
    assert real_occlusion_episodes([m0, m0], 1) == []
    assert Davis.object_ids(np.array([[0, 2], [3, 0]], np.uint8)) == [2, 3]


def test_primary_object_is_largest():
    m = np.zeros((10, 10), np.uint8)
    m[0, 0] = 1
    m[2:8, 2:8] = 3
    m[9, 5:8] = 2
    assert Davis.primary_object(m) == 3
    assert Davis.primary_object(np.zeros((4, 4), np.uint8)) is None
