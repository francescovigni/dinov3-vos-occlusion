import numpy as np

from dvos.davis import Davis, real_occlusion_episodes


def test_real_occlusion_episodes():
    m1 = np.ones((4, 4), np.uint8)
    m0 = np.zeros((4, 4), np.uint8)
    masks = [m0, m1, m1, m0, m0, m1, m0, m1, m0]
    assert real_occlusion_episodes(masks, 1) == [(3, 5), (6, 7)]
    assert real_occlusion_episodes([m0, m0], 1) == []
    assert Davis.object_ids(np.array([[0, 2], [3, 0]], np.uint8)) == [2, 3]
