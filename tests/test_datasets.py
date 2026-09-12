import numpy as np

from dvos.datasets import annotation_gaps, box_mask, prepare_sequence


def test_box_mask_fills_and_clips():
    m = box_mask((10, 12), [(2, 3, 6, 7), (-5, 8, 4, 40)])
    assert m[3:7, 2:6].min() == 1 and m[8:, :4].min() == 2 and m.max() == 2
    assert m[0, 0] == 0 and m.shape == (10, 12)


def test_annotation_gaps():
    assert annotation_gaps([False, True, True, False, False, True, False]) == [(3, 5)]
    assert annotation_gaps([True, True]) == [] and annotation_gaps([False, False]) == []


class _Fake:
    name, gt_kind, sequences = "fake", "mask", ["s"]

    def load(self, seq):
        masks = [np.zeros((4, 4), np.uint8) for _ in range(10)]
        for t in range(3, 10):
            masks[t][1:3, 1:3] = 1
        return [np.full((4, 4, 3), t, np.uint8) for t in range(10)], masks


def test_prepare_sequence_trims_strides_and_caps():
    imgs, masks, recipe = prepare_sequence(_Fake(), "s", stride=2, max_frames=3)
    assert recipe == dict(start=3, stride=2, max_frames=3)
    assert [int(i[0, 0, 0]) for i in imgs] == [3, 5, 7] and all(m.max() == 1 for m in masks)
    imgs, _, recipe = prepare_sequence(_Fake(), "s", trim=False)
    assert len(imgs) == 10 and recipe["start"] == 0
