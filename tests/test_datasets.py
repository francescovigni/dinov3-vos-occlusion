from dvos.datasets import annotation_gaps, box_mask


def test_box_mask_fills_and_clips():
    m = box_mask((10, 12), [(2, 3, 6, 7), (-5, 8, 4, 40)])
    assert m[3:7, 2:6].min() == 1 and m[8:, :4].min() == 2 and m.max() == 2
    assert m[0, 0] == 0 and m.shape == (10, 12)


def test_annotation_gaps():
    assert annotation_gaps([False, True, True, False, False, True, False]) == [(3, 5)]
    assert annotation_gaps([True, True]) == [] and annotation_gaps([False, False]) == []
