import numpy as np

from dvos.metrics import (
    boundary_f,
    jaccard,
    leak_ratio,
    recovery_delay,
    sequence_jf,
    visibility_auc,
)


def square(h=64, w=64, y0=16, y1=48, x0=16, x1=48):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_jaccard_known_overlap():
    a, b = square(), square(x0=32, x1=64)
    assert abs(jaccard(a, b) - (16 * 32) / (32 * 32 + 16 * 32)) < 1e-9
    assert jaccard(np.zeros((4, 4), bool), np.zeros((4, 4), bool)) == 1.0
    assert jaccard(square(), np.zeros((64, 64), bool)) == 0.0


def test_boundary_f_identity_and_disjoint():
    assert boundary_f(square(), square()) == 1.0
    assert boundary_f(square(), np.zeros((64, 64), bool)) == 0.0
    shifted = square(x0=17, x1=49)
    assert 0.5 < boundary_f(square(), shifted) <= 1.0


def test_sequence_jf_skips_frame_zero():
    preds = [square(), square(), np.zeros((64, 64), bool)]
    gts = [square(), square(), square()]
    r = sequence_jf(preds, gts)
    assert np.isnan(r["J_per_frame"][0])
    assert r["J_per_frame"][1] == 1.0 and r["J_per_frame"][2] == 0.0
    assert abs(r["J"] - 0.5) < 1e-9


def test_recovery_delay():
    j = [float("nan"), 1.0, 0.0, 0.0, 0.2, 0.7, 0.9]
    assert recovery_delay(j, [(2, 4)]) == [1]
    assert recovery_delay(j, [(2, 7)]) == [None]


def test_leak_ratio():
    pred, occ = square(), square(x0=32, x1=64)
    assert abs(leak_ratio(pred, occ) - 0.5) < 1e-9
    assert leak_ratio(np.zeros((8, 8), bool), occ[:8, :8]) == 0.0


def test_visibility_auc():
    assert visibility_auc(np.array([0.1, 0.9, 0.8, 0.2]), np.array([0, 1, 1, 0])) == 1.0
    assert visibility_auc(np.array([0.9, 0.1]), np.array([0, 1])) == 0.0
    assert np.isnan(visibility_auc(np.array([0.5, 0.5]), np.array([1, 1])))
