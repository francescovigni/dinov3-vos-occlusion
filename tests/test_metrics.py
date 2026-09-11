import numpy as np
import pytest

from dvos.metrics import (
    boundary_f,
    hidden_frames,
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


def test_sequence_jf_scores_frames_1_to_t_minus_2():
    preds = [square(), square(), np.zeros((64, 64), bool), square()]
    gts = [square(), square(), square(), square()]
    r = sequence_jf(preds, gts)
    assert np.isnan(r["J_per_frame"][0])
    assert r["J_per_frame"][1:] == [1.0, 0.0, 1.0]  # every frame after the first
    assert abs(r["J"] - 0.5) < 1e-9  # frames 1 and 2 only: last frame excluded
    assert abs(r["F"] - 0.5) < 1e-9


def test_boundary_f_matches_reference_implementation():
    pytest.importorskip("davis2017", reason="optional GPL reference package not installed")
    from davis2017.metrics import db_eval_boundary

    rng = np.random.default_rng(0)
    for _ in range(20):
        h, w = 48, 80
        a = np.zeros((h, w), bool)
        b = np.zeros((h, w), bool)
        y0, x0 = rng.integers(0, 30), rng.integers(0, 50)
        a[y0 : y0 + 15, x0 : x0 + 25] = True
        b[y0 + rng.integers(-4, 5) : y0 + 16, x0 + rng.integers(-4, 5) : x0 + 27] = True
        b[rng.integers(0, h), rng.integers(0, w)] = True  # a stray pixel
        ours = boundary_f(a, b)
        ref = float(db_eval_boundary(b.astype(np.uint8), a.astype(np.uint8)))
        assert abs(ours - ref) < 1e-9, (ours, ref)


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


def test_hidden_frames_uses_fraction_or_empty_mask():
    vis = np.ones((4, 4, 4), bool)
    vis[3] = False
    frac = [0.0, 0.95, 0.5, 0.0]
    assert hidden_frames(vis, frac, thr=0.9).tolist() == [False, True, False, True]
    assert hidden_frames(vis, None).tolist() == [False, False, False, True]
