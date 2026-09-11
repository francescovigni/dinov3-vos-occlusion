import numpy as np

from dvos.calibrate import best_threshold


def test_best_threshold_separates_hidden_from_visible():
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    hidden = np.array([True, True, True, False, False, False])
    thr, ba = best_threshold(scores, hidden)
    assert 0.3 < thr <= 0.8 and ba == 1.0
    assert best_threshold(scores, np.zeros(6, bool)) == (0.5, float("nan")) or np.isnan(
        best_threshold(scores, np.zeros(6, bool))[1]
    )
