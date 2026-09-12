import numpy as np

from dvos.features import load_feats, n_channels, n_frames, save_feats


def test_quantised_roundtrip_is_close(tmp_path):
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(5, 8, 6, 7)).astype(np.float32) * 3.0
    save_feats(tmp_path, feats, quantize=True)
    back = load_feats(tmp_path)
    assert back.shape == feats.shape and back.dtype == np.float32
    rel = np.abs(back - feats).max() / np.abs(feats).max()
    assert rel < 0.01
    assert n_frames(tmp_path) == 5 and n_channels(tmp_path) == 8
    assert load_feats(tmp_path, [1, 3]).shape == (2, 8, 6, 7)


def test_fp16_path_still_works(tmp_path):
    feats = np.ones((3, 4, 2, 2), np.float32)
    save_feats(tmp_path, feats, quantize=False)
    assert (tmp_path / "feats.npy").exists()
    assert load_feats(tmp_path, slice(0, 2)).shape == (2, 4, 2, 2) and n_frames(tmp_path) == 3
