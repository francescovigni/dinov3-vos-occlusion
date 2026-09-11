from pathlib import Path

from dvos.config import feature_dir, load_config


def test_default_config_loads_and_is_consistent():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    assert cfg.data.size[0] % 16 == 0 and cfg.data.size[1] % 16 == 0
    assert "clean" in cfg.data.variants and all(
        v == "clean" or v.startswith("occ") for v in cfg.data.variants
    )
    assert 0.0 <= cfg.model.vis_gate <= 1.0 and cfg.model.memory_max >= 2
    assert cfg.train.teacher_forcing_end <= cfg.train.teacher_forcing_start
    assert (
        feature_dir(cfg, "val", "bear", "occ0")
        == Path(cfg.data.features_root) / "val" / "bear" / "occ0"
    )
