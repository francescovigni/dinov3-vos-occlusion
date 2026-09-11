"""Tiny YAML config loader with attribute access."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml


def _ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_ns(v) for v in d]
    return d


def load_config(path: str | Path) -> SimpleNamespace:
    with open(path) as f:
        return _ns(yaml.safe_load(f))


def feature_dir(cfg, split: str, seq: str, variant: str) -> Path:
    return Path(cfg.data.features_root) / split / seq / variant
