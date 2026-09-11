"""End-to-end plumbing on a synthetic mini-DAVIS with a weight-free backbone."""

import json
import sys

import numpy as np
import pytest
import torch

from dvos import evaluate, extract, figures, train
from helpers import FakeBackbone, write_config, write_mini_davis


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("mini")
    davis = write_mini_davis(root / "DAVIS")
    cfg = write_config(root / "cfg.yaml", davis, root / "features")
    return dict(root=root, cfg=cfg)


def run(monkeypatch, module, argv):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    module.main()


def test_01_extract(pipeline, monkeypatch):
    monkeypatch.setattr(extract, "load_dinov3", lambda *a, **k: FakeBackbone().eval())
    for split in ("val", "train"):
        run(monkeypatch, extract, ["--config", str(pipeline["cfg"]), "--split", split])
    d = pipeline["root"] / "features" / "val" / "seqc" / "occ0"
    feats = np.load(d / "feats.npy")
    meta = json.loads((d / "meta.json").read_text())
    masks = np.load(d / "masks.npz")
    assert feats.shape == (8, 16, 6, 8) and feats.dtype == np.float16
    assert meta["target_id"] == 1 and meta["episode"][0] >= 2
    assert masks["visible"].shape == (8, 96, 128)
    s, e = meta["episode"]
    assert max(meta["fraction"][s:e]) > 0.5
    assert not (masks["visible"] & ~masks["full"]).any()
    clean = json.loads((d.parent / "clean" / "meta.json").read_text())
    assert clean["episode"] is None and max(clean["fraction"]) == 0.0


def test_02_extract_is_idempotent_and_seeded(pipeline, monkeypatch):
    d = pipeline["root"] / "features" / "val" / "seqc" / "occ0"
    before = (d / "meta.json").read_text()
    monkeypatch.setattr(extract, "load_dinov3", lambda *a, **k: FakeBackbone().eval())
    run(monkeypatch, extract, ["--config", str(pipeline["cfg"]), "--split", "val"])
    assert (d / "meta.json").read_text() == before
    assert extract.seq_seed("bear", 0) != extract.seq_seed("bear", 1) != extract.seq_seed("bmx", 1)


def test_02b_stride_and_variant_override(pipeline, monkeypatch, tmp_path):
    cfg = write_config(tmp_path / "cfg.yaml", pipeline["root"] / "DAVIS", tmp_path / "features")
    monkeypatch.setattr(extract, "load_dinov3", lambda *a, **k: FakeBackbone().eval())
    run(
        monkeypatch,
        extract,
        ["--config", str(cfg), "--split", "val", "--stride", "2", "--variants", "clean"],
    )
    d = tmp_path / "features" / "val" / "seqc"
    assert sorted(p.name for p in d.iterdir()) == ["clean"]
    meta = json.loads((d / "clean" / "meta.json").read_text())
    assert meta["stride"] == 2 and meta["n_frames"] == 4
    assert np.load(d / "clean" / "feats.npy").shape[0] == 4


def test_03_baseline(pipeline, monkeypatch):
    out = pipeline["root"] / "runs" / "baseline"
    run(
        monkeypatch,
        evaluate,
        [
            "--config",
            str(pipeline["cfg"]),
            "--method",
            "baseline",
            "--variant",
            "occ0",
            "--out",
            str(out),
            "--save-masks",
        ],
    )
    m = json.loads((out / "metrics.json").read_text())
    s = m["summary"]
    assert s["n_seq"] == 1 and 0.0 <= s["J"] <= 1.0 and 0.0 <= s["F"] <= 1.0
    assert s["n_episodes"] == 1 and s["vis_gate"] is None
    assert (out / "masks" / "seqc.npz").exists() and (out / "summary.md").exists()


def test_04_train_and_eval_model(pipeline, monkeypatch):
    out = pipeline["root"] / "runs" / "head"
    run(
        monkeypatch,
        train,
        ["--config", str(pipeline["cfg"]), "--out", str(out), "--variants", "clean", "occ0"],
    )
    ck = torch.load(out / "model.pt", map_location="cpu")
    assert ck["c_in"] == 16 and "kv.key.weight" in ck["model"] and 0.0 <= ck["val_J"] <= 1.0
    assert (out / "last.pt").exists()
    log = (out / "log.csv").read_text().splitlines()
    assert log[0] == "epoch,loss,loss_mask,loss_vis,teacher_forcing,val_J" and len(log) == 2
    ev = pipeline["root"] / "runs" / "head_occ0"
    run(
        monkeypatch,
        evaluate,
        [
            "--config",
            str(pipeline["cfg"]),
            "--method",
            "model",
            "--checkpoint",
            str(out / "model.pt"),
            "--variant",
            "occ0",
            "--out",
            str(ev),
            "--save-masks",
        ],
    )
    s = json.loads((ev / "metrics.json").read_text())["summary"]
    assert s["vis_gate"] == 0.5 and s["keep_first"] is True and 0.0 <= s["JF"] <= 1.0
    ev2 = pipeline["root"] / "runs" / "head_occ0_fifo"
    run(
        monkeypatch,
        evaluate,
        [
            "--config",
            str(pipeline["cfg"]),
            "--method",
            "model",
            "--checkpoint",
            str(out / "model.pt"),
            "--variant",
            "occ0",
            "--out",
            str(ev2),
            "--vis-gate",
            "0",
            "--no-keep-first",
        ],
    )
    s2 = json.loads((ev2 / "metrics.json").read_text())["summary"]
    assert s2["vis_gate"] == 0.0 and s2["keep_first"] is False


def test_04b_gated_baseline_and_gate_by_j(pipeline, monkeypatch):
    from dvos import calibrate

    out = pipeline["root"] / "runs" / "head"
    run(
        monkeypatch,
        calibrate,
        ["--config", str(pipeline["cfg"]), "--run", str(out), "--objective", "j"],
    )
    g = json.loads((out / "gate_j.json").read_text())
    assert 0.0 <= g["vis_gate"] <= 1.0 and len(g["curve"]) == 10
    ev = pipeline["root"] / "runs" / "gated_occ0"
    run(
        monkeypatch,
        evaluate,
        [
            "--config",
            str(pipeline["cfg"]),
            "--method",
            "gated_baseline",
            "--checkpoint",
            str(out / "model.pt"),
            "--variant",
            "occ0",
            "--out",
            str(ev),
            "--vis-gate",
            str(g["vis_gate"]),
        ],
    )
    s = json.loads((ev / "metrics.json").read_text())["summary"]
    assert s["method"] == "gated_baseline" and 0.0 <= s["JF"] <= 1.0


def test_05_figure(pipeline, monkeypatch):
    png = pipeline["root"] / "runs" / "qual.png"
    run(
        monkeypatch,
        figures,
        [
            "--config",
            str(pipeline["cfg"]),
            "--run",
            str(pipeline["root"] / "runs" / "head_occ0"),
            "--out",
            str(png),
            "--width",
            "128",
        ],
    )
    assert png.exists() and png.stat().st_size > 2000


def test_06_train_refuses_without_features(pipeline, monkeypatch, tmp_path):
    cfg = write_config(tmp_path / "cfg.yaml", pipeline["root"] / "DAVIS", tmp_path / "nofeatures")
    with pytest.raises(SystemExit):
        run(monkeypatch, train, ["--config", str(cfg), "--out", str(tmp_path / "run")])
