import json
import runpy
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "episode_split.py"


def test_split_j_partitions_frames(tmp_path, monkeypatch):
    feats = tmp_path / "features" / "val" / "seqx" / "occ0"
    feats.mkdir(parents=True)
    T, H, W = 12, 8, 8
    gt = np.zeros((T, H, W), np.uint8)
    gt[:, 2:6, 2:6] = 1
    gt[4:7] = 0  # hidden during the episode (4, 7)
    np.savez_compressed(feats / "masks.npz", visible=gt, full=gt, occluder=np.zeros_like(gt))
    run = tmp_path / "runs" / "r"
    (run / "masks").mkdir(parents=True)
    pred = gt.copy()
    pred[8:] = 0  # wrong after the episode, right inside it (both empty -> J=1)
    np.savez_compressed(run / "masks" / "seqx.npz", pred=pred, vis=np.ones(T))
    (run / "metrics.json").write_text(
        json.dumps(
            dict(
                summary=dict(split="val", variant="occ0"),
                sequences=[dict(seq="seqx", episodes=[[4, 7]])],
            )
        )
    )
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.yaml").write_text(
        f"data: {{features_root: {tmp_path / 'features'}}}\n"
    )
    monkeypatch.chdir(tmp_path)
    ns = runpy.run_path(str(SCRIPT), run_name="not_main")
    r = ns["split_j"](run, ns["load_config"]("configs/default.yaml"), after=10)
    assert r["n_inside"] == 3 and r["n_post"] == 5 and r["n_outside"] == 3
    assert r["inside"] == 1.0 and r["outside"] == 1.0
    assert abs(r["post"] - 1 / 5) < 1e-9  # frame 7 right, frames 8..11 wrong
