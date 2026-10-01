"""The real-camera evaluator, on a scene whose answers are known by construction.

Camera scenes have no ground truth, so `tools/scene_eval.py` derives it from the target's
hue and calls a frame hidden when that blob disappears. Those two steps carry the whole
result, which is why they get a test: a synthetic ball that is visible, then covered for
exactly three frames, then visible again.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
H, W = 240, 320
HIDDEN = (5, 7)  # inclusive, by construction
N_FRAMES = 12


def _load_scene_eval():
    spec = importlib.util.spec_from_file_location("scene_eval", ROOT / "tools" / "scene_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_scene(tmp: Path) -> tuple[Path, Path]:
    """Orange ball on grey, covered by a grey card for frames 5-7, with matching features."""
    scene, feats = tmp / "scene", tmp / "feats"
    scene.mkdir(parents=True)
    feats.mkdir(parents=True)
    h, w = H // 16, W // 16
    rng = np.random.default_rng(0)
    ball = rng.normal(size=384)
    ball /= np.linalg.norm(ball)
    bg = rng.normal(size=384)
    bg /= np.linalg.norm(bg)

    for t in range(N_FRAMES):
        img = np.full((H, W, 3), 120, np.uint8)
        cx = 160 + 3 * t
        hidden = HIDDEN[0] <= t <= HIDDEN[1]
        if hidden:
            cv2.rectangle(img, (cx - 60, 40), (cx + 60, 200), (90, 90, 90), -1)
        else:
            cv2.circle(img, (cx, 120), 28, (30, 140, 245), -1)  # BGR orange
        cv2.imwrite(str(scene / f"frame_{t:04d}.png"), img)

        f = np.repeat(bg[:, None, None], h * w, axis=1).reshape(384, h, w).copy()
        if not hidden:
            py, px = 120 * h // H, cx * w // W
            f[:, max(py - 1, 0) : py + 2, max(px - 1, 0) : px + 2] = ball[:, None, None]
        f += 0.01 * rng.normal(size=f.shape)
        f /= np.linalg.norm(f, axis=0, keepdims=True)
        np.save(feats / f"feat_{t:04d}.npy", f.astype(np.float16))
    return scene, feats


def test_target_mask_finds_the_ball_and_nothing_else():
    mod = _load_scene_eval()
    img = np.full((H, W, 3), 120, np.uint8)
    cv2.circle(img, (160, 120), 28, (30, 140, 245), -1)
    m = mod.target_mask(img)
    assert m.sum() > 0
    # Within a few percent of a disc of radius 28, and centred on it.
    assert abs(m.sum() - np.pi * 28**2) / (np.pi * 28**2) < 0.1
    ys, xs = np.nonzero(m)
    assert abs(xs.mean() - 160) < 2 and abs(ys.mean() - 120) < 2
    # A frame with no orange in it yields nothing, which is what marks a frame hidden.
    assert mod.target_mask(np.full((H, W, 3), 120, np.uint8)).sum() == 0


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([False, True, True, False], [(1, 2)]),
        ([True, False, True], [(0, 0), (2, 2)]),
        ([False, True, True], [(1, 2)]),  # run that reaches the end
        ([False, False], []),
    ],
)
def test_contiguous_runs(flags, expected):
    assert _load_scene_eval().contiguous_runs(flags) == expected


def test_scene_eval_recovers_the_constructed_episode(tmp_path):
    scene, feats = _build_scene(tmp_path)
    out = tmp_path / "run"
    rc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "scene_eval.py"),
            "--feats",
            str(feats),
            "--scene",
            str(scene),
            "--out",
            str(out),
        ],
        cwd=ROOT,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, rc.stderr
    summary = json.loads((out / "scene_metrics.json").read_text())["summary"]

    assert summary["frames"] == N_FRAMES
    assert summary["hidden_frames"] == HIDDEN[1] - HIDDEN[0] + 1
    assert summary["episodes"] == [list(HIDDEN)]
    assert summary["never_recovered"] == 0
    # The mask must actually follow the ball while it is visible, and let go while it is not.
    assert summary["J_visible_mean"] > 0.4
    assert summary["ghost_area_while_hidden"] < 0.25
