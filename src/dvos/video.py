"""Overlay videos of a run: prediction, ground truth, occluder and the visibility verdict.

Per frame: prediction filled red, ground-truth visible mask in green, occluder outline in
yellow, a status bar with frame index, J, the visibility score as a gauge and its verdict
against the run's gate, and an "occluder present" tag during the episode. ``--compare``
puts a second run side by side (typically the zero-shot baseline on the left).

Writes an MP4 (mp4v) and, with ``--gif``, a downscaled GIF for READMEs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dvos.config import feature_dir, load_config
from dvos.extract import build_bank
from dvos.figures import overlay, sequence_images
from dvos.metrics import jaccard

WHITE, GREY, GREEN, RED, AMBER = (
    (255, 255, 255),
    (150, 150, 150),
    (60, 220, 60),
    (230, 60, 60),
    (240, 200, 40),
)
BAR_H = 34


def status_bar(
    width: int,
    seq: str,
    name: str,
    t: int,
    n: int,
    j: float,
    vis: float,
    gate: float | None,
    in_episode: bool,
) -> np.ndarray:
    bar = np.zeros((BAR_H, width, 3), np.uint8)
    hidden = gate is not None and vis < gate
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.5
    title = f"{name}  {seq}  t={t:>3}/{n - 1}  J={j:.2f}"
    cv2.putText(bar, title, (6, 22), font, fs, WHITE, 1, cv2.LINE_AA)
    tw = cv2.getTextSize(title, font, fs, 1)[0][0]
    # visibility gauge + verdict, right-aligned; gauge width shrinks on narrow frames
    gw = 120 if width >= 900 else 70
    x1 = width - 70
    x0, y0, y1 = x1 - gw, 9, 25
    cv2.rectangle(bar, (x0, y0), (x1, y1), GREY, 1)
    fill = int((x1 - x0 - 2) * float(np.clip(vis, 0, 1)))
    cv2.rectangle(bar, (x0 + 1, y0 + 1), (x0 + 1 + fill, y1 - 1), RED if hidden else GREEN, -1)
    if gate is not None:
        gx = x0 + 1 + int((x1 - x0 - 2) * gate)
        cv2.line(bar, (gx, y0 - 3), (gx, y1 + 3), WHITE, 1)
    cv2.putText(
        bar,
        "HIDDEN" if hidden else "VISIBLE",
        (x1 + 6, 22),
        font,
        fs,
        RED if hidden else GREEN,
        1,
        cv2.LINE_AA,
    )
    if in_episode:
        tag = "OCCLUDER PRESENT" if x0 - (6 + tw + 12) > 150 else "OCCL."
        tag_w = cv2.getTextSize(tag, font, fs, 1)[0][0]
        x = min(max(6 + tw + 12, (width - tag_w) // 2), x0 - tag_w - 8)
        if x > 6 + tw:
            cv2.putText(bar, tag, (x, 22), font, fs, AMBER, 1, cv2.LINE_AA)
    return bar


def load_run(run: Path, seq: str) -> tuple[dict, np.ndarray, np.ndarray, float | None]:
    metrics = json.loads((run / "metrics.json").read_text())
    row = next(r for r in metrics["sequences"] if r["seq"] == seq)
    m = np.load(run / "masks" / f"{seq}.npz")
    gate = metrics["summary"].get("vis_gate")
    if gate is None and metrics["summary"]["method"] == "baseline":
        gate = 0.5  # the baseline's score is the max foreground probability; 0.5 is its nominal cut
    return row, m["pred"].astype(bool), m["vis"], gate


def render_frames(
    cfg, run: Path, seq: str, split: str, name: str, imgs: list[np.ndarray], masks, episode
) -> list[np.ndarray]:
    row, pred, vis, gate = load_run(run, seq)
    gt, occ = masks["visible"].astype(bool), masks["occluder"].astype(bool)
    frames = []
    for t in range(len(imgs)):
        tile = overlay(imgs[t], pred[t], gt[t], occ[t])
        j = jaccard(pred[t], gt[t])
        in_ep = bool(episode) and episode[0] <= t < episode[1]
        bar = status_bar(tile.shape[1], seq, name, t, len(imgs), j, float(vis[t]), gate, in_ep)
        frames.append(np.concatenate([bar, tile], axis=0))
    return frames


def write_mp4(frames: list[np.ndarray], out: Path, fps: int) -> None:
    """RGB frames -> H.264 mp4.

    OpenCV's ``mp4v`` writer produces a structurally valid MPEG-4 stream that QuickTime
    renders as a flat green screen, so ffmpeg is used when it is on PATH and the fourcc
    writer is only the fallback. Dimensions are padded to even numbers because H.264
    cannot encode odd ones.
    """
    h, w = frames[0].shape[:2]
    out.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-movflags",
            "+faststart",
            str(out),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for f in frames:
                proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
            proc.stdin.close()
            if proc.wait() == 0:
                return
        except (BrokenPipeError, OSError):
            proc.kill()
        # fall through to the fourcc writer if ffmpeg failed for any reason

    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"cannot open video writer for {out}")
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def write_gif(frames: list[np.ndarray], out: Path, fps: int, width: int, every: int = 2) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    ims = []
    for f in frames[::every]:
        scale = width / f.shape[1]
        small = cv2.resize(f, (width, int(f.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        ims.append(Image.fromarray(small).quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    ims[0].save(
        out,
        save_all=True,
        append_images=ims[1:],
        duration=int(1000 * every / fps),
        loop=0,
        optimize=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", required=True, help="evaluate output dir with metrics.json and masks/")
    ap.add_argument(
        "--compare", default=None, help="second run shown on the left (e.g. the baseline)"
    )
    ap.add_argument("--seq", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True, help="output .mp4")
    ap.add_argument("--gif", default=None, help="also write a downscaled GIF here")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--gif-width", type=int, default=480)
    ap.add_argument("--names", nargs="*", default=None, help="labels for --compare and --run")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Path(args.run)
    variant = json.loads((run / "metrics.json").read_text())["summary"]["variant"]
    fdir = feature_dir(cfg, args.split, args.seq, variant)
    meta = json.loads((fdir / "meta.json").read_text())
    masks = np.load(fdir / "masks.npz")
    bank = build_bank(cfg) if variant != "clean" else None
    imgs = sequence_images(cfg, args.split, args.seq, variant, bank, meta["target_id"], meta)
    names = args.names or (["baseline", "head"] if args.compare else [run.name])
    if args.compare:
        left = render_frames(
            cfg, Path(args.compare), args.seq, args.split, names[0], imgs, masks, meta["episode"]
        )
        right = render_frames(
            cfg, run, args.seq, args.split, names[1], imgs, masks, meta["episode"]
        )
        frames = [np.concatenate([a, b], axis=1) for a, b in zip(left, right, strict=True)]
    else:
        frames = render_frames(
            cfg, run, args.seq, args.split, names[0], imgs, masks, meta["episode"]
        )
    write_mp4(frames, Path(args.out), args.fps)
    print(f"wrote {args.out} ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]})")
    if args.gif:
        write_gif(frames, Path(args.gif), args.fps, args.gif_width)
        print(f"wrote {args.gif}")


if __name__ == "__main__":
    main()
