"""Record a short scene from the Jetson's CSI camera to a PNG sequence.

Runs on the device under Python 3.6 with the JetPack OpenCV build — no pip installs.
Capture is deliberately separate from inference: the camera findings need correct mask
propagation on real frames, not fast propagation, and the per-frame latency is measured
on its own elsewhere. Scenes are tens of frames, so recording them costs nothing and
removes the real-time constraint from everything downstream.

    python3 tools/jetson_capture.py --scene occlusion --frames 60
    python3 tools/jetson_capture.py --scene pan --frames 60 --fps 60    # motion blur
    python3 tools/jetson_capture.py --scene exposure --frames 90        # auto-exposure step

Writes scenes/<name>/frame_%04d.png plus meta.json with the real capture rate.
"""

import argparse
import json
import os
import time

import cv2

# IMX219 (Raspberry Pi Camera v2) on CSI port 0. nvarguscamerasrc keeps the ISP in the
# loop, which is the point: auto-exposure and auto-white-balance are part of what the
# DAVIS-trained pipeline has never seen.
PIPELINE = (
    "nvarguscamerasrc sensor-id={sensor} ! "
    "video/x-raw(memory:NVMM),width={cw},height={ch},framerate={fps}/1 ! "
    "nvvidconv flip-method={flip} ! "
    "video/x-raw,width={w},height={h},format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=2 sync=false"
)


def open_camera(sensor, cw, ch, w, h, fps, flip):
    pipe = PIPELINE.format(sensor=sensor, cw=cw, ch=ch, w=w, h=h, fps=fps, flip=flip)
    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(
            "could not open the CSI camera.\npipeline: " + pipe + "\n"
            "check: ls /dev/video0, and that nothing else holds the sensor "
            "(sudo systemctl restart nvargus-daemon)"
        )
    return cap, pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="occlusion | pan | exposure | <name>")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--sensor", type=int, default=0)
    ap.add_argument("--capture-size", default="1280x720", help="sensor mode")
    ap.add_argument("--size", default="1280x720", help="size written to disk")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--flip", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=15, help="frames dropped so AE/AWB settle")
    ap.add_argument("--out", default="scenes")
    args = ap.parse_args()

    cw, ch = [int(v) for v in args.capture_size.lower().split("x")]
    w, h = [int(v) for v in args.size.lower().split("x")]
    outdir = os.path.join(args.out, args.scene)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    cap, pipe = open_camera(args.sensor, cw, ch, w, h, args.fps, args.flip)
    try:
        for _ in range(args.warmup):
            ok, _frame = cap.read()
            if not ok:
                raise RuntimeError("camera opened but delivered no frames during warm-up")

        stamps = []
        t0 = time.time()
        for i in range(args.frames):
            ok, frame = cap.read()
            if not ok:
                print(f"dropped at frame {i}, stopping")
                break
            stamps.append(time.time() - t0)
            cv2.imwrite(os.path.join(outdir, f"frame_{i:04d}.png"), frame)
        elapsed = time.time() - t0
    finally:
        cap.release()

    n = len(stamps)
    real_fps = (n - 1) / (stamps[-1] - stamps[0]) if n > 1 else 0.0
    meta = {
        "scene": args.scene,
        "frames": n,
        "requested_fps": args.fps,
        "measured_fps": round(real_fps, 2),
        "wall_s": round(elapsed, 2),
        "size": [h, w],
        "capture_size": [ch, cw],
        "pipeline": pipe,
        "note": "write-to-disk is in the loop, so measured_fps is a capture+PNG rate, "
        "not the sensor's ceiling",
    }
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{n} frames to {outdir} at {real_fps:.1f} fps (incl. PNG write)")


if __name__ == "__main__":
    main()
