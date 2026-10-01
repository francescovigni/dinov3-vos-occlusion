"""Will this scene work, before recording 200 frames of it?

Object size is necessary and not sufficient. What decides whether occlusion can be measured is
the **margin**: how much more similar the object is to the reference than the most
object-looking part of the background. Measured on two clips of the same object:

    68 px object, margin 0.252  ->  occlusion estimate r = 0.94
    54 px object, margin 0.142  ->  occlusion estimate r = 0.19, unusable at any threshold

The threshold has to sit between object and background. Below about 0.2 there is no room for
it, and no amount of tuning recovers that — a full sweep was run and every setting failed.

Captures a few frames, runs the engine, prints the margin.

    python3 tools/check_margin.py --engine engines/<...>.plan --ref-feat feats/reference/feat_0000.npy \\
        --ref-mask scenes/reference/mask.png
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jetson_infer as ji  # noqa: E402  - same directory on the device

TARGET_LO = np.array([5, 120, 100], np.uint8)
TARGET_HI = np.array([25, 255, 255], np.uint8)
GOOD_MARGIN = 0.20


def largest_blob(bgr):
    m = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), TARGET_LO, TARGET_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    if n <= 1:
        return None
    return lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--ref-feat", required=True)
    ap.add_argument("--ref-mask", required=True)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--scene", default=None, help="score an existing scene instead of capturing")
    args = ap.parse_args()

    if args.scene:
        tmp = args.scene
        paths = sorted(glob.glob(os.path.join(tmp, "frame_*.png")))[: args.frames]
        if not paths:
            raise SystemExit("no frames in " + tmp)
    else:
        tmp = tempfile.mkdtemp(prefix="margin_")
        print("capturing %d frames..." % args.frames)
        rc = subprocess.call(
            [
                sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "jetson_capture.py"),
                "--scene",
                os.path.basename(tmp),
                "--frames",
                str(args.frames),
                "--out",
                os.path.dirname(tmp),
            ]
        )
        if rc != 0:
            raise SystemExit("capture failed")
        paths = sorted(glob.glob(os.path.join(tmp, "frame_*.png")))

    ref = np.load(args.ref_feat).astype(np.float32)
    C, h, w = ref.shape
    mask = cv2.imread(args.ref_mask, cv2.IMREAD_GRAYSCALE) > 127
    small = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    rf = ref.reshape(C, -1)
    rf = rf / (np.linalg.norm(rf, axis=0, keepdims=True) + 1e-8)
    proto = rf[:, small.reshape(-1)].mean(1)
    proto = proto / (np.linalg.norm(proto) + 1e-8)

    engine = ji.load_engine(args.engine)
    context = engine.create_execution_context()
    in_idx = [i for i in range(engine.num_bindings) if engine.binding_is_input(i)][0]
    out_idx = [i for i in range(engine.num_bindings) if not engine.binding_is_input(i)][0]
    in_shape = tuple(engine.get_binding_shape(in_idx))
    out_shape = tuple(engine.get_binding_shape(out_idx))
    host_in = np.zeros(in_shape, np.float32)
    host_out = np.zeros(out_shape, np.float32)
    d_in, d_out = ji.cuda_malloc(host_in.nbytes), ji.cuda_malloc(host_out.nbytes)
    bindings = [0] * engine.num_bindings
    bindings[in_idx] = int(d_in.value)
    bindings[out_idx] = int(d_out.value)

    ons, bgs, spans = [], [], []
    try:
        for p in paths:
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            gt = largest_blob(bgr)
            if gt is None:
                print("  %s: object not found by the hue threshold" % os.path.basename(p))
                continue
            H, W = bgr.shape[:2]
            host_in[...] = ji.preprocess(bgr, in_shape[2], in_shape[3])
            ji.copy_to_device(d_in, host_in)
            context.execute_v2(bindings)
            ji._cudart.cudaDeviceSynchronize()
            ji.copy_to_host(host_out, d_out)
            f = host_out[0].reshape(C, -1)
            f = f / (np.linalg.norm(f, axis=0, keepdims=True) + 1e-8)
            sim = (proto @ f).reshape(h, w)
            su = cv2.resize(sim, (W, H), interpolation=cv2.INTER_LINEAR)
            ons.append(float(su[gt].max()))
            bgs.append(float(np.percentile(su[~gt], 99)))
            ys, xs = np.nonzero(gt)
            spans.append(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
    finally:
        ji._cudart.cudaFree(d_in)
        ji._cudart.cudaFree(d_out)

    if not ons:
        raise SystemExit("the object was never found; fix lighting or the hue window first")
    on, bg = float(np.mean(ons)), float(np.mean(bgs))
    margin = on - bg
    print("\nobject span   %.0f px" % np.mean(spans))
    print("on-target     %.3f" % on)
    print("background    %.3f  (99th percentile)" % bg)
    print("MARGIN        %.3f" % margin)
    if margin >= GOOD_MARGIN:
        print("\nOK — record. (0.252 gave an occlusion estimate of r=0.94)")
    else:
        print("\nTOO THIN — do not record yet. (0.142 gave r=0.19, unusable at any threshold)")
        print("move the object closer, and put it against something less like it in colour.")


if __name__ == "__main__":
    main()
