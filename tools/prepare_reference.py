"""Turn a photograph of the object into a reference the tracker can use.

Runs on the device under Python 3.6 with the JetPack OpenCV, because the features have to come
from the same engine at the same input size as the scene.

Two things must be true of the reference and neither is automatic:

* **the geometry must match the scene**, so the photo is letterboxed (never squashed) into the
  scene's frame size — a near-square photo forced to 16:9 stretches the object and the silhouette
  with it;
* **the object must subtend roughly what it subtends in the video.** Measured on a 10.9x gap:
  rescaling the reference keeps the margin over background at 0.244, leaving it at full size
  drops it to 0.178.

    python3 tools/prepare_reference.py --photo refphoto.jpeg --target-px 70
    python3 jetson_infer.py --engine engines/<...>.plan \\
        --scene scenes/reference --out feats/reference

Writes scenes/reference/frame_0000.png and scenes/reference/mask.png.
"""

import argparse
import os

import cv2
import numpy as np

TARGET_LO = np.array([5, 120, 100], np.uint8)
TARGET_HI = np.array([25, 255, 255], np.uint8)


def largest_blob(bgr, lo, hi):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, lo, hi)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    if n <= 1:
        return None
    return lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", required=True)
    ap.add_argument("--out", default="scenes/reference")
    ap.add_argument("--size", default="1280x720", help="scene frame size, WxH")
    ap.add_argument(
        "--target-px",
        type=float,
        default=70.0,
        help="how many pixels the object should span, i.e. what it spans in the video",
    )
    ap.add_argument("--mask", default=None, help="PNG mask; without it the hue window is used")
    ap.add_argument("--hue", default="5,25", help="hue window lo,hi for the object")
    ap.add_argument("--pad-value", type=int, default=128)
    args = ap.parse_args()

    W, H = [int(v) for v in args.size.lower().split("x")]
    lo, hi = TARGET_LO.copy(), TARGET_HI.copy()
    lo[0], hi[0] = [int(v) for v in args.hue.split(",")]

    img = cv2.imread(args.photo, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit("cannot read " + args.photo)

    mask = (
        cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE) > 127
        if args.mask
        else largest_blob(img, lo, hi)
    )
    if mask is None or mask.sum() == 0:
        raise SystemExit(
            "no object found in the photo. Pass --mask with a PNG, or widen --hue; "
            "the silhouette is the whole point and must not be guessed."
        )
    ys, xs = np.nonzero(mask)
    span = float(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
    scale = args.target_px / span
    print(
        "object spans %.0f px in the photo, scaling by %.3f to reach %.0f px"
        % (span, scale, args.target_px)
    )

    small = cv2.resize(
        img,
        (max(int(round(img.shape[1] * scale)), 1), max(int(round(img.shape[0] * scale)), 1)),
        interpolation=cv2.INTER_AREA,
    )
    small_m = cv2.resize(
        mask.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    if small.shape[0] > H or small.shape[1] > W:
        raise SystemExit(
            "rescaled photo is larger than the scene frame; raise --size or lower --target-px"
        )

    canvas = np.full((H, W, 3), args.pad_value, np.uint8)
    canvas_m = np.zeros((H, W), np.uint8)
    y0, x0 = (H - small.shape[0]) // 2, (W - small.shape[1]) // 2
    canvas[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small
    canvas_m[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small_m * 255

    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    cv2.imwrite(os.path.join(args.out, "frame_0000.png"), canvas)
    cv2.imwrite(os.path.join(args.out, "mask.png"), canvas_m)
    area = int((canvas_m > 0).sum())
    print(
        "wrote %s/frame_0000.png and mask.png — object %d px (%.2f%% of frame)"
        % (args.out, area, 100.0 * area / (H * W))
    )
    print(
        "next: python3 jetson_infer.py --engine <plan> --scene %s --out feats/reference" % args.out
    )


if __name__ == "__main__":
    main()
