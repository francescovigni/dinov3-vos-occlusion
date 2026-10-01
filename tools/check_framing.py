"""Is the object big enough to track? Answer before recording, not after.

The encoder sees a 1/16 patch grid: at 480x864 one patch covers 24 px of a 1280x720 frame.
An object spanning 68 px covers ~6 patches and tracks well; 16 px covers less than one and
nothing tracks at all, by any method — two clips were recorded and thrown away learning that.

Run this, watch the number, move the object closer until it reads OK, then record.

    python3 tools/check_framing.py                 # repeats until Ctrl-C
    python3 tools/check_framing.py --min-span 50
"""

import argparse
import time

import cv2
import numpy as np

PIPELINE = (
    "nvarguscamerasrc sensor-id={s} ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
    "nvvidconv flip-method=0 ! video/x-raw,width=1280,height=720,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=2 sync=false"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", type=int, default=0)
    ap.add_argument("--hue", default="5,25")
    ap.add_argument("--sat-min", type=int, default=120)
    ap.add_argument("--val-min", type=int, default=100)
    ap.add_argument("--min-span", type=float, default=50.0, help="px; 68 is what worked")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between reports")
    args = ap.parse_args()

    h0, h1 = [int(v) for v in args.hue.split(",")]
    lo = np.array([h0, args.sat_min, args.val_min], np.uint8)
    hi = np.array([h1, 255, 255], np.uint8)

    cap = cv2.VideoCapture(PIPELINE.format(s=args.sensor), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise SystemExit("cannot open the CSI camera")
    print("move the object until the span reads OK, then Ctrl-C and record.")
    print("target >= %.0f px span (the clip that worked was 68 px)\n" % args.min_span)
    try:
        last = 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if time.time() - last < args.every:
                continue
            last = time.time()
            m = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lo, hi)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
            if n <= 1:
                print("  no object found — too dark, too washed out, or out of the hue window")
                continue
            i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            area = int(stats[i, cv2.CC_STAT_AREA])
            span = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            patches = area / (24.0 * 24.0)  # one patch at 480x864 covers ~24x24 px of the frame
            verdict = (
                "OK"
                if span >= args.min_span
                else ("CLOSER — %.0f%% of target" % (100 * span / args.min_span))
            )
            print(
                "  span %3d px, area %5d px, ~%4.1f patches at 480x864   %s"
                % (span, area, patches, verdict)
            )
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
