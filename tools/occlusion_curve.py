"""Estimated against true occluded fraction: the figure a partial-occlusion clip exists for.

A clip where the object is fully hidden and then fully visible only says whether the tracker
notices an occlusion. A clip where it is covered a quarter, a half, three quarters says whether
the estimate is *calibrated* — which is the part that was weakest: the earlier run read 1.00
occluded while the truth was 0.48, because patches straddling object and occluder stop matching
before the object is really hidden.

    python tools/occlusion_curve.py --run runs/sil_partial --out docs/figures/occlusion_curve.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    data = json.loads((args.run / "silhouette_metrics.json").read_text())
    rows = data["frames"]
    est = np.array([r["est_occluded"] for r in rows], float)
    true = np.array([r["true_occluded"] for r in rows], float)
    ok = np.isfinite(est) & np.isfinite(true)
    est, true = est[ok], true[ok]

    print("%-14s %7s %9s %9s" % ("true occluded", "frames", "mean est", "mean err"))
    for lo, hi in BINS:
        sel = (true >= lo) & (true < hi)
        if not sel.any():
            print("%4.0f-%3.0f %%      %7d %9s %9s" % (lo * 100, hi * 100, 0, "-", "-"))
            continue
        print(
            "%4.0f-%3.0f %%      %7d %9.3f %+9.3f"
            % (lo * 100, hi * 100, sel.sum(), est[sel].mean(), (est[sel] - true[sel]).mean())
        )
    # Two populations, because they answer different questions and a single r hides it:
    # every frame (does the estimate reach 1.0 when the object is gone?) and only the frames
    # where something is still visible (is the *fraction* right?). silhouette_track reports
    # the second; quoting one as if it were the other is how these numbers get oversold.
    r = float(np.corrcoef(est, true)[0, 1]) if len(est) > 2 else float("nan")
    print(
        "\nall frames          r=%.3f  MAE=%.3f  bias=%+.3f"
        % (r, np.abs(est - true).mean(), (est - true).mean())
    )
    part = true < 1.0
    if part.sum() > 2:
        rp = float(np.corrcoef(est[part], true[part])[0, 1])
        print(
            "partially visible   r=%.3f  MAE=%.3f  bias=%+.3f  (%d frames)"
            % (
                rp,
                np.abs(est[part] - true[part]).mean(),
                (est[part] - true[part]).mean(),
                part.sum(),
            )
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=160)
    t = np.arange(len(true))
    ax1.plot(t, true, color="#1f4e79", lw=1.6, label="true (hue ground truth)")
    ax1.plot(t, est, color="#b03030", lw=1.4, ls="--", label="estimated")
    ax1.set_xlabel("frame")
    ax1.set_ylabel("occluded fraction")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

    ax2.scatter(true, est, s=14, alpha=0.6, color="#1f4e79", zorder=3)
    ax2.plot([0, 1], [0, 1], color="#999", lw=1, ls=":", zorder=2)
    ax2.set_xlabel("true occluded fraction")
    ax2.set_ylabel("estimated")
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(alpha=0.25)
    ax2.set_title(f"r={r:.3f}  MAE={np.abs(est - true).mean():.3f}", fontsize=9)

    fig.suptitle(args.label or f"Occlusion estimate — {args.run.name}", fontsize=10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out)
    print("wrote " + str(args.out))


if __name__ == "__main__":
    main()
