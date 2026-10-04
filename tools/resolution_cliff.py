#!/usr/bin/env python3
"""The benchmark blind spot, as one figure.

    python tools/resolution_cliff.py --out docs/figures/resolution_cliff.png

Four numbers, all from the tables in docs/edge.md ("The real Pareto" and "The
resolution cliff"), which are themselves produced by tools/accuracy_sweep.sh and
tools/scene_eval.py. Hardcoded here rather than re-derived: they come from two
different measurement rigs (workstation sweep, camera scene) and a CSV pipeline
for four values would be more code than the figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

HI, LO = "480x864", "384x672"
SERIES = {HI: "#2a78d6", LO: "#eb6834"}  # dataviz categorical slots 1, 2

# (group label, sub-label, {resolution: score})
PANELS = [
    ("DAVIS 2017 val", "benchmark J&F, 30 sequences", {HI: 0.767, LO: 0.733}),
    ("Real camera scene", "J after occlusion, 2 episodes", {HI: 0.646, LO: 0.006}),
]
VERDICT = [
    "costs 3.4 points —\nlooks like a rounding error",
    "costs the whole track —\n0 of 2 re-acquired",
]

# Stacked bands, so nothing can collide: header line, verdicts, then the bars.
Y_RULE, Y_VERDICT, Y_MAX = 1.20, 1.12, 1.26


def plot(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    width, gap = 0.3, 0.02

    for g, (_, _, scores) in enumerate(PANELS):
        for s, (res, val) in enumerate(scores.items()):
            x = g + (s - 0.5) * (width + gap)
            ax.bar(x, val, width, color=SERIES[res], zorder=3, label=res if g == 0 else None)
            ax.text(x, val + 0.02, f"{val:.3f}", ha="center", fontsize=9.5, color="#0b0b0b")
            ax.text(x, -0.045, res, ha="center", fontsize=8.5, color="#52514e")

    for g, (label, sub, _) in enumerate(PANELS):
        ax.text(g, -0.115, label, ha="center", fontsize=10.5, color="#0b0b0b")
        ax.text(g, -0.165, sub, ha="center", fontsize=8, color="#52514e")
        ax.text(g, Y_VERDICT, VERDICT[g], ha="center", fontsize=9, color="#0b0b0b", va="top")

    # The thing the figure is about: one decision, two verdicts.
    ax.plot([-0.16, 1.16], [Y_RULE, Y_RULE], color="#c9c8c0", lw=1, zorder=2)
    ax.text(
        0.5,
        Y_RULE + 0.015,
        "dropping input resolution to buy 2x the frame rate",
        ha="center",
        fontsize=9,
        color="#52514e",
    )

    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, Y_MAX)
    ax.set_xticks([])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylabel("IoU-based score (0–1)")
    ax.set_title("The benchmark is blind to the failure that matters", fontsize=12, pad=26)
    ax.legend(frameon=False, fontsize=9, loc="center", bbox_to_anchor=(0.80, 0.42))
    ax.grid(axis="y", alpha=0.25, zorder=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_bounds(0, 0.8)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor="#fcfcfb")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/figures/resolution_cliff.png"))
    plot(ap.parse_args().out)


if __name__ == "__main__":
    main()
