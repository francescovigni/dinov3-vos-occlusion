"""Merge the device sweep and the accuracy sweep into one Pareto figure and one table.

Frame rate and J&F only mean something on the same axes, so this is the only place either
number is allowed to appear on its own.

    python tools/edge_pareto.py \
        --device artifacts/sweep.csv \
        --accuracy artifacts/accuracy_sweep.csv \
        --out docs/figures/edge_pareto.png --table docs/edge_pareto.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REALTIME_FPS = 25.0
# DAVIS val, zero-shot k-NN propagation at the config size, from runs/baseline_clean.
REFERENCE = {"size": "480x864", "tokens": 1620, "JF": 0.7668}


def read_csv(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line)
    return list(csv.DictReader(rows))


def merge(device: Path, accuracy: Path) -> list[dict]:
    acc = {r["size"]: float(r["JF"]) for r in read_csv(accuracy)}
    acc.setdefault(REFERENCE["size"], REFERENCE["JF"])
    out = []
    for r in read_csv(device):
        if r.get("status") != "ok":
            out.append({"size": r["size"], "tokens": int(r["tokens"]), "status": r["status"]})
            continue
        out.append(
            {
                "size": r["size"],
                "tokens": int(r["tokens"]),
                "fps": float(r["throughput_qps"]),
                "ms": float(r["gpu_mean_ms"]),
                "build_s": int(r["build_s"]),
                "jf": acc.get(r["size"]),
                "status": "ok",
            }
        )
    return sorted(out, key=lambda d: d["tokens"])


def table(rows: list[dict], precision: str = "FP32") -> str:
    head = (
        f"| input | patch tokens | J&F (DAVIS val) | fps (Nano, {precision}) | GPU ms/frame | engine build |\n"
        "|---|---|---|---|---|---|\n"
    )
    body = []
    for r in rows:
        if r["status"] != "ok":
            body.append(f"| {r['size']} | {r['tokens']} | — | **{r['status']}** | — | — |")
            continue
        jf = f"{r['jf']:.3f}" if r.get("jf") is not None else "—"
        body.append(
            f"| {r['size']} | {r['tokens']} | {jf} | {r['fps']:.2f} | {r['ms']:.0f} | {r['build_s']}s |"
        )
    return head + "\n".join(body) + "\n"


def plot(rows: list[dict], out: Path, precision: str = "FP32") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in rows if r["status"] == "ok" and r.get("jf") is not None]
    if not ok:
        print("nothing measured yet, skipping figure")
        return
    fps = [r["fps"] for r in ok]
    jf = [r["jf"] for r in ok]

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=160)
    ax.plot(fps, jf, "-o", color="#1f4e79", zorder=3)
    for r in ok:
        ax.annotate(
            f"{r['size']}\n{r['tokens']} tok",
            (r["fps"], r["jf"]),
            textcoords="offset points",
            xytext=(7, -4),
            fontsize=7.5,
            color="#333",
        )
    ax.axvline(REALTIME_FPS, color="#b03030", ls="--", lw=1, zorder=2)
    ax.text(
        REALTIME_FPS * 0.96,
        min(jf) + 0.02,
        "25 fps",
        rotation=90,
        ha="right",
        fontsize=8,
        color="#b03030",
    )
    failed = [r for r in rows if r["status"] != "ok"]
    if failed:
        names = ", ".join(f"{r['size']} ({r['tokens']} tok)" for r in failed)
        ax.text(
            0.015,
            0.035,
            f"no point for {names}: engine build failed,\nCUDA launch watchdog during tactic timing",
            transform=ax.transAxes,
            fontsize=7.5,
            color="#b03030",
            va="bottom",
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(axis="x", which="both", labelsize=8)
    ax.set_xlabel(f"frames per second (Jetson Nano, TensorRT {precision}, 921 MHz)")
    ax.set_ylabel("J&F, DAVIS 2017 val, zero-shot propagation")
    ax.set_title("Frozen DINOv3 ViT-S/16: what the edge costs", fontsize=10)
    ax.grid(alpha=0.25, zorder=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=Path, default=Path("artifacts/sweep.csv"))
    ap.add_argument("--accuracy", type=Path, default=Path("artifacts/accuracy_sweep.csv"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures/edge_pareto.png"))
    ap.add_argument("--table", type=Path, default=Path("docs/edge_pareto.md"))
    args = ap.parse_args()

    # The precision is recorded in the device CSV; the header must not lie about it.
    precision = next(
        (r.get("precision", "") for r in read_csv(args.device) if r.get("precision")), ""
    )
    rows = merge(args.device, args.accuracy)
    md = table(rows, precision.upper() or "FP32")
    print(md)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(md)
    plot(rows, args.out, precision.upper() or "FP32")

    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        best = max(ok, key=lambda r: r.get("jf") or 0)
        print(
            f"best measured quality: {best['size']} J&F {best.get('jf')} at {best['fps']:.2f} fps"
        )
        print(f"gap to 25 fps from there: {REALTIME_FPS / best['fps']:.0f}x")


if __name__ == "__main__":
    main()
