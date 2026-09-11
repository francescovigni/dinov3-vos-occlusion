#!/usr/bin/env python3
"""Render the README/article result tables from runs/*/metrics.json (stdout, Markdown).

Sections: main comparison across variants, occlusion-episode split, v4 ablations.
Usage: render_results.py [runs_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dvos.config import feature_dir, load_config  # noqa: E402
from dvos.metrics import jaccard  # noqa: E402

ROWS = [
    ("baseline", "zero-shot k-NN propagation"),
    ("gated", "zero-shot propagation + learned visibility gate"),
    ("v1_head", "head v1 (soft memory masks)"),
    ("v2_head", "head v2 (+ hard masks, gapped clips, held-out selection)"),
    ("v3_head", "head v3 (+ position channels, locality window)"),
    ("head", "head v4 (+ zero-shot prior, calibrated gate)"),
]
VARIANTS = ["clean", "occ0", "occ1"]


def load(runs: Path, name: str) -> dict | None:
    p = runs / name / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def f(x, nd=3) -> str:
    return "–" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def main_table(runs: Path) -> list[str]:
    out = [
        "| method | clean J&F | occ0 J&F | occ1 J&F | occ0 leak | occ0 vis AUC | occ0 never recovered |",
        "|---|---|---|---|---|---|---|",
    ]
    for prefix, label in ROWS:
        m = {v: load(runs, f"{prefix}_{v}") for v in VARIANTS}
        if not any(m.values()):
            continue
        s = {v: (m[v] or {}).get("summary", {}) for v in VARIANTS}
        out.append(
            f"| {label} | {f(s['clean'].get('JF'))} | {f(s['occ0'].get('JF'))} | {f(s['occ1'].get('JF'))} | "
            f"{f(s['occ0'].get('leak'))} | {f(s['occ0'].get('vis_auc'))} | "
            f"{s['occ0'].get('never_recovered', '–')} / {s['occ0'].get('n_episodes', '–')} |"
        )
    return out


def split_table(runs: Path, cfg, variant: str = "occ0", after: int = 10) -> list[str]:
    out = [
        f"| method ({variant}) | J inside episode | J {after} frames after | J elsewhere |",
        "|---|---|---|---|",
    ]
    for prefix, label in ROWS:
        run = runs / f"{prefix}_{variant}"
        m = load(runs, run.name)
        if m is None or not (run / "masks").exists():
            continue
        inside, post, outside = [], [], []
        for row in m["sequences"]:
            fdir = feature_dir(cfg, m["summary"]["split"], row["seq"], variant)
            gt = __import__("numpy").load(fdir / "masks.npz")["visible"].astype(bool)
            pred = (
                __import__("numpy").load(run / "masks" / f"{row['seq']}.npz")["pred"].astype(bool)
            )
            eps = [tuple(e) for e in row["episodes"]]
            for t in range(1, len(gt)):
                j = jaccard(pred[t], gt[t])
                if any(a <= t < b for a, b in eps):
                    inside.append(j)
                elif any(b <= t < b + after for _, b in eps):
                    post.append(j)
                else:
                    outside.append(j)
        mean = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
        out.append(f"| {label} | {f(mean(inside))} | {f(mean(post))} | {f(mean(outside))} |")
    return out


def ablation_table(runs: Path) -> list[str]:
    rows = [
        ("head_occ0", "v4, calibrated gate, frame 0 permanent"),
        ("head_gatej_occ0", "v4, gate chosen by J"),
        ("abl_ungated_occ0", "v4, ungated writes"),
        ("abl_fifo_occ0", "v4, FIFO memory (frame 0 evictable)"),
        ("abl_cleanonly_occ0", "v4 trained on clean features only"),
    ]
    out = ["| ablation (occ0) | J&F | leak | vis AUC | never recovered |", "|---|---|---|---|---|"]
    for name, label in rows:
        m = load(runs, name)
        if m is None:
            continue
        s = m["summary"]
        out.append(
            f"| {label} | {f(s['JF'])} | {f(s['leak'])} | {f(s['vis_auc'])} | {s['never_recovered']} / {s['n_episodes']} |"
        )
    clean = load(runs, "abl_cleanonly_clean")
    if clean:
        out.append(
            f"| v4 trained on clean only, evaluated on clean | {f(clean['summary']['JF'])} | – | – | – |"
        )
    return out


START, END = "<!-- results:start -->", "<!-- results:end -->"


def render(runs: Path) -> str:
    cfg = load_config("configs/default.yaml")
    parts = [
        "#### Main comparison (DAVIS 2017 val, 30 sequences, largest object on frame 0)",
        "",
        *main_table(runs),
        "",
        "#### Where the accuracy goes (occ0, per-frame J)",
        "",
        *split_table(runs, cfg),
        "",
        "#### v4 ablations",
        "",
        *ablation_table(runs),
    ]
    return "\n".join(parts)


def inject(text: str, block: str) -> str:
    a, b = text.index(START), text.index(END)
    return text[: a + len(START)] + "\n" + block + "\n" + text[b:]


def main() -> None:
    argv = sys.argv[1:]
    targets = [argv[i + 1] for i, a in enumerate(argv) if a == "--inject"]
    positional = [
        a
        for i, a in enumerate(argv)
        if not a.startswith("--") and (i == 0 or argv[i - 1] != "--inject")
    ]
    runs = Path(positional[0]) if positional else Path("runs")
    block = render(runs)
    for t in targets:
        path = Path(t)
        path.write_text(inject(path.read_text(), block))
        print(f"injected into {t}")
    if not targets:
        print(block)


if __name__ == "__main__":
    main()
