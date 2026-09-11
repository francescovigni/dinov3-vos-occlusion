#!/usr/bin/env python3
"""Collect every runs/*/metrics.json into one Markdown table (stdout)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(x) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return "–" if x != x else f"{x:.3f}"
    return str(x)


def main(root: str = "runs") -> None:
    rows = []
    for m in sorted(Path(root).glob("*/metrics.json")):
        s = json.loads(m.read_text())["summary"]
        rows.append((m.parent.name, s))
    if not rows:
        print("no runs found", file=sys.stderr)
        return
    print(
        "| run | method | variant | gate | keep first | J | F | J&F | leak | vis AUC | recovery median | never recovered / episodes |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, s in rows:
        print(
            f"| {name} | {s['method']} | {s['variant']} | {fmt(s.get('vis_gate'))} | {fmt(s.get('keep_first'))} | "
            f"{fmt(s['J'])} | {fmt(s['F'])} | {fmt(s['JF'])} | {fmt(s['leak'])} | {fmt(s['vis_auc'])} | "
            f"{fmt(s['recovery_median'])} | {s['never_recovered']} / {s['n_episodes']} |"
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
