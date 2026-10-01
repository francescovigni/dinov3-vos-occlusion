#!/usr/bin/env bash
# The accuracy axis of the edge Pareto plot: zero-shot J&F on DAVIS val at each input size
# the Jetson engines were built for. Frame rate without this is a meaningless number.
#
# Runs on the workstation, not the device. 480x864 is already in runs/baseline_clean.
# Features are cached per size and deleted straight after scoring — the full set would be
# ~6 GB and this machine runs close to full.
#
#   ./tools/accuracy_sweep.sh                      # the four smaller sizes
#   SIZES="256x448" ./tools/accuracy_sweep.sh      # one size
#
# Output: artifacts/accuracy_sweep.csv

set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONPATH=src
SIZES="${SIZES:-384x672 320x576 256x448 192x336}"
OUT=artifacts/accuracy_sweep.csv
mkdir -p artifacts

[ -f "$OUT" ] || echo "size,tokens,J,F,JF,n_seq,extract_s,eval_s" > "$OUT"

for size in $SIZES; do
  h=${size%x*}; w=${size#*x}
  tokens=$(( (h / 16) * (w / 16) ))
  cfg="configs/_sweep_${size}.yaml"
  feats="data/features_sweep_${size}"
  run="runs/sweep_${size}"

  "$PY" - "$cfg" "$h" "$w" "$feats" <<'PY'
import sys, yaml
cfg, h, w, feats = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
c = yaml.safe_load(open("configs/default.yaml"))
c["data"]["size"] = [h, w]
c["data"]["features_root"] = feats
c["data"]["variants"] = ["clean"]
yaml.safe_dump(c, open(cfg, "w"), sort_keys=False)
PY

  echo "=== $size ($tokens tokens) ==="
  t0=$(date +%s)
  "$PY" -m dvos.extract --config "$cfg" --split val --variants clean > "artifacts/extract_${size}.log" 2>&1
  t1=$(date +%s)
  "$PY" -m dvos.evaluate --config "$cfg" --method baseline --variant clean --out "$run" \
        > "artifacts/eval_${size}.log" 2>&1
  t2=$(date +%s)

  "$PY" - "$run/metrics.json" "$size" "$tokens" "$((t1 - t0))" "$((t2 - t1))" "$OUT" <<'PY'
import json, sys
m, size, tokens, ex, ev, out = sys.argv[1:]
s = json.load(open(m))["summary"]
with open(out, "a") as f:
    f.write(f"{size},{tokens},{s['J']:.4f},{s['F']:.4f},{s['JF']:.4f},{s['n_seq']},{ex},{ev}\n")
print(f"  -> J&F {s['JF']:.4f}  (extract {ex}s, eval {ev}s)")
PY

  rm -rf "$feats"          # ~2.5 GB at 480p; this machine has no room to keep five of them
done

echo; echo "--- $OUT ---"; cat "$OUT"
