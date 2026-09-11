#!/usr/bin/env bash
# Zero-shot propagation + the head's learned visibility gate. Runs after scripts/run_all.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONPATH=src
CFG=${CFG:-configs/default.yaml}
LOG=runs/pipeline.log
stage() {
  local name=$1 check=$2; shift 2
  if [ -e "$check" ]; then echo "MILESTONE $name skipped (exists)" | tee -a "$LOG"; return 0; fi
  echo "MILESTONE $name start $(date +%H:%M:%S)" | tee -a "$LOG"
  if "$@" > "runs/$name.log" 2>&1; then echo "MILESTONE $name ok $(date +%H:%M:%S)" | tee -a "$LOG"
  else echo "FAILED $name (see runs/$name.log)" | tee -a "$LOG"; grep -A 8 Traceback "runs/$name.log" | tail -10 | tee -a "$LOG"; exit 1; fi
}
stage calibrate_j runs/head/gate_j.json $PY -m dvos.calibrate --config $CFG --run runs/head --objective j
GATE=$($PY -c 'import json; print(json.load(open("runs/head/gate_j.json"))["vis_gate"])')
echo "MILESTONE gate_j=$GATE" | tee -a "$LOG"
for v in clean occ0 occ1; do
  stage gated_$v runs/gated_$v/metrics.json $PY -m dvos.evaluate --config $CFG --method gated_baseline --checkpoint runs/head/model.pt --variant $v --out runs/gated_$v --save-masks --vis-gate $GATE
done
for v in occ0 occ1; do
  stage head_gatej_$v runs/head_gatej_$v/metrics.json $PY -m dvos.evaluate --config $CFG --method model --checkpoint runs/head/model.pt --variant $v --out runs/head_gatej_$v --vis-gate $GATE
done
$PY scripts/collect_results.py > runs/results.md
echo "DONE-GATED $(date +%H:%M:%S)" | tee -a "$LOG"
