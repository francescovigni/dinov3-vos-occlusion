#!/usr/bin/env bash
# Full pipeline, idempotent: each stage is skipped when its output already exists.
# Milestones go to runs/pipeline.log; per-stage output to runs/<stage>.log.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONPATH=src
CFG=${CFG:-configs/default.yaml}
LOG=runs/pipeline.log
mkdir -p runs docs/figures

stage() {  # stage <name> <done-check-path> <command...>
  local name=$1 check=$2; shift 2
  if [ -e "$check" ]; then echo "MILESTONE $name skipped (exists)" | tee -a "$LOG"; return 0; fi
  echo "MILESTONE $name start $(date +%H:%M:%S)" | tee -a "$LOG"
  if "$@" > "runs/$name.log" 2>&1; then
    echo "MILESTONE $name ok $(date +%H:%M:%S)" | tee -a "$LOG"
  else
    echo "FAILED $name (see runs/$name.log)" | tee -a "$LOG"; grep -A 8 Traceback "runs/$name.log" | tail -10 | tee -a "$LOG"; exit 1
  fi
}

stage extract_val   data/features/val/.done   sh -c "$PY -m dvos.extract --config $CFG --split val && touch data/features/val/.done"
stage extract_train data/features/train/.done sh -c "$PY -m dvos.extract --config $CFG --split train --stride 2 --variants clean occ0 && touch data/features/train/.done"
for v in clean occ0 occ1; do
  stage baseline_$v runs/baseline_$v/metrics.json $PY -m dvos.evaluate --config $CFG --method baseline --variant $v --out runs/baseline_$v --save-masks
done
stage train_head runs/head/model.pt $PY -m dvos.train --config $CFG --out runs/head
for v in clean occ0 occ1; do
  stage head_$v runs/head_$v/metrics.json $PY -m dvos.evaluate --config $CFG --method model --checkpoint runs/head/model.pt --variant $v --out runs/head_$v --save-masks
done
for v in occ0 occ1; do
  stage abl_ungated_$v runs/abl_ungated_$v/metrics.json $PY -m dvos.evaluate --config $CFG --method model --checkpoint runs/head/model.pt --variant $v --out runs/abl_ungated_$v --vis-gate 0
  stage abl_fifo_$v    runs/abl_fifo_$v/metrics.json    $PY -m dvos.evaluate --config $CFG --method model --checkpoint runs/head/model.pt --variant $v --out runs/abl_fifo_$v --no-keep-first
done
stage train_cleanonly runs/head_cleanonly/model.pt $PY -m dvos.train --config $CFG --out runs/head_cleanonly --variants clean
for v in clean occ0 occ1; do
  stage abl_cleanonly_$v runs/abl_cleanonly_$v/metrics.json $PY -m dvos.evaluate --config $CFG --method model --checkpoint runs/head_cleanonly/model.pt --variant $v --out runs/abl_cleanonly_$v
done
stage fig_head     docs/figures/qualitative_occ0.png          $PY -m dvos.figures --config $CFG --run runs/head_occ0     --out docs/figures/qualitative_occ0.png
stage fig_baseline docs/figures/qualitative_baseline_occ0.png $PY -m dvos.figures --config $CFG --run runs/baseline_occ0 --out docs/figures/qualitative_baseline_occ0.png
$PY scripts/collect_results.py > runs/results.md
echo "DONE $(date +%H:%M:%S)" | tee -a "$LOG"
