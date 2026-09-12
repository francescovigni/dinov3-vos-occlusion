#!/usr/bin/env bash
# Colonoscopy study, idempotent. Caches at stride 2 with per-video caps (disk), uint8 features.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONPATH=src
CFG=configs/polyp.yaml
PG=configs/polypgen.yaml
LOG=runs/polyp_pipeline.log
mkdir -p runs docs/figures docs/videos
stage() {
  local name=$1 check=$2; shift 2
  if [ -e "$check" ]; then echo "MILESTONE $name skipped (exists)" | tee -a "$LOG"; return 0; fi
  echo "MILESTONE $name start $(date +%H:%M:%S)" | tee -a "$LOG"
  if "$@" > "runs/$name.log" 2>&1; then echo "MILESTONE $name ok $(date +%H:%M:%S)" | tee -a "$LOG"
  else echo "FAILED $name (see runs/$name.log)" | tee -a "$LOG"; grep -A 8 Traceback "runs/$name.log" | tail -10 | tee -a "$LOG"; exit 1; fi
}
# features: val = Test split, stride 2, first 80 cached frames per video; train = TrainValid, stride 2, 50 frames
stage polyp_extract_val   data/features_polyp/val/.done   sh -c "$PY -m dvos.extract --config $CFG --split val --stride 2 --max-frames 80 && touch data/features_polyp/val/.done"
stage polyp_extract_train data/features_polyp/train/.done sh -c "$PY -m dvos.extract --config $CFG --split train --stride 2 --max-frames 50 --variants clean occ0 && touch data/features_polyp/train/.done"
stage polypgen_extract_val data/features_polypgen/val/.done sh -c "$PY -m dvos.extract --config $PG --split val --variants clean occ0 && touch data/features_polypgen/val/.done"
for v in clean occ0; do
  stage polyp_baseline_$v runs/polyp_baseline_$v/metrics.json $PY -m dvos.evaluate --config $CFG --method baseline --variant $v --out runs/polyp_baseline_$v --save-masks
  stage polypgen_baseline_$v runs/polypgen_baseline_$v/metrics.json $PY -m dvos.evaluate --config $PG --method baseline --variant $v --out runs/polypgen_baseline_$v --save-masks
done
stage polyp_train runs/polyp_head/model.pt $PY -m dvos.train --config $CFG --out runs/polyp_head
stage polyp_calibrate runs/polyp_head/gate.json $PY -m dvos.calibrate --config $CFG --run runs/polyp_head
GATE=$($PY -c 'import json; print(json.load(open("runs/polyp_head/gate.json"))["vis_gate"])')
echo "MILESTONE polyp_gate=$GATE" | tee -a "$LOG"
for v in clean occ0; do
  stage polyp_head_$v runs/polyp_head_$v/metrics.json $PY -m dvos.evaluate --config $CFG --method model --checkpoint runs/polyp_head/model.pt --variant $v --out runs/polyp_head_$v --save-masks --vis-gate $GATE
  stage polypgen_head_$v runs/polypgen_head_$v/metrics.json $PY -m dvos.evaluate --config $PG --method model --checkpoint runs/polyp_head/model.pt --variant $v --out runs/polypgen_head_$v --save-masks --vis-gate $GATE
done
stage polyp_abl_ungated runs/polyp_abl_ungated_occ0/metrics.json $PY -m dvos.evaluate --config $CFG --method model --checkpoint runs/polyp_head/model.pt --variant occ0 --out runs/polyp_abl_ungated_occ0 --vis-gate 0
stage polyp_abl_fifo    runs/polyp_abl_fifo_occ0/metrics.json    $PY -m dvos.evaluate --config $CFG --method model --checkpoint runs/polyp_head/model.pt --variant occ0 --out runs/polyp_abl_fifo_occ0 --no-keep-first --vis-gate $GATE
stage polyp_abl_ungated_clean runs/polyp_abl_ungated_clean/metrics.json $PY -m dvos.evaluate --config $CFG --method model --checkpoint runs/polyp_head/model.pt --variant clean --out runs/polyp_abl_ungated_clean --vis-gate 0
stage polyp_fig_head docs/figures/polyp_qualitative_clean.png $PY -m dvos.figures --config $CFG --run runs/polyp_head_clean --out docs/figures/polyp_qualitative_clean.png
stage polyp_fig_base docs/figures/polyp_qualitative_baseline_clean.png $PY -m dvos.figures --config $CFG --run runs/polyp_baseline_clean --out docs/figures/polyp_qualitative_baseline_clean.png
echo "DONE-POLYP $(date +%H:%M:%S)" | tee -a "$LOG"
