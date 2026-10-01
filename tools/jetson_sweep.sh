#!/usr/bin/env bash
# Build and benchmark one TensorRT engine per exported ONNX, on the device, and emit CSV.
#
# Runs on a Jetson Nano (JetPack 4.6, TensorRT 8.2): FP16 is the best this SoC offers —
# compute capability 5.3 has no INT8 path — so --fp16 is the only precision flag here.
# No Python, no pip, nothing installed: trtexec does build and timing on its own.
#
#   ./jetson_sweep.sh ~/vos-edge            # onnx/ in, engines/ logs/ sweep.csv out
#
# Engines are kept so the camera stage can reuse them; delete engines/ to force a rebuild.

set -u
ROOT="${1:-$HOME/vos-edge}"
TRTEXEC=/usr/src/tensorrt/bin/trtexec
WORKSPACE_MB="${WORKSPACE_MB:-1024}"
ITERATIONS="${ITERATIONS:-50}"
AVGRUNS="${AVGRUNS:-10}"
# fp16 is 1.4x faster here and returns NaN for every pixel: TensorRT 8.2 has no native
# LayerNorm, the decomposed variance term overflows fp16, and trtexec never looks at the
# output it just timed. fp32 is the only precision on this board that computes the function.
PRECISION="${PRECISION:-fp32}"
case "$PRECISION" in
  fp16) PREC_FLAG="--fp16" ;;
  fp32) PREC_FLAG="" ;;
  *) echo "PRECISION must be fp16 or fp32"; exit 2 ;;
esac

cd "$ROOT" || exit 1
mkdir -p engines logs
CSV="$ROOT/sweep_${PRECISION}.csv"

# Clock state decides the numbers, so record it next to them.
gpu_cur=$(cat /sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq 2>/dev/null || echo NA)
gpu_max=$(cat /sys/devices/gpu.0/devfreq/57000000.gpu/max_freq 2>/dev/null || echo NA)
cpu_cur=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo NA)
cpu_gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo NA)
{
  echo "# $(cat /sys/firmware/devicetree/base/model | tr -d '\0') precision=$PRECISION"
  echo "# $(head -1 /etc/nv_tegra_release)"
  echo "# trtexec: $($TRTEXEC --help 2>&1 | grep -m1 -i 'TensorRT.*version' || echo 'TensorRT 8.2.1')"
  echo "# gpu_cur_hz=$gpu_cur gpu_max_hz=$gpu_max cpu0_cur_khz=$cpu_cur governor=$cpu_gov"
  echo "# date=$(date -Is)"
  echo "size,tokens,precision,build_s,throughput_qps,gpu_mean_ms,gpu_median_ms,host_mean_ms,status"
} > "$CSV"

for onnx in onnx/*.onnx; do
  [ -e "$onnx" ] || { echo "no ONNX in $ROOT/onnx"; exit 1; }
  base=$(basename "$onnx" .onnx)
  size=$(echo "$base" | sed -E 's/.*_([0-9]+x[0-9]+)(_raw)?$/\1/')
  h=${size%x*}; w=${size#*x}
  tokens=$(( (h / 16) * (w / 16) ))
  log="logs/${base}_${PRECISION}.log"
  plan="engines/${base}_${PRECISION}.plan"

  echo "=== $size ($tokens tokens) ==="
  if [ -s "$plan" ] && grep -q "Throughput:" "$log" 2>/dev/null; then
    # A build is 8-14 minutes on this board; never pay it twice for the same engine.
    echo "  reusing the engine and log already on disk"
    rc=0
    build_s=0
  else
    t0=$(date +%s)
    "$TRTEXEC" --onnx="$onnx" $PREC_FLAG --workspace="$WORKSPACE_MB" \
               --iterations="$ITERATIONS" --avgRuns="$AVGRUNS" \
               --saveEngine="$plan" > "$log" 2>&1
    rc=$?
    build_s=$(( $(date +%s) - t0 ))
  fi

  if [ $rc -ne 0 ]; then
    status="FAIL_rc$rc"
    if grep -qiE "out of memory|bad_alloc|Killed" "$log"; then status="OOM"; fi
    if grep -qiE "unsupported|No importer|parsing" "$log"; then status="PARSE"; fi
    if grep -qi "launch timed out" "$log"; then status="LAUNCH_TIMEOUT"; fi
    echo "$size,$tokens,$PRECISION,$build_s,,,,,$status" >> "$CSV"
    echo "  -> $status (see $log)"
    continue
  fi

  # trtexec 8.2 prints: "Throughput: N qps", "GPU Compute Time: min = .. mean = X ms, median = Y ms"
  qps=$(grep -m1 -i "Throughput:" "$log" | sed -E 's/.*Throughput: *([0-9.]+).*/\1/')
  gpu_mean=$(grep -m1 -i "GPU Compute Time:" "$log" | sed -E 's/.*mean = ([0-9.]+) ms.*/\1/')
  gpu_med=$(grep -m1 -i "GPU Compute Time:" "$log" | sed -E 's/.*median = ([0-9.]+) ms.*/\1/')
  host_mean=$(grep -m1 -iE "^\[I\] (Host )?Latency:" "$log" | sed -E 's/.*mean = ([0-9.]+) ms.*/\1/')
  echo "$size,$tokens,$PRECISION,$build_s,${qps:-},${gpu_mean:-},${gpu_med:-},${host_mean:-},ok" >> "$CSV"
  echo "  -> ${qps:-?} qps, gpu mean ${gpu_mean:-?} ms, built in ${build_s}s"
done

echo
echo "--- $CSV ---"
cat "$CSV"
