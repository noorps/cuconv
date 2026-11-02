#!/usr/bin/env bash
# One command: correctness, benchmark, profiler, chart, feature maps.
# Usage: bash run.sh [label]   (results land in results/<label>/)
set -euo pipefail
cd "$(dirname "$0")"
LABEL="${1:-$(date +%Y%m%d-%H%M)}"
PY="${PYTHON:-python3}"
$PY -m pip install -q ninja   # torch builds the extension with ninja; Colab lacks it
echo "== results/$LABEL"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,memory.total --format=csv
$PY bench/check.py --label "$LABEL"
$PY bench/bench.py --label "$LABEL"
$PY bench/ncu_profile.py --label "$LABEL"
$PY bench/plot.py --label "$LABEL"
$PY bench/featuremaps.py --label "$LABEL"
echo "== done: results/$LABEL"
ls -la "results/$LABEL"
