#!/usr/bin/env bash
# GPU parity of the periodic ball (results/pball_parity.py gpu) and, as a regression check of the CUDA kernels, of the
# periodic two-slit band (results/pslits_parity.py gpu), on a fresh Colab VM, default T4.
#   ./run_colab_pball.sh [GPU]   -> results/pball_parity_gpu.json, results/pslits_parity_gpu.json (+ logs)
set -euo pipefail
GPU="${1:-T4}"; HERE="$(cd "$(dirname "$0")" && pwd)"; SESSION="blml-pball-$(date +%Y%m%d-%H%M%S)"; REMOTE=/content/blml
colab new -s "$SESSION" --gpu "$GPU"
trap 'colab stop -s "$SESSION" || true' EXIT
echo "import os; [os.makedirs('$REMOTE/'+d, exist_ok=True) for d in ('results',)]" | colab exec -s "$SESSION"
for f in "$HERE"/blml_*.py "$HERE"/selector.py; do colab upload -s "$SESSION" "$f" "$REMOTE/$(basename "$f")"; done
for t in ${TESTS:-pball_parity pslits_parity}; do
  colab upload -s "$SESSION" "$HERE/results/$t.py" "$REMOTE/results/$t.py"
  echo "import subprocess, sys; r = subprocess.run([sys.executable, 'results/$t.py', 'gpu'], cwd='$REMOTE', capture_output=True, text=True); print(r.stdout[-6000:]); print(r.stderr[-4000:])" \
    | colab exec -s "$SESSION" --timeout 3600 2>&1 | tee "$HERE/results/${t}_gpu.log"
  colab download -s "$SESSION" "$REMOTE/results/${t}_gpu.json" "$HERE/results/${t}_gpu.json" || true
done
