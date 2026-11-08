#!/usr/bin/env bash
# GPU parity of the periodic two-slit kernel (results/pslits_parity.py gpu) on a fresh Colab VM, default T4.
#   ./run_colab_parity.sh [GPU]   -> results/pslits_parity_gpu.json (+ log)
set -euo pipefail
GPU="${1:-T4}"; HERE="$(cd "$(dirname "$0")" && pwd)"; SESSION="blml-parity-$(date +%Y%m%d-%H%M%S)"; REMOTE=/content/blml
colab new -s "$SESSION" --gpu "$GPU"
trap 'colab stop -s "$SESSION" || true' EXIT
echo "import os; [os.makedirs('$REMOTE/'+d, exist_ok=True) for d in ('results',)]" | colab exec -s "$SESSION"
for f in "$HERE"/blml_*.py "$HERE"/selector.py; do colab upload -s "$SESSION" "$f" "$REMOTE/$(basename "$f")"; done
colab upload -s "$SESSION" "$HERE/results/pslits_parity.py" "$REMOTE/results/pslits_parity.py"
echo "import subprocess, sys; r = subprocess.run([sys.executable, 'results/pslits_parity.py', 'gpu'], cwd='$REMOTE', capture_output=True, text=True); print(r.stdout[-6000:]); print(r.stderr[-4000:])" \
  | colab exec -s "$SESSION" --timeout 3600 2>&1 | tee "$HERE/results/pslits_parity_gpu.log"
colab download -s "$SESSION" "$REMOTE/results/pslits_parity_gpu.json" "$HERE/results/pslits_parity_gpu.json"
