#!/usr/bin/env bash
# T4 run of results/complex/gpu_check.py; outputs to outputs/<timestamp>/
set -euo pipefail
GPU="${1:-T4}"; HERE="$(cd "$(dirname "$0")/../.." && pwd)"; DATA="$HOME/data_repo/caiso"
SESSION="blml-$(date +%Y%m%d-%H%M%S)"; OUT="$HERE/outputs/${SESSION#blml-}"; REMOTE=/content/blml
colab new -s "$SESSION" --gpu "$GPU"
trap 'colab stop -s "$SESSION" || true' EXIT
echo "import os; [os.makedirs('$REMOTE/'+d, exist_ok=True) for d in ('code','data','outputs')]" | colab exec -s "$SESSION"
for f in blml_trivial.py blml_gpu.py selector.py caiso_data.py; do colab upload -s "$SESSION" "$HERE/$f" "$REMOTE/code/$f"; done
for f in caiso_3d_price.npy caiso_3d_hour.npy caiso_3d_doy.npy; do colab upload -s "$SESSION" "$DATA/$f" "$REMOTE/data/$f"; done
mkdir -p "$OUT"
{ echo "import os; os.environ.update(BLML_CODE_DIR='$REMOTE/code', BLML_DATA_DIR='$REMOTE/data', BLML_OUT_DIR='$REMOTE/outputs', ENGINE='gpu')"; cat "$HERE/results/complex/gpu_check.py"; } > "$OUT/job.py"
colab exec -s "$SESSION" --timeout 3600 -f "$OUT/job.py" 2>&1 | tee "$OUT/log.txt"
echo "import shutil; shutil.make_archive('$REMOTE/outputs', 'zip', '$REMOTE/outputs')" | colab exec -s "$SESSION"
colab download -s "$SESSION" "$REMOTE/outputs.zip" "$OUT/outputs.zip"
unzip -o -q "$OUT/outputs.zip" -d "$OUT" && rm "$OUT/outputs.zip"
echo "Outputs in $OUT"
