#!/usr/bin/env bash
# Run blml_gpu_exp_2d3d4d.py on a fresh Colab A100 and fetch the outputs.
#   ./run_colab.sh [GPU]        GPU defaults to A100 (T4, L4, G4, H100 also work)
# Code comes from ~/code_repo/blml and data from ~/data_repo/caiso (the source of truth).
# Outputs land in ~/code_repo/blml/outputs/<timestamp>/.
set -euo pipefail

GPU="${1:-A100}"
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="$HOME/data_repo/caiso"
SESSION="blml-$(date +%Y%m%d-%H%M%S)"
OUT="$HERE/outputs/${SESSION#blml-}"
REMOTE=/content/blml

colab new -s "$SESSION" --gpu "$GPU"
trap 'colab stop -s "$SESSION" || true' EXIT

# The upload API does not create parent directories.
echo "import os; [os.makedirs('$REMOTE/'+d, exist_ok=True) for d in ('code','data','outputs')]" | colab exec -s "$SESSION"

for f in blml_trivial.py blml_gpu.py selector.py caiso_data.py; do colab upload -s "$SESSION" "$HERE/$f" "$REMOTE/code/$f"; done
for f in caiso_3d_price.npy caiso_3d_hour.npy caiso_3d_doy.npy; do colab upload -s "$SESSION" "$DATA/$f" "$REMOTE/data/$f"; done

mkdir -p "$OUT"
colab exec -s "$SESSION" --timeout 3600 -f "$HERE/blml_gpu_exp_2d3d4d.py" 2>&1 | tee "$OUT/log.txt"

echo "import shutil; shutil.make_archive('$REMOTE/outputs', 'zip', '$REMOTE/outputs')" | colab exec -s "$SESSION"
colab download -s "$SESSION" "$REMOTE/outputs.zip" "$OUT/outputs.zip"
unzip -o -q "$OUT/outputs.zip" -d "$OUT" && rm "$OUT/outputs.zip"
echo "Outputs in $OUT"
