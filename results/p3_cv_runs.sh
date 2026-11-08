#!/bin/zsh
# Paper 3, double slit, 1-d protocol with K-fold CV (5 folds on train + validation) and the one-SE rule for every
# hyperparameter (BLML fc, KDE / adaptive-KDE bandwidths): KDE, adaptive KDE and complex BLML with the EM flat
# background (as the KDE's), complex BLML with the background mode N = 0, 1, 2, and streaming BLML with N = 0 and 1.
# Usage: results/p3_cv_runs.sh <set>   -> results/dslit_cv/<set>/{flat,bg0,bg1,bg2,stream0,stream1}
set -e; cd ~/code_repo/blml; S=$1
export BLML_DATA_DIR=~/data_repo/dslit_gpu BLML_ENGINE=cpu BLML_COLLAPSE=1 BLML_SELECT=cv OMP_NUM_THREADS=1 BLML_SETS=$S
PY="nice .venv/bin/python dslit_job.py"; R=results/dslit_cv/$S
mkdir -p $R/flat; cp results/dslit_1d/$S/flat/${S}_fraunhofer.json $R/flat/
BLML_OUT_DIR=$R/flat BLML_BG_FLAT=1 BLML_PARTS=kde,complex,akde ${=PY}
for N in 1 0 2; do
  mkdir -p $R/bg$N; cp $R/flat/${S}_fraunhofer.json $R/bg$N/
  BLML_OUT_DIR=$R/bg$N BLML_BG_HARM=$N,$N BLML_PARTS=complex_bg ${=PY}
done
for N in 0 1; do
  mkdir -p $R/stream$N; BLML_OUT_DIR=$R/stream$N BLML_BG_HARM=$N,$N BLML_PARTS=stream_bg ${=PY}
done
echo $S done
