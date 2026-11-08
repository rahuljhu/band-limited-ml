#!/bin/zsh
# Paper 3, double slit: complex BLML with the two-slit aperture kernel (fill searched with fc by 5-fold CV, one-SE
# rule ordered by the aperture's measure fc fill) and the background mode N = 1. Usage: results/p3_slits_run.sh <set>
set -e; cd ~/code_repo/blml; S=$1; R=results/dslit_cv/$S/slits1; mkdir -p $R
cp results/dslit_1d/$S/flat/${S}_fraunhofer.json $R/
BLML_DATA_DIR=~/data_repo/dslit_gpu BLML_ENGINE=cpu BLML_COLLAPSE=1 BLML_SELECT=cv OMP_NUM_THREADS=1 BLML_SETS=$S \
  BLML_OUT_DIR=$R BLML_BG_HARM=1,1 BLML_FILLS=1.0,0.7,0.5,0.4,0.33,0.25 BLML_PARTS=complex_bg nice .venv/bin/python dslit_job.py
