#!/bin/zsh
# Paper 3 overnight, laptop (4 threads, niced): ghost through the 1-d protocol (bg N = 1, 0, 2), then the build-up study.
set -e; cd ~/code_repo/blml
export BLML_DATA_DIR=~/data_repo/dslit_gpu BLML_ENGINE=cpu BLML_COLLAPSE=1 OMP_NUM_THREADS=4
for N in 1 0 2; do
  D=results/dslit_1d/ghost/bg$N; mkdir -p $D; cp results/dslit_1d/ghost/flat/ghost_fraunhofer.json $D/
  BLML_SETS=ghost BLML_OUT_DIR=$D BLML_BG_HARM=$N,$N BLML_PARTS=complex_bg nice .venv/bin/python dslit_job.py
done
BLML_BG_HARM=1,1 P3_OUT=results nice .venv/bin/python p3_buildup.py photon ghost electron photon_mm
echo all done
