#!/bin/zsh
# Paper 3, double slit, 1-d protocol (collapsed across the fringes): references / KDE / histogram with a flat
# background estimated from the record (EM; the references' weight jointly), complex BLML with the background as a
# dirichlet mode of N = 0, 1, 2 harmonics (fc: full-grid argmax on validation), rank 2 + background at N = 1.
set -e
export BLML_DATA_DIR=~/data_repo/dslit_gpu BLML_ENGINE=cpu BLML_COLLAPSE=1 OMP_NUM_THREADS=4
PY="nice .venv/bin/python dslit_job.py"
for S in photon electron photon_mm; do
  F=results/dslit_1d/$S/flat; mkdir -p $F
  BLML_SETS=$S BLML_OUT_DIR=$F BLML_BG_FLAT=1 BLML_PARTS=fraunhofer,fresnel,kde,hist ${=PY}
  for N in 0 1 2; do
    D=results/dslit_1d/$S/bg$N; mkdir -p $D; cp $F/${S}_fraunhofer.json $D/
    P=complex_bg; [ $N = 1 ] && P=complex_bg,rank2_bg
    BLML_SETS=$S BLML_OUT_DIR=$D BLML_BG_HARM=$N,$N BLML_PARTS=$P ${=PY}
  done
done
