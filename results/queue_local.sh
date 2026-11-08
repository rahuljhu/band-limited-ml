#!/bin/zsh
# Local queue, 2026-11-01 (<= 6 cores with the z-stack window run's 3): after the single-image all-starts runs finish,
# the blurred physics references (3 sets) and the exchangeability tests (real data), then the positive controls.
cd ~/code_repo/blml
while pgrep -f sec42_allstarts >/dev/null; do sleep 30; done
export BLML_DATA_DIR=~/data_repo/dslit_gpu BLML_ENGINE=cpu BLML_COLLAPSE=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
( for S in photon electron photon_mm; do
    BLML_SETS=$S BLML_OUT_DIR=results/dslit_1d/$S/flat BLML_BG_FLAT=1 BLML_PARTS=fraunhofer_blur,fresnel_blur nice .venv/bin/python dslit_job.py
  done; nice .venv/bin/python results/dslit_1d_table.py ) > results/dslit_blur.log 2>&1 &
nice .venv/bin/python results/exch_test.py photon real > results/exch_photon_real.log 2>&1 &
nice .venv/bin/python results/exch_test.py electron real > results/exch_electron_real.log 2>&1 &
wait
nice .venv/bin/python results/exch_test.py photon drift > results/exch_photon_drift.log 2>&1 &
nice .venv/bin/python results/exch_test.py photon copy > results/exch_photon_copy.log 2>&1 &
wait
echo queue done
