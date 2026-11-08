#!/bin/zsh
# 2-d rank test on both photon records, 2 threads each
cd ~/code_repo/blml
for s in photon photon_mm; do OMP_NUM_THREADS=2 nice .venv/bin/python results/dslit_rank2d.py $s > results/dslit_rank2d/$s.log 2>&1 & done
wait
