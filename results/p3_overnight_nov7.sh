#!/usr/bin/env bash
# Paper 3, overnight chain of 2026-11-07 (Mac, at most 6 workers, nice). Each step waits for the previous one;
# every script skips results that already exist, so a rerun resumes. Progress: results/p3_overnight_nov7.log
cd "$(dirname "$0")/.." || exit 1
export BLML_DATA_DIR=~/data_repo/dslit_gpu
L=results/p3_overnight_nov7.log
say() { echo "[$(date +%H:%M)] $*" >> $L; }
while pgrep -x xargs > /dev/null; do sleep 30; done                      # 0. single-mode N<=2 fits, electrons (running)
say "0 done: slit_main_n012 $(ls results/p3_slit_main_n012 | wc -l), electron $(ls results/p3_partial3_real | grep -c 'electron.*json')"
for m in 2 4 8 16 32 64; do echo $m; done | nice xargs -P 6 -L 1 sh -c \
  'P3_NMULT=$0 .venv/bin/python p3_partial3_real.py photon_mm 0 0 > /dev/null 2>>results/p3_partial3_real/err_mm_big.log'
say "1 done: multimode size sweep $(ls results/p3_partial3_real | grep -c '_x')"
for r in 0 1 2 3 4; do echo $r; done | nice xargs -P 5 -L 1 sh -c \
  'P3_WIDE=1024 P3_NS=0,1,2 P3_OUT=results/p3_slit_main_n012 .venv/bin/python p3_slit_main.py photon 0 $0 > /dev/null 2>>results/sm_wide_err.log'
say "2 done (diagnosis): crop-free photon replicas $(ls results/p3_slit_main_n012 | grep -c _w1024); part b waits for the fix"
