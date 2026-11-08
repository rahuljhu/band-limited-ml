#!/usr/bin/env bash
# Paper 3, 2026-11-08: phase/contrast scores on the cropped benchmark (CV over N), then the benchmark with a flat
# background (N = 0, the synthetic protocol) and its scores. Mac, 6 workers, nice. Log results/p3_overnight_nov7.log.
cd "$(dirname "$0")/.." || exit 1
export BLML_DATA_DIR=~/data_repo/dslit_gpu
L=results/p3_overnight_nov7.log
say() { echo "[$(date +%H:%M)] $*" >> $L; }
jobs_() { for c in sym asym partial; do for b in 0 0.1; do for n in 100000 10000 1000; do for r in 0 1 2; do echo "$c $b $n $r"; done; done; done; done; }
jobs_ | nice xargs -P 6 -L 1 .venv/bin/python p3_cbench_phase.py > /dev/null 2>> results/p3_phase_err.log
say "d1 done: phase scores (CV N)"
jobs_ | CB_N=0 CB_OUT=results/p3_cbench_n0 nice xargs -P 6 -L 1 .venv/bin/python p3_cbench.py > /dev/null 2>> results/p3_phase_err.log
jobs_ | CB_OUT=results/p3_cbench_n0 nice xargs -P 6 -L 1 .venv/bin/python p3_cbench_phase.py > /dev/null 2>> results/p3_phase_err.log
say "d2 done: cropped benchmark N=0 $(ls results/p3_cbench_n0 | wc -l)"
