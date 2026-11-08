#!/usr/bin/env bash
# Paper 3 redo (PLAN_v5): Fig. 1 crop-free ladder, then Fig. 2 the same truth on a 7-fringe window. Mac, 6 workers, nice.
cd "$(dirname "$0")/.." || exit 1
export BLML_DATA_DIR=~/data_repo/dslit_gpu
L=results/p3_overnight_nov7.log
say() { echo "[$(date +%H:%M)] $*" >> $L; }
say "r0 start: ladder"
{ for n in 1000000 300000 100000 30000 10000 3000 1000; do for c in sym asym partial; do for b in 0 0.1; do for r in 0 1 2; do echo "$c $b $n $r"; done; done; done; done; } \
  | nice xargs -P 6 -L 1 sh -c '.venv/bin/python p3_ladder.py "$@" > /dev/null 2>>results/p3_ladder_err.log' _
say "r1 done: ladder $(ls results/p3_ladder | wc -l)"
jobs_() { for n in 100000 10000 1000; do for c in sym asym partial; do for b in 0 0.1; do for r in 0 1 2; do echo "$c $b $n $r"; done; done; done; done; }
export CB_TRUTH=tapered CB_M=3 CB_N=0 CB_OUT=results/p3_cbench_tap
jobs_ | nice xargs -P 6 -L 1 .venv/bin/python p3_cbench.py > /dev/null 2>>results/p3_ladder_err.log
jobs_ | nice xargs -P 6 -L 1 .venv/bin/python p3_cbench_phase.py > /dev/null 2>>results/p3_ladder_err.log
say "r2 done: cropped tapered $(ls results/p3_cbench_tap | wc -l)"
