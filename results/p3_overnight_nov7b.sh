#!/usr/bin/env bash
# Paper 3, overnight chain part b (after the crop-normalisation diagnosis and, if it held, the periodic slit-kernel fix):
# crop-free multimode, benchmark ladder (n = 1e6, V = 0.15 / 0.3), the N = 0 pass. Log: results/p3_overnight_nov7.log
cd "$(dirname "$0")/.." || exit 1
export BLML_DATA_DIR=~/data_repo/dslit_gpu
L=results/p3_overnight_nov7.log
say() { echo "[$(date +%H:%M)] $*" >> $L; }
for m in 1 4 16; do echo $m; done | nice xargs -P 3 -L 1 sh -c \
  'P3_WIDE=1024 P3_NMULT=$0 .venv/bin/python p3_partial3_real.py photon_mm 0 0 > /dev/null 2>>results/p3_partial3_real/err_mm_wide.log'
say "2 done: crop-free multimode $(ls results/p3_partial3_real | grep -c '_w1024')"
export BLML_OUT_DIR=results/dbench DB_WORKERS=6
DB_CASES=sym,asym,defocus,partial DB_NS=1000000 nice .venv/bin/python p3_dbench_job.py >> results/dbench/ladder_job.log 2>&1
DB_CASES=partial_v0.15,partial_v0.3 DB_NS=1000,10000,100000,1000000 nice .venv/bin/python p3_dbench_job.py >> results/dbench/ladder_job.log 2>&1
say "3 done: benchmark n=1e6 and V=0.15/0.3 cases ($(grep -c ERR results/dbench/ladder_job.log) errors)"
DB_FLAT_ONLY=1 DB_CASES=sym,asym,defocus,partial,partial_v0.15,partial_v0.3 DB_NS=1000,10000,100000,1000000 \
  nice .venv/bin/python p3_dbench_job.py >> results/dbench/flat_job.log 2>&1
say "4 done: N=0 pass $(ls results/dbench/p3_dbench | grep -c '_s0.json') ($(grep -c ERR results/dbench/flat_job.log) errors)"
say "all done"
