#!/usr/bin/env bash
# Paper 3, 2026-11-08 part b, after the N = 0 cropped benchmark: real-record recovery scores, wide-window pseudo-true.
cd "$(dirname "$0")/.." || exit 1
export BLML_DATA_DIR=~/data_repo/dslit_gpu
L=results/p3_overnight_nov7.log
say() { echo "[$(date +%H:%M)] $*" >> $L; }
until grep -q "d2 done" $L; do sleep 30; done
{ for s in photon_mm photon ghost electron; do echo "M $s"; done; echo "P sym"; echo "P asym"; } | nice xargs -P 6 -L 1 sh -c \
  'if [ "$0" = M ]; then .venv/bin/python p3_periodic_metrics.py "$1" > results/p3_periodic_metrics_$1.log 2>&1; else .venv/bin/python p3_dbench_pseudo.py "$1" > results/dbench/pseudo_$1.log 2>&1; fi'
say "d3 done: real recovery scores $(ls results/p3_periodic_metrics | wc -l), pseudo-true $(ls results/dbench/pseudo_*.json | wc -l)"
