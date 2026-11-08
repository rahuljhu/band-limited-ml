#!/bin/zsh
# Paper 3: synthetic replicas of one double-slit record, reps 0..2 (p3_synth_replica.py). Usage: <set> [variant]
cd ~/code_repo/blml
for r in 0 1 2; do OMP_NUM_THREADS=1 nice .venv/bin/python p3_synth_replica.py $1 $r ${2:-base} || exit 1; done
