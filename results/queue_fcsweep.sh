#!/usr/bin/env bash
# Cutoff-sensitivity sweep, after the big-n run has produced all its files.
cd "$(dirname "$0")/.."
until [ "$(ls results/p1big/big 2>/dev/null | wc -l | tr -d ' ')" -ge 6 ]; do sleep 120; done
sleep 60
./colab_job.sh start p1fc p1_fcsweep.py
