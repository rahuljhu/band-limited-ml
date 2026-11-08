#!/usr/bin/env bash
# After the T2 job releases its session: final systems experiments + T3 rate study.
cd "$(dirname "$0")/.."
sleep 120
while [ -f results/.p1t2.session ]; do sleep 60; done
BLML_P1_TRUTHS=T3 ./colab_job.sh start p1x p1_experiments.py
