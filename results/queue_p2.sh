#!/usr/bin/env bash
# Paper 2 job queue: start each job when one of the two Colab slots is free.
cd "$(dirname "$0")/.."
slot() { while [ "$(pgrep -f 'colab_job.sh (start|resume) ' | wc -l | tr -d ' ')" -ge 2 ]; do sleep 60; done; }
slot; nohup ./colab_job.sh start p2gef p2_gefcom.py >> results/p2gef_follow.log 2>&1 & sleep 90
slot; nohup ./colab_job.sh start p2syn p2_selector_synth.py >> results/p2syn_follow.log 2>&1 & sleep 90
slot; BLML_P2_RUNS=daysplit,rolling BLML_P2_MODELS=BLML-auto,BLML-auto-asinh \
  nohup ./colab_job.sh start p2auto p2_experiments.py >> results/p2auto_follow.log 2>&1 &
