#!/usr/bin/env bash
# T3 cutoff sweep below f_c = 1 (f_c = 0.25..0.7), once a Colab slot is free.
cd "$(dirname "$0")/.."
while [ "$(pgrep -f 'colab_job.sh (start|resume) ' | wc -l | tr -d ' ')" -ge 2 ]; do sleep 60; done
BLML_FC_MULTS=0.125,0.175,0.25,0.35 BLML_FC_SUB=fcsweep_lo BLML_FC_TRUTHS=T3 \
  ./colab_job.sh start p1fclo p1_fcsweep.py >> results/p1fclo_follow.log 2>&1
