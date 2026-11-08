#!/usr/bin/env bash
# T3 rate study at low cutoffs (bias floor), once a Colab slot is free (after the cutoff sweep).
cd "$(dirname "$0")/.."
while pgrep -f "colab_job.sh (start|resume) p1fc( |\$)" >/dev/null; do sleep 120; done
sleep 30
BLML_P1_PARTS=rate BLML_P1_TRUTHS=T3 BLML_P1_DLIST=1,2,4,8 BLML_P1_FC_MULTS=0.25,0.35,0.5,0.7 \
  ./colab_job.sh start p1t3fc p1_experiments.py >> results/p1t3fc_follow.log 2>&1
