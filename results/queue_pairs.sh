#!/usr/bin/env bash
# Pair-density experiment, after the day-split hll jobs have been launched.
cd "$(dirname "$0")/.."
until [ -d results/p2dayhldiff ] || pgrep -f "colab_job.sh (start|resume) p2dayhldiff" >/dev/null; do sleep 60; done
sleep 120
while [ "$(pgrep -fl 'colab_job.sh (start|resume) ' | awk '{print $5}' | sort -u | wc -l | tr -d ' ')" -ge 2 ]; do sleep 60; done
nohup ./colab_job.sh start p2pairs p2_pairs.py >> results/p2pairs_follow.log 2>&1
