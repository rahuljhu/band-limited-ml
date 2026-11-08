#!/usr/bin/env bash
cd "$(dirname "$0")/.."
while [ "$(colab sessions 2>/dev/null | grep -c '^\[blml-')" -ge 2 ]; do sleep 20; done
./colab_job.sh start p2modecheck p2_modecheck.py >> results/p2modecheck_follow.log 2>&1
