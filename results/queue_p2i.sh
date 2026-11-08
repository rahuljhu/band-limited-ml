#!/usr/bin/env bash
cd "$(dirname "$0")/.."
slot() { while [ "$(pgrep -fl 'colab_job.sh (start|resume) ' | awk '{print $5}' | sort -u | wc -l | tr -d ' ')" -ge 2 ]; do sleep 20; done; }
go() { slot; nohup ./colab_job.sh start "$@" >> "results/$1_follow.log" 2>&1 & sleep 60; }
BLML_P2_RUNS=daysplit BLML_P2_COVSETS=hl,hll go p2day5 p2_experiments.py
BLML_P2_RUNS=rollfwd BLML_P2_COVSETS=hl,hll BLML_P2_MODELS=BLML-auto go p2fwd5 p2_experiments.py
go p2pairs p2_pairs.py
wait
