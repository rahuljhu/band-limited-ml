#!/usr/bin/env bash
cd "$(dirname "$0")/.."
slot() { while [ "$(colab sessions 2>/dev/null | grep -c '^\[blml-')" -ge 2 ]; do sleep 20; done; }
go() { slot; nohup ./colab_job.sh start "$@" >> "results/$1_follow.log" 2>&1 & sleep 90; }
BLML_P2_RUNS=rollfwd BLML_P2_COVSETS=hl,hll BLML_P2_MODELS=BLML-auto go p2fwd5 p2_experiments.py
go p2pairs p2_pairs.py
wait
