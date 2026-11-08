#!/usr/bin/env bash
# Day-split runs without day of year: hll for every model; y - lag24 variants on hl.
cd "$(dirname "$0")/.."
slot() { while [ "$(pgrep -fl 'colab_job.sh (start|resume) ' | awk '{print $5}' | sort -u | wc -l | tr -d ' ')" -ge 2 ]; do sleep 60; done; }
slot; BLML_P2_RUNS=daysplit BLML_P2_COVSETS=hll \
  BLML_P2_MODELS=BLML-auto,BLML-auto-diff,GMM-diff,CondKDE,CondKNN,GMM,Climatology,LightGBM-quantile,QuantileForest,CatBoost-Normal,MDN,NGBoost-Normal,TabPFN \
  nohup ./colab_job.sh start p2dayhll p2_experiments.py >> results/p2dayhll_follow.log 2>&1 & sleep 120
slot; BLML_P2_RUNS=daysplit BLML_P2_COVSETS=hl BLML_P2_MODELS=BLML-auto-diff,GMM-diff \
  nohup ./colab_job.sh start p2dayhldiff p2_experiments.py >> results/p2dayhldiff_follow.log 2>&1 &
wait
