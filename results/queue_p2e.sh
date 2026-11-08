#!/usr/bin/env bash
# Grid fast-path check first; the day-split hll runs and the pair experiment only if it passes.
cd "$(dirname "$0")/.."
slot() { while [ "$(pgrep -fl 'colab_job.sh (start|resume) ' | awk '{print $5}' | sort -u | wc -l | tr -d ' ')" -ge 2 ]; do sleep 60; done; }
slot; ./colab_job.sh start p2gridtest p2_gridtest.py >> results/p2gridtest_follow.log 2>&1
grep -q '"passed": true' results/p2gridtest/gridtest.json 2>/dev/null || { echo "grid test FAILED; queue stopped"; exit 1; }
slot; BLML_P2_RUNS=daysplit BLML_P2_COVSETS=hll \
  BLML_P2_MODELS=BLML-auto,BLML-auto-diff,GMM-diff,CondKDE,CondKNN,GMM,Climatology,LightGBM-quantile,QuantileForest,CatBoost-Normal,MDN,NGBoost-Normal,TabPFN \
  nohup ./colab_job.sh start p2dayhll p2_experiments.py >> results/p2dayhll_follow.log 2>&1 & sleep 120
slot; BLML_P2_RUNS=daysplit BLML_P2_COVSETS=hl BLML_P2_MODELS=BLML-auto-diff,GMM-diff \
  nohup ./colab_job.sh start p2dayhldiff p2_experiments.py >> results/p2dayhldiff_follow.log 2>&1 & sleep 120
slot; nohup ./colab_job.sh start p2pairs p2_pairs.py >> results/p2pairs_follow.log 2>&1 &
wait
