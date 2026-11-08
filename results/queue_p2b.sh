#!/usr/bin/env bash
# After queue_p2.sh has launched p2auto: baselines on the day-ahead (lagged-price) covariate sets.
cd "$(dirname "$0")/.."
until pgrep -f "colab_job.sh (start|resume) p2auto( |\$)" >/dev/null || [ -d results/p2auto ]; do sleep 60; done
slot() { while [ "$(pgrep -f 'colab_job.sh (start|resume) ' | wc -l | tr -d ' ')" -ge 2 ]; do sleep 60; done; }
sleep 120
slot; BLML_P2_RUNS=daysplit,rolling BLML_P2_COVSETS=hl,hdl,hdll \
  BLML_P2_MODELS=CondKDE,CondKNN,GMM,Climatology,LightGBM-quantile,QuantileForest,CatBoost-Normal,MDN,NGBoost-Normal \
  nohup ./colab_job.sh start p2lag p2_experiments.py >> results/p2lag_follow.log 2>&1 & sleep 120
slot; BLML_P2_RUNS=daysplit,rolling BLML_P2_COVSETS=hl,hdl,hdll BLML_P2_MODELS=TabPFN \
  nohup ./colab_job.sh start p2lagtab p2_experiments.py >> results/p2lagtab_follow.log 2>&1 &
