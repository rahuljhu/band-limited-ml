#!/usr/bin/env bash
# Paper 2 queue (asinh variant dropped): restart GEFCom, then CAISO auto, lag baselines, lag TabPFN.
cd "$(dirname "$0")/.."
slot() { while [ "$(pgrep -f 'colab_job.sh (start|resume) ' | wc -l | tr -d ' ')" -ge 2 ]; do sleep 60; done; }
slot; nohup ./colab_job.sh start p2gef p2_gefcom.py >> results/p2gef_follow.log 2>&1 & sleep 120
slot; BLML_P2_RUNS=daysplit,rolling BLML_P2_MODELS=BLML-auto \
  nohup ./colab_job.sh start p2auto p2_experiments.py >> results/p2auto_follow.log 2>&1 & sleep 120
slot; BLML_P2_RUNS=daysplit,rolling BLML_P2_COVSETS=hl,hdl,hdll \
  BLML_P2_MODELS=CondKDE,CondKNN,GMM,Climatology,LightGBM-quantile,QuantileForest,CatBoost-Normal,MDN,NGBoost-Normal \
  nohup ./colab_job.sh start p2lag p2_experiments.py >> results/p2lag_follow.log 2>&1 & sleep 120
slot; BLML_P2_RUNS=daysplit,rolling BLML_P2_COVSETS=hl,hdl,hdll BLML_P2_MODELS=TabPFN \
  nohup ./colab_job.sh start p2lagtab p2_experiments.py >> results/p2lagtab_follow.log 2>&1 &
wait
