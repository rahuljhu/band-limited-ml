#!/usr/bin/env bash
# LL-only runs with the joint/marginal BLML conditional. Smoke test gates the rest.
cd "$(dirname "$0")/.."
slot() { while [ "$(pgrep -fl 'colab_job.sh (start|resume) ' | awk '{print $5}' | sort -u | wc -l | tr -d ' ')" -ge 2 ]; do sleep 30; done; }
go() { slot; nohup ./colab_job.sh start "$@" >> "results/$1_follow.log" 2>&1 & sleep 90; }
./colab_job.sh start p2smoke p2_smoke.py >> results/p2smoke_follow.log 2>&1
grep -q '"passed": true' results/p2smoke/smoke.json 2>/dev/null || { echo "smoke test FAILED; queue stopped"; exit 1; }
BLML_P2_RUNS=daysplit BLML_P2_COVSETS=hl,hll BLML_P2_MODELS=BLML-auto,BLML-auto-diff,GMM-diff go p2dayblml p2_experiments.py
BLML_P2_RUNS=daysplit BLML_P2_COVSETS=hll BLML_P2_MODELS=CondKDE,CondKNN,GMM,Climatology,CatBoost-Normal,MDN,NGBoost-Normal,TabPFN go p2dayhll p2_experiments.py
BLML_P2_RUNS=rollfwd BLML_P2_COVSETS=hl,hll BLML_P2_MODELS=BLML-auto,BLML-auto-diff go p2fwdblml p2_experiments.py
go p2pairs p2_pairs.py
BLML_P2_MODELS=BLML-daily,BLML-daily-fixed go p2dailyblml p2_daily.py
BLML_P2_MODELS=MDN,GMM,CondKNN,CatBoost-Normal,TabPFN go p2daily p2_daily.py
wait
