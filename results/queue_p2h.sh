#!/usr/bin/env bash
# After the smoke test: all models on the day split (hl, hll) in one job (same GPU, comparable timings), then the rest.
cd "$(dirname "$0")/.."
slot() { while [ "$(pgrep -fl 'colab_job.sh (start|resume) ' | awk '{print $5}' | sort -u | wc -l | tr -d ' ')" -ge 2 ]; do sleep 30; done; }
go() { slot; nohup ./colab_job.sh start "$@" >> "results/$1_follow.log" 2>&1 & sleep 90; }
while pgrep -f "colab_job.sh (start|resume) p2smoke" >/dev/null; do sleep 20; done
grep -q '"passed": true' results/p2smoke/smoke.json 2>/dev/null || { echo "smoke test FAILED; queue stopped"; exit 1; }
BLML_P2_RUNS=daysplit BLML_P2_COVSETS=hl,hll \
  BLML_P2_MODELS=BLML-auto,BLML-auto-diff,GMM-diff,CondKDE,CondKNN,GMM,Climatology,CatBoost-Normal,MDN,NGBoost-Normal,TabPFN go p2dayall p2_experiments.py
BLML_P2_RUNS=rollfwd BLML_P2_COVSETS=hl,hll BLML_P2_MODELS=BLML-auto,BLML-auto-diff go p2fwdblml p2_experiments.py
go p2pairs p2_pairs.py
BLML_P2_MODELS=BLML-daily,BLML-daily-fixed go p2dailyblml p2_daily.py
BLML_P2_MODELS=MDN,GMM,CondKNN,CatBoost-Normal,TabPFN go p2daily p2_daily.py
wait
