#!/usr/bin/env bash
# When the p1 job finishes T1 (d=8, n=1e5), stop it (p1t2 covers T2) and rerun p2.
cd "$(dirname "$0")/.."
until [ -f results/p1/r1/T1_d8_n100000.json ] || grep -q "saved r1/T1_d8_n100000" results/p1_follow.log 2>/dev/null; do sleep 60; done
./colab_job.sh stop p1; pkill -f "colab_job.sh resume p1$"
./colab_job.sh start p2 p2_experiments.py
