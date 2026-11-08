#!/usr/bin/env bash
# p1t3fc8 covers d=8, so stop p1t3fc once its d=4 cells are saved.
cd "$(dirname "$0")/.."
until grep -q "saved r1/T3_d4_n100000" results/p1t3fc_follow.log 2>/dev/null; do sleep 60; done
sleep 120
./colab_job.sh fetch p1t3fc; ./colab_job.sh stop p1t3fc
