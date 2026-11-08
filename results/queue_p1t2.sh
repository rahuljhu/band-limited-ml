#!/usr/bin/env bash
# Start the T2 rate study once the p2 job has released its Colab session.
cd "$(dirname "$0")/.."
while [ -f results/.p2.session ]; do sleep 60; done
BLML_P1_PARTS=rate BLML_P1_TRUTHS=T2 ./colab_job.sh start p1t2 p1_experiments.py
