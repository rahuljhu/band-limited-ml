#!/usr/bin/env bash
# Big-n BLML runs (asymptotic slope) once p1x and the TabPFN queue have released their sessions.
cd "$(dirname "$0")/.."
sleep 300
while [ -f results/.p1x.session ] || pgrep -f queue_tabpfn.sh >/dev/null; do sleep 60; done
BLML_P1_PARTS=bign ./colab_job.sh start p1big p1_experiments.py
