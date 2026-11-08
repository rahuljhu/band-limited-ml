#!/usr/bin/env bash
# Run only TabPFN (Paper 2) once p1x frees a Colab session, then merge into results/p2.
cd "$(dirname "$0")/.."
while [ -f results/.p1x.session ]; do sleep 60; done
BLML_P2_MODELS=TabPFN ./colab_job.sh start p2tab p2_experiments.py
mkdir -p results/p2/p2 && cp results/p2tab/p2/*.json results/p2/p2/
