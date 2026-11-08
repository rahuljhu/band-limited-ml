#!/usr/bin/env bash
# Crop-bias diagnostic for the z-stack (paper 3, Sec. 6): joint BLML and Zernike ML fitted on 31^2 (regression vs the
# published run), 41^2 and 51^2 windows, all scored on the central 31^2 pixels. Beads 0-5.
cd "$(dirname "$0")/../.."
M=blml_stack,ml_zernike,neighbour_hist
ZS_OUT=results/complex/zstack_h15check/ ZS_METHODS=$M ZS_H=15 .venv/bin/python results/complex/zstack.py 0
for h in 20 25; do ZS_METHODS=$M ZS_H=$h .venv/bin/python results/complex/zstack.py 0 1 2 3 4 5; done
