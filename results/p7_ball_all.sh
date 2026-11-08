#!/bin/zsh
# Paper 7: every isolated bead paired against paper 3's ball-kernel BLML (two streams of 2 threads).
cd ~/code_repo/blml
for s in 0 1; do (
  for k in $(seq $s 2 52); do
    [ -f results/p7_beads/ball_b${k}.json ] && continue
    OMP_NUM_THREADS=2 nice .venv/bin/python results/p7_beads_ball.py $k 2>&1 | grep -v '^info'
  done ) &
done
wait
