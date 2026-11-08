#!/bin/zsh
# Paper 7 bead sweep: beads 0, 1, 3 x n = 1e3 .. 1e5, full (seed 0), three streams of 2 threads (6 cores).
cd ~/code_repo/blml
for k in 0 1 3; do
  ( for n in 1000 3000 10000 30000 100000 full; do
      [ -f results/p7_beads/b${k}_${n}_0.json ] && continue
      OMP_NUM_THREADS=2 nice .venv/bin/python results/p7_beads.py $k $n 0 2>&1 | grep -v "^\s*\["
    done ) &
done
wait
