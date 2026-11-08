#!/bin/zsh
# Paper 7: every isolated bead at its full counts (seed 0), three streams of 2 threads.
cd ~/code_repo/blml
NB=$(.venv/bin/python -c "import numpy as np; print(len(np.load('results/psf/beads_iso.npy')))")
for s in 0 1 2; do
  ( for k in $(seq $s 3 $((NB - 1))); do
      [ -f results/p7_beads/b${k}_full_0.json ] && continue
      OMP_NUM_THREADS=2 nice .venv/bin/python results/p7_beads.py $k full 0 2>&1 | grep -v "^\s*\["
    done ) &
done
wait
