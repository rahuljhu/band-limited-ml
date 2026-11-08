#!/bin/zsh
# every cell of p3_phase_frame.py with the periodic ball (P3_KERNEL=pball -> results/complex/phase_frame3), two at a
# time with 3 threads each (<= 6 cores), skipping finished cells
cd ~/code_repo/blml
for n in 10000 100000 1000000; do for c in coma astig defocus mixed; do for s in 0 1 2; do echo "$c $n $s"; done; done; done \
  | while read c n s; do [ -f results/complex/phase_frame3/${c}_${n}_${s}.json ] || echo "$c $n $s"; done \
  | P3_KERNEL=pball OMP_NUM_THREADS=3 NUMBA_NUM_THREADS=3 nice xargs -P ${NPAR:-2} -L 1 .venv/bin/python results/complex/p3_phase_frame.py 2>&1 | grep -v '^info'
echo all done
