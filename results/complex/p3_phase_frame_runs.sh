#!/bin/zsh
# all cells of p3_phase_frame.py, two at a time (3 threads each: <= 6 cores), skipping finished ones
cd ~/code_repo/blml
cells=()
for n in 1e4 1e5 1e6; do for c in coma astig defocus mixed; do for s in 0 1 2; do cells+=("$c $n $s"); done; done; done
run() { set -- ${=1}; f=results/complex/phase_frame2/$1_$(python3 -c "print(int(float('$2')))")_$3.json; [ -f $f ] || OMP_NUM_THREADS=3 nice .venv/bin/python results/complex/p3_phase_frame.py $1 $2 $3 2>&1 | grep -v '^info'; }
i=0
for cell in $cells; do
  run "$cell" & ; i=$((i+1)); if [ $((i % 2)) -eq 0 ]; then wait; fi
done; wait; echo all done
