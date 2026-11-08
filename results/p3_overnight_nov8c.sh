#!/usr/bin/env bash
# Paper 3, night of 2026-11-07/08, part c (after the crop diagnosis): B. the cropped benchmark (periodic vs line kernel
# vs KDE on a record-like crop), C. the real records with the periodic kernel (splits + replicas), then part b
# (A. wide-window benchmark extras: n = 1e6, V = 0.15 / 0.3, the N = 0 pass). Mac, 6 workers, nice. Log as before.
cd "$(dirname "$0")/.." || exit 1
export BLML_DATA_DIR=~/data_repo/dslit_gpu
L=results/p3_overnight_nov7.log
say() { echo "[$(date +%H:%M)] $*" >> $L; }
until grep -q "2 done (diagnosis)" $L; do sleep 30; done
say "c start"
{ for c in sym asym partial; do for b in 0 0.1; do for n in 100000 10000 1000; do for r in 0 1 2; do echo "B $c $b $n $r"; done; done; done; done
  for s in photon ghost photon_mm electron; do for sd in 0 1 2 3 4 5; do echo "C $s $sd"; done; for r in 0 1 2 3 4; do echo "C $s 0 $r"; done; done
} | nice xargs -P 6 -L 1 sh -c 'if [ "$0" = B ]; then .venv/bin/python p3_cbench.py "$@"; else .venv/bin/python p3_periodic_main.py "$@"; fi > /dev/null 2>>results/p3_night_c_err.log'
say "c done: cropped benchmark $(ls results/p3_cbench | wc -l), periodic real $(ls results/p3_periodic_main | wc -l)"
bash results/p3_overnight_nov7b.sh
