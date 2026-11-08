#!/bin/sh
# Nov 8b: after the ladder rescore (LD_RESCORE=1), the cropped tapered rescore (phi_rel, shift-aligned ap_err) and the
# n -> inf floor of the periodic class. Run from the repo root.
cnt() { .venv/bin/python -c "import json,glob;print(sum('model_p' in json.load(open(f)) for f in glob.glob('results/p3_ladder/*.json')))"; }
while [ "$(cnt)" -lt 126 ]; do sleep 30; done
echo "[$(date +%H:%M)] r3 ladder rescored" >> results/p3_overnight_nov7.log
export CB_TRUTH=tapered CB_M=3 CB_N=0 CB_OUT=results/p3_cbench_tap
{ ls results/p3_cbench_tap/*.json | sed -E 's#.*/([a-z]+)_bg([0-9.]+)_n([0-9]+)_([0-9]+)\.json#\1 \2 \3 \4#'
  for c in sym asym partial; do for b in 0 0.1; do echo "$c $b 0 0"; done; done; } \
  | nice xargs -P 6 -L 1 .venv/bin/python p3_cbench_phase.py > /dev/null 2>>results/p3_ladder_err.log
echo "[$(date +%H:%M)] r4 cropped rescored + floor" >> results/p3_overnight_nov7.log
