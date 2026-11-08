"""Pairs: joint test LL per model and paired per-charge-day tests vs a reference.
    .venv/bin/python quick_pairs.py [--ref BLML-joint]     (newest results/*/pairs/<model>.json wins)"""
import os, glob, json, argparse, collections
import numpy as np
from scipy import stats
ap = argparse.ArgumentParser(); ap.add_argument('--ref', default='BLML-joint'); a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
for f in sorted(glob.glob(os.path.join(HERE, 'results', '*', 'pairs', '*.json')), key=os.path.getmtime):
    r = json.load(open(f))
    if 'per_pair' in r:
        R[r['model']] = r
def daily(r):
    lp = np.array(r['per_pair']['lp1']) + np.array(r['per_pair']['lp2'])
    g = collections.defaultdict(list)
    for d, v in zip(r['day1'], lp):
        g[d].append(v)
    return {d: np.mean(v) for d, v in g.items()}
ref = daily(R[a.ref])
print(f'pairs: joint test LL (nats/pair); paired test over charge days vs {a.ref} (diff = ref - model, 95% CI)')
for name, r in sorted(R.items(), key=lambda kv: -kv[1]['ll']):
    if name == a.ref:
        print(f"  {name:18s} {r['ll']:8.3f}  n={r['n_test']}  days={len(ref)}"); continue
    dm = daily(r); days = sorted(set(ref) & set(dm))
    d = np.array([ref[k] - dm[k] for k in days])
    t = stats.ttest_1samp(d, 0.0)
    print(f"  {name:18s} {r['ll']:8.3f}  n={r['n_test']}  diff {d.mean():+.3f} ±{1.96 * d.std(ddof=1) / np.sqrt(len(d)):.3f}  p={t.pvalue:.3g}")
