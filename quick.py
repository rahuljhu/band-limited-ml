"""Quick look at Paper 2 results: test LL per model and day-level paired tests vs a reference model.

    .venv/bin/python quick.py daysplit hl,hll            # run prefix, covariate sets
    .venv/bin/python quick.py fwd hl,hll --ref BLML-auto-diff
Reads every results/*/**/<run>*_<covset>_<model>.json (newest file wins for duplicates). For runs with
several folds (fwd2024-06, ..., expanding2024-06, ...) the folds are pooled.
"""
import sys, os, json, glob, collections, argparse
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument('run'); ap.add_argument('covsets'); ap.add_argument('--ref', default='BLML-auto')
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
files = {}
for f in glob.glob(os.path.join(HERE, 'results', '*', '**', f'{a.run}*.json'), recursive=True):
    base = os.path.basename(f)[:-5]
    run, cs, model = base.split('_', 2)
    if cs not in a.covsets.split(','):
        continue
    key = (run, cs, model)
    if key not in files or os.path.getmtime(f) > os.path.getmtime(files[key]):
        files[key] = f
data = collections.defaultdict(dict)          # (covset, model) -> {time: logpdf}
for (run, cs, model), f in files.items():
    r = json.load(open(f))
    po = r.get('per_obs', {})
    if 'logpdf' not in po:
        continue
    data[(cs, model)].update(zip(r['test_times'], po['logpdf']))
def daily(d):
    out = collections.defaultdict(list)
    for t, v in d.items():
        out[t[:10]].append(v)
    return {k: np.mean(v) for k, v in out.items()}
for cs in a.covsets.split(','):
    models = sorted([m for (c, m) in data if c == cs], key=lambda m: -np.mean(list(data[(cs, m)].values())))
    if not models:
        continue
    print(f'\n== {a.run} | {cs}   (test LL, nats/hour; paired day-level test vs {a.ref}: diff = ref - model)')
    ref = daily(data[(cs, a.ref)]) if (cs, a.ref) in data else None
    for m in models:
        ll = np.mean(list(data[(cs, m)].values()))
        line = f'  {m:22s} {ll:8.3f}  n={len(data[(cs, m)])}'
        if ref is not None and m != a.ref:
            dm = daily(data[(cs, m)]); days = sorted(set(ref) & set(dm))
            x = np.array([ref[d] - dm[d] for d in days])
            line += f'   diff {x.mean():+.3f} ±{1.96 * x.std(ddof=1) / np.sqrt(len(x)):.3f}  p={stats.ttest_1samp(x, 0).pvalue:.3g}'
        print(line)
