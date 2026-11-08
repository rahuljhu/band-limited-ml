"""Paper 3 diagnostic: is BLML's gap to the KDE on the double-slit records in the fit or in the band selection?
For each record, with the main split (64/16/20, seed 0) and the background mode N=1: over the fc grid, the validation
log-likelihood (fit on training) and the test log-likelihood (fit on training + validation), against the KDE over its
bandwidth grid scored the same two ways. -> results/p3_diag_band/<set>.json
Run from the repo root: BLML_DATA_DIR=~/data_repo/dslit_gpu BLML_ENGINE=cpu BLML_COLLAPSE=1 BLML_BG_HARM=1,1
  .venv/bin/python results/p3_diag_band.py <set> [<set> ...]"""
import os, sys, json, numpy as np
os.environ.setdefault('BLML_COLLAPSE', '1'); os.environ.setdefault('BLML_BG_HARM', '1,1'); os.environ.setdefault('BLML_ENGINE', 'cpu')
sys.path.insert(0, '.')
import dslit_job as dj
_argv, sys.argv = sys.argv, sys.argv[:1]                 # p3_buildup runs the sets named on its command line
from p3_buildup import setup, counts
sys.argv = _argv
OUT = 'results/p3_diag_band'; os.makedirs(OUT, exist_ok=True)
for name in sys.argv[1:]:
    dj.BFIX = np.array([np.nan])
    x, W, bflat, trv, te = setup(name)
    u = np.random.default_rng(0).random(len(x)); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))
    TR, VA, TE = (counts(x, np.nonzero(part == k)[0], W) for k in range(3)); TV = TR + VA
    rows = []
    for f in dj.FC_GRID[dj.FC_GRID >= 2.0 / W]:
        P, _, b1 = dj.bg_fit(TR, [f], (1,), True, 1, max_sweeps=5000, tol=1e-6)
        Q, _, b2 = dj.bg_fit(TV, [f], (1,), True, 1, max_sweeps=5000, tol=1e-6)
        rows.append(dict(fc=float(f), val=dj.ll(P, VA), test=dj.ll(Q, TE), beta=float(b2)))
        print(name, {k: round(v, 5) for k, v in rows[-1].items()}, flush=True)
    kde = []
    for h in (0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16):
        kde.append(dict(h=h, val=dj.ll(dj.em(TR, bflat, dj.kde_q((h,)))[0], VA), test=dj.ll(dj.em(TV, bflat, dj.kde_q((h,)))[0], TE)))
    hist = dj.ll(TV + 0.5, TE)
    json.dump(dict(set=name, n_val=float(VA.sum()), n_test=float(TE.sum()), blml=rows, kde=kde, hist=hist),
              open(f'{OUT}/{name}.json', 'w'), indent=1)
