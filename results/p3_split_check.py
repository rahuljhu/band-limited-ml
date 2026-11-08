"""Paper 3: is a real record's BLML - KDE gap a property of the split? The record's particles re-split at random
(64/16/20, seeds 1..5; seed 0 is the paper's split) and both estimators refitted by the protocol (5-fold CV, one-SE rule):
complex BLML with the background mode N = 1 (L-BFGS) and the KDE with the flat background by EM. Prints the test
log-likelihood difference per particle and its paired SE. Usage: python results/p3_split_check.py <set> [seeds]"""
import os, sys, numpy as np
os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_SOLVER', 'lbfgs')
name = sys.argv[1]; seeds = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else '0,1,2,3,4,5').split(',')]
sys.argv = sys.argv[:1]; sys.path.insert(0, '.')
import dslit_job as dj
_a, sys.argv = sys.argv, sys.argv[:1]
from p3_buildup import setup
x, W, bflat, trv, te = setup(name); n = len(x); dj.BFIX = np.array([np.nan])
G = list(dj.FC_GRID[dj.FC_GRID >= 2.0 / W]); HG = [(h,) for h in (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24)]
for sd in seeds:
    rng = np.random.default_rng(sd); u = rng.random(n); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))
    fold = np.random.default_rng(1000 + sd).integers(0, 5, n)
    cnt = lambda m: np.bincount(x[m], minlength=W).astype(float)
    TV, TE = cnt(part < 2), cnt(part == 2)
    FOLDS = [(cnt((part < 2) & (fold != k)), cnt((part < 2) & (fold == k))) for k in range(5)]
    f_, _ = dj.cv_select(FOLDS, lambda f, T: dj.bg_fit(T, [f], (1,), True, 1, max_sweeps=5000, tol=1e-6)[0], G, patience=6)
    B = dj.bg_fit(TV, [f_], (1,), True, 1, max_sweeps=5000, tol=1e-6)[0]
    h, _ = dj.cv_select(FOLDS, lambda h_, T: dj.em(T, bflat, dj.kde_q(h_))[0], HG[::-1])
    K = dj.em(TV, bflat, dj.kde_q(h))[0]
    d = np.repeat(np.log(B / B.sum()) - np.log(K / K.sum()), TE.astype(int))
    print(f'{name} seed {sd}: BLML fc {f_:.4f}  KDE h {h[0]:g}  BLML - KDE {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}', flush=True)
