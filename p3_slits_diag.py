"""Paper 3: why does the two-slit kernel select a wide aperture on the real records? Partial coherence: the intensity is
V |g1 + g2|^2 + (1 - V)(|g1|^2 + |g2|^2), whose incoherent part no single field on the slits produces. Candidate models,
all by the protocol's 5-fold CV and one-SE rule over (fc, fill), compared by their CV log-likelihood on the same folds:
  K1-N<n>  one slit-kernel mode + the background mode of n harmonics (the incoherent envelopes are smooth: a few harmonics)
  K2-N1    two slit-kernel modes (a shared kernel) + the background mode N = 1 (the partially coherent two-slit model)
fc on the grid within 0.8-1.6 times the record's plain-band CV fc, fill on a fine grid. Records: the real data sets
(paper split) and the synthetic partially coherent truth of p3_dbench (V = 0.6, true fill 0.45, n = 10^4).
  [SPLIT_SEED=s] BLML_DATA_DIR=~/data_repo/dslit_gpu python p3_slits_diag.py <set> <config>
  -> results/p3_slits_diag/<set>_<config>[_s<s>].json"""
import os, sys, json, time, numpy as np
os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_SOLVER', 'lbfgs')
name, cfg = sys.argv[1], sys.argv[2]; sys.argv = sys.argv[:1]; sys.path.insert(0, '.')
import dslit_job as dj
SEED = int(os.environ.get('SPLIT_SEED', '0'))     # the train / validation / test split (0: the paper's)
OUT = 'results/p3_slits_diag'; os.makedirs(OUT, exist_ok=True); fn = f'{OUT}/{name}_{cfg}' + (f'_s{SEED}' if SEED else '') + '.json'
if os.path.exists(fn):
    sys.exit(0)
K = int(cfg[1]); NBG = int(cfg.split('-N')[1])
FILLS = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.36, 0.33, 0.3, 0.25]
if name == 'synth_partial':                       # p3_dbench's partial case, bg 0, n = 10^4, rep 0
    W, NF = 768, 1 << 16; nu = np.fft.fftfreq(NF); D_, A_, V = 0.017, 0.005, 0.6
    sl = lambda c: np.where(np.abs(nu - c) / (A_ / 2) < 1, np.cos(np.pi * np.abs(nu - c) / (A_ / 2) / 2) ** 2, 0.0)
    fld = lambda A: np.fft.fftshift(np.fft.ifft(A))[NF // 2 - W // 2:NF // 2 + W // 2]
    g1, g2 = fld(sl(-D_ / 2)), fld(sl(D_ / 2)); P = V * np.abs(g1 + g2) ** 2 + (1 - V) * (np.abs(g1) ** 2 + np.abs(g2) ** 2)
    P = P / P.sum(); rng = np.random.default_rng(7); x = rng.choice(W, size=10000, p=P)
    u = rng.random(len(x)); fc_ref = None
else:
    _a, sys.argv = sys.argv, sys.argv[:1]
    from p3_buildup import setup
    x, W, bflat, trv, te = setup(name); u = np.random.default_rng(SEED).random(len(x))
    fc_ref = json.load(open(f'results/p3synth_lbfgs/bg1/{name}/{name}_complex_bg.json'))['fc'][0]
part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2)); fold = np.random.default_rng(1000 + SEED).integers(0, 5, len(x))
cnt = lambda m: np.bincount(x[m], minlength=W).astype(float)
TV, TE = cnt(part < 2), cnt(part == 2)
FOLDS = [(cnt((part < 2) & (fold != k)), cnt((part < 2) & (fold == k))) for k in range(5)]
G = list(dj.FC_GRID[dj.FC_GRID >= 2.0 / W])
if fc_ref is None:                                # the plain band's CV fc (N = 1), as the reference for the fc window
    fc_ref, _ = dj.cv_select(FOLDS, lambda f, T: dj.bg_fit(T, [f], (1,), True, 1, max_sweeps=5000, tol=1e-6)[0], G, patience=6)
Gs = [float(f) for f in G if 0.8 * fc_ref <= f <= 1.6 * fc_ref]
fit = lambda T, f, w, init=None: dj.bg_fit(T, [f], (NBG,), True, K, init=init, max_sweeps=5000, tol=1e-6, fill=w)
t0 = time.time(); SL = {}
for k, (T, _) in enumerate(FOLDS):                # per fill, increasing fc, warm-started
    for w in FILLS:
        prev = None
        for f in Gs:
            P_, prev, _ = fit(T, f, w, prev); SL[(f, w, k)] = P_
fold_of = {id(T): k for k, (T, _) in enumerate(FOLDS)}
C_ = sorted(((f, w) for f in Gs for w in FILLS), key=lambda c: (c[0] * c[1], c[0]))
(fs, fill), tab = dj.cv_select(FOLDS, lambda c, T: SL[(c[0], c[1], fold_of[id(T)])], C_)
best = max(tab, key=lambda r: r['cv'])
Q, m, beta = fit(TV, fs, fill)
res = dict(set=name, config=cfg, seed=SEED, K=K, N=NBG, fc=fs, fill=fill, fc_ref=fc_ref, cv_at_pick=next(r['cv'] for r in tab if r['cand'] == (fs, fill)),
           cv_best=best['cv'], best_cand=list(best['cand']), test=dj.ll(Q, TE), beta=beta, s=time.time() - t0,
           cv_by_fill={str(w): max(r['cv'] for r in tab if r['cand'][1] == w) for w in FILLS})
json.dump(res, open(fn, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'cv_by_fill'}))
