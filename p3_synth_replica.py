"""Paper 3: synthetic replicas of the double-slit records. Truth: the record's Fresnel fit with its flat background
(results/dslit_1d/<set>/flat/<set>_fresnel_P.npy, copied to the data folder as dslit_<set>_fresnel_P.npy) on the record's crop, so the pattern is cut where the record's is;
the record's particle count, split 64/16/20, the 5-fold CV and one-SE rule of dslit_job (BLML_SELECT=cv). Estimators:
KDE with the flat background by EM, complex BLML with the background mode N = 1, complex BLML with the EM flat
background (fc by CV), and streaming BLML with the background mode N = 0, 1 (fc learned online; train + validation in
arrival order, batches of BLML_STREAM_NB) with the final batch fit at the learned fc. Scores: test log-likelihood per particle (each fit renormalised on the crop) and KL from the truth.
Usage: BLML_DATA_DIR=~/data_repo/dslit_gpu .venv/bin/python p3_synth_replica.py <set> <rep> [variant]
  -> $BLML_OUT_DIR (default results)/p3_synth/<set>_<variant>_<rep>.json (skipped if it exists)
variants: base (the Fresnel truth on the record's crop), wide (the same truth on a window twice as wide, W/2 more
on each side: the pattern no longer cut; the same expected count inside the old crop)."""
import os, sys, json, time, numpy as np
os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_COLLAPSE', '1')
name, rep = sys.argv[1], int(sys.argv[2]); variant = sys.argv[3] if len(sys.argv) > 3 else 'base'
sys.argv = sys.argv[:1]; sys.path.insert(0, '.')
import dslit_job as dj
OUT = os.path.join(os.environ.get('BLML_OUT_DIR', 'results'), 'p3_synth'); os.makedirs(OUT, exist_ok=True)
if os.path.exists(f'{OUT}/{name}_{variant}_{rep}.json'): sys.exit(0)
P = np.load(os.path.join(dj.DATA, f'dslit_{name}_' + ('eq_' if variant == 'eq' else '') + 'fresnel_P.npy')).astype(float)
P = P / P.sum(); W = len(P)                       # variant eq: the Fresnel fit with equal slits (rho = 1, V free)
n = len(np.load(os.path.join(dj.DATA, f'dslit_{name}_points.npy')))
if variant == 'wide':                             # the same truth on a window W/2 wider on each side: the Fresnel field
    import json as _j                             # beyond the crop and the flat background at the same level per pixel,
    th = np.array(_j.load(open(f'results/dslit_1d/{name}/flat/{name}_fresnel.json'))['theta'])   # so the expected
    E = W // 2; dj.BFIX = np.array([np.nan])      # count inside the old crop is the record's n
    beta = float(np.atleast_1d(dj.ref_P(th, np.ones((1, W)) / W, 0, True)[2])[0])
    th[0] += E; q = dj.ref_parts(th, (W + 2 * E,), 0, True)[0]; q = q / q[E:E + W].sum()
    Pu = beta / W + (1 - beta) * q
    assert np.abs(Pu[E:E + W] - P).max() < 1e-6 * P.max(), 'wide truth does not match the crop truth'
    n = int(round(n * Pu.sum())); P = Pu / Pu.sum(); W = len(P)
rng = np.random.default_rng(100 + rep)
x = rng.choice(W, size=n, p=P); part = rng.choice(3, size=n, p=[0.64, 0.16, 0.20])   # particles in arrival order
TR, VA, TE = (np.bincount(x[part == k], minlength=W).astype(float) for k in range(3)); TV = TR + VA
fold = np.stack([rng.multinomial(int(c), [0.2] * 5) for c in TV], 1).astype(float)          # (5, W)
FOLDS = [(TV - fold[k], fold[k]) for k in range(5)]
dj.BFIX = np.array([np.nan]); bflat = np.ones((1, W)) / W
kl = lambda Q: float((P * np.log(P / np.maximum(Q / Q.sum(), 1e-300))).sum())
res = dict(set=name, rep=rep, variant=variant, n=n, truth_test=dj.ll(P, TE)); t0 = time.time()
HG = [(h,) for h in (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24)]
h, _ = dj.cv_select(FOLDS, lambda h_, T: dj.em(T, bflat, dj.kde_q(h_))[0], HG[::-1])
Q = dj.em(TV, bflat, dj.kde_q(h))[0]; res['kde'] = dict(h=h[0], test=dj.ll(Q, TE), kl=kl(Q))
G = list(dj.FC_GRID[dj.FC_GRID >= 2.0 / W])
f_, _ = dj.cv_select(FOLDS, lambda f, T: dj.bg_fit(T, [f], (1,), True, 1, max_sweeps=dj.SEARCH_SWEEPS, tol=1e-6)[0], G)
Q, _, beta = dj.bg_fit(TV, [f_], (1,), True, 1, max_sweeps=5000, tol=1e-5)
res['blml_bg1'] = dict(fc=float(f_), test=dj.ll(Q, TE), kl=kl(Q), beta=beta)
f_, _ = dj.cv_select(FOLDS, lambda f, T: dj.em(T, bflat, dj.blml_q([f], 'complex'), tol=1e-6)[0], G)
Q = dj.em_polish(TV, bflat, np.array([f_]), 'complex')[0]
res['blml_em'] = dict(fc=float(f_), test=dj.ll(Q, TE), kl=kl(Q))
xs = x[part < 2]; NB = int(os.environ.get('BLML_STREAM_NB', '400'))   # streaming, train + validation in arrival order
batches = [np.bincount(xs[b0:b0 + NB], minlength=W).astype(float) for b0 in range(0, len(xs), NB)]
for N in (0, 1):
    m, Q, beta = dj.stream_bg_fit(batches, (W,), (N,)); fs = float(m._fc_now()[0][0])
    Qr, _, br = dj.bg_fit(TV, [fs], (N,), True, 1, max_sweeps=5000, tol=1e-5)
    res[f'stream_bg{N}'] = dict(fc=fs, test=dj.ll(Q, TE), kl=kl(Q), beta=beta, refit_test=dj.ll(Qr, TE), refit_kl=kl(Qr), epochs=m.epochs_)
res['s'] = time.time() - t0
json.dump(res, open(f'{OUT}/{name}_{variant}_{rep}.json', 'w'), indent=1)
print(json.dumps(res))
