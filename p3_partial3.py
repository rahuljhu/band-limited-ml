"""Paper 3: the partially coherent synthetic case of p3_dbench.py, I = V |g1 + g2|^2 + (1 - V)(|g1|^2 + |g2|^2), fitted
by the model that contains it: one two-slit mode (fc, fill) for the coherent part, two modes of the single-slit band for
the incoherent part, and the background mode N = 1 ('three') or a flat one, N = 0 ('three_flat': the N = 1 band is
about the slit band, so with N = 1 the background mode takes the incoherent part). A slit field g1 = e^{2 pi i c x} h1(x) with h1 of band a has
|g1|^2 = |h1|^2, so the single-slit modes are sinc modes of fc a = fill fc / 2 (the carrier drops out of the intensity);
g1, g2 have disjoint spectra, so V = Q_coh / (Q_coh + Q_1 + Q_2). Against it, the one-mode fit (K1-N1). Same data and
split as p3_dbench (case partial, background share bg, n, rep); (fc, fill) by 5-fold CV and the one-SE rule (parsimony
fc fill), fc within 0.8-1.6 times the plain band's CV fc.
  python p3_partial3.py <n> <rep> <bg>   -> results/p3_partial3/n<n>_r<rep>_bg<bg>.json"""
import os, sys, json, time, numpy as np
os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_SOLVER', 'lbfgs')
N, REP, BG = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]); sys.argv = sys.argv[:1]; sys.path.insert(0, '.')
import dslit_job as dj, blml_trivial as bt
from blml_binned import BLMLBinned
OUT = 'results/p3_partial3'; os.makedirs(OUT, exist_ok=True); fn = f'{OUT}/n{N}_r{REP}_bg{BG:g}.json'
if os.path.exists(fn):
    sys.exit(0)
W, NF = 768, 1 << 16; D_, A_, V = 0.017, 0.005, 0.6                  # as p3_dbench: fill 2a/(d+a) = 0.4545
nu = np.fft.fftfreq(NF)
slit = lambda c: np.where(np.abs(nu - c) / (A_ / 2) < 1, np.cos(np.pi * np.abs(nu - c) / (A_ / 2) / 2) ** 2, 0.0)
field = lambda A: np.fft.fftshift(np.fft.ifft(A))[NF // 2 - W // 2:NF // 2 + W // 2] * NF
g1, g2 = field(slit(-D_ / 2)), field(slit(D_ / 2))
I = V * np.abs(g1 + g2) ** 2 + (1 - V) * (np.abs(g1) ** 2 + np.abs(g2) ** 2)
P = (1 - BG) * I / I.sum() + BG / W; P = P / P.sum()
rng = np.random.default_rng(1000 * REP + N % 997)
x = rng.choice(W, size=N, p=P); part = rng.choice(3, size=N, p=[0.64, 0.16, 0.20])
TV, TE = np.bincount(x[part < 2], minlength=W).astype(float), np.bincount(x[part == 2], minlength=W).astype(float)
fold = rng.integers(0, 5, size=N)
FOLDS = [(np.bincount(x[(part < 2) & (fold != k)], minlength=W).astype(float),
          np.bincount(x[(part < 2) & (fold == k)], minlength=W).astype(float)) for k in range(5)]
fold_of = {id(T): k for k, (T, _) in enumerate(FOLDS)}
kl = lambda Q: float((P * np.log(P / np.maximum(Q / Q.sum(), 1e-300))).sum())
FILLS = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.36, 0.33, 0.3, 0.25]
G = list(dj.FC_GRID[dj.FC_GRID >= 2.0 / W]); dj.BFIX = np.array([np.nan])
fc_ref, _ = dj.cv_select(FOLDS, lambda f, T: dj.bg_fit(T, [f], (1,), True, 1, max_sweeps=5000, tol=1e-6)[0], G, patience=6)
Gs = [float(f) for f in G if 0.8 * fc_ref <= f <= 1.6 * fc_ref]
def modes(fc, w, three, nbg=1):
    m = dj.bg_modes([fc], (W,), (nbg,), 1, w)                       # [two-slit mode, background mode]
    return m[:1] + ([bt.make_modes([w * fc / 2], ['sinc'])[0]] * 2 if three else []) + m[1:]
def fit(T, fc, w, three, init=None, nbg=1):
    m = BLMLBinned(tol=1e-6, max_sweeps=5000, ll_tol=1e-10, engine=dj.ENGINE, solver=dj.SOLVER).fit(
        T, modes(fc, w, three, nbg), complex_g=True, init=init)
    U = dj._np(m.Se_); E = dj._np(m.e_hat_); Q = np.array([float(np.real(np.sum(np.conj(E[k]) * U[k]))) for k in range(len(E))])
    Pm = np.maximum(dj._np(m.density_), 1e-300); return Pm / Pm.sum(), m, Q / Q.sum()
res = dict(n=N, rep=REP, bg=BG, V=V, fill_true=2 * A_ / (D_ + A_), a_true=A_, d_true=D_, fc_ref=fc_ref); t0 = time.time()
for lab, three, nbg in (('one', False, 1), ('three', True, 1), ('three_flat', True, 0)):
    SL = {}
    for k, (T, _) in enumerate(FOLDS):                              # per fill, increasing fc, warm-started
        for w in FILLS:
            prev = None
            for f in Gs:
                P_, m_, _ = fit(T, f, w, three, prev, nbg); prev = m_; SL[(f, w, k)] = P_
    C_ = sorted(((f, w) for f in Gs for w in FILLS), key=lambda c: (c[0] * c[1], c[0]))
    (fc, w), tab = dj.cv_select(FOLDS, lambda c, T: SL[(c[0], c[1], fold_of[id(T)])], C_)
    Q, m, sh = fit(TV, fc, w, three, None, nbg); a = w * fc / 2
    r = dict(fc=fc, fill=w, a=a, d=fc - a, cv=max(rr['cv'] for rr in tab), test=dj.ll(Q, TE), kl=kl(Q), share=sh.tolist(),
             cv_by_fill={str(v): max(rr['cv'] for rr in tab if rr['cand'][1] == v) for v in FILLS})
    if three:
        r['V_hat'] = float(sh[0] / sh[:3].sum())
    res[lab] = r
res['s'] = time.time() - t0
json.dump(res, open(fn, 'w'), indent=1)
print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != 'cv_by_fill'} if isinstance(v, dict) else v) for k, v in res.items()}))
