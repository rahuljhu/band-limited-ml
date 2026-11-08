"""Paper 3: the periodic two-slit kernel (kind 'dirichlet:<fill>', period = the window) on a double-slit record or its
replica, against the line two-slit kernel ('slits') and the KDE on the same window. The window: the record's crop
trimmed to centre +- (m + 1/2) P (P the fringe period, centre and P from the record's Fresnel fit, m the largest that
fits), so both edges sit on matching troughs and the periodic field joins smoothly. Periodic candidates are harmonic
ranges lo <= |k| <= hi (fc = 2 hi / L, fill = 1 - (lo - 1/2) / hi; lo = 0 the plain band), hi from 0.8 to 2 times the
physical band; the background mode N in {0, 1, 2} (periodic over the window, as always). Every choice by 5-fold CV
with the one-SE rule (parsimony: fewest harmonics); split and replicas as p3_slit_main.py.
  python p3_periodic_main.py <set> <seed> [replica]   -> results/p3_periodic_main/<set>[_rep<k>]_s<seed>.json"""
import os, sys, json, time, numpy as np
os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_SOLVER', 'lbfgs'); os.environ.setdefault('BLML_COLLAPSE', '1')
name, SEED = sys.argv[1], int(sys.argv[2]); REP = int(sys.argv[3]) if len(sys.argv) > 3 else None
sys.argv = sys.argv[:1]; sys.path.insert(0, '.')
import dslit_job as dj
OUT = 'results/p3_periodic_main'; os.makedirs(OUT, exist_ok=True)
fn = f'{OUT}/{name}' + ('' if REP is None else f'_rep{REP}') + f'_s{SEED}.json'
if os.path.exists(fn):
    sys.exit(0)
fr = json.load(open(f'results/dslit_1d/{name}/flat/{name}_fresnel.json'))['params']
Pf, r, c0 = fr['P'], fr['a_over_d'], fr['s0']; phys = (1 + r) / Pf
if REP is None:
    from p3_buildup import setup
    x, W = setup(name)[:2]; Ptrue = None
else:
    Ptrue = np.load(os.path.join(dj.DATA, f'dslit_{name}_fresnel_P.npy')).astype(float); Ptrue /= Ptrue.sum(); W = len(Ptrue)
    n = len(np.load(os.path.join(dj.DATA, f'dslit_{name}_points.npy')))
    x = np.random.default_rng(100 + REP).choice(W, size=n, p=Ptrue)
m_ = int(min(c0, W - c0) / Pf - 0.5); a0, a1 = int(round(c0 - (m_ + 0.5) * Pf)), int(round(c0 + (m_ + 0.5) * Pf))
keep = (x >= a0) & (x < a1); x = x[keep] - a0; L = a1 - a0
if Ptrue is not None:
    Ptrue = Ptrue[a0:a1] / Ptrue[a0:a1].sum()
u = np.random.default_rng(SEED).random(len(x)); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))
fold = np.random.default_rng(1000 + SEED).integers(0, 5, len(x))
cnt = lambda msk: np.bincount(x[msk], minlength=L).astype(float)
TV, TE = cnt(part < 2), cnt(part == 2)
FOLDS = [(cnt((part < 2) & (fold != k)), cnt((part < 2) & (fold == k))) for k in range(5)]
fold_of = {id(T): k for k, (T, _) in enumerate(FOLDS)}
kl = (lambda Q: float((Ptrue * np.log(Ptrue / np.maximum(Q / Q.sum(), 1e-300))).sum())) if Ptrue is not None else (lambda Q: None)
dj.BFIX = np.array([np.nan]); t0 = time.time()
res = dict(set=name, seed=SEED, rep=REP, window=[a0, a1], L=L, m=m_, n_kept=int(len(x)), n_record=int(len(keep)), phys=phys)
from blml_binned import BLMLBinned
import blml_trivial as bt
def pfit(T, hi, lo, N, init=None):
    fc = (2 * hi + 0.5) / L; w = 1.0 if lo == 0 else 1.0 - (lo - 0.5) / (hi + 0.25)
    sig = {0: dict(kind='dirichlet', fc=fc, period=float(L), **({} if w >= 1 else dict(fill=w)))}
    bg = {0: dict(kind='dirichlet', fc=(N + 0.5) * 2 / L, period=float(L))}
    m = BLMLBinned(tol=1e-6, max_sweeps=5000, ll_tol=1e-10, engine=dj.ENGINE, solver=dj.SOLVER).fit(T, [sig, bg], complex_g=True, init=init)
    U = dj._np(m.Se_); E = dj._np(m.e_hat_); Qs = np.array([float(np.real(np.sum(np.conj(E[k]) * U[k]))) for k in range(2)])
    P_ = np.maximum(dj._np(m.density_), 1e-300); return P_ / P_.sum(), m, float(Qs[1] / Qs.sum())
his = [h for h in range(1, int(2.0 * phys * L / 2) + 2) if h >= 0.8 * phys * L / 2 - 0.5]
best = None
for N in (0, 1, 2):
    C_ = sorted(((hi, lo) for hi in his for lo in range(0, hi + 1) if lo == 0 or lo >= 1), key=lambda c: (c[0] - c[1] + 1, c[0]))
    SL = {}
    for k, (T, _) in enumerate(FOLDS):
        for c in C_:
            SL[(c, k)] = pfit(T, c[0], c[1], N)[0]
    (hi, lo), tab = dj.cv_select(FOLDS, lambda c, T: SL[(c, fold_of[id(T)])], C_)
    cvb = max(rr['cv'] for rr in tab); res[f'per_N{N}'] = dict(hi=hi, lo=lo, cv=cvb)
    if best is None or cvb > best[0]:
        best = (cvb, N, hi, lo)
_, N, hi, lo = best
Q, m, beta = pfit(TV, hi, lo, N); Lp = np.log(np.maximum(Q, 1e-300))
w_eff = 1.0 if lo == 0 else 1.0 - (lo - 0.5) / (hi + 0.25)
res['periodic'] = dict(N=N, hi=hi, lo=lo, fc=(2 * hi + 0.5) / L, fill=w_eff, ratio=(2 * hi + 0.5) / L / phys, beta=beta, test=dj.ll(Q, TE), kl=kl(Q))
FILLS = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.36, 0.33, 0.3, 0.25]           # the line two-slit kernel on the same window
G = [float(f) for f in dj.FC_GRID if 0.8 * phys <= f <= 2.0 * phys]
lfit = lambda T, f, w, N, init=None: dj.bg_fit(T, [f], (N,), True, 1, init=init, max_sweeps=5000, tol=1e-6, fill=w)
bestl = None
for N in (0, 1, 2):
    SL = {}
    for k, (T, _) in enumerate(FOLDS):
        for w in FILLS:
            prev = None
            for f in G:
                P_, prev, _ = lfit(T, f, w, N, prev); SL[(f, w, k)] = P_
    C_ = sorted(((f, w) for f in G for w in FILLS), key=lambda c: (c[0] * c[1], c[0]))
    (fc, w), tab = dj.cv_select(FOLDS, lambda c, T: SL[(c[0], c[1], fold_of[id(T)])], C_)
    cvb = max(rr['cv'] for rr in tab)
    if bestl is None or cvb > bestl[0]:
        bestl = (cvb, N, fc, w)
_, N, fc, w = bestl
Q, _, beta = lfit(TV, fc, w, N); Ll = np.log(np.maximum(Q, 1e-300))
res['line'] = dict(N=N, fc=fc, fill=w, ratio=fc / phys, beta=beta, test=dj.ll(Q, TE), kl=kl(Q), cv=bestl[0])
bflat = np.ones((1, L)) / L
HG = [(h,) for h in (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24)]
h, _ = dj.cv_select(FOLDS, lambda h_, T: dj.em(T, bflat, dj.kde_q(h_))[0], HG[::-1])
Qk = dj.em(TV, bflat, dj.kde_q(h))[0]; Lk = np.log(np.maximum(Qk / Qk.sum(), 1e-300)); res['kde'] = dict(h=h[0], test=dj.ll(Qk, TE), kl=kl(Qk))
for a, La in (('periodic', Lp), ('line', Ll)):
    d = np.repeat(La - Lk, TE.astype(int)); res[f'{a}_minus_kde'] = dict(diff=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))))
d = np.repeat(Lp - Ll, TE.astype(int)); res['periodic_minus_line'] = dict(diff=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))))
res['s'] = time.time() - t0
json.dump(res, open(fn, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if not k.startswith('per_N')}))
