"""Paper 3, cropped benchmark: what a record-like crop determines. Truth: the far field of two sharp slits (a/d = 0.2,
fringe period P = 56 px, centre at 256) on a 512-px detector, about 9 fringes, as the photon records; cases sym (equal
slits), asym (amplitudes 1 and RHO, relative phase PHI) and partial (equal slits, visibility V); a flat background
share bg. The window is trimmed to centre +- (m + 1/2) P, both edges on troughs (L = 9 P = 504 px), so each slit's
centre falls halfway between two harmonics of the window. Estimators, each by 5-fold CV with the one-SE rule on
train + validation (64/16/20 split): the periodic two-slit kernel ('dirichlet:<fill>' over the window; harmonic range
lo <= |k| <= hi and background harmonics N in {0, 1, 2}, or CB_N), the line two-slit kernel ('slits'; fc, fill, N) and the KDE
(flat background by EM); for partial also the periodic three-mode model at the periodic fit's harmonics (V_hat
its coherent share). Scores: test log-likelihood and KL from the window's truth; for the two kernels, from the
coherent mode's field on the window (its Fourier series a_k, k > 0 one slit and k < 0 the other): the slit separation
d (|a_k|^2 centroids), the amplitude ratio rho (side energies; min(rho, 1/rho), the twin swaps them) and the relative
phase phi = arg(sum_{k>0} a_k / sum_{k<0} a_k) (invariant to the global phase and the twin), each against the same
quantity of the true field on the window.
  CB_TRUTH=tapered: Fig. 1's truth (p3_ladder.py) on 2048 px, with CB_M=4 the window of nine fringes.
  [CB_N=0 CB_OUT=<dir> CB_TRUTH=tapered CB_M=4] python p3_cbench.py <case> <bg> <n> <rep>   -> results/p3_cbench/<case>_bg<bg>_n<n>_<rep>.json"""
import os, sys, json, time, numpy as np
os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_SOLVER', 'lbfgs'); os.environ.setdefault('BLML_COLLAPSE', '1')
case, BG, N_, REP = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); sys.argv = sys.argv[:1]
sys.path.insert(0, '.')
import dslit_job as dj
from blml_binned import BLMLBinned
OUT = os.environ.get('CB_OUT', 'results/p3_cbench'); NS_ = tuple(int(v) for v in os.environ.get('CB_N', '0,1,2').split(',')); os.makedirs(OUT, exist_ok=True); fn = f'{OUT}/{case}_bg{BG:g}_n{N_}_{REP}.json'
if os.path.exists(fn):
    sys.exit(0)
RHO, PHI, V = (0.6, 1.0, 1.0) if case == 'asym' else (1.0, 0.0, 0.6) if case == 'partial' else (1.0, 0.0, 1.0)
if os.environ.get('CB_TRUTH', 'sharp') == 'tapered':   # Fig. 1's truth (p3_ladder.py): cos^2-tapered slits, d 0.017, a 0.005
    D_, A_, NF = 0.017, 0.005, 1 << 16; W, C0 = 2048, 1024.0; Pf, R = 1 / D_, A_ / D_; nu_ = np.fft.fftfreq(NF)
    sl = lambda c: np.where(np.abs(nu_ - c) < A_ / 2, np.cos(np.pi * np.abs(nu_ - c) / A_) ** 2, 0.0)
    fl = lambda A: np.fft.fftshift(np.fft.ifft(A))[NF // 2 - W // 2:NF // 2 + W // 2] * NF
    g1, g2 = fl(sl(-D_ / 2)), fl(sl(D_ / 2)); field = g1 + RHO * np.exp(1j * PHI) * g2
    I = V * np.abs(field) ** 2 + (1 - V) * (np.abs(g1) ** 2 + RHO ** 2 * np.abs(g2) ** 2)
else:
    W, Pf, R, C0 = 512, 56.0, 0.2, 256.0
    lg = lambda p: float(np.log(p / (1 - p))) if p < 1 else 30.0
    th = np.array([C0, Pf, lg(R), 0.0, np.log(RHO), PHI, lg(V), 0.0, 0.0])
    I, field, _ = dj.ref_parts(th, (W,), 0, True)
P = (1 - BG) * I / I.sum() + BG / W; P /= P.sum()
m_ = int(os.environ.get('CB_M', '0')) or int(min(C0, W - C0) / Pf - 0.5); a0, a1 = int(round(C0 - (m_ + 0.5) * Pf)), int(round(C0 + (m_ + 0.5) * Pf)); L = a1 - a0
rng = np.random.default_rng(1000 * REP + N_ % 997)
x = rng.choice(W, size=N_, p=P); x = x[(x >= a0) & (x < a1)] - a0
Pw = P[a0:a1] / P[a0:a1].sum(); gw = field[a0:a1]
part = rng.choice(3, size=len(x), p=[0.64, 0.16, 0.20]); fold = rng.integers(0, 5, size=len(x))
cnt = lambda msk: np.bincount(x[msk], minlength=L).astype(float)
TV, TE = cnt(part < 2), cnt(part == 2)
FOLDS = [(cnt((part < 2) & (fold != k)), cnt((part < 2) & (fold == k))) for k in range(5)]
fold_of = {id(T): k for k, (T, _) in enumerate(FOLDS)}
kl = lambda Q: float((Pw * np.log(Pw / np.maximum(Q / Q.sum(), 1e-300))).sum())
phys = (1 + R) / Pf; dj.BFIX = np.array([np.nan])
def slit_metrics(g):
    a = np.fft.fft(g); k = np.fft.fftfreq(L) * L; pos, neg = (k > 0) & (k < L / 4), (k < 0) & (k > -L / 4)
    e = np.abs(a) ** 2; cp_, cn = (e[pos] * k[pos]).sum() / e[pos].sum(), (e[neg] * k[neg]).sum() / e[neg].sum()
    rho = np.sqrt(e[pos].sum() / e[neg].sum())
    return dict(d=(cp_ - cn) / L, rho=float(min(rho, 1 / rho)), phi=float(np.angle(a[pos].sum() / a[neg].sum())))
Mt = slit_metrics(gw)
res = dict(case=case, bg=BG, n=N_, rep=REP, L=L, n_kept=int(len(x)), truth=Mt, d_nominal=1 / Pf, truth_test=dj.ll(Pw, TE))
def score(name, Q, g=None, **ex):
    r = dict(test=dj.ll(Q, TE), kl=kl(Q), **ex)
    if g is not None:
        M = slit_metrics(g)
        r.update(M, d_err=abs(M['d'] / Mt['d'] - 1), rho_err=abs(M['rho'] - Mt['rho']),
                 phi_err=float(abs(np.angle(np.exp(1j * (M['phi'] - Mt['phi']))))))
    res[name] = r
line = lambda m: np.asarray(dj._np(m.Se_))[0] / m.n_
def pfit(T, hi, lo, N, init=None):
    fc = (2 * hi + 0.5) / L; w = 1.0 if lo == 0 else 1.0 - (lo - 0.5) / (hi + 0.25)
    sig = {0: dict(kind='dirichlet', fc=fc, period=float(L), **({} if w >= 1 else dict(fill=w)))}
    bg = {0: dict(kind='dirichlet', fc=(N + 0.5) * 2 / L, period=float(L))}
    m = BLMLBinned(tol=1e-6, max_sweeps=5000, ll_tol=1e-10, engine=dj.ENGINE, solver=dj.SOLVER).fit(T, [sig, bg], complex_g=True, init=init)
    P_ = np.maximum(dj._np(m.density_), 1e-300); return P_ / P_.sum(), m
t0 = time.time()
his = [h for h in range(1, int(2.0 * phys * L / 2) + 2) if h >= 0.8 * phys * L / 2 - 0.5]
best = None
for N in NS_:
    C_ = sorted(((hi, lo) for hi in his for lo in range(0, hi + 1)), key=lambda c: (c[0] - c[1] + 1, c[0]))
    SL = {(c, k): pfit(T, c[0], c[1], N)[0] for k, (T, _) in enumerate(FOLDS) for c in C_}
    c, tab = dj.cv_select(FOLDS, lambda c, T: SL[(c, fold_of[id(T)])], C_)
    cvb = max(rr['cv'] for rr in tab)
    if best is None or cvb > best[0]:
        best = (cvb, N, c)
_, N, (hi, lo) = best
Q, m = pfit(TV, hi, lo, N); score('periodic', Q, line(m), N=N, hi=hi, lo=lo)
if case == 'partial':                             # the partially coherent model on the window: the coherent periodic mode
    hp, lp = hi, max(lo, 1)                       # at its (hi, lo), two single-slit modes (baseband of the slit's harmonic
    sig = {0: dict(kind='dirichlet', fc=(2 * hp + 0.5) / L, period=float(L), fill=1.0 - (lp - 0.5) / (hp + 0.25))}
    one = {0: dict(kind='dirichlet', fc=(2 * (hp - lp) + 0.5) / L, period=float(L))}   # width, |k| <= hi - lo) + flat bg
    bgf = {0: dict(kind='dirichlet', fc=0.5 * 2 / L, period=float(L))}
    m3 = BLMLBinned(tol=1e-6, max_sweeps=5000, ll_tol=1e-10, engine=dj.ENGINE, solver=dj.SOLVER).fit(TV, [sig, one, one, bgf], complex_g=True)
    U = dj._np(m3.Se_); E = dj._np(m3.e_hat_); sh = np.array([float(np.real(np.sum(np.conj(E[k]) * U[k]))) for k in range(4)]); sh /= sh.sum()
    Q3 = np.maximum(dj._np(m3.density_), 1e-300); score('periodic_three', Q3 / Q3.sum(), line(m3), share=sh.tolist(), V_hat=float(sh[0] / sh[:3].sum()))
FILLS = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.36, 0.33, 0.3, 0.25]
G = [float(f) for f in dj.FC_GRID if 0.8 * phys <= f <= 2.0 * phys]
lfit = lambda T, f, w, N, init=None: dj.bg_fit(T, [f], (N,), True, 1, init=init, max_sweeps=5000, tol=1e-6, fill=w)
bestl = None
for N in NS_:
    SL = {}
    for k, (T, _) in enumerate(FOLDS):
        for w in FILLS:
            prev = None
            for f in G:
                P_, prev, _ = lfit(T, f, w, N, prev); SL[(f, w, k)] = P_
    C_ = sorted(((f, w) for f in G for w in FILLS), key=lambda c: (c[0] * c[1], c[0]))
    c, tab = dj.cv_select(FOLDS, lambda c, T: SL[(c[0], c[1], fold_of[id(T)])], C_)
    cvb = max(rr['cv'] for rr in tab)
    if bestl is None or cvb > bestl[0]:
        bestl = (cvb, N, c)
_, N, (fc, w) = bestl
Q, m, _ = lfit(TV, fc, w, N); score('line', Q, line(m), N=N, fc=fc, fill=w)
bflat = np.ones((1, L)) / L
HG = [(h,) for h in (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24)]
h, _ = dj.cv_select(FOLDS, lambda h_, T: dj.em(T, bflat, dj.kde_q(h_))[0], HG[::-1])
score('kde', dj.em(TV, bflat, dj.kde_q(h))[0], h=h[0])
res['s'] = time.time() - t0
json.dump(res, open(fn, 'w'), indent=1)
print('done', fn, f"{res['s']:.0f}s", {k: {a: (round(b, 4) if isinstance(b, float) else b) for a, b in res[k].items() if a in ('test', 'kl', 'd_err', 'rho_err', 'phi_err')} for k in ('periodic', 'line', 'kde')})
