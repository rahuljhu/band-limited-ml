"""Paper 3, Fig. 1: the prior ladder on a crop-free line. Truth: two cos^2-tapered slits (centres +-D_/2, width A_
cycles/px; fill w = 2a/(d+a) = 0.4545) on W = 2048 bins (4e-6 of the mass outside: nothing is cut that 1e6 particles resolve; a wider line only adds background-only bins, i.e. variance),
cases sym (equal slits), asym (amplitude RHO, phase PHI), partial (equal slits, visibility V); a flat background share
bg. Split 64/16/20; every choice by 5-fold CV with the one-SE rule. Estimators, from the weakest prior to the strongest
(all complex BLML with a flat background mode, N = 0):
  kde      smoothness only (flat background by EM)
  band     the aperture's extent: the sinc band, fc by CV
  shape    two slits of unknown geometry: the two-slit kernel, (fc, fill) by CV
  known    two slits of known geometry: the two-slit kernel at the true (fc, fill)
  three    the partially coherent model at the known geometry: two-slit mode + one mode per slit (coherent share V_hat)
  model    the full physical model: tapered slits with free centre, d, a, rho, phi, V and background, by ML (the bound)
  model_p  the same with the prior V = 1 (sym, asym) or rho = 1 (partial): only gamma is identified otherwise
Scores: KL from the truth, test log-likelihood; for fields the slit centre and width (moments of |A|^2 per side) and
the aperture error (unit norm, min over global phase and twin); asym: relative slit phase and amplitude ratio (side
energies, min(rho, 1/rho)); every estimator: the fringe contrast gamma = 2 ||F_d|| / ||F_0|| of the signal density
(F its Fourier transform, F_0 the lobe |nu| < a, F_d the lobe at |nu - d| < a, background share removed), which for
slits of one profile is 2 rho V / (1 + rho^2).
  python p3_ladder.py <case> <bg> <n> <rep>   -> results/p3_ladder/<case>_bg<bg>_n<n>_<rep>.json"""
import os, sys, json, time, numpy as np
os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_COLLAPSE', '1'); os.environ.setdefault('BLML_SOLVER', 'lbfgs')
case, BG, N, REP = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); sys.argv = sys.argv[:1]
sys.path.insert(0, '.')
import dslit_job as dj
from scipy.optimize import minimize, minimize_scalar
OUT = os.environ.get('LD_OUT', 'results/p3_ladder'); os.makedirs(OUT, exist_ok=True); fn = f'{OUT}/{case}_bg{BG:g}_n{N}_{REP}.json'
RESCORE = os.environ.get('LD_RESCORE') == '1'       # refit at the stored CV choices (no CV) and rescore the fields
if os.path.exists(fn) and not RESCORE:
    sys.exit(0)
OLD = json.load(open(fn)) if RESCORE and os.path.exists(fn) else None
if RESCORE and OLD is None:
    sys.exit(0)
W, NF = int(os.environ.get('LD_W', '2048')), 1 << 16
D_, A_, RHO, PHI, V = 0.017, 0.005, 0.6, 1.0, 0.6
FC0, W0 = D_ + A_, 2 * A_ / (D_ + A_)            # the true two-slit band and fill
FILLS = [1.0, 0.7, 0.55, 0.45, 0.35, 0.25]
nu = np.fft.fftfreq(NF)
def slit(c, a=A_):
    u = np.abs(nu - c) / (a / 2); return np.where(u < 1, np.cos(np.pi * u / 2) ** 2, 0.0)
def field(A, x0=0.0):                             # the far field on the W bins, centre x0 (px) from the window centre
    g = np.fft.ifft(A * np.exp(-2j * np.pi * nu * x0)); return np.fft.fftshift(g)[NF // 2 - W // 2:NF // 2 + W // 2] * NF
def intensity(d, a, rho, phi, Vv, x0=0.0):
    g1, g2 = field(slit(-d / 2, a), x0), field(slit(d / 2, a), x0)
    return Vv * np.abs(g1 + rho * np.exp(1j * phi) * g2) ** 2 + (1 - Vv) * (np.abs(g1) ** 2 + rho ** 2 * np.abs(g2) ** 2), g1 + rho * np.exp(1j * phi) * g2
rho_t, phi_t, V_t = (RHO, PHI, 1.0) if case == 'asym' else (1.0, 0.0, V if case == 'partial' else 1.0)
I, gtrue = intensity(D_, A_, rho_t, phi_t, V_t)
P = (1 - BG) * I / I.sum() + BG / W; P /= P.sum()
rng = np.random.default_rng(1000 * REP + N % 997)
x = rng.choice(W, size=N, p=P); part = rng.choice(3, size=N, p=[0.64, 0.16, 0.20])
TR, VA, TE = (np.bincount(x[part == k], minlength=W).astype(float) for k in range(3)); TV = TR + VA
fold = rng.integers(0, 5, size=N)
FOLDS = [(np.bincount(x[(part < 2) & (fold != k)], minlength=W).astype(float),
          np.bincount(x[(part < 2) & (fold == k)], minlength=W).astype(float)) for k in range(5)]
fold_of = {id(T): k for k, (T, _) in enumerate(FOLDS)}
dj.BFIX = np.array([np.nan]); bflat = np.ones((1, W)) / W
kl = lambda Q: float((P * np.log(P / np.maximum(Q / Q.sum(), 1e-300))).sum())
L8 = 8 * W; nuL = np.fft.fftfreq(L8)
def aperture(g):
    return np.fft.fft(np.r_[g, np.zeros(L8 - W)]) * np.exp(2j * np.pi * nuL * (W // 2))
def moments(A):
    out = []
    for sgn in (-1, 1):
        m = (np.sign(nuL) == sgn) & (np.abs(nuL) < 0.5); p = np.abs(A[m]) ** 2; v = nuL[m]
        c = (p * v).sum() / p.sum(); out += [abs(c), np.sqrt((p * (v - c) ** 2).sum() / p.sum()), p.sum()]
    return np.array(out)
Aref = aperture(gtrue) if case != 'partial' else None; M_true = moments(aperture(field(slit(-D_ / 2) + slit(D_ / 2))))
nuW = np.fft.fftfreq(W)
def gamma(Q, beta):                               # the fringe contrast of the signal density (background share removed)
    Qs = np.maximum(Q / Q.sum() - beta / W, 0); F = np.fft.fft(Qs)
    l0, ld = np.abs(nuW) < A_, np.abs(np.abs(nuW) - D_) < A_
    return float(np.sqrt((np.abs(F[ld]) ** 2).sum() / 2) * 2 / np.sqrt((np.abs(F[l0]) ** 2).sum()))
G_T = gamma(P, BG)
tW = np.arange(W) - W // 2
def field_metrics(g):
    A = aperture(g); A = A / np.linalg.norm(A); mo = moments(A); out = {}
    out['centre_err'] = float(np.mean(np.abs(mo[[0, 3]] / M_true[[0, 3]] - 1)))
    # a linear phase on g (a shift of A in nu) leaves |g|^2 unchanged: score the separation, and align the shift below
    out['sep_err'] = float(abs((mo[0] + mo[3]) / (M_true[0] + M_true[3]) - 1))
    out['width_err'] = float(np.mean(np.abs(mo[[1, 4]] / M_true[[1, 4]] - 1)))
    r = np.sqrt(mo[5] / mo[2]); out['rho'] = float(min(r, 1 / r))
    if Aref is None:
        return out
    R = Aref / np.linalg.norm(Aref); best = None
    for tw in (False, True):
        gt = np.conj(g) if tw else g                  # the twin: conj(g) <-> conj(A(-nu))
        Ad = lambda s: (lambda X: X / np.linalg.norm(X))(aperture(gt * np.exp(2j * np.pi * s * tW)))
        r = minimize_scalar(lambda s: -abs(np.vdot(Ad(s), R)), bounds=(-3.0 / W, 3.0 / W), method='bounded', options=dict(xatol=1e-8))
        B = Ad(r.x); ip = np.vdot(B, R); err = float(np.sqrt(max(2 - 2 * abs(ip), 0.0)))
        if best is None or err < best[0]:
            best = (err, B * np.exp(1j * np.angle(ip)), r.x)
    out['aperture_err'] = best[0]; B = best[1]; out['shift'] = float(best[2])
    if case == 'asym':
        s1, s2 = (nuL < 0) & (np.abs(nuL + D_ / 2) < A_ / 2), (nuL > 0) & (np.abs(nuL - D_ / 2) < A_ / 2)
        ph = lambda X: np.angle(np.sum(X[s2]) / np.sum(X[s1]))
        out['phase_err'] = float(abs(np.angle(np.exp(1j * (ph(B) - ph(R))))))
        out['rho_err'] = abs(out['rho'] - RHO)
    return out
res = dict(case=case, bg=BG, n=N, rep=REP, W=W, truth_test=dj.ll(P, TE), gamma_true=G_T); t0 = time.time()
def score(name, Q, beta, g=None, **extra):
    r = dict(kl=kl(Q), test=dj.ll(Q, TE), beta=float(beta), gamma=gamma(Q, beta), **extra); r['gamma_err'] = abs(r['gamma'] - G_T)
    if g is not None:
        r.update(field_metrics(g))
    res[name] = r; print(f'{time.time() - t0:.0f}s', name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
line = lambda m: np.asarray(dj._np(m.Se_))[0] / m.n_
# kde
HG = [(h,) for h in (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24)]
if OLD:
    res['kde'] = OLD['kde']
else:
    h, _ = dj.cv_select(FOLDS, lambda h_, T: dj.em(T, bflat, dj.kde_q(h_))[0], HG[::-1])
    Qk, _, bk, _ = dj.em(TV, bflat, dj.kde_q(h)); score('kde', Qk, np.atleast_1d(bk)[0], h=h[0])
# band
G = [float(f) for f in dj.FC_GRID if 0.5 * FC0 <= f <= 2.0 * FC0]
WARM = {}
def warm(T, fit):
    r = fit(WARM.get(id(T))); WARM[id(T)] = r[1]; return r[0]
f_ = OLD['band']['fc'] if OLD else dj.cv_select(FOLDS, lambda f, T: warm(T, lambda i: dj.bg_fit(T, [f], (0,), True, 1, init=i, max_sweeps=5000, tol=1e-6)), G, patience=6)[0]
Q, m, beta = dj.bg_fit(TV, [f_], (0,), True, 1, max_sweeps=5000, tol=1e-6); score('band', Q, beta, line(m), fc=float(f_))
# shape: (fc, fill) by CV, fc in 0.8-1.6 times the band's, per fill warm-started over fc
Gs = [f for f in G if 0.8 * f_ <= f <= 1.6 * f_] or [f_]
C_ = sorted(((float(f), w) for f in Gs for w in FILLS), key=lambda c: (c[0] * c[1], c[0]))
SL = {}
for k, (T, _) in enumerate(FOLDS if not OLD else []):
    for w in FILLS:
        prev = None
        for f in Gs:
            P_, prev, _ = dj.bg_fit(T, [f], (0,), True, 1, init=prev, max_sweeps=5000, tol=1e-6, fill=w); SL[(f, w, k)] = P_
(fs, fill) = (OLD['shape']['fc'], OLD['shape']['fill']) if OLD else dj.cv_select(FOLDS, lambda c, T: SL[(c[0], c[1], fold_of[id(T)])], C_)[0]
Q, m, beta = dj.bg_fit(TV, [fs], (0,), True, 1, max_sweeps=5000, tol=1e-6, fill=fill); score('shape', Q, beta, line(m), fc=fs, fill=fill)
# known geometry; the three-mode model at it
Q, m, beta = dj.bg_fit(TV, [FC0], (0,), True, 1, max_sweeps=5000, tol=1e-6, fill=W0); score('known', Q, beta, line(m))
Q, m, beta = dj.bg_fit(TV, [FC0], (0,), True, 1, max_sweeps=5000, tol=1e-6, fill=W0, incoh=2)
U = dj._np(m.Se_); E = dj._np(m.e_hat_); sh = np.array([float(np.real(np.sum(np.conj(E[k]) * U[k]))) for k in range(len(E))]); sh /= sh.sum()
score('three', Q, beta, line(m), V_hat=float(sh[0] / sh[:-1].sum()), share=sh.tolist())
# the full physical model (tapered slits): ML over (x0, d, a, log rho, phi, logit V, logit beta), started at the truth
lg = lambda p: float(np.log(p / (1 - p))) if 0 < p < 1 else (8.0 if p >= 1 else -8.0)
def model(th):
    x0, d, a, lr, ph, lv, lb = th; Vv = 1 / (1 + np.exp(-lv)); b = 1 / (1 + np.exp(-lb))
    Ii, gg = intensity(d, abs(a), np.exp(lr), ph, Vv, x0); Q = (1 - b) * Ii / Ii.sum() + b / W; return Q / Q.sum(), gg, b
nll = lambda th: -float((TV * np.log(np.maximum(model(th)[0], 1e-300))).sum()) / TV.sum()
th0 = np.array([0.0, D_, A_, np.log(rho_t), phi_t, lg(V_t if V_t < 1 else 0.999), lg(max(BG, 1e-3))])
r = minimize(nll, th0, method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-7, fatol=1e-10))
Q, gg, b = model(r.x); score('model', Q, b, gg, V_fit=float(1 / (1 + np.exp(-r.x[5]))), rho_fit=float(np.exp(r.x[3])))
# with rho and V both free only gamma is identified (exact for equal profiles): the model under the prior that
# identifies the other one, V = 1 (sym, asym: rho estimated) or rho = 1 (partial: V estimated)
fix = (3, 0.0) if case == 'partial' else (5, 8.0)
put = lambda t: np.insert(t, fix[0], fix[1])
r = minimize(lambda t: nll(put(t)), np.delete(th0, fix[0]), method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-7, fatol=1e-10))
th = put(r.x); Q, gg, b = model(th); score('model_p', Q, b, gg, V_fit=float(1 / (1 + np.exp(-th[5]))), rho_fit=float(np.exp(th[3])))
res['s'] = time.time() - t0; json.dump(res, open(fn, 'w'), indent=1); print('done', fn, f"{res['s']:.0f}s")
