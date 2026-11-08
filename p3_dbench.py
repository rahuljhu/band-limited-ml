"""Paper 3, synthetic diffraction benchmark: one data set per call, a far field of two tapered slits on a 1-d line of
W bins (slit profile cos^2 across each slit, so the field decays inside the window and no estimator meets a cut).
  python p3_dbench.py <case> <bg> <n> <rep>   -> $BLML_OUT_DIR/p3_dbench/<case>_bg<bg>_n<n>_<rep>.json (skipped if present)
cases (suffix _sharp: rectangular slits, window DB_W = 1536; DB_HIO_ONLY=1: HIO only, with 2000-iteration and
KDE-input variants, -> ..._hio.json; DB_FLAT_ONLY=1: only the flat-background (N = 0) fits blml0 (sinc band),
blml_slits0 (two-slit kernel,
N = 0) and blml_three (every case: that mode + two single-slit modes of band fill fc / 2 + N = 0: the partially coherent model;
V_hat its coherent share), fc in 0.7-2 times the main run's band fc, -> ..._s0.json): sym (equal slits, real field), asym (amplitudes 1, RHO, relative phase PHI), defocus (quadratic aperture phase
BETA at the slit edges), partial (visibility V = 0.6, or partial_v<V>: V |g1 + g2|^2 + (1 - V)(|g1|^2 + |g2|^2), no single field). bg: the flat
background share (0 or 0.1). Split 64/16/20 of the n particles; every hyperparameter by 5-fold CV on train +
validation with the one-SE rule (dslit_job.cv_select); fits by BLMLBinned(solver=BLML_SOLVER, default lbfgs).
Estimators: complex BLML (sinc band, background mode of one harmonic, N = 1), complex BLML with the two-slit kernel (fill and fc in
0.7-2 times the band's fc; per fill the fc grid warm-started), real positive-root BLML (flat background by EM), KDE (flat background by
EM), histogram, the Fraunhofer and Fresnel two-slit models, and HIO phase retrieval on the training histogram given
the true slit support or the BLML band (HIO_STARTS starts, HIO_ITERS HIO + 100 error-reduction iterations, the best
Poisson likelihood). Scores: KL from the true pixel probabilities and the test log-likelihood per particle; for the
estimators with a field (complex BLML, slit kernel, real BLML's root, the physics models, HIO): the aperture error
min over the global phase and the twin of ||A_hat - A|| (both unit L2, the transforms of the windowed fields), the
slit centre and width (moments of |A|^2 per half-axis) relative to the truth's, the relative slit phase (asym) and
the defocus coefficient (defocus) against the truth's."""
import os, sys, json, time, numpy as np
os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_COLLAPSE', '1'); os.environ.setdefault('BLML_SOLVER', 'lbfgs')
case, BG, N, REP = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); sys.argv = sys.argv[:1]
SHARP = case.endswith('_sharp'); base = case[:-len('_sharp')] if SHARP else case     # <case>_sharp: rectangular slits
HIO_ONLY = os.environ.get('DB_HIO_ONLY', '0') == '1'      # only HIO, with its robustness variants (2000 iterations, KDE input)
FLAT_ONLY = os.environ.get('DB_FLAT_ONLY', '0') == '1'    # only the flat-background slit fits (N = 0; one or three modes)
sys.path.insert(0, '.')
import dslit_job as dj
OUT = os.path.join(os.environ.get('BLML_OUT_DIR', 'results'), 'p3_dbench'); os.makedirs(OUT, exist_ok=True)
fn = f'{OUT}/{case}_bg{BG:g}_n{N}_{REP}' + ('_hio' if HIO_ONLY else '_s0' if FLAT_ONLY else '') + '.json'
if os.path.exists(fn):
    sys.exit(0)
W, NF = int(os.environ.get('DB_W', 1536 if SHARP else 768)), 1 << 16   # sharp slits: a wider window for the slow tails
D_, A_ = 0.017, 0.005                          # slit centres at +-D_/2, widths A_ (cycles / px: the field's spectrum)
RHO, PHI, BETA, V = 0.6, 1.0, 2.0, 0.6
if base.startswith('partial_v'):                  # partial_v<V>: the partial case at visibility V
    V = float(base[len('partial_v'):]); base = 'partial'
FILLS = [1.0, 0.7, 0.55, 0.45, 0.35, 0.25]; HIO_STARTS, HIO_ITERS = int(os.environ.get('HIO_STARTS', '20')), 400
nu = np.fft.fftfreq(NF)
def slit(c):
    u = np.abs(nu - c) / (A_ / 2); return np.where(u < 1, 1.0 if SHARP else np.cos(np.pi * u / 2) ** 2, 0.0)
def field(A):
    return np.fft.fftshift(np.fft.ifft(A))[NF // 2 - W // 2:NF // 2 + W // 2] * NF
A1, A2 = slit(-D_ / 2), slit(D_ / 2)
if base == 'sym':
    Atrue = A1 + A2
elif base == 'asym':
    Atrue = A1 + RHO * np.exp(1j * PHI) * A2
elif base == 'defocus':
    Atrue = (A1 + A2) * np.exp(1j * BETA * (nu / (D_ / 2 + A_ / 2)) ** 2)
elif base == 'partial':
    Atrue = None
else:
    raise SystemExit(f'case {case}?')
if Atrue is None:
    g1, g2 = field(A1), field(A2); I = V * np.abs(g1 + g2) ** 2 + (1 - V) * (np.abs(g1) ** 2 + np.abs(g2) ** 2); gtrue = None
else:
    gtrue = field(Atrue); I = np.abs(gtrue) ** 2
P = (1 - BG) * I / I.sum() + BG / W; P = P / P.sum()
rng = np.random.default_rng(1000 * REP + N % 997)
x = rng.choice(W, size=N, p=P); part = rng.choice(3, size=N, p=[0.64, 0.16, 0.20])
TR, VA, TE = (np.bincount(x[part == k], minlength=W).astype(float) for k in range(3)); TV = TR + VA
fold = rng.integers(0, 5, size=N)
FOLDS = [(np.bincount(x[(part < 2) & (fold != k)], minlength=W).astype(float),
          np.bincount(x[(part < 2) & (fold == k)], minlength=W).astype(float)) for k in range(5)]
dj.BFIX = np.array([np.nan]); bflat = np.ones((1, W)) / W
kl = lambda Q: float((P * np.log(P / np.maximum(Q / Q.sum(), 1e-300))).sum())
L8 = 8 * W; nuL = np.fft.fftfreq(L8)
def aperture(g):                                   # the windowed field's transform on a fine grid (x centred at 0)
    return np.fft.fft(np.r_[g, np.zeros(L8 - W)]) * np.exp(2j * np.pi * nuL * (W // 2))
Aref = None if gtrue is None else aperture(gtrue)
def moments(A):
    out = []
    for sgn in (-1, 1):
        m = (np.sign(nuL) == sgn) & (np.abs(nuL) < 0.5); p = np.abs(A[m]) ** 2; v = nuL[m]
        c = (p * v).sum() / p.sum(); out += [abs(c), np.sqrt((p * (v - c) ** 2).sum() / p.sum())]
    return np.array(out)                           # centre and width of each half
M_true = moments(aperture(field(A1 + A2)))         # the slit geometry (amplitudes aside)
def field_metrics(g):
    A = aperture(g); A = A / np.linalg.norm(A); out = dict()
    mo = moments(A); out['centre_err'] = float(np.mean(np.abs(mo[[0, 2]] / M_true[[0, 2]] - 1)))
    out['width_err'] = float(np.mean(np.abs(mo[[1, 3]] / M_true[[1, 3]] - 1)))
    if Aref is None:
        return out
    R = Aref / np.linalg.norm(Aref); best = None
    for tw in (False, True):
        B = np.conj(np.roll(A[::-1], 1)) if tw else A     # twin: conj(A(-nu))
        ip = np.vdot(B, R); err = float(np.sqrt(max(2 - 2 * abs(ip), 0.0)))
        if best is None or err < best[0]:
            best = (err, B * np.exp(1j * np.angle(ip)))
    out['aperture_err'] = best[0]; B = best[1]
    if base == 'asym':
        s1, s2 = (nuL < 0) & (np.abs(nuL + D_ / 2) < A_ / 2), (nuL > 0) & (np.abs(nuL - D_ / 2) < A_ / 2)
        ph = lambda X: np.angle(np.sum(X[s2]) / np.sum(X[s1]))
        out['phase_err'] = float(abs(np.angle(np.exp(1j * (ph(B) - ph(R))))))
    if base == 'defocus':
        m = np.abs(R) > 0.05 * np.abs(R).max()
        o = np.argsort(nuL[m]); v = nuL[m][o] / (D_ / 2 + A_ / 2)
        def curv(X):                               # the quadratic coefficient of the unwrapped aperture phase, |X|^2-weighted
            return np.polyfit(v, np.unwrap(np.angle(X[m][o])), 2, w=np.abs(X[m][o]))[0]
        out['defocus_err'] = float(abs(curv(B) - curv(R)) / BETA)
    return out
res = dict(case=case, bg=BG, n=N, rep=REP, W=W, truth_test=dj.ll(P, TE)); t0 = time.time()
def score(name, Q, g=None, **extra):
    r = dict(kl=kl(Q), test=dj.ll(Q, TE), **extra)
    if g is not None:
        r.update(field_metrics(g))
    res[name] = r; print(f'{time.time() - t0:.0f}s', name, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
G = list(dj.FC_GRID[dj.FC_GRID >= 2.0 / W])
line = lambda m: np.asarray(dj._np(m.Se_))[0] / m.n_       # mode 0's field on the lattice
fc_sinc = None
if FLAT_ONLY:
    f_, _ = dj.cv_select(FOLDS, lambda f, T: dj.bg_fit(T, [f], (0,), True, 1, max_sweeps=5000, tol=1e-6)[0], G, patience=6)
    Q, m, beta = dj.bg_fit(TV, [f_], (0,), True, 1, max_sweeps=5000, tol=1e-6)
    score('blml0', Q, line(m), fc=float(f_), beta=beta)
    fc_sinc = json.load(open(f'{OUT}/{case}_bg{BG:g}_n{N}_{REP}.json'))['blml']['fc']   # the slit grids as in the main run
    Gs = [f for f in G if 0.7 * fc_sinc <= f <= 2.0 * fc_sinc]
    C_ = sorted(((float(f), w) for f in Gs for w in FILLS), key=lambda c: (c[0] * c[1], c[0]))
    fold_of = {id(T): k for k, (T, _) in enumerate(FOLDS)}
    for nm, inc in (('blml_slits0', 0), ('blml_three', 2)):
        fit = lambda T, c, init=None: dj.bg_fit(T, [c[0]], (0,), True, 1, init=init, max_sweeps=5000, tol=1e-6, fill=c[1], incoh=inc)
        SL = {}
        for k, (T, _) in enumerate(FOLDS):
            for w in FILLS:
                prev = None
                for f in Gs:
                    P_, prev, _ = fit(T, (float(f), w), init=prev); SL[(float(f), w, k)] = P_
        (fs, fill), _ = dj.cv_select(FOLDS, lambda c, T: SL[(c[0], c[1], fold_of[id(T)])], C_)
        Q, m, beta = fit(TV, (fs, fill))
        U = dj._np(m.Se_); E = dj._np(m.e_hat_); sh = np.array([float(np.real(np.sum(np.conj(E[k]) * U[k]))) for k in range(len(E))])
        sh = sh / sh.sum(); ex = dict(V_hat=float(sh[0] / sh[:-1].sum()), share=sh.tolist()) if inc else {}
        score(nm, Q, line(m), fc=fs, fill=fill, beta=beta, **ex)
    res['s'] = time.time() - t0; json.dump(res, open(fn, 'w'), indent=1); print('done', fn, f"{res['s']:.0f}s"); sys.exit(0)
if not HIO_ONLY:
    # complex BLML, sinc band, constant background mode
    WARM = {}                                          # per fold: the previous candidate's fit (a warm start)
    def warm(T, fit):
        r = fit(WARM.get(id(T))); WARM[id(T)] = r[1]; return r[0]
    f_, _ = dj.cv_select(FOLDS, lambda f, T: warm(T, lambda i: dj.bg_fit(T, [f], (1,), True, 1, init=i, max_sweeps=5000, tol=1e-6)), G, patience=6)
    Q, m, beta = dj.bg_fit(TV, [f_], (1,), True, 1, max_sweeps=5000, tol=1e-6)
    score('blml', Q, line(m), fc=float(f_), beta=beta); fc_sinc = float(f_)
    # complex BLML, the two-slit kernel: (fc, fill) by CV, parsimony by the aperture's measure fc fill; two starts
    def slits_fit(T, c, init=None, tol=1e-6):
        return dj.bg_fit(T, [c[0]], (1,), True, 1, init=init, max_sweeps=5000, tol=tol, fill=c[1])
    Gs = [f for f in G if 0.7 * fc_sinc <= f <= 2.0 * fc_sinc]
    C_ = sorted(((float(f), w) for f in Gs for w in FILLS), key=lambda c: (c[0] * c[1], c[0]))
    SL = {}                                            # every candidate on every fold: per fill in increasing fc, each fit
    for k, (T, _) in enumerate(FOLDS):                 # warm-started from the previous fc (one start: L-BFGS reaches the
        for w in FILLS:                                # same optimum from every start)
            prev = None
            for f in Gs:
                P_, prev, _ = slits_fit(T, (float(f), w), init=prev); SL[(float(f), w, k)] = P_
    fold_of = {id(T): k for k, (T, _) in enumerate(FOLDS)}
    (fs, fill), _ = dj.cv_select(FOLDS, lambda c, T: SL[(c[0], c[1], fold_of[id(T)])], C_)
    Q, m, beta = slits_fit(TV, (fs, fill))
    score('blml_slits', Q, line(m), fc=fs, fill=fill, beta=beta)
    # real BLML, the positive root, flat background by EM
    f_, _ = dj.cv_select(FOLDS, lambda f, T: dj.em(T, bflat, dj.blml_q([f], 'positive'), tol=1e-6)[0], G, patience=6)
    Q, q, beta, mr = dj.em_polish(TV, bflat, np.array([f_]), 'positive')
    score('blml_real', Q, line(mr), fc=float(f_), beta=float(np.atleast_1d(beta)[0]))
    # KDE, flat background by EM; histogram
    HG = [(h,) for h in (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24)]
    h, _ = dj.cv_select(FOLDS, lambda h_, T: dj.em(T, bflat, dj.kde_q(h_))[0], HG[::-1])
    score('kde', dj.em(TV, bflat, dj.kde_q(h))[0], h=h[0]); score('hist', TV + 0.5)
    # the two-slit physics models (flat background)
    for nm, fr in (('fraunhofer', False), ('fresnel', True)):
        th = dj.ref_fit(TV, bflat, 0, fr, th0=res.get('_th_fr'))
        if not fr:
            res['_th_fr'] = th
        Q = dj.ref_P(th, bflat, 0, fr)[0]; gq = dj.ref_parts(th, (W,), 0, fr)[1]
        score(nm, Q, gq)
    res.pop('_th_fr', None)
# HIO on the training histogram: support = the true slits (|A| > 1e-3 max) or the BLML band, on the W-point grid
nuW = np.fft.fftfreq(W)
At = (A1 + A2) if Atrue is None else Atrue
supp_true = np.interp(nuW, np.fft.fftshift(nu), np.fft.fftshift(np.abs(At))) > 1e-3 * np.abs(At).max()
def hio(amp, supp, seed, iters=HIO_ITERS):
    r = np.random.default_rng(seed); ph = np.exp(2j * np.pi * r.random(W))
    g = np.fft.ifftshift(amp * ph); Aw = np.fft.fft(g); Aw = np.where(supp, Aw, 0)
    for it in range(iters + 100):
        g = np.fft.ifft(Aw); g = np.fft.ifftshift(amp) * np.exp(1j * np.angle(g))
        An = np.fft.fft(g)
        Aw = np.where(supp, An, Aw - 0.9 * An) if it < iters else np.where(supp, An, 0)
    return np.fft.fftshift(np.fft.ifft(Aw))
amp = np.sqrt(TV / TV.sum())
runs = [('hio_true_support', supp_true, amp, HIO_ITERS)]
if fc_sinc is not None:
    runs.append(('hio_band', np.abs(nuW) <= fc_sinc / 2, amp, HIO_ITERS))
if HIO_ONLY:                                       # robustness: longer runs, and the KDE's smoothed amplitude as input
    HG = [(h,) for h in (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24)]
    h, _ = dj.cv_select(FOLDS, lambda h_, T: dj.em(T, bflat, dj.kde_q(h_))[0], HG[::-1])
    Qk = dj.em(TV, bflat, dj.kde_q(h))[0]
    runs += [('hio_true_support_2000', supp_true, amp, 2000), ('hio_true_support_kde', supp_true, np.sqrt(Qk / Qk.sum()), HIO_ITERS),
             ('hio_true_support_kde_2000', supp_true, np.sqrt(Qk / Qk.sum()), 2000)]
for nm, supp, am, iters in runs:
    best = None
    for sd in range(HIO_STARTS):
        g = hio(am, supp, sd, iters); Q = np.abs(g) ** 2 + 1e-12; L = dj.ll(Q, TV)
        if best is None or L > best[0]:
            best = (L, g)
    score(nm, np.abs(best[1]) ** 2 + 1e-12, best[1], iters=iters)
res['s'] = time.time() - t0
json.dump(res, open(fn, 'w'), indent=1)
print('done', fn, f"{res['s']:.0f}s")
