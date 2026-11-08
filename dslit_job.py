"""Paper 3, double slit (photons and electrons): one protocol for every method, on the CPU or a Colab GPU
(BLML_DATA_SRC=~/data_repo/dslit_gpu ./colab_job.sh start dslit dslit_job.py T4; locally BLML_DATA_DIR=~/data_repo/dslit_gpu
BLML_OUT_DIR=results/dslit BLML_ENGINE=cpu python dslit_job.py). Data from results/dslit_export.py.

Model for every method: P = sum_j beta_j b_j + (1 - sum_j beta_j) q on the crop lattice: b_j the dataset's background
shapes (results/dslit_export.py, from the source papers: photons, a flat floor plus the multimode pattern; electrons, the
sideband profile), beta_j their weights (measured outside the signal where possible, else fitted: NaN in
dslit_<set>_beta.npy), q the method's density; q and the free weights by maximum likelihood on the training particles
(EM: responsibilities R_j = beta_j b_j / P, q refitted on the weights N (1 - sum_j R_j)). Particles are split at random 0.64 / 0.16 / 0.20 (rng 0) into train /
validation / test; hyperparameters (BLML fc per axis, KDE bandwidths) maximise the validation log-likelihood (fc:
search_fc, the selector's search with warm-started candidates; KDE: a grid); the final fit uses train + validation; the score is the test log-likelihood per particle.
Methods (BLML_PARTS, one output file each per set; finished ones are skipped):
  fraunhofer, fresnel   the physics reference. Field along the fringe axis s (pixels), slits at xi = -1/2, +1/2 (units of
                 the separation d), width r = a / d:
                   u_j(s) = int_slit j exp(i pi eps xi^2) exp(-2 pi i (s - s0) xi / P) dxi,
                   I(s) = V |u_1 + rho e^{i phi} u_2|^2 + (1 - V)(|u_1|^2 + rho^2 |u_2|^2)
                 (P the fringe period; eps the aperture quadratic phase of Fresnel propagation, its Fresnel number;
                 eps = 0 is Fraunhofer; V < 1 partial coherence), times a Gaussian profile across. Poisson ML.
  positive       BLML real, positive root;  complex  BLML complex g;  signed  BLML real with the signs of the fitted
                 Fresnel field (its real part after removing the global phase that maximises it): the physics-signs reference
  kde            Gaussian product kernel;  akde  Abramson's adaptive Gaussian KDE (1-d lines);  hist  add-1/2 histogram (no background term)
  real_bg, complex_bg, rank2_bg   the background as one more mode (bg_fit: a dirichlet kernel periodic over the crop,
                 harmonics fixed by BLML_BG_HARM, default 0,0: flat, as the source papers' offsets; 'search': chosen on
                 validation; f = sum |g_k|^2 + |g_bg|^2, no EM, beta = the background mode's share); real_bg
                 is real rank 2 (the signal mode may change sign: not the positive root, which exists for one mode only)
  rank1, rank2, rank3   complex BLML of rank K (f = sum_k |g_k|^2, one shared kernel; fc started from rank K-1's): the
                 coherence test. A partially coherent two-slit beam has rank <= 2 (every source point's field lies in the span
                 of the two single-slit fields)
  stream_positive, stream_complex   fc learned by streaming epochs with forgetting and the EM background (stream_em), then the final
                 fit at that fc; the same test score (an alternative to the fc search)
  stream         BLMLBinnedStreaming, complex, Adam on log fc, particles in arrival order (BLML_STREAM_NB per batch)
  stream_bg      BLMLBinnedStreaming.fit on train + validation in arrival order (BLML_STREAM_NB per batch): complex signal
                 (fc selected at the burn-in, BLML_STREAM_BURN counts, then Adam over epochs with the leave-batch-out
                 gradient, tol 1e-4 with the parameters stationary) plus the fixed background mode (BLML_BG_HARM)
BLML_BG_FLAT=1: the EM / reference background is one flat shape with its weight fitted (estimated from the record;
no sideband measurement, no multimode template). BLML_COLLAPSE=1 (stripes: the electrons): the counts and background shapes summed across the fringes, every method on
the 1-d lattice along them (the marginal of |psi|^2, band-limited with the same band; the references without the
profile across). Outputs (BLML_OUT_DIR): <set>_<part>.json (test_ll, beta, hyperparameters, ...), <set>_<part>_P.npy (the fitted pixel
probabilities, for paired test standard errors), and for BLML / the references the field along the fringe axis through
the profile maximum: <set>_<part>_line.npy (complex)."""
import os, sys, json, time, glob, numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
CODE = os.environ.get('BLML_CODE_DIR', '.'); DATA = os.environ.get('BLML_DATA_DIR', '.'); OUT = os.environ.get('BLML_OUT_DIR', 'results/dslit')
sys.path.insert(0, CODE)
import blml_trivial as bt, selector as sel
from blml_binned import BLMLBinned, BLMLBinnedStreaming, _np
ENGINE = os.environ.get('BLML_ENGINE', 'gpu')
PARTS = os.environ.get('BLML_PARTS', 'fraunhofer,fresnel,hist,kde,positive,signed,complex,stream').split(',')
SETS = [s for s in (os.environ.get('BLML_SETS') or 'photon,electron').split(',') if s]
COLLAPSE = os.environ.get('BLML_COLLAPSE', '0') == '1'                 # stripes (electrons): 1-d along the fringes
BG_FLAT = os.environ.get('BLML_BG_FLAT', '0') == '1'                   # EM / references: a flat background, weight fitted
_bh = os.environ.get('BLML_BG_HARM', '0,0')                            # background-mode harmonics per dim, fixed (the
BG_HARM = None if _bh == 'search' else tuple(int(v) for v in _bh.split(','))   # model's choice; 'search': on validation)
SELECT = os.environ.get('BLML_SELECT', 'val')                          # hyperparameters: 'val' the validation maximum; 'cv'
CV_K = int(os.environ.get('BLML_CV_K', '5'))                           # K-fold CV on train + validation, the one-SE rule (1-d)
_fl = os.environ.get('BLML_FILLS')                                     # the signal kernel: the sinc (None) or the two-slit
FILLS = None if not _fl else [float(v) for v in _fl.split(',')]       # aperture kernel, its fill searched with fc (CV)
SEARCH_SWEEPS = int(os.environ.get('BLML_SEARCH_SWEEPS', '5000'))      # sweep cap of the CV candidates' fits
SOLVER = os.environ.get('BLML_SOLVER', 'cd')                            # BLMLBinned solver of complex / rank K fits: 'cd' | 'lbfgs'
os.makedirs(OUT, exist_ok=True)

def ll(P, N):
    P = P / P.sum(); return float((N * np.log(np.maximum(P, 1e-300))).sum() / N.sum())

EM_SWEEPS = int(os.environ.get("BLML_EM_SWEEPS", "20"))                 # sweeps per EM step (generalised EM), then a polish
BFIX = None                                                               # (J,) background weights, NaN = fitted
TOTAL_N = [None]                                                          # the current set's particles (stream_em)
def bw(beta, b):
    """beta_j b_j for (J,) weights and (J, ...) shapes."""
    return np.asarray(beta, float).reshape((-1,) + (1,) * (b.ndim - 1)) * b
def em(N, b, fit_q, iters=300, tol=1e-9, st0=None):
    """Background weights beta_j and q by EM for P = sum_j beta_j b_j + (1 - sum_j beta_j) q (b: (J, H, W) shapes);
    weights given in BFIX stay fixed (measured outside the signal), NaN ones are fitted. fit_q(weights, previous state)
    -> (q normalised, state)."""
    J = len(b); fixed = np.isfinite(BFIX); beta = np.where(fixed, BFIX, 0.2 / J); prev, st = -np.inf, st0
    R = bw(beta, np.ones_like(b))                                          # start: a flat share per component
    for _ in range(iters):
        q, st = fit_q(N * np.clip(1 - R.sum(0), 0.0, 1.0), st); bg = bw(beta, b)   # clip: round-off when the background takes ~all
        P = bg.sum(0) + (1 - beta.sum()) * q; R = bg / P
        beta = np.where(fixed, BFIX, (N[None] * R).sum(tuple(range(1, b.ndim))) / N.sum())
        P = bw(beta, b).sum(0) + (1 - beta.sum()) * q; L = ll(P, N)
        if L - prev < tol: break
        prev = L
    return P, q, beta, st

# ---------------------------------------------------------------- physics reference
XI = (np.arange(48) + 0.5) / 48 - 0.5                                     # midpoints across one slit (units of its width)
def ref_parts(th, shape, axis, fresnel, sig=None):
    """sig: a Gaussian blur along the fringes (pixels; source size, detector blur), on a line padded by 5 sig so the
    detector's edges see the pattern beyond them."""
    s0, P, lr, eps, lrho, phi, V, t0, lst = th[:9]
    r = 1 / (1 + np.exp(-lr)); eps = eps if fresnel else 0.0; rho = np.exp(lrho); V = 1 / (1 + np.exp(-V))   # V by its logit
    pad = 0 if sig is None else int(np.ceil(5 * sig))
    s = np.arange(-pad, shape[axis] + pad) - s0; t = None if len(shape) == 1 else np.arange(shape[1 - axis]) - t0
    u = []
    for c in (-0.5, 0.5):
        xi = c + r * XI
        u.append((np.exp(1j * np.pi * eps * xi ** 2)[None, :] * np.exp(-2j * np.pi * np.outer(s, xi) / P)).mean(1) * r)
    field = u[0] + rho * np.exp(1j * phi) * u[1]
    I = V * np.abs(field) ** 2 + (1 - V) * (np.abs(u[0]) ** 2 + rho ** 2 * np.abs(u[1]) ** 2)
    if sig is not None:
        k = np.exp(-0.5 * (np.arange(-pad, pad + 1) / sig) ** 2); I = np.convolve(I, k / k.sum(), 'same')[pad:pad + shape[axis]]
        field = field[pad:pad + shape[axis]]
    if len(shape) == 1:                                                    # collapsed: the marginal along the fringes
        return I / I.sum(), field, None
    prof = np.exp(-0.5 * (t / np.exp(lst)) ** 2)
    Q = np.outer(I, prof) if axis == 0 else np.outer(prof, I)
    return Q / Q.sum(), field, prof
def ref_P(th, b, axis, fresnel):
    sig = float(np.exp(th[9 + len(b)])) if len(th) > 9 + len(b) else None    # the blur, after the background weights
    q, _, _ = ref_parts(th, b.shape[1:], axis, fresnel, sig); fixed = np.isfinite(BFIX); e = np.exp(th[9:9 + len(b)])
    free = (1 - BFIX[fixed].sum()) * e / (1 + e.sum())                        # free weights share what the fixed leave
    beta = np.where(fixed, BFIX, free)
    return bw(beta, b).sum(0) + (1 - beta.sum()) * q, q, beta
def ref_fit(N, b, axis, fresnel, th0=None, blur=False):
    prof_s = N if N.ndim == 1 else N.sum(1 - axis)            # b: (J, H, W), or (J, W) collapsed
    s0 = float(np.argmax(gaussian_filter(prof_s.astype(float), 3)))
    t0 = 0.0 if N.ndim == 1 else float(np.argmax(gaussian_filter(N.sum(axis).astype(float), 2)))
    # fringe period from the autocorrelation of the fringe-axis profile
    p = prof_s - gaussian_filter(prof_s.astype(float), 25); ac = np.correlate(p, p, 'full')[len(p) - 1:]
    P0 = float(np.argmax(ac[10:len(p) // 3]) + 10)
    # Fresnel starts at the Fraunhofer optimum (eps = 0: nested, so its training likelihood can only improve) and at eps = +-1
    if blur:                                                   # from the unblurred optimum (nested: sig -> 0 is it)
        starts = [np.r_[th0, np.log(s_)] for s_ in (0.3, 1.0, 3.0)]
    else:
        starts = [np.r_[th0[:3], e, th0[4:]] for e in (0.0, 1.0, -1.0)] if th0 is not None else [np.r_[s0 + ds, P0, np.log(0.2 / 0.8), 0.0, 0.0, 0.0, 2.0, t0, np.log(3.0), np.full(len(b), -1.0)]
                                               for ds in (-P0 / 4, 0, P0 / 4)]
    best = None
    for st in starts:
        st = np.array(st, float)
        f = lambda t: -ll(ref_P(t, b, axis, fresnel)[0], N)
        r = minimize(f, st, method='Nelder-Mead', options=dict(maxiter=40000, maxfev=40000, xatol=1e-5, fatol=1e-10, adaptive=True))
        r = minimize(f, r.x, method='Nelder-Mead', options=dict(maxiter=40000, maxfev=40000, xatol=1e-6, fatol=1e-11, adaptive=True))
        if best is None or r.fun < best.fun: best = r
    return best.x
def ref_desc(th, fresnel):
    r = 1 / (1 + np.exp(-th[2])); return dict(s0=float(th[0]), P=float(th[1]), a_over_d=float(r), eps=float(th[3]) if fresnel else 0.0,
                                                rho=float(np.exp(th[4])), phi=float(np.angle(np.exp(1j * th[5]))), V=float(1 / (1 + np.exp(-th[6]))),
                                                t0=float(th[7]), sigma_t=float(np.exp(th[8])))

# ---------------------------------------------------------------- BLML / KDE fits of q
SEARCH_DTYPE = np.float32 if os.environ.get('BLML_DTYPE', 'float64') == 'float32' else np.float64   # the fc search; final fits float64
def blml_q(fc, kind, signs=None, max_sweeps=None, K=1, dtype=np.float64):
    max_sweeps = max_sweeps or EM_SWEEPS
    def fit(w, m):
        modes = bt.make_modes(list(fc), ['sinc'] * len(fc))
        m = BLMLBinned(tol=1e-5, max_sweeps=max_sweeps, ll_tol=1e-10, engine=ENGINE, dtype=dtype, solver=SOLVER).fit(
            w, modes if K == 1 else [modes[0]] * K, complex_g=(kind == 'complex' or K > 1), signs=signs, init=m)
        q = np.maximum(_np(m.density_), 1e-300); return q / q.sum(), m
    return fit
def em_polish(N, b, fc, kind, K=1, signs=None, st0=None):
    """The final fit at the selected fc: generalised EM (EM_SWEEPS sweeps per step, warm-started), then EM steps with
    every refit run to convergence (tol 1e-5, 3000 sweeps), so the reported fit is the converged one."""
    P, q, beta, m = em(N, b, blml_q(fc, kind, signs, K=K), st0=st0)
    full = blml_q(fc, kind, signs, max_sweeps=3000, K=K)
    return em(N, b, lambda w, st: full(w, st if st is not None else m))
def search_fc(TR, VA, b, kind, K=1, signs=None, start=None, tag=''):
    """fc per axis by the selector's search (bracket by doubling from fc0 = 0.008, below every physical band here, then
    refinement) on the validation log-likelihood, each candidate a generalised-EM fit on the training counts warm-started
    from the previous candidate's coefficients, at EM tolerance 1e-6 and SEARCH_DTYPE (the final fit is float64, to
    convergence). Returns fc, the validation score, the path and the last fit (a warm start for the final fit)."""
    path, last = [], [None]
    def obj(fcv):
        if np.any(np.asarray(fcv) > 0.5) or np.any(np.asarray(fcv) < 2.0 / max(TR.shape)): return -np.inf
        P, q, beta, last[0] = em(TR, b, blml_q(fcv, kind, signs, K=K, dtype=SEARCH_DTYPE), tol=1e-6, st0=last[0])
        v = ll(P, VA); path.append((list(map(float, fcv)), v)); return v
    D = TR.ndim
    fc, vll = sel._search(obj, D, ['sinc'] * D, [None] * D, fc0=0.008, tag=tag,
                          factors=(np.sqrt(2.0), 2.0 ** 0.25, 2.0 ** 0.125), start=start)
    return np.asarray(fc, float), vll, path, last[0]
def stream_em(order, img, b, kind, NB, fc0=(0.05, 0.05), burn_in=1000):
    """fc learned by streaming epochs: BLMLBinnedStreaming (Adam on log fc along the implicit prequential gradient)
    with forgetting, NB particles per batch, each batch weighted by its signal share 1 - r (r the background
    responsibility under the current P = sum_j beta_j b_j + (1 - sum beta) q; for the mixture d log P = (1 - r) d log q,
    so the learner ascends the mixture's prequential score). Epoch 1 in arrival order, later epochs in a fresh random
    batch order; decay d with window 1 / (1 - d) = a quarter epoch (>= 4 batches), so a batch's own earlier copy weighs
    d^B ~ e^-4 when it returns and the prequential gradient stays nearly out of sample. The free background weights get
    one EM update per batch on the decayed counts. Stops when the epoch-mean fc moves < 1% (or BLML_STREAM_EPOCHS);
    the fc returned is the last epoch's geometric mean. Returns fc, P, beta, the fit, the history."""
    J = len(b); fixed = np.isfinite(BFIX); beta = np.where(fixed, BFIX, 0.2 / J)
    nb = int(np.ceil(len(order) / NB)); W = max(4.0, nb / 4.0); d = 1.0 - 1.0 / W
    s_ = BLMLBinnedStreaming(decay=d, burn_in=burn_in, lr=0.1, param_every=1, tol=1e-5, max_sweeps=3000, ll_tol=1e-9, solver=SOLVER,
                             batch_sweeps=EM_SWEEPS, engine=ENGINE, dtype=SEARCH_DTYPE)
    rng = np.random.default_rng(1); Ncum = 0.0; hist = []; q = None; prev = None
    for ep in range(int(os.environ.get('BLML_STREAM_EPOCHS', '20'))):
        idx = np.arange(nb) if ep == 0 else rng.permutation(nb); fcs = []
        for bi in idx:
            sel_ = np.zeros(len(TOTAL_N[0]), bool); sel_[order[bi * NB:(bi + 1) * NB]] = True; C = img(sel_)
            Ncum = d * Ncum + C
            if q is None:
                s_.partial_fit(C * (1 - beta.sum()), bt.make_modes(list(fc0), ['sinc', 'sinc']), complex_g=(kind == 'complex')); pre = np.nan
            else:
                P = (beta[:, None, None] * b).sum(0) + (1 - beta.sum()) * q
                pre = ll(P, C); s_.partial_fit(C * np.clip(1 - (beta[:, None, None] * b).sum(0) / P, 0.0, 1.0))
            q = np.maximum(_np(s_.density_), 1e-300); q = q / q.sum()
            P = (beta[:, None, None] * b).sum(0) + (1 - beta.sum()) * q
            beta = np.where(fixed, BFIX, (Ncum[None] * (beta[:, None, None] * b / P)).sum((1, 2)) / Ncum.sum())
            fcs.append(np.asarray(s_._fc_now()[0], float))
            hist.append(dict(epoch=ep, n=float(Ncum.sum()), preq=pre, fc=fcs[-1].tolist(), beta=beta.tolist()))
        cur = np.exp(np.mean(np.log(fcs), 0))
        print(f'  stream {kind} epoch {ep}: mean fc {np.round(cur, 4)} beta {np.round(beta, 3)}', flush=True)
        if prev is not None and np.max(np.abs(np.log(cur / prev))) < 0.01: break
        prev = cur
    P = (beta[:, None, None] * b).sum(0) + (1 - beta.sum()) * q
    return cur, P, beta, s_, hist
def bg_modes(fc, SH, nbg, K=1, fill=None, incoh=0):
    """K signal modes (sinc x sinc, one shared fc; 1-d with fill: the two-slit kernel 'slits') and one background
    mode: dirichlet x dirichlet with the crop as its period, nbg = (harmonics per dim); N = 0 is a constant. The
    background mode's normaliser is the crop integral. incoh (1-d with fill): that many single-slit modes after the
    signal modes, sinc of the slit band a = fill fc / 2 (|g_j|^2 of a slit's own field: its carrier drops out), the
    incoherent part of a partially coherent source."""
    D = len(SH); sig = bt.make_modes(list(fc), ['sinc'] * D if fill is None else [f'slits:{float(fill)!r}'])[0]
    one = [bt.make_modes([float(fill) * fc[0] / 2], ['sinc'])[0]] * incoh if incoh else []
    bg = bt.make_modes([(nbg[j] + 0.5) * 2 / SH[j] for j in range(D)], ['dirichlet'] * D, periods=[float(v) for v in SH])[0]
    return [sig] * K + one + [bg]
def bg_fit(N, fc, nbg, complex_g, K=1, init=None, dtype=np.float64, max_sweeps=3000, tol=1e-5, fill=None, seed=0, incoh=0):
    """f = sum_k |g_k|^2 + |g_bg|^2 by BLMLBinned (rank K + incoh + 1, per-mode kernels), no EM: the background weight
    is the background mode's share Q_bg / Q of the fit. seed: the start's random phases. incoh: see bg_modes."""
    m = BLMLBinned(tol=tol, max_sweeps=max_sweeps, ll_tol=1e-10, engine=ENGINE, dtype=dtype, solver=SOLVER, seed=seed).fit(
        N, bg_modes(fc, N.shape, nbg, K, fill, incoh), complex_g=complex_g, init=init)
    U = _np(m.Se_); E = _np(m.e_hat_); Q = np.array([float(np.real(np.sum(np.conj(E[k]) * U[k]))) for k in range(len(E))])
    P = np.maximum(_np(m.density_), 1e-300); return P / P.sum(), m, float(Q[-1] / Q.sum())
def stream_bg_fit(batches, SH, nb_):
    """Streaming BLML (BLMLBinnedStreaming.fit) on count batches in arrival order: a complex signal mode, its fc selected at
    the burn-in (BLML_STREAM_BURN counts) and then learned over epochs by the leave-batch-out gradient (tol 1e-4 with the
    parameters stationary, at most BLML_STREAM_EPOCHS), plus the fixed background mode of nb_ harmonics per dim over the
    crop SH. Returns the fit, its pixel probabilities and the background share."""
    D = len(SH)
    sig = {j: dict(kind='sinc') for j in range(D)}                       # fc left out: selected at the burn-in
    bgm = {j: dict(kind='dirichlet', period=float(SH[j]), fc=dict(value=(nb_[j] + 0.5) * 2 / SH[j], trainable=False)) for j in range(D)}
    m = BLMLBinnedStreaming(burn_in=int(os.environ.get('BLML_STREAM_BURN', '1000')), lr=0.05, param_every=1, tol=1e-6,
                            max_sweeps=5000, engine=ENGINE, trace_trainable=True, solver=SOLVER).fit(
        batches, [sig, bgm], epochs=int(os.environ.get('BLML_STREAM_EPOCHS', '20')), complex_g=True, tol=1e-4, tol_check='exact')
    U = _np(m.Se_); E = _np(m.e_hat_); Q = np.array([float(np.real(np.sum(np.conj(E[k]) * U[k]))) for k in range(len(E))])
    P = np.maximum(_np(m.density_), 1e-300)
    return m, P / P.sum(), float(Q[-1] / Q.sum())
FC_GRID = 0.008 * 2.0 ** (np.arange(0, 49) / 8.0); FC_GRID = FC_GRID[FC_GRID <= 0.5]   # 1-d fc grid, step 2^(1/8)
def cv_select(folds, logp, cands, patience=None, z_stop=3.0):
    """K-fold cross-validation with the one-standard-error rule. folds: [(fit counts, held-out counts)]; logp(cand,
    counts) -> log P on the lattice; cands in order of parsimony (BLML: increasing fc, nested spaces; KDE: decreasing h).
    Each candidate's held-out log-likelihood per particle over the folds; the first candidate whose paired difference to
    the best is within one standard error of it (the SE of the per-particle difference). patience: stop the scan once
    that many consecutive candidates after the best so far are each more than z_stop SE below it (they and the rest are
    not evaluated; past the band the score only falls). Returns the chosen candidate and the table."""
    n = sum(float(V.sum()) for _, V in folds); L = {}; tot = {}
    def diff_se(c, b):
        mu = tot[c] - tot[b]
        var = sum(float((V * (l - lb - mu) ** 2).sum()) for (_, V), l, lb in zip(folds, L[c], L[b])) / (n - 1)
        return mu, np.sqrt(var / n)
    run = 0
    for c in cands:
        L[c] = [np.log(np.maximum(P, 1e-300) / P.sum()) for P in (logp(c, T) for T, _ in folds)]
        tot[c] = sum(float((V * l).sum()) for (_, V), l in zip(folds, L[c])) / n
        b = max(tot, key=tot.get)
        if patience is not None:
            mu, se = diff_se(c, b) if c != b else (0.0, 0.0)
            run = run + 1 if (c != b and mu < -z_stop * se) else 0
            if run >= patience:
                break
    cands = [c for c in cands if c in tot]
    cb = max(tot, key=tot.get); tab = []
    for c in cands:
        mu = tot[c] - tot[cb]
        var = sum(float((V * (l - lb - mu) ** 2).sum()) for (_, V), l, lb in zip(folds, L[c], L[cb])) / (n - 1)
        tab.append(dict(cand=c, cv=tot[c], diff=mu, se=float(np.sqrt(var / n))))
    pick = next(r['cand'] for r in tab if r['diff'] >= -r['se'])
    return pick, tab
def bg_search(TR, VA, complex_g, K=1, tag='', NBG=(0, 1, 2, 3)):
    """Signal fc by the selector with one background harmonic per dim, refining from fc = 0.05 per axis (where the
    signal mode is not yet the background mode's twin; from very low fc the two are alike and the background takes
    everything), each candidate warm-started from the previous one unless that one's signal share collapsed (< 1%: a
    mode at zero stays at zero under the update), SEARCH_DTYPE; then the background harmonics on validation over
    NBG x NBG at that fc, then the selector refining fc from there."""
    last = [None]; path = []
    def run(nbg, start):
        def obj(fcv):
            if np.any(np.asarray(fcv) > 0.5) or np.any(np.asarray(fcv) < 2.0 / max(TR.shape)): return -np.inf
            P, m, beta = bg_fit(TR, fcv, nbg, complex_g, K, init=last[0], dtype=SEARCH_DTYPE, max_sweeps=500, tol=1e-4)
            last[0] = m if beta < 0.99 else None
            v = ll(P, VA); path.append((list(map(float, fcv)), list(nbg), v, beta)); return v
        D = TR.ndim
        return sel._search(obj, D, ['sinc'] * D, [None] * D, fc0=0.008, tag=tag, factors=(np.sqrt(2.0), 2.0 ** 0.25, 2.0 ** 0.125), start=start)
    if BG_HARM is not None:                                            # fixed background shape: the signal fc only
        nb_ = BG_HARM[-TR.ndim:]                                       # collapsed: the harmonic along the fringes
        if TR.ndim == 1:                                               # 1-d: the argmax over the full grid (fits take
            G = FC_GRID[FC_GRID >= 2.0 / len(TR)]                      # seconds; the doubling bracket can step over the
            for f in G:                                                # band and settle on a plateau above it)
                P, m, beta = bg_fit(TR, [f], nb_, complex_g, K, init=None, max_sweeps=5000, tol=1e-6)
                path.append(([float(f)], list(nb_), ll(P, VA), beta))
            k = int(np.argmax([p_[2] for p_ in path])); return np.array(path[k][0]), nb_, path[k][2], path, {}
        fc, v = run(nb_, None); return np.asarray(fc, float), nb_, v, path, {}
    fc, v = run((1, 1), None); fc = np.asarray(fc, float)              # bracket up from fc0 = 0.008 (collapses restart cold)
    grid = {}
    for ny in NBG:
        for nx in NBG:
            P, m, beta = bg_fit(TR, fc, (ny, nx), complex_g, K, init=None, dtype=SEARCH_DTYPE, max_sweeps=500, tol=1e-4)
            grid[(ny, nx)] = ll(P, VA)
    nbg = max(grid, key=grid.get)
    fc, v = run(nbg, fc)
    return np.asarray(fc, float), nbg, v, path, {f'{k[0]},{k[1]}': g for k, g in grid.items()}
def kde_q(h):
    def fit(w, st):
        q = np.maximum(gaussian_filter(w, h, mode='constant'), 1e-300); return q / q.sum(), None
    return fit
def akde_q(h, alpha):
    """Abramson's adaptive KDE on the lattice: Gaussian kernels of width h (pilot / geometric mean of the pilot)^-alpha
    at each bin, the pilot the fixed-width KDE of width h on the same weights."""
    def fit(w, st):
        w = np.asarray(w, float); pil = np.maximum(gaussian_filter(w, h, mode='constant'), 1e-300)
        occ = w > 0; gm = np.exp((w[occ] * np.log(pil[occ])).sum() / w[occ].sum())
        q = np.zeros(w.shape); idx = np.argwhere(occ)
        for i in idx:                                    # one Gaussian per occupied bin, its own width
            hi = h * (pil[tuple(i)] / gm) ** (-alpha); e = np.zeros(w.shape); e[tuple(i)] = w[tuple(i)]
            q += gaussian_filter(e, hi, mode='constant')
        q = np.maximum(q, 1e-300); return q / q.sum(), None
    return fit
def line_of(m, axis, t0, k=None):
    G = _np(m.Se_) / m.n_; t = int(round(t0))
    L = G if G.ndim == 2 else (G[:, :, t] if axis == 0 else G[:, t, :])   # collapsed: the whole line
    return L[0] if k is None else L                                          # mode 0, or every mode (rank K)

def run(name):
    global BFIX
    BFIX = np.atleast_1d(np.load(os.path.join(DATA, f'dslit_{name}_beta.npy'))).astype(float)
    pts = np.load(os.path.join(DATA, f'dslit_{name}_points.npy')); b = np.load(os.path.join(DATA, f'dslit_{name}_b.npy'))
    axis = int(np.load(os.path.join(DATA, f'dslit_{name}_axis.npy'))); b = b.reshape((-1,) + b.shape[-2:]); SH = b.shape[1:]
    ij = [np.clip(np.floor(pts[:, k] + 0.5).astype(int), 0, SH[k - 1] - 1) for k in (1, 2)]
    if BG_FLAT:                                          # the background estimated from the record: one flat shape, weight fitted
        b = np.ones((1,) + SH) / np.prod(SH); BFIX = np.array([np.nan])
    if COLLAPSE:                                         # a stripe: the counts summed across the fringes, 1-d along them
        b = b.sum(2 - axis); b = b / b.sum(1, keepdims=True); ij = [ij[axis]]; SH = b.shape[1:]; axis = 0
    u = np.random.default_rng(0).random(len(pts)); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))
    TOTAL_N[0] = pts
    def img(s):
        C = np.zeros(SH); np.add.at(C, tuple(x[s] for x in ij), 1); return C
    TR, VA, TE = img(part == 0), img(part == 1), img(part == 2); TV = TR + VA
    if SELECT == 'cv':                                   # folds of the train + validation particles
        fold = np.random.default_rng(2).integers(0, CV_K, len(pts))
        FOLDS = [(img((part < 2) & (fold != k)), img((part < 2) & (fold == k))) for k in range(CV_K)]
    path = lambda p: os.path.join(OUT, f'{name}_{p}.json')
    base = dict(set=name, n_train=int(TR.sum()), n_val=int(VA.sum()), n_test=int(TE.sum()), shape=list(SH), axis=axis, collapsed=COLLAPSE,
                bg_flat=BG_FLAT, bg_harm=None if BG_HARM is None else list(BG_HARM), select=SELECT)
    def save(p, d, line=None, P=None):
        json.dump(dict(base, part=p, **d), open(path(p), 'w'), indent=1)
        if P is not None: np.save(os.path.join(OUT, f'{name}_{p}_P.npy'), (P / P.sum()).astype(np.float32))   # for paired test SEs
        if line is not None: np.save(os.path.join(OUT, f'{name}_{p}_line.npy'), line)
        print(name, p, {k: v for k, v in d.items() if k not in ('val_path', 'history')}, flush=True)
    for p in PARTS:
        if os.path.exists(path(p)): continue
        t0 = time.time()
        if p in ('fraunhofer_blur', 'fresnel_blur'):
            base_p = p[:-5]; th0 = np.array(json.load(open(path(base_p)))['theta'])
            th = ref_fit(TV, b, axis, base_p == 'fresnel', th0, blur=True); sig = float(np.exp(th[9 + len(b)]))
            P, q, beta = ref_P(th, b, axis, base_p == 'fresnel'); _, field, _ = ref_parts(th, SH, axis, base_p == 'fresnel', sig)
            save(p, dict(test_ll=ll(P, TE), beta=np.atleast_1d(beta).tolist(), theta=th.tolist(), params=dict(ref_desc(th, base_p == 'fresnel'), sigma=sig),
                         train_ll=ll(P, TV), s=time.time() - t0), field, P)
        elif p in ('fraunhofer', 'fresnel'):
            th0 = np.array(json.load(open(path('fraunhofer')))['theta']) if p == 'fresnel' and os.path.exists(path('fraunhofer')) else None
            th = ref_fit(TV, b, axis, p == 'fresnel', th0)
            P, q, beta = ref_P(th, b, axis, p == 'fresnel'); _, field, _ = ref_parts(th, SH, axis, p == 'fresnel')
            save(p, dict(test_ll=ll(P, TE), beta=np.atleast_1d(beta).tolist(), theta=th.tolist(), params=ref_desc(th, p == 'fresnel'), train_ll=ll(P, TV), s=time.time() - t0), field, P)
        elif p == 'hist':
            save(p, dict(test_ll=ll(TV + 0.5, TE), beta=[0.0], s=time.time() - t0), None, TV + 0.5)
        elif p == 'kde':
            import itertools
            HG = list(itertools.product(*[(0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24)] * TR.ndim))
            tab = None
            if SELECT == 'cv':
                h, tab = cv_select(FOLDS, lambda h_, T: em(T, b, kde_q(h_))[0], HG[::-1])
            else:
                val = {','.join(map(str, h)): ll(em(TR, b, kde_q(h))[0], VA) for h in HG}
                h = tuple(float(v) for v in max(val, key=val.get).split(','))
            P, q, beta, _ = em(TV, b, kde_q(h)); save(p, dict(test_ll=ll(P, TE), beta=np.atleast_1d(beta).tolist(), h=list(h), h_at_grid_edge=bool(24.0 in h), cv=tab, s=time.time() - t0), None, P)
        elif p == 'akde':                                # Abramson adaptive KDE: (h, alpha) on validation, flat background by EM
            HG = [(h, a) for h in (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24) for a in (0.25, 0.5)]
            tab = None
            if SELECT == 'cv':                           # parsimony: the widest pilot, then the least adaptive
                (h, a), tab = cv_select(FOLDS, lambda c, T: em(T, b, akde_q(*c))[0], sorted(HG, key=lambda c: (-c[0], c[1])))
            else:
                val = {f'{h},{a}': ll(em(TR, b, akde_q(h, a))[0], VA) for h, a in HG}
                h, a = (float(v) for v in max(val, key=val.get).split(','))
            P, q, beta, _ = em(TV, b, akde_q(h, a)); save(p, dict(test_ll=ll(P, TE), beta=np.atleast_1d(beta).tolist(), h=h, alpha=a, cv=tab, s=time.time() - t0), None, P)
        elif p in ('positive', 'signed', 'complex'):
            ref = json.load(open(path('fraunhofer')))
            signs = None
            if p == 'signed':
                _, field, prof = ref_parts(np.array(ref['theta']), SH, axis, False)
                k = np.argmax(np.abs(field)); fr = np.real(field * np.exp(-1j * np.angle(field[k])))
                sg = np.where(fr >= 0, 1.0, -1.0)
                signs = sg if len(SH) == 1 else (np.tile(sg[:, None], (1, SH[1])) if axis == 0 else np.tile(sg[None, :], (SH[0], 1)))
            if SELECT == 'cv' and TR.ndim == 1:              # the fc grid, each fold's fit by EM with the background
                G = FC_GRID[FC_GRID >= 2.0 / len(TR)]
                f_, vpath = cv_select(FOLDS, lambda f, T: em(T, b, blml_q([f], p, signs), tol=1e-6)[0], list(G))
                fc, vll, st = np.array([f_]), None, None
            else:
                fc, vll, vpath, st = search_fc(TR, VA, b, p, signs=signs, tag=f'{name}:{p}')
            P, q, beta, m = em_polish(TV, b, fc, p, signs=signs, st0=st)
            save(p, dict(test_ll=ll(P, TE), beta=np.atleast_1d(beta).tolist(), fc=list(map(float, fc)), val_ll=vll, val_path=vpath,
                         certificate=float(m.certificate()) if p == 'complex' else None, s=time.time() - t0), line_of(m, axis, ref['params']['t0']), P)
        elif p.startswith('rank') and not p.endswith('_bg'):                # rank K complex (shared kernel, f = sum_k |g_k|^2)
            K = int(p[4:]); t_ref = 0.0 if TV.ndim == 1 else float(np.argmax(gaussian_filter(TV.sum(axis), 2)))   # profile maximum across
            prev = path(f'rank{K - 1}') if K > 1 else None
            start = np.array(json.load(open(prev))['fc']) if prev and os.path.exists(prev) else None
            fc, vll, vpath, st = search_fc(TR, VA, b, 'complex', K=K, start=start, tag=f'{name}:{p}')
            P, q, beta, m = em_polish(TV, b, fc, 'complex', K=K, st0=None)
            save(p, dict(test_ll=ll(P, TE), beta=np.atleast_1d(beta).tolist(), beta_fixed=np.isfinite(BFIX).tolist(), fc=list(map(float, fc)), val_ll=vll,
                         val_path=vpath, t_line=t_ref, s=time.time() - t0), line_of(m, axis, t_ref, k='all'), P)
        elif p in ('real_bg', 'complex_bg') or (p.startswith('rank') and p.endswith('_bg')):   # background as a mode
            cg = p != 'real_bg'; K = int(p[4:-3]) if p.startswith('rank') else 1
            fill = None
            if SELECT == 'cv' and TR.ndim == 1 and K == 1 and FILLS:   # (fc, fill) by CV; parsimony: the aperture's
                nbg = BG_HARM[-1:]; G = FC_GRID[FC_GRID >= 2.0 / len(TR)]  # measure fc fill (nested spaces)
                C_ = sorted(((float(f), w) for f in G for w in FILLS), key=lambda c: (c[0] * c[1], c[0]))
                (f_, fill), vpath = cv_select(FOLDS, lambda c, T: bg_fit(T, [c[0]], nbg, cg, 1, max_sweeps=SEARCH_SWEEPS, tol=1e-6, fill=c[1])[0], C_)
                fc, vll, grid = np.array([f_]), None, {}
            elif SELECT == 'cv' and TR.ndim == 1 and K == 1:   # the fc grid by K-fold CV, the background harmonics fixed
                nbg = BG_HARM[-1:]; G = FC_GRID[FC_GRID >= 2.0 / len(TR)]
                f_, vpath = cv_select(FOLDS, lambda f, T: bg_fit(T, [f], nbg, cg, 1, max_sweeps=SEARCH_SWEEPS, tol=1e-6)[0], list(G))
                fc, vll, grid = np.array([f_]), None, {}
            else:
                fc, nbg, vll, vpath, grid = bg_search(TR, VA, cg, K, tag=f'{name}:{p}')
            P, m, beta = bg_fit(TV, fc, nbg, cg, K, max_sweeps=5000, tol=1e-5, fill=fill)
            t_ref = float(json.load(open(path('fraunhofer')))['params']['t0'])
            save(p, dict(test_ll=ll(P, TE), beta=[beta], fc=list(map(float, fc)), fill=fill, bg_harmonics=list(nbg), val_ll=vll,
                         val_grid=grid, val_path=vpath, stop=m.stop_reason_, s=time.time() - t0), line_of(m, axis, t_ref, k='all' if K > 1 else None), P)
        elif p.startswith('stream_') and p != 'stream_bg':                  # one streaming pass with the EM background
            kind = p[7:]; NB = int(os.environ.get('BLML_STREAM_NB', '400')); trv = np.nonzero(part < 2)[0]   # arrival order
            fc_t, P, beta, s_, hist = stream_em(trv, img, b, kind, NB)
            P, q, beta, m = em_polish(TV, b, fc_t, kind, st0=s_)
            save(p, dict(test_ll=ll(P, TE), beta=np.atleast_1d(beta).tolist(), fc=list(map(float, fc_t)), history=hist,
                         certificate=float(m.certificate()) if kind == 'complex' else None, s=time.time() - t0),
                 line_of(m, axis, json.load(open(path('fraunhofer')))['params']['t0']), P)
        elif p == 'stream_bg':                                   # streaming: complex + the fixed background mode
            NB = int(os.environ.get('BLML_STREAM_NB', '400')); trv = np.nonzero(part < 2)[0]   # train + val, arrival order
            batches = [img(np.isin(np.arange(len(pts)), trv[b0:b0 + NB])) for b0 in range(0, len(trv), NB)]
            nb_ = BG_HARM[-TR.ndim:]
            m, P, beta_s = stream_bg_fit(batches, SH, nb_); fc_s = m._fc_now()[0]
            Pr, mr, beta_r = bg_fit(TV, fc_s, nb_, True, 1, max_sweeps=5000, tol=1e-5)   # the final batch fit at the learned fc
            hist = [dict(n=h['n'], preq=h['preq'], fc=h['fc'][0], revisit=h.get('revisit', False)) if 'n' in h else h for h in m.history_]
            save(p, dict(test_ll=ll(P, TE), beta=[beta_s], fc=list(map(float, fc_s)), bg_harmonics=list(nb_),
                         refit_test_ll=ll(Pr, TE), refit_beta=beta_r,
                         burn_in_fc=[h['fc'][0] for h in m.history_ if h.get('burn_in')][0], epochs=m.epochs_, converged=m.converged_,
                         residual=m.residual_, n_batches=len(batches), history=hist, s=time.time() - t0),
                 line_of(m, axis, 0.0), P)
        elif p == 'stream':
            NB = int(os.environ.get('BLML_STREAM_NB', '400'))
            m = BLMLBinnedStreaming(burn_in=1000, lr=0.1, param_every=3, tol=1e-5, max_sweeps=3000, ll_tol=1e-9, batch_sweeps=300, engine=ENGINE, solver=SOLVER)
            for b0 in range(0, len(pts), NB):
                s = np.zeros(len(pts), bool); s[b0:b0 + NB] = True; C = img(s)
                if m.n_batches_ == 0: m.partial_fit(C, bt.make_modes([0.05, 0.05], ['sinc', 'sinc']), complex_g=True)
                else: m.partial_fit(C)
            save(p, dict(history=m.history_, fc=[list(map(float, f)) for f in m._fc_now()], s=time.time() - t0))

if __name__ == '__main__':
    for name in SETS:
        if os.path.exists(os.path.join(DATA, f'dslit_{name}_points.npy')): run(name)
