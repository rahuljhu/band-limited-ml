"""Shared code for paper 3 (bead PSFs as photon densities).
Pixel data: counts N_b on the window lattice Z (pixel centres, unit spacing). Every method returns
P, a probability vector over the window pixels (sums to 1); score = mean log P over held-out photons.
Methods
  blml_window(fc)   f_true = g^2 band-limited on R^2; data density f_true 1_W / sum_W f_true; L-BFGS from BLMLQuick on counts
  blml_plain(fc)    BLMLQuick on counts (normalised on R^2), renormalised over the window
  histogram         add-1/2 (Krichevsky-Trofimov) pixel histogram
  kde(h)            Gaussian smoothing of the count image, reflecting boundary
  gmm_bg(K)         K-component GMM + uniform background, weighted EM on pixel centres
Hyper-parameters (fc, h, K) are chosen on validation counts; fits are refit on train+val."""
import numpy as np, torch
from scipy.ndimage import gaussian_filter
import blml_trivial as bt
from blml_trivial import make_modes
bt.DIRICHLET_LEGACY = False
torch.set_default_dtype(torch.float64)
NA, LAM, PX = 1.3, 0.515, 0.065
FC_PHYS = 2 * NA / LAM * PX            # intensity band limit 2NA/lambda in cycles/pixel (lambda assumed)
GAIN_CORR = 1.438                       # photon-transfer correction of the stored 0.51 photons/ADU
FC_GRID = np.round(np.geomspace(0.12, 0.5, 14), 4)

def lattice(H):
    ax = np.arange(-H, H + 1.0); X, Y = np.meshgrid(ax, ax)
    return np.column_stack([X.ravel(), Y.ravel()])

def bead_counts(k, H=15, D='results/psf/'):
    im = np.asarray(np.load(D + 'stack_photons.npy', mmap_mode='r')[5]); y0, x0 = np.load(D + 'beads_iso.npy')[k]
    raw = im[y0 - H:y0 + H + 1, x0 - H:x0 + H + 1].astype(float)
    return np.round(np.maximum(raw / GAIN_CORR, 0)).astype(np.int64), (int(y0), int(x0))

def split(cnt, seed=0, f=(0.64, 0.16)):
    """Per-pixel binomial thinning into independent train / val / test (valid for Poisson counts)."""
    rng = np.random.default_rng(seed)
    tr = rng.binomial(cnt, f[0]); rest = cnt - tr; va = rng.binomial(rest, min(1.0, f[1] / (1 - f[0])))
    return tr, va, rest - va

def thin(N, q, seed):
    return np.random.default_rng(seed).binomial(N, q)

def score(P, N):
    N = np.asarray(N).ravel(); return float((N * np.log(np.maximum(P, 1e-300))).sum() / N.sum())

def kl(p, P):
    p = np.asarray(p).ravel(); m = p > 0; return float((p[m] * np.log(p[m] / np.maximum(P[m], 1e-300))).sum())

# ---------------- BLML ----------------
def _K(Z, fc):
    return (fc[0] * np.sinc(fc[0] * (Z[:, None, 0] - Z[None, :, 0]))) * (fc[1] * np.sinc(fc[1] * (Z[:, None, 1] - Z[None, :, 1])))

def blml_plain(Z, N, fc):
    m = bt.BLMLQuick(dtype=np.float64).fit(Z, bt.with_lattice(make_modes(list(fc), None, None, d=Z.shape[1]), 1.0), sample_weight=np.asarray(N).ravel().astype(float))
    P = m.predict(Z); return P / P.sum(), m

def blml_window(Z, N, fc, iters=4):
    P0, m = blml_plain(Z, N, fc)
    idx = {tuple(z): i for i, z in enumerate(Z)}; a0 = np.zeros(len(Z))
    for zb, eb in zip(m.Z_, m.e_hat_): a0[idx[tuple(zb)]] = eb / m.n_samples_
    Kt = torch.tensor(_K(Z, fc)); Nt = torch.tensor(np.asarray(N).ravel().astype(float)); a = torch.tensor(a0, requires_grad=True)
    def probs(a):
        g2 = (Kt @ a) ** 2; return g2 / g2.sum()
    opt = torch.optim.LBFGS([a], max_iter=300, tolerance_grad=1e-11, tolerance_change=1e-13, history_size=30, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); L = -(Nt * torch.log(probs(a) + 1e-300)).sum() / Nt.sum(); L.backward(); return L
    for _ in range(iters): opt.step(closure)
    with torch.no_grad(): P = probs(a).numpy()
    return P, a.detach().numpy()

def select_fc(Z, tr, va, fitter, grid=FC_GRID):
    """Shared fc on the grid, then two per-axis coordinate passes; returns (fc, n_fits)."""
    cache = {}
    def vl(fc):
        if fc not in cache: cache[fc] = score(fitter(Z, tr, fc)[0], va)
        return cache[fc]
    fc = [max((vl((f, f)), f) for f in grid)[1]] * 2
    for _ in range(2):
        for a in range(2):
            fc[a] = max((vl(tuple(fc[:a] + [f] + fc[a + 1:])), f) for f in grid)[1]
    return tuple(fc), len(cache)

def bl_ridge(Z, N, fc, lam):
    """Band-limited ridge fit of the square-root histogram: c = (S + lam I)^-1 sqrt(N/n), i.e.
    min ||g(z) - sqrt(N/n)||^2 + lam * int g^2 over the plane; P = g(z)^2 normalised over the window."""
    S = _K(Z, fc); u = np.sqrt(np.asarray(N).ravel() / np.sum(N))
    g = S @ np.linalg.solve(S + lam * np.eye(len(Z)), u); P = g ** 2
    return P / P.sum(), None

RIDGE_LAM = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1]

def select_ridge(Z, tr, va, grid=FC_GRID, lams=RIDGE_LAM):
    best = max((score(bl_ridge(Z, tr, (f, f), l)[0], va), f, l) for f in grid for l in lams)
    _, f, lam = best; fc = [f, f]
    for _ in range(2):
        for a in range(2):
            fc[a] = max((score(bl_ridge(Z, tr, tuple(fc[:a] + [g] + fc[a + 1:]), lam)[0], va), g) for g in grid)[1]
    lam = max((score(bl_ridge(Z, tr, tuple(fc), l)[0], va), l) for l in lams)[1]
    return tuple(fc), lam

def blml_bg(Z, N, fc, iters=60, tol=1e-7):
    """BLML (normalised on the plane) + uniform background over the window, EM with BLMLQuick on counts in the M-step."""
    N = np.asarray(N).ravel().astype(float); B = len(Z); pi = 0.1; r = np.full(B, pi); prev = -np.inf; m = None
    for _ in range(iters):            # M-step warm-starts from the previous one (same occupied bins: N > 0)
        w = N * (1 - r); m = bt.BLMLQuick(dtype=np.float64).fit(Z, bt.with_lattice(make_modes(list(fc), None, None, d=Z.shape[1]), 1.0), sample_weight=np.where(w > 1e-6, w, 0), init=m)
        q = m.predict(Z); q = q / q.sum(); P = pi / B + (1 - pi) * q
        r = (pi / B) / P; pi = float((N * r).sum() / N.sum()); P = pi / B + (1 - pi) * q
        ll = score(P, N)
        if ll - prev < tol: break
        prev = ll
    return P / P.sum(), pi

def blml_complex(Z, N, fc, kinds=None):
    """Complex BLML: f = |g|^2 with g complex band-limited (phases estimated; BLMLQuick on counts complex_g)."""
    m = bt.BLMLQuick(dtype=np.float64, tol=1e-5).fit(Z, bt.with_lattice(make_modes(list(fc), kinds, None), 1.0), sample_weight=np.asarray(N).ravel().astype(float), complex_g=True)
    P = m.predict(Z); return P / P.sum(), m

def blml_complex_bg(Z, N, fc, iters=60, tol=1e-7, kinds=None):
    """Complex BLML + uniform background over the window, EM with complex BLMLQuick on counts in the M-step."""
    N = np.asarray(N).ravel().astype(float); B = len(Z); pi = 0.1; r = np.full(B, pi); prev = -np.inf; m = None
    for it in range(iters):           # start search on the first M-step only; later ones warm-start (same bins: N > 0)
        w = N * (1 - r)
        m = bt.BLMLQuick(dtype=np.float64, tol=1e-5).fit(Z, bt.with_lattice(make_modes(list(fc), kinds, None), 1.0), sample_weight=np.where(w > 1e-6, w, 0), complex_g=True, init=m)
        q = m.predict(Z); q = q / q.sum(); P = pi / B + (1 - pi) * q
        r = (pi / B) / P; pi = float((N * r).sum() / N.sum()); P = pi / B + (1 - pi) * q
        ll = score(P, N)
        if ll - prev < tol: break
        prev = ll
    blml_complex_bg.last_model = m; blml_complex_bg.n_iter = it + 1
    return P / P.sum(), pi

# ---------------- baselines ----------------
def histogram(N):
    N = np.asarray(N).ravel().astype(float) + 0.5; return N / N.sum()

def kde(N, h, shape):
    s = np.maximum(gaussian_filter(np.asarray(N).reshape(shape).astype(float), h, mode='reflect').ravel(), 1e-12)
    return s / s.sum()

def gmm_bg(Z, N, K, iters=200, em_iters=40, seed=1):
    """GMM(K) + uniform background over the window, weighted EM on pixel centres."""
    w = np.asarray(N).ravel().astype(float); B = len(Z); g = np.random.default_rng(seed)
    c = (w[:, None] * Z).sum(0) / w.sum(); mu = c + g.normal(0, 2.0, (K, 2)) * (K > 1)
    C = np.repeat(np.eye(2)[None] * 4.0, K, 0); a = np.full(K, 1.0 / K); pi = 0.1
    def comp(mu, C, a):
        return np.stack([a[j] * np.exp(-0.5 * np.einsum('bi,ij,bj->b', Z - mu[j], np.linalg.inv(C[j]), Z - mu[j]))
                         / (2 * np.pi * np.sqrt(np.linalg.det(C[j]))) for j in range(K)])
    for _ in range(em_iters):
        dens = comp(mu, C, a); q = dens.sum(0); q = q / q.sum()
        r = (pi / B) / (pi / B + (1 - pi) * q); pi = float((w * r).sum() / w.sum()); wb = w * (1 - r)
        for _ in range(iters // em_iters + 1):
            dens = comp(mu, C, a); R = dens / np.maximum(dens.sum(0), 1e-300); Wk = R * wb; a = Wk.sum(1) / wb.sum()
            for j in range(K):
                s = max(Wk[j].sum(), 1e-12); mu[j] = (Wk[j][:, None] * Z).sum(0) / s; d = Z - mu[j]
                C[j] = (Wk[j][:, None, None] * d[:, :, None] * d[:, None, :]).sum(0) / s + 0.05 * np.eye(2)
    q = comp(mu, C, a).sum(0); q = q / q.sum(); P = pi / B + (1 - pi) * q
    return P / P.sum()

KDE_H = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
GMM_K = [1, 2, 3, 5, 8]

def run_all(Z, tr, va, te=None, shape=None, truth=None, fc_phys=FC_PHYS, methods=None):
    """Fit every method with validation selection, refit on train+val; returns {method: dict(ll / kl, hyper)}."""
    tv = tr + va; out = {}; methods = methods or ['BL-ridge', 'BL-ridge@phys', 'BLML+bg', 'BLML+bg@phys', 'BLML-W', 'BLML-W@phys', 'BLML', 'histogram', 'KDE', 'GMM+bg', 'Gaussian+bg']
    def rec(name, P, **hp):
        d = dict(hp)
        if te is not None: d['ll'] = score(P, te)
        if truth is not None: d['kl'] = kl(truth, P)
        out[name] = d; out[name]['_P'] = P
    if 'BL-ridge' in methods:
        fc, lam = select_ridge(Z, tr, va); rec('BL-ridge', bl_ridge(Z, tv, fc, lam)[0], fc=list(fc), lam=lam)
    if 'BL-ridge@phys' in methods:
        lam = max((score(bl_ridge(Z, tr, (fc_phys, fc_phys), l)[0], va), l) for l in RIDGE_LAM)[1]
        rec('BL-ridge@phys', bl_ridge(Z, tv, (fc_phys, fc_phys), lam)[0], fc=[fc_phys] * 2, lam=lam)
    if 'BLML+bg' in methods:
        fc, nf = select_fc(Z, tr, va, blml_bg); P, pi = blml_bg(Z, tv, fc); rec('BLML+bg', P, fc=list(fc), pi=pi)
    if 'BLML+bg@phys' in methods:
        P, pi = blml_bg(Z, tv, (fc_phys, fc_phys)); rec('BLML+bg@phys', P, fc=[fc_phys] * 2, pi=pi)
    if 'BLML-W' in methods:
        fc, nf = select_fc(Z, tr, va, blml_window); rec('BLML-W', blml_window(Z, tv, fc)[0], fc=list(fc), fits=nf)
    if 'BLML-W@phys' in methods:
        rec('BLML-W@phys', blml_window(Z, tv, (fc_phys, fc_phys))[0], fc=[fc_phys] * 2)
    if 'cBLML' in methods:
        fc, nf = select_fc(Z, tr, va, blml_complex); rec('cBLML', blml_complex(Z, tv, fc)[0], fc=list(fc))
    if 'cBLML@phys' in methods:
        rec('cBLML@phys', blml_complex(Z, tv, (fc_phys, fc_phys))[0], fc=[fc_phys] * 2)
    if 'cBLML+bg' in methods:
        fc, nf = select_fc(Z, tr, va, blml_complex_bg); P, pi = blml_complex_bg(Z, tv, fc); rec('cBLML+bg', P, fc=list(fc), pi=pi)
    if 'cBLML+bg@phys' in methods:
        P, pi = blml_complex_bg(Z, tv, (fc_phys, fc_phys)); rec('cBLML+bg@phys', P, fc=[fc_phys] * 2, pi=pi)
    if 'BLML' in methods:
        fc, nf = select_fc(Z, tr, va, blml_plain); rec('BLML', blml_plain(Z, tv, fc)[0], fc=list(fc))
    if 'histogram' in methods:
        rec('histogram', histogram(tv))
    if 'KDE' in methods:
        h = max((score(kde(tr, h, shape), va), h) for h in KDE_H)[1]; rec('KDE', kde(tv, h, shape), h=h)
    if 'GMM+bg' in methods:
        K = max((score(gmm_bg(Z, tr, K), va), K) for K in GMM_K)[1]; rec('GMM+bg', gmm_bg(Z, tv, K), K=K)
    if 'Gaussian+bg' in methods:
        rec('Gaussian+bg', gmm_bg(Z, tv, 1))
    return out
