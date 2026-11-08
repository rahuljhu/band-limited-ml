"""Paper 3, Sec. 6: pupil phase from a defocus stack (localize-psf bead z-stack, 18 planes, dz = 0.2496 um).
Joint complex BLML: one g in PW_disk (the in-focus field), plane z sees g_z = D_z g, D_z the defocus propagator
(transfer exp(i psi_z(nu)), psi_z = 2 pi z sqrt((n_i/lam)^2 - |nu|^2), piston removed). By the representer argument
g = (1/n) sum_j e_j phi_j, phi_j = D_{-z_j} K(. - x_j), with the Hermitian Gram S_ij = T_{z_i - z_j}(x_i - x_j),
T_D(d) = int_disk exp(2 pi i nu.d + i psi_D(nu)) dnu; the stationarity condition e_b conj((Se)_b) = n w_b and the
coordinate update are those of complex BLML. Background: a uniform floor per plane (EM, as for the beads).
Baselines on the same training planes: Gerchberg-Saxton z-stack retrieval (same disk support and propagator), the
single-plane complex fit (focus plane) propagated to the other planes, and the mean of the neighbouring training
planes' histograms. Score: each held-out plane's counts are thinned 20/80; the floor of every method is fitted on the
20 %, and the mean log-probability per photon is taken on the 80 %.
Usage: python results/complex/zstack.py [bead ...]   (default: all 53)
env ZS_H: window half-width (15); fits on (2 ZS_H + 1)^2 are scored on the central (2 ZS_SCORE_H + 1)^2 pixels (15),
renormalised there, so window sizes compare on the same test photons; ZS_METHODS: comma list (all)."""
import sys, json, os, numpy as np, numba
sys.path.insert(0, '.'); sys.path.insert(0, 'results/psf')
from scipy.optimize import minimize_scalar

NA, LAM, PX, NI, DZ = 1.3, 0.515, 0.065, 1.518, 0.2496
RHO = NA / LAM * PX                   # pupil radius, cycles / px (= FC_PHYS / 2)
H = int(os.environ.get('ZS_H', '15')); W = 2 * H + 1
HS = int(os.environ.get('ZS_SCORE_H', '15')); METHODS = os.environ.get('ZS_METHODS', 'blml_stack,blml_focus,gs,ml_zernike,ml_pixel,neighbour_hist').split(',')
TRAIN, TEST, FOCUS = [3, 5, 7], [2, 4, 6, 8], 5
GAIN_CORR = 1.438
OUT = os.environ.get('ZS_OUT') or ('results/complex/zstack/' if H == 15 else f'results/complex/zstack_h{H}/')
ZETA = dict(blml=1.0, gs=2.5, zernike=1.5, pixel=0.75)   # defocus scales chosen on validation, beads 0-4 (zstack_select.json)

def psi(nu2, dz_um):
    k = (NI / LAM) ** 2 - nu2 / PX ** 2
    return 2 * np.pi * dz_um * (np.sqrt(np.maximum(k, 0)) - NI / LAM)

def tables(kmax, L=1024, dz=DZ):
    """T_k(d) for defocus k*DZ, k = -kmax..kmax, offsets d in [-(W-1), W-1]^2 -> (2kmax+1, 2W-1, 2W-1)."""
    nu = np.fft.fftfreq(L); NX, NY = np.meshgrid(nu, nu, indexing='ij'); nu2 = NX ** 2 + NY ** 2
    disk = nu2 <= RHO ** 2; out = []
    for k in range(-kmax, kmax + 1):
        T = np.fft.ifft2(np.where(disk, np.exp(1j * psi(nu2, k * dz)), 0)) * L * L / L ** 2   # sum * dnu^2 (dnu = 1/L)
        T = np.fft.fftshift(T)[L // 2 - (W - 1):L // 2 + W, L // 2 - (W - 1):L // 2 + W]
        out.append(T)
    return np.array(out)

TAB = tables(6)
_TABS = {}
def set_zeta(zeta):
    """Defocus scale: focal shift = zeta * stage step (index mismatch)."""
    global TAB
    if zeta not in _TABS:
        _TABS[zeta] = tables(6, dz=zeta * DZ)
    TAB = _TABS[zeta]
OFF = np.array([(i, j) for i in range(W) for j in range(W)])      # window pixel (row, col)

def gram(planes_i, planes_j):
    """S[a, b] = T_{z_a - z_b}(x_a - x_b) for pixels of planes_i (rows) and planes_j (columns)."""
    zi = np.repeat(planes_i, W * W); zj = np.repeat(planes_j, W * W)
    xi = np.tile(OFF, (len(planes_i), 1)); xj = np.tile(OFF, (len(planes_j), 1))
    k = zi[:, None] - zj[None, :] + 6
    dx = xi[:, None, 0] - xj[None, :, 0] + W - 1; dy = xi[:, None, 1] - xj[None, :, 1] + W - 1
    return TAB[k, dx, dy]

@numba.njit(cache=True)
def sweeps(S, w, n, e, u, tol, maxit):
    B = len(w)
    for it in range(maxit):
        for i in range(B):
            if w[i] <= 0:
                continue
            s = S[i, i].real; r = u[i] - s * e[i]; ar = abs(r); c = n * w[i]
            t = 2 * c / (ar + np.sqrt(ar * ar + 4 * s * c))
            new = t * r / ar if ar > 0 else np.sqrt(c / s) + 0j
            d = new - e[i]
            if d != 0:
                for k in range(B):
                    u[k] += S[k, i] * d
                e[i] = new
        q = 0.0
        for i in range(B):
            q += (np.conj(e[i]) * u[i]).real
        sc = n / np.sqrt(q)
        for i in range(B):
            e[i] *= sc; u[i] *= sc
        err = 0.0
        for i in range(B):
            if w[i] > 0:
                err = max(err, abs(e[i] * np.conj(u[i]) / (n * w[i]) - 1))
        if err < tol:
            return it + 1, err
    return maxit, err

def blml_stack(N, planes, iters=8, tol=1e-6, seed=0):
    """Joint complex BLML + uniform floor per plane (EM). N: (len(planes), W*W) counts. Returns e, floors."""
    S = np.ascontiguousarray(gram(planes, planes)); B = S.shape[0]; P = len(planes)
    Nf = N.ravel().astype(float); pi = np.full(P, 0.3)
    rng = np.random.default_rng(seed); e = None
    for _ in range(iters):
        if e is None:
            wq = Nf * (1 - np.repeat(pi, W * W))
        else:
            q = (np.abs(S @ e) ** 2).reshape(P, -1); q /= q.sum(1, keepdims=True)
            Pm = pi[:, None] / (W * W) + (1 - pi[:, None]) * q
            r = (pi[:, None] / (W * W)) / Pm; wq = (N * (1 - r)).ravel(); pi = (N * r).sum(1) / N.sum(1)
        n = wq.sum(); w = wq
        if e is None:
            e = np.sqrt(n * w / S[0, 0].real) * np.exp(1j * rng.normal(0, 0.1, B))
        u = S @ e
        sweeps(S, w, n, e, u, tol, 2000)
    return e, pi

def predict_blml(e, planes, z):
    g = gram([z], planes) @ e; q = np.abs(g) ** 2; return q / q.sum()

def gs(N, planes, pi, iters=300, L=128, zeta=1.0):
    """Gerchberg-Saxton over defocus planes: pupil on an L x L grid, disk support, data amplitudes on the window
    (background floor removed), free outside the window."""
    nu = np.fft.fftfreq(L); NX, NY = np.meshgrid(nu, nu, indexing='ij'); nu2 = NX ** 2 + NY ** 2
    disk = nu2 <= RHO ** 2; c0 = L // 2 - H
    amp = []
    for p, z in enumerate(planes):
        a = np.maximum(N[p] - pi[p] * N[p].sum() / (W * W), 0).reshape(W, W); amp.append(np.sqrt(a / a.sum()))
    P = disk.astype(complex)
    for _ in range(iters):
        acc = 0
        for p, z in enumerate(planes):
            h = np.fft.fftshift(np.fft.ifft2(P * np.exp(1j * psi(nu2, (z - FOCUS) * zeta * DZ))))
            win = h[c0:c0 + W, c0:c0 + W]; sc = np.sqrt((np.abs(win) ** 2).sum())
            h[c0:c0 + W, c0:c0 + W] = amp[p] * sc * np.exp(1j * np.angle(win))
            acc = acc + np.fft.fft2(np.fft.ifftshift(h)) * np.exp(-1j * psi(nu2, (z - FOCUS) * zeta * DZ))
        P = np.where(disk, acc / len(planes), 0)
    def pred(z):
        h = np.fft.fftshift(np.fft.ifft2(P * np.exp(1j * psi(nu2, (z - FOCUS) * zeta * DZ))))[c0:c0 + W, c0:c0 + W]
        q = np.abs(h.ravel()) ** 2; return q / q.sum()
    return P, pred

def zernike(nmax, r, th):
    """Noll-ordered Zernike polynomials up to radial order nmax on the unit disk (piston included)."""
    from math import factorial
    out = []
    for nn in range(nmax + 1):
        for m in range(-nn, nn + 1, 2):
            R = sum((-1) ** k * factorial(nn - k) / (factorial(k) * factorial((nn + abs(m)) // 2 - k) * factorial((nn - abs(m)) // 2 - k))
                    * r ** (nn - 2 * k) for k in range((nn - abs(m)) // 2 + 1))
            out.append(R * (np.cos(m * th) if m >= 0 else np.sin(-m * th)))
    return np.array(out)

def ml_pupil(N, planes, zeta=1.0, L=128, basis='pixel', nz=(7, 4), iters=300, P0=None, seed=None):
    """Poisson maximum-likelihood pupil retrieval (autodiff, L-BFGS): plane z pixel probabilities
    pi_z/B + (1 - pi_z) |F^-1[P e^{i psi_z}]|^2 / sum_W, pupil on the disk. basis 'pixel': free complex pupil samples;
    'zernike': P = exp(A + i Phi), A and Phi Zernike series up to radial orders nz = (phase, amplitude)."""
    import torch
    nu = np.fft.fftfreq(L); NX, NY = np.meshgrid(nu, nu, indexing='ij'); nu2 = NX ** 2 + NY ** 2
    disk = nu2 <= RHO ** 2; c0 = L // 2 - H
    prop = [torch.tensor(np.where(disk, np.exp(1j * psi(nu2, (z - FOCUS) * zeta * DZ)), 0)) for z in planes]
    Nt = torch.tensor(np.asarray(N, float))
    idx = torch.tensor(np.flatnonzero(disk.ravel()))
    if basis == 'pixel':
        init = np.ones(disk.sum(), complex) if P0 is None else P0.ravel()[disk.ravel()]
        x = torch.tensor(np.r_[init.real, init.imag], requires_grad=True)
        def pupil(x):
            m = len(x) // 2; v = torch.complex(x[:m], x[m:])
            return torch.zeros(L * L, dtype=torch.complex128).index_put((idx,), v).reshape(L, L)
    else:
        r = np.sqrt(nu2) / RHO; th = np.arctan2(NY, NX)
        Zp = torch.tensor(zernike(nz[0], r, th)[:, disk]); Za = torch.tensor(zernike(nz[1], r, th)[:, disk])
        x0 = np.zeros(len(Zp) + len(Za))
        if seed is not None:
            x0[3:len(Zp)] = np.random.default_rng(seed).normal(0, 0.3, len(Zp) - 3)     # random phase, piston/tilt 0
        x = torch.tensor(x0, requires_grad=True)
        def pupil(x):
            v = torch.exp(Za.T @ x[len(Zp):] + 1j * (Zp.T @ x[:len(Zp)]))
            return torch.zeros(L * L, dtype=torch.complex128).index_put((idx,), v).reshape(L, L)
    a = torch.zeros(len(planes), dtype=torch.float64, requires_grad=True)       # logit floors
    def probs(x, a, P_list=prop):
        P = pupil(x); out = []
        for p, pr in enumerate(P_list):
            h = torch.fft.fftshift(torch.fft.ifft2(P * pr))[c0:c0 + W, c0:c0 + W].reshape(-1)
            q = h.abs() ** 2; q = q / q.sum(); f = torch.sigmoid(a[p] - 2)
            out.append(f / (W * W) + (1 - f) * q)
        return out
    opt = torch.optim.LBFGS([x, a], max_iter=iters, line_search_fn='strong_wolfe', tolerance_grad=1e-9, tolerance_change=1e-12)
    def closure():
        opt.zero_grad(); loss = -sum((Nt[p] * torch.log(q + 1e-300)).sum() for p, q in enumerate(probs(x, a))) / Nt.sum()
        loss.backward(); return loss
    prev = np.inf
    for _ in range(50):                                   # restart L-BFGS until the loss stops moving
        loss = float(opt.step(closure).detach())
        if prev - loss < 1e-10:
            break
        prev = loss
    ml_pupil.loss = loss
    with torch.no_grad():
        P = pupil(x).numpy()
    def pred(z):
        h = np.fft.fftshift(np.fft.ifft2(P * np.where(disk, np.exp(1j * psi(nu2, (z - FOCUS) * zeta * DZ)), 0)))[c0:c0 + W, c0:c0 + W]
        q = np.abs(h.ravel()) ** 2; return q / q.sum()
    return P, pred

def ml_zernike(N, planes, zeta=1.0, starts=6):
    """Zernike ML pupil, best training likelihood over random starts."""
    best = None
    for sd in range(starts):
        P, pred = ml_pupil(N, planes, zeta=zeta, basis='zernike', nz=(6, 4), seed=sd)
        if np.isfinite(ml_pupil.loss) and (best is None or ml_pupil.loss < best[0]):   # a diverged start has loss nan
            best = (ml_pupil.loss, P, pred)
    return best[1], best[2]

def fit_floor(q, Nn):
    f = lambda a: -(Nn * np.log(a / len(q) + (1 - a) * q + 1e-300)).sum()
    return minimize_scalar(f, bounds=(1e-4, 0.9999), method='bounded').x

CROP = (np.abs(np.arange(W)[:, None] - H) <= HS) & (np.abs(np.arange(W)[None, :] - H) <= HS)

def score(q, Nn, Ns):
    if HS < H:                                            # the central window, renormalised there
        m = CROP.ravel(); q, Nn, Ns = q[m] / q[m].sum(), Nn[m], Ns[m]
    a = fit_floor(q, Nn); P = a / len(q) + (1 - a) * q
    return float((Ns * np.log(P)).sum() / Ns.sum()), float(a)

def run(bead):
    st = np.load('results/psf/stack_photons.npy', mmap_mode='r'); y0, x0 = np.load('results/psf/beads_iso.npy')[bead]
    cnt = {z: np.round(np.maximum(np.asarray(st[z, y0 - H:y0 + H + 1, x0 - H:x0 + H + 1], float) / GAIN_CORR, 0))
           .astype(np.int64).ravel() for z in TRAIN + TEST}
    rng = np.random.default_rng(100 + bead)
    hold = {z: rng.binomial(cnt[z], 0.2) for z in TEST}                       # floor-fitting part
    Ntr = np.array([cnt[z] for z in TRAIN])
    import time
    T = {}
    set_zeta(ZETA['blml']); t = time.time(); e, pi = blml_stack(Ntr, TRAIN); T['blml_stack'] = time.time() - t
    e1 = P = Pz = Pp = None; pred = {'blml_stack': lambda z: predict_blml(e, TRAIN, z)}
    if 'blml_focus' in METHODS:
        e1, pi1 = blml_stack(Ntr[[TRAIN.index(FOCUS)]], [FOCUS])
        tw = []                               # single-plane fit: choose the twin (conj e) on the other training planes
        for ee in (e1, np.conj(e1)):
            tw.append(sum(score(predict_blml(ee, [FOCUS], z), Ntr[i], Ntr[i])[0] for i, z in enumerate(TRAIN) if z != FOCUS))
        e1 = e1 if tw[0] >= tw[1] else np.conj(e1); pred['blml_focus'] = lambda z: predict_blml(e1, [FOCUS], z)
    if 'gs' in METHODS:
        t = time.time(); P, pred['gs'] = gs(Ntr, TRAIN, pi, zeta=ZETA['gs']); T['gs'] = time.time() - t
    if 'ml_zernike' in METHODS:
        t = time.time(); Pz, pred['ml_zernike'] = ml_zernike(Ntr, TRAIN, zeta=ZETA['zernike']); T['ml_zernike'] = time.time() - t
    if 'ml_pixel' in METHODS:
        t = time.time(); Pp, pred['ml_pixel'] = ml_pupil(Ntr, TRAIN, zeta=ZETA['pixel']); T['ml_pixel'] = time.time() - t
    res = dict(bead=bead, H=H, score_H=HS, pi_train=pi.tolist(), planes={})
    for z in TEST:
        Nn, Ns = hold[z], cnt[z] - hold[z]
        dmin = min(abs(t - z) for t in TRAIN); nb = [t for t in TRAIN if abs(t - z) == dmin]
        hist = sum(cnt[t] / cnt[t].sum() for t in nb) / len(nb)
        pred['neighbour_hist'] = lambda z, hist=hist: hist
        res['planes'][z] = {k: score(pred[k](z), Nn, Ns)[0] for k in METHODS if k in pred}
    res['time_s'] = T
    os.makedirs(OUT, exist_ok=True)
    np.savez(OUT + f'bead{bead}.npz', **{k: v for k, v in dict(e=e, e1=e1, P_gs=P, P_zernike=Pz, P_pixel=Pp).items() if v is not None})
    json.dump(res, open(OUT + f'bead{bead}.json', 'w'))
    print(json.dumps(res), flush=True)

if __name__ == '__main__':
    for b in (map(int, sys.argv[1:]) if len(sys.argv) > 1 else range(53)):
        run(b)
