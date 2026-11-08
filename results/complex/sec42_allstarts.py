"""Paper 3, Sec. 4.2: single-image phase retrieval with every start polished. The earlier search (sec42_starts.py)
polished only the top 3 HIO and top 2 symmetry-breaking starts with the point-data solver; here all of them (30 HIO
from random pupil phases, 8 symmetry-breaking kicks of the positive root) are polished with BLMLBinned on the 61 x 61
pixel lattice (exact ball kernel, complex, tol 1e-6), and so is the truth start, so every likelihood is the same
engine's. Keeps the fit of highest likelihood. Same data as sec42_phase.py (n photons at the pixel centres).
Usage: sec42_allstarts.py <case> <n> <seed>  ->  results/complex/sec42_all/<case>_<n>_<seed>.json"""
import sys, os, json, time, types, numpy as np
src = open('results/complex/sec42_phase.py').read(); exec(src.split("\nm = bt.BLMLQuick")[0])
from blml_binned import BLMLBinned, _np
OUT = 'results/complex/sec42_all/'; os.makedirs(OUT, exist_ok=True)
W = 2 * H + 1; Ng = N.reshape(W, W)
M = 128; nu = np.fft.fftfreq(M); NX, NY = np.meshgrid(nu, nu, indexing='ij')
pr = np.hypot(NX, NY) / RHO; pt = np.arctan2(NY, NX); disk = pr <= 1
Amag = np.sqrt(Ng.astype(float)); box = (slice(0, W), slice(0, W))


def project_data(g):                                   # |g| = scale sqrt(N) on the window, phase kept
    gb = g[box]; s = np.sqrt((np.abs(gb) ** 2).sum() / (Amag ** 2).sum())
    g = g.copy(); g[box] = Amag * s * np.exp(1j * np.angle(gb)); return g


def hio(P, n_hio, n_er, beta=0.9):
    for it in range(n_hio + n_er):
        Pn = np.fft.fft2(project_data(np.fft.ifft2(P)))
        P = np.where(disk, Pn, (P - beta * Pn) if it < n_hio else 0)
    return P * disk


def zern_rms(sqrt_density):
    """rms Zernike error of the pupil of g (piston and tilt removed, up to the twin), as in sec42_phase.py."""
    L = 128; dx = 0.25; xs = np.arange(-L, L) * dx; X, Y = np.meshgrid(xs, xs, indexing='ij')
    g = sqrt_density(np.c_[X.ravel() + H, Y.ravel() + H]).reshape(X.shape)
    Ph = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g))); kk = np.fft.fftshift(np.fft.fftfreq(2 * L, d=dx))
    QX, QY = np.meshgrid(kk, kk, indexing='ij'); qr = np.hypot(QX, QY) / RHO; qt = np.arctan2(QY, QX); inside = qr <= 0.9
    amp = np.abs(Ph[inside]); wts = amp / amp.sum()
    A = np.stack([f(qr[inside], qt[inside]) for _, f, _ in ZN], 1); est = np.zeros(len(ZN))
    for _ in range(20):
        res = np.angle(Ph[inside] * np.exp(-1j * (A @ est)))
        est += np.linalg.lstsq(A * np.sqrt(wts)[:, None], res * np.sqrt(wts), rcond=None)[0]
    true = np.array([coef.get(nm, 0.0) for nm, _, _ in ZN]); par = np.array([pp for _, _, pp in ZN]); keep = par != 'nuis'
    tw = est.copy(); tw[par == 'even'] *= -1
    return float(min(np.sqrt(np.mean((est - true)[keep] ** 2)), np.sqrt(np.mean((tw - true)[keep] ** 2))))


def polish(gb):
    """BLMLBinned from the start g at the pixels (e_b = N_b / conj g_b, the warm-start convention)."""
    e0 = np.where(Ng > 0, Ng / np.conj(np.where(np.abs(gb) == 0, 1e-12, gb)), 0)[None]
    m = BLMLBinned(tol=1e-6, max_sweeps=20000, ll_tol=1e-12).fit(Ng, make_modes([FC, FC], ['ball', 'ball']), complex_g=True,
                                                               init=types.SimpleNamespace(e_hat_=e0, shape_=Ng.shape))
    return m


rng = np.random.default_rng(100 + seed); t0 = time.time(); starts = []
starts += [('hio', hio(disk * np.exp(2j * np.pi * rng.random((M, M))), 400, 100)) for _ in range(30)]
G0 = np.zeros((M, M), complex); G0[box] = Amag; P0 = np.fft.fft2(G0) * disk; Zf = dict((nm, f) for nm, f, _ in ZN)
starts += [('symbreak', hio(P0 * np.exp(1j * s * Zf[nm](pr, pt)), 0, 100))
           for nm in ['defocus', 'astig_0', 'astig_45', 'spherical'] for s in (1, -1)]
rows = []
mt = polish(hw.reshape(W, W)); truth = dict(ll=mt.loglik_ * mt.n_, rms=zern_rms(mt.sqrt_density), sweeps=mt.n_sweeps_)
for fam, P in starts:
    m = polish(np.fft.ifft2(P)[box])
    rows.append(dict(family=fam, ll_minus_truth=m.loglik_ * m.n_ - truth['ll'], rms=zern_rms(m.sqrt_density),
                     sweeps=m.n_sweeps_, conv=bool(m.converged_)))
best = max(rows, key=lambda r: r['ll_minus_truth'])
out = dict(case=case, n=n, seed=seed, truth=truth, best=best, n_starts=len(rows),
           n_reach_truth=int(sum(r['ll_minus_truth'] > -1.0 for r in rows)), rows=rows, t_total=time.time() - t0)
json.dump(out, open(f'{OUT}{case}_{n}_{seed}.json', 'w'), indent=1)
print(case, n, seed, 'best dLL %+.1f rms %.3f' % (best['ll_minus_truth'], best['rms']), 'truth rms %.3f' % truth['rms'],
      'starts within 1 nat of truth: %d/%d' % (out['n_reach_truth'], len(rows)), '%.0fs' % out['t_total'], flush=True)
