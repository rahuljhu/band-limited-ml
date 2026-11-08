"""Paper 3, electron double slit: is the central component of complex BLML's |FT g| (at xi = 0, between the slits at
+-d/2) physics or method? Three tests at the complex fit's fc (results/dslit_cpu/electron_complex.json), each scored by
train / test log-likelihood and the heights of |FT g| along the fringes at xi / d = -1/2, 0, +1/2 (relative to the max):
  1. background: beta fitted (EM) vs fixed at the sideband measurement (independent of the fit);
  2. ambiguity: 6 starts (phase_noise 0.1-1.0, seeds) for each beta; equal likelihood with different spectra means the
     component is not identified by the intensity;
  3. partial coherence: rank 2 (two complex fields, f = |g_1|^2 + |g_2|^2), beta fixed; the spectra of both modes.
Output results/dslit_cpu/electron_center_diag.json"""
import os, sys, json, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_binned import BLMLBinned, _np
D = os.path.expanduser('~/data_repo/dslit_gpu/'); OUT = 'results/dslit_cpu'
pts = np.load(D + 'dslit_electron_points.npy'); b = np.load(D + 'dslit_electron_b.npy'); SH = b.shape; axis = 1
ij = [np.clip(np.floor(pts[:, k] + 0.5).astype(int), 0, SH[k - 1] - 1) for k in (1, 2)]
u = np.random.default_rng(0).random(len(pts)); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))
def img(s):
    C = np.zeros(SH); np.add.at(C, (ij[0][s], ij[1][s]), 1); return C
TV, TE = img(part < 2), img(part == 2)
cx = json.load(open(f'{OUT}/electron_complex.json')); fc = cx['fc']; fr = json.load(open(f'{OUT}/electron_fraunhofer.json'))['params']
# sideband background level: electrons in the sidebands (rows 140-169, 200-229, 60 rows) within the crop columns, scaled to 18 rows
ev = np.load(os.path.expanduser('~/data_repo/unl_dslit/electrons.npy')); x, y = ev[:, 1], ev[:, 2]
side = (((y >= 140) & (y < 170)) | ((y >= 200) & (y < 230))) & (x >= 63.5) & (x < 599.5)
BETA_SB = float(side.sum() * SH[0] / 60 / len(pts))
def ll(P, N):
    P = P / P.sum(); return float((N * np.log(np.maximum(P, 1e-300))).sum() / N.sum())
def fit(N, beta_fixed, K, seed, pn, iters=200, tol=1e-9):
    beta = 0.2 if beta_fixed is None else beta_fixed; r = np.full(SH, beta); m = None; prev = -np.inf
    modes = bt.make_modes(list(fc), ['sinc', 'sinc']) if K == 1 else [bt.make_modes(list(fc), ['sinc', 'sinc'])[0]] * K
    for _ in range(iters):
        m = BLMLBinned(tol=1e-5, max_sweeps=3000, ll_tol=1e-10, seed=seed, phase_noise=pn).fit(N * (1 - r), modes, complex_g=True, init=m)
        q = np.maximum(_np(m.density_), 1e-300); q = q / q.sum(); P = beta * b + (1 - beta) * q
        r = beta * b / P
        if beta_fixed is None: beta = float((N * r).sum() / N.sum()); P = beta * b + (1 - beta) * q
        L = ll(P, N)
        if L - prev < tol: break
        prev = L
    return P, beta, m
def spectra(m):
    t = int(round(fr['t0'])); G = _np(m.Se_)[:, t, :] / m.n_; pad = 8 * SH[1]
    xi = np.fft.fftfreq(pad) * fr['P']; out = []
    for g in G:
        A = np.abs(np.fft.fft(g, pad)); A = A / A.max()
        out.append({f'{c:+.1f}': float(A[np.abs(xi - c) < 0.12].max()) for c in (-0.5, 0.0, 0.5)})
    return out
res = dict(fc=fc, beta_sideband=BETA_SB, runs=[])
for tag, bf, K in (('beta fitted', None, 1), ('beta fixed (sideband)', BETA_SB, 1), ('rank 2, beta fixed', BETA_SB, 2)):
    for seed, pn in ((0, 0.1), (1, 0.3), (2, 0.3), (3, 1.0), (4, 1.0), (5, 1.0)):
        P, beta, m = fit(TV, bf, K, seed, pn)
        row = dict(test=tag, seed=seed, phase_noise=pn, beta=beta, train_ll=ll(P, TV), test_ll=ll(P, TE),
                   cert=float(m.certificate()) if K == 1 else None, spectra=spectra(m))
        res['runs'].append(row); print(json.dumps(row), flush=True)
json.dump(res, open(f'{OUT}/electron_center_diag.json', 'w'), indent=1)
