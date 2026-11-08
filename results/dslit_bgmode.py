"""Paper 3, electron double slit: the background as an incoherent BLML mode with a periodic (dirichlet) kernel whose
period is the crop, f = |g_signal|^2 + |g_bg|^2 (BLMLBinned rank 2, per-mode kernels; complex). With N harmonics per
dim the background mode is a periodic band-limited function on the crop (N = 0: constant); its normaliser is the crop
integral, while the signal mode's is the plane's, so slow crop-wide structure costs the background mode nothing and the
signal mode a normaliser. No EM, no measured level: beta = Q_bg / Q. Signal fc fixed at the EM fit's
(results/dslit_electron/electron_complex.json). Prints test LL per electron (same split as dslit_job.py) and beta."""
import os, sys, json, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_binned import BLMLBinned, _np
D = os.path.expanduser('~/data_repo/dslit_gpu/')
pts = np.load(D + 'dslit_electron_points.npy'); SH = np.load(D + 'dslit_electron_b.npy').shape[-2:]
ij = [np.clip(np.floor(pts[:, k] + 0.5).astype(int), 0, SH[k - 1] - 1) for k in (1, 2)]
u = np.random.default_rng(0).random(len(pts)); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))
def img(s):
    C = np.zeros(SH); np.add.at(C, (ij[0][s], ij[1][s]), 1); return C
TR, VA, TE, TV = img(part == 0), img(part == 1), img(part == 2), img(part < 2)
fc = json.load(open('results/dslit_electron/electron_complex.json'))['fc']
def ll(P, N):
    P = P / P.sum(); return float((N * np.log(np.maximum(P, 1e-300))).sum() / N.sum())
sig = bt.make_modes(list(fc), ['sinc', 'sinc'])[0]
for Nx in (0, 1, 2, 3):
    bg = bt.make_modes([(0 + 0.5) * 2 / SH[0], (Nx + 0.5) * 2 / SH[1]], ['dirichlet', 'dirichlet'], periods=[float(SH[0]), float(SH[1])])[0]
    res = {}
    for lab, N in (('val', TR), ('test', TV)):
        m = BLMLBinned(tol=1e-5, max_sweeps=5000, ll_tol=1e-10).fit(N, [sig, bg], complex_g=True)
        U = _np(m.Se_); E = _np(m.e_hat_); Q = [float(np.real(np.sum(np.conj(E[k]) * U[k]))) for k in range(2)]
        P = _np(m.density_); res[lab] = (ll(P, VA if lab == 'val' else TE), Q[1] / sum(Q), m.stop_reason_)
    print(f"background harmonics along x N={Nx}: val {res['val'][0]:.5f}  test {res['test'][0]:.5f}  beta {res['test'][1]:.3f}  ({res['test'][2]})", flush=True)
