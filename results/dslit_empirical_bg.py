"""Paper 3 / 7, electron double slit: the background as one more BLML mode, with BLMLEmpirical's window normalisation
(f a density on the observed crop: sum_b f_b = 1), in place of EM with a measured background shape and level.
  f = sum_k |g_k|^2,  the signal modes at the complex ML fit's fc (results/dslit_cpu/electron_complex.json), plus one
  background mode band-limited to fc_bg (a few cycles across the crop: slow, incoherent with the signal).
A slow, incoherent addition is what the background is; under the window normalisation a near-constant mode costs nothing
at the crop edge, which the plane normalisation (BLMLBinned, int g^2 = 1) cannot offer.
Models: signal K = 1 or 2 complex modes, with or without the background mode; fc_bg on validation over a small grid.
Reports: the background fraction (sum |g_bg|^2 / sum f over the crop) against the sideband measurement, test LL per
electron (the window-normalised f at the test electrons' pixels), and |FT g| of each signal mode along the fringes at
xi / d = -1/2, 0, +1/2. lam = 1e-8 (numerical only). Output results/dslit_cpu/electron_empirical_bg.json"""
import os, sys, json, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_binned import BLMLEmpirical, _np
D = os.path.expanduser('~/data_repo/dslit_gpu/'); OUT = 'results/dslit_cpu'
pts = np.load(D + 'dslit_electron_points.npy'); SH = np.load(D + 'dslit_electron_b.npy').shape
BETA_SB = float(np.load(D + 'dslit_electron_beta.npy'))
ij = [np.clip(np.floor(pts[:, k] + 0.5).astype(int), 0, SH[k - 1] - 1) for k in (1, 2)]
u = np.random.default_rng(0).random(len(pts)); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))
def img(s):
    C = np.zeros(SH); np.add.at(C, (ij[0][s], ij[1][s]), 1); return C
TR, VA, TE, TV = img(part == 0), img(part == 1), img(part == 2), img(part < 2)
fc_sig = json.load(open(f'{OUT}/electron_complex.json'))['fc']; fr = json.load(open(f'{OUT}/electron_fraunhofer.json'))['params']
def ll(P, N):
    P = P / P.sum(); return float((N * np.log(np.maximum(P, 1e-300))).sum() / N.sum())
def fit(N, K, fc_bg, seed=0):
    modes = [bt.make_modes(list(fc_sig), ['sinc', 'sinc'])[0]] * K
    if fc_bg is not None:
        modes = modes + [bt.make_modes([0.02, fc_bg], ['sinc', 'sinc'])[0]]   # across the 18 rows: near-flat
    m = BLMLEmpirical(reg='ridge', lam=1e-8, tol=1e-6, max_iter=5000, max_outer=500, seed=seed, retrieve=True).fit(N, modes, complex_g=True)
    U = _np(m.Se_); F = (np.abs(U) ** 2); P = F.sum(0); P = P / P.sum()
    bg = float(F[-1].sum() / F.sum()) if fc_bg is not None else 0.0
    return m, P, bg
def spectra(m, K):
    t = int(round(fr['t0'])); G = _np(m.Se_)[:K, t, :]; pad = 8 * SH[1]; xi = np.fft.fftfreq(pad) * fr['P']; out = []
    for g in G:
        A = np.abs(np.fft.fft(g, pad)); A = A / A.max(); out.append({f'{c:+.1f}': round(float(A[np.abs(xi - c) < 0.12].max()), 2) for c in (-0.5, 0.0, 0.5)})
    return out
res = dict(beta_sideband=BETA_SB, fc_signal=fc_sig, runs=[])
for K in (1, 2):
    for fc_bg in (None, 0.004, 0.008, 0.016):
        m, P, bg = fit(TR, K, fc_bg); v = ll(P, VA)
        m2, P2, bg2 = fit(TV, K, fc_bg)
        row = dict(K=K, fc_bg=fc_bg, val_ll=v, test_ll=ll(P2, TE), bg_fraction=bg2, outer=int(m2.n_outer_), spectra=spectra(m2, K))
        res['runs'].append(row); print(json.dumps(row), flush=True)
json.dump(res, open(f'{OUT}/electron_empirical_bg.json', 'w'), indent=1)
