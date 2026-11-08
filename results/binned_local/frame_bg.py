"""Full 2048^2 frame, complex BLMLBinned on the CPU (float64) with the background as one more mode: the signal (ball,
physical band 2NA/lambda) plus a dirichlet mode with the base harmonic only (N = 0: a constant field, periodic over the
frame), f = |g_sig|^2 + |g_bg|^2. Same split and stopping as frame_cpu.py (seed 0, 70/10/20; 4000 sweeps), so the two
differ only by the background mode. Output results/binned_local/frame_bg.json / .npz"""
import sys, json, time, numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, 'results/psf')
import blml_trivial as bt, psfkit as pk
from blml_binned import BLMLBinned, _np
FC = 2 * 1.3 / 0.515 * 0.065
N = np.round(np.maximum(np.asarray(np.load('results/psf/stack_photons.npy', mmap_mode='r')[5]) / pk.GAIN_CORR, 0)).astype(np.int64)
r = np.random.default_rng(0); tr = r.binomial(N, 0.7); rest = N - tr; va = r.binomial(rest, 1 / 3); te = rest - va
sc = lambda P, M: float((M * np.log(np.maximum(P, 1e-300))).sum() / M.sum())
L = float(N.shape[0])
sig = bt.make_modes([FC, FC], ['ball', 'ball'])[0]
bg = bt.make_modes([0.5 * 2 / L, 0.5 * 2 / L], ['dirichlet', 'dirichlet'], periods=[L, L])[0]
t = time.time(); m = BLMLBinned(tol=1e-4, max_sweeps=4000, ll_tol=1e-9, verbose=True).fit(tr, [sig, bg], complex_g=True)
f = _np(m.density_); P = f / f.sum(); E, U = _np(m.e_hat_), _np(m.Se_)
Q = np.array([float(np.real(np.sum(np.conj(E[k]) * U[k]))) for k in range(len(E))])
out = dict(time_s=time.time() - t, sweeps=m.n_sweeps_, stop=m.stop_reason_, residual=m.residual_, shares=(Q / Q.sum()).tolist(),
           train=sc(P, tr), val=sc(P, va), test=sc(P, te), rho_history=[float(x) for x in m.rho_history_[::10]])
print(json.dumps({k: v for k, v in out.items() if 'history' not in k}))
json.dump(out, open('results/binned_local/frame_bg.json', 'w'), indent=1)
np.savez_compressed('results/binned_local/frame_bg.npz', density=f.astype(np.float32))
