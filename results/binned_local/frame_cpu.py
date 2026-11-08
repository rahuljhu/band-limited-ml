"""Full 2048^2 frame, complex rank-1 BLMLBinned on the CPU (float64), with the host normal-equation Anderson step;
same split as binned_gpu_job.py (seed 0, 70/10/20). Output results/binned_local/frame_cpu.json / .npz"""
import sys, json, time, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_binned import BLMLBinned
FC = 2 * 1.3 / 0.515 * 0.065
N = np.load('/private/tmp/claude-501/-Users-ra/e6cd45ce-b6f4-42cc-be9a-45c946facadf/scratchpad/gpu_data/psf_plane5.npy').astype(np.int64)
r = np.random.default_rng(0); tr = r.binomial(N, 0.7); rest = N - tr; va = r.binomial(rest, 1 / 3); te = rest - va
sc = lambda P, M: float((M * np.log(np.maximum(P, 1e-300))).sum() / M.sum())
t = time.time(); m = BLMLBinned(tol=1e-4, max_sweeps=4000, ll_tol=1e-9, verbose=True).fit(tr, bt.make_modes([FC, FC], ['ball', 'ball']), complex_g=True)
f = m.density_; P = f / f.sum()
out = dict(time_s=time.time() - t, sweeps=m.n_sweeps_, stop=m.stop_reason_, residual=m.residual_, anderson=m.n_anderson_, damped=m.n_damped_,
           gradient=m.n_gradient_, train=sc(P, tr), val=sc(P, va), test=sc(P, te), rho_history=[float(x) for x in m.rho_history_[::10]],
           merit_per_photon=[float(x) / m.n_ for x in m.merit_history_[::10]])
print(json.dumps({k: v for k, v in out.items() if 'history' not in k and 'merit' not in k}))
json.dump(out, open('results/binned_local/frame_cpu.json', 'w'), indent=1)
np.savez_compressed('results/binned_local/frame_cpu.npz', density=f.astype(np.float32))
