"""Streaming vs one-shot on a 257 x 257 crop of the bead frame (around bead 0), complex signal (ball, 2NA/lambda) plus the
constant dirichlet background mode: BLMLBinned.fit on train against BLMLBinnedStreaming.fit on the same train counts in
10 binomial batches (fc fixed: fc_update=None; epochs until the exact residual < tol). Same split as the frame (70/10/20).
-> results/binned_local/stream_vs_batch_crop.json"""
import sys, json, time, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_binned import BLMLBinned, BLMLBinnedStreaming, _np
N = np.load('/private/tmp/claude-501/-Users-ra/e6cd45ce-b6f4-42cc-be9a-45c946facadf/scratchpad/gpu_data/psf_plane5.npy')[533:790, 940:1197].astype(np.int64)
r = np.random.default_rng(0); tr = r.binomial(N, 0.7); rest = N - tr; va = r.binomial(rest, 1 / 3); te = rest - va
rb = np.random.default_rng(5); rest = tr.copy(); batches = []
for k in range(10):
    b = rb.binomial(rest, 1.0 / (10 - k)); batches.append(b.astype(float)); rest = rest - b
L = 257.0; FC = 2 * 1.3 / 0.515 * 0.065
modes = [bt.make_modes([FC, FC], ['ball', 'ball'])[0], bt.make_modes([1 / L, 1 / L], ['dirichlet', 'dirichlet'], periods=[L, L])[0]]
sc = lambda m, M: float((M * np.log(np.maximum((lambda f: f / f.sum())(_np(m.density_)), 1e-300))).sum() / M.sum())
def shares(m):
    E, U = _np(m.e_hat_), _np(m.Se_); Q = np.array([float(np.real(np.sum(np.conj(E[k]) * U[k]))) for k in range(len(E))]); return (Q / Q.sum()).round(4).tolist()
out = {}
t = time.time(); m = BLMLBinned(tol=1e-4, max_sweeps=4000).fit(tr.astype(float), modes, complex_g=True)
out['batch'] = dict(s=time.time() - t, sweeps=m.n_sweeps_, stop=m.stop_reason_, residual=float(m.residual_), test=sc(m, te), shares=shares(m)); print('batch', out['batch'], flush=True)
for bs in (100, 300):
    t = time.time(); s = BLMLBinnedStreaming(tol=1e-4, fc_update=None, batch_sweeps=bs, max_sweeps=4000).fit(batches, modes, epochs=20, complex_g=True, tol=1e-4, tol_check='exact')
    out[f'stream_bs{bs}'] = dict(s=time.time() - t, epochs=s.epochs_, converged=s.converged_, residual=float(s.residual_), test=sc(s, te), shares=shares(s),
                                 minus_batch=sc(s, te) - out['batch']['test']); print(f'stream bs{bs}', out[f'stream_bs{bs}'], flush=True)
json.dump(out, open('results/binned_local/stream_vs_batch_crop.json', 'w'), indent=1)
