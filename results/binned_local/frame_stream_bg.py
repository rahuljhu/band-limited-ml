"""Full 2048^2 frame, streaming complex BLML with the background mode and the cut-off learned: the signal (ball) starts
at the physical band 2NA/lambda and follows Adam on the leave-batch-out prequential score (implicit gradient); the
background is the dirichlet base harmonic (fixed). Train counts (seed 0, 70/10/20 as frame_cpu.py) in 10 binomial
batches (rng 5); revisits until the exact residual < tol with the cut-off stationary, at most EPOCHS.
env BLML_EPOCHS (6), BLML_BATCH_SWEEPS (100). Output results/binned_local/frame_stream_bg.json / .npz"""
import os, sys, json, time, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'results/psf')
import blml_trivial as bt, psfkit as pk
from blml_binned import BLMLBinnedStreaming, _np
FC = 2 * 1.3 / 0.515 * 0.065; EP = int(os.environ.get('BLML_EPOCHS', '6')); BS = int(os.environ.get('BLML_BATCH_SWEEPS', '100'))
N = np.round(np.maximum(np.asarray(np.load('results/psf/stack_photons.npy', mmap_mode='r')[5]) / pk.GAIN_CORR, 0)).astype(np.int64)
r = np.random.default_rng(0); tr = r.binomial(N, 0.7); rest = N - tr; va = r.binomial(rest, 1 / 3); te = rest - va
rb = np.random.default_rng(5); rest = tr.copy(); batches = []
for k in range(10):
    b = rb.binomial(rest, 1.0 / (10 - k)); batches.append(b.astype(float)); rest = rest - b
L = float(N.shape[0])
sig = {(0, 1): dict(kind='ball', fc=FC)}                                    # trainable (Adam on log fc)
bg = {j: dict(kind='dirichlet', period=L, fc=dict(value=1.0 / L, trainable=False)) for j in (0, 1)}
sc = lambda P, M: float((M * np.log(np.maximum(P, 1e-300))).sum() / M.sum())
def on_epoch(ep, m):
    f = _np(m.density_); print(f'epoch {ep + 1}: fc {np.ravel(m._fc_now()[0]).round(4).tolist()} residual {m._residual():.2e} '
                               f'test {sc(f / f.sum(), te):.6f} {time.time() - t:.0f}s', flush=True)
t = time.time()
m = BLMLBinnedStreaming(burn_in=0, fc_update='adam', lr=0.05, param_every=1, batch_sweeps=BS, max_sweeps=4000, tol=1e-4,
                        verbose=True).fit(batches, [sig, bg], epochs=EP, complex_g=True, tol=1e-4, tol_check='exact', on_epoch=on_epoch)
f = _np(m.density_); P = f / f.sum(); E, U = _np(m.e_hat_), _np(m.Se_)
Q = np.array([float(np.real(np.sum(np.conj(E[k]) * U[k]))) for k in range(len(E))])
out = dict(time_s=time.time() - t, epochs=m.epochs_, converged=m.converged_, fc=np.ravel(m._fc_now()[0]).tolist(), fc_phys=FC,
           shares=(Q / Q.sum()).tolist(), train=sc(P, tr), val=sc(P, va), test=sc(P, te), history=m.history_)
print(json.dumps({k: v for k, v in out.items() if k != 'history'}), flush=True)
json.dump(out, open('results/binned_local/frame_stream_bg.json', 'w'), indent=1, default=float)
np.savez_compressed('results/binned_local/frame_stream_bg.npz', density=f.astype(np.float32))
