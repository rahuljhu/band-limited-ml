"""Full 2048^2 bead frame on the GPU, complex BLML with the background as a second mode (dirichlet base harmonic, fixed):
  oneshot  BLMLBinned, signal ball at the physical band 2NA/lambda (as results/binned_local/frame_bg.py)
  stream   BLMLBinnedStreaming, signal fc learned by Adam on the leave-batch-out score (as frame_stream_bg.py),
           10 binomial batches (rng 5) of the train counts, revisited until the exact residual < tol, at most EPOCHS
Split seed 0, 70/10/20 (as frame_cpu.py). Each part writes <part>_bg.json / .npz into BLML_OUT_DIR, skipped if present.
Run: BLML_DATA_SRC=<dir with psf_plane5.npy> BLML_PARTS=oneshot,stream ./colab_job.sh start frame_bg frame_bg_gpu_job.py T4
env BLML_EPOCHS (8), BLML_BATCH_SWEEPS (100), BLML_REVISIT_SWEEPS (4000), BLML_MAX_SWEEPS (20000, one-shot),
BLML_DTYPE (float32)."""
import os, sys, json, time, subprocess
OUT = os.environ.get('BLML_OUT_DIR', 'outputs_frame_bg'); CODE = os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('BLML_DATA_DIR', '.'); ENGINE = os.environ.get('BLML_ENGINE', 'gpu')
PARTS = os.environ.get('BLML_PARTS', 'oneshot,stream').split(',')
os.makedirs(OUT, exist_ok=True)
if ENGINE == 'gpu' and subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda13x', 'numba'], check=True)
sys.path.insert(0, CODE)
import numpy as np
import blml_trivial as bt
from blml_binned import BLMLBinned, BLMLBinnedStreaming, _np
dt = np.float64 if os.environ.get('BLML_DTYPE', 'float32') == 'float64' else np.float32
EP = int(os.environ.get('BLML_EPOCHS') or '8'); BS = int(os.environ.get('BLML_BATCH_SWEEPS') or '100')
RV = int(os.environ.get('BLML_REVISIT_SWEEPS') or '4000')     # cap per batch revisit (float32 cannot reach tol: keep it short)
FC = 2 * 1.3 / 0.515 * 0.065
N = np.load(os.path.join(DATA, 'psf_plane5.npy')).astype(np.int64); L = float(N.shape[0])
r = np.random.default_rng(0); tr = r.binomial(N, 0.7); rest = N - tr; va = r.binomial(rest, 1 / 3); te = rest - va
sc = lambda P, M: float((M * np.log(np.maximum(P, 1e-300))).sum() / M.sum())
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def shares(m):
    E, U = _np(m.e_hat_), _np(m.Se_); Q = np.array([float(np.real(np.sum(np.conj(E[k]) * U[k]))) for k in range(len(E))])
    return (Q / Q.sum()).tolist()


def save(name, out, f):
    json.dump(out, open(os.path.join(OUT, name + '.json'), 'w'), indent=1, default=float)
    np.savez_compressed(os.path.join(OUT, name + '.npz'), density=np.asarray(f, np.float32))


if 'oneshot' in PARTS and not os.path.exists(os.path.join(OUT, 'oneshot_bg.json')):
    sig = bt.make_modes([FC, FC], ['ball', 'ball'])[0]
    bg = bt.make_modes([0.5 * 2 / L, 0.5 * 2 / L], ['dirichlet', 'dirichlet'], periods=[L, L])[0]
    t = time.time()
    m = BLMLBinned(engine=ENGINE, dtype=dt, tol=1e-4, max_sweeps=int(os.environ.get('BLML_MAX_SWEEPS') or '20000'),
                   ll_tol=1e-9, verbose=True).fit(tr, [sig, bg], complex_g=True)
    f = _np(m.density_); P = f / f.sum()
    out = dict(time_s=time.time() - t, sweeps=m.n_sweeps_, stop=m.stop_reason_, residual=m.residual_, fc=FC, shares=shares(m),
               train=sc(P, tr), val=sc(P, va), test=sc(P, te), rho_history=[float(x) for x in m.rho_history_[::10]])
    log('oneshot ' + json.dumps({k: v for k, v in out.items() if 'history' not in k}, default=float)); save('oneshot_bg', out, f)

if 'stream' in PARTS and not os.path.exists(os.path.join(OUT, 'stream_bg.json')):
    rb = np.random.default_rng(5); rest = tr.copy(); batches = []
    for k in range(10):
        b = rb.binomial(rest, 1.0 / (10 - k)); batches.append(b.astype(float)); rest = rest - b
    sig = {(0, 1): dict(kind='ball', fc=FC)}                                    # trainable (Adam on log fc)
    bg = {j: dict(kind='dirichlet', period=L, fc=dict(value=1.0 / L, trainable=False)) for j in (0, 1)}
    t = time.time(); rows = []

    def on_epoch(ep, m):
        f = _np(m.density_); P = f / f.sum()
        rows.append(dict(epoch=ep + 1, fc=np.ravel(m._fc_now()[0]).tolist(), residual=float(m._residual()), shares=shares(m),
                         val=sc(P, va), test=sc(P, te), time_s=time.time() - t))
        log('epoch ' + json.dumps(rows[-1], default=float)); save('stream_bg.partial', dict(epochs=rows), f)
    m = BLMLBinnedStreaming(engine=ENGINE, dtype=dt, burn_in=0, fc_update='adam', lr=0.05, param_every=1, batch_sweeps=BS,
                            max_sweeps=RV, tol=1e-4, verbose=True).fit(batches, [sig, bg], epochs=EP, complex_g=True, tol=1e-4,
                                                                        tol_check='exact', on_epoch=on_epoch)
    f = _np(m.density_); P = f / f.sum()
    out = dict(time_s=time.time() - t, epochs=m.epochs_, converged=m.converged_, fc=np.ravel(m._fc_now()[0]).tolist(), fc_phys=FC,
               shares=shares(m), train=sc(P, tr), val=sc(P, va), test=sc(P, te), per_epoch=rows, history=m.history_)
    log('stream ' + json.dumps({k: v for k, v in out.items() if k not in ('history', 'per_epoch')}, default=float)); save('stream_bg', out, f)
log('done')
