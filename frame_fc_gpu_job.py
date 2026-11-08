"""Full 2048^2 bead frame on the GPU: does held-out likelihood on the whole frame select the microscope's band?
Complex BLML (ball kernel, no background mode: on the frame it adds nothing, frame_bg_gpu_job.py) at a grid of fc
around 2NA/lambda, warm-started from the neighbouring fc (physical band first, then outwards); validation and test
log-likelihood per photon, and the global-optimality certificate at the physical band. Split seed 0, 70/10/20.
Run: BLML_DATA_SRC=<dir with psf_plane5.npy> ./colab_job.sh start frame_fc frame_fc_gpu_job.py T4
env BLML_FC_MULTS (0.8,0.9,1,1.1,1.22,1.35), BLML_MAX_SWEEPS (8000 per fc after the first; the first 20000)."""
import os, sys, json, time, subprocess
OUT = os.environ.get('BLML_OUT_DIR', 'outputs_frame_fc'); CODE = os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('BLML_DATA_DIR', '.'); ENGINE = os.environ.get('BLML_ENGINE', 'gpu')
os.makedirs(OUT, exist_ok=True)
if ENGINE == 'gpu' and subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda13x', 'numba'], check=True)
sys.path.insert(0, CODE)
import numpy as np
import blml_trivial as bt
from blml_binned import BLMLBinned, _np
dt = np.float64 if os.environ.get('BLML_DTYPE', 'float32') == 'float64' else np.float32
FC = 2 * 1.3 / 0.515 * 0.065
MULTS = [float(x) for x in (os.environ.get('BLML_FC_MULTS') or '0.8,0.9,1,1.1,1.22,1.35').split(',')]
CAP = int(os.environ.get('BLML_MAX_SWEEPS') or '8000')
N = np.load(os.path.join(DATA, 'psf_plane5.npy')).astype(np.int64)
r = np.random.default_rng(0); tr = r.binomial(N, 0.7); rest = N - tr; va = r.binomial(rest, 1 / 3); te = rest - va
sc = lambda P, M: float((M * np.log(np.maximum(P, 1e-300))).sum() / M.sum())
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

order = sorted(MULTS, key=lambda x: (abs(np.log(x)), x))          # physical first, then outwards
done = {}
for k in order:
    name = f'fc_{k:.2f}'
    if os.path.exists(os.path.join(OUT, name + '.json')):
        continue
    near = min((d for d in done), key=lambda d: abs(np.log(d / k)), default=None)
    t = time.time()
    m = BLMLBinned(engine=ENGINE, dtype=dt, tol=1e-4, max_sweeps=20000 if near is None else CAP, ll_tol=1e-9,
                   verbose=True).fit(tr, bt.make_modes([k * FC, k * FC], ['ball', 'ball']), complex_g=True,
                                     init=None if near is None else done[near])
    f = _np(m.density_); P = f / f.sum()
    out = dict(mult=k, fc=k * FC, fc_phys=FC, time_s=time.time() - t, sweeps=m.n_sweeps_, stop=m.stop_reason_,
               residual=m.residual_, warm_from=near, train=sc(P, tr), val=sc(P, va), test=sc(P, te))
    if k == 1.0:
        t = time.time(); out['certificate'] = m.certificate(); out['certificate_s'] = time.time() - t
    log(name + ' ' + json.dumps(out, default=float))
    json.dump(out, open(os.path.join(OUT, name + '.json'), 'w'), indent=1, default=float)
    np.savez_compressed(os.path.join(OUT, name + '.npz'), density=f.astype(np.float32))
    done[k] = m
    for d in [d for d in done if d != k]:                          # keep only the latest fits on the device
        if abs(np.log(d / k)) > 0.15:
            del done[d]
log('done')
