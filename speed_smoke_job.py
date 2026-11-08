"""Speed smoke test on an A100: the paper's timing protocol (p1_experiments.py expB: T1, d = 3, seed 2, the true cut-off,
BLMLQuick on the GPU, synchronised wall time after a warm-up), float32 only, at n = 3e4, 5e4, 1e5, plus n = 5e4 at
d = 8. Paper (A100, float32): 0.86 s at 3e4, 3.86 s at 1e5, 9 sweeps. Writes OUT/speed.json."""
import os, sys, json, time, subprocess
OUT = os.environ.get('BLML_OUT_DIR', 'outputs_speed'); CODE = os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT, exist_ok=True)
if subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda13x', 'numba'], check=True)
sys.path.insert(0, CODE)
import numpy as np, cupy as cp
import blml_trivial as bt
from blml_trivial import make_modes, BLMLQuick
from p1_truth import Truth, FC_TRUE
bt.DIRICHLET_LEGACY = False
sync = lambda: cp.cuda.Stream.null.synchronize()
def timed(f):
    sync(); t0 = time.time(); r = f(); sync(); return r, time.time() - t0
print('GPU:', cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), flush=True)
BLMLQuick(engine='gpu', dtype=np.float32).fit(Truth('T1', 2).sample(np.random.default_rng(0), 500), make_modes(FC_TRUE, ['sinc'] * 2, None))
rows = []
for n, d in [(30000, 3), (50000, 3), (100000, 3), (50000, 8)]:
    X = Truth('T1', d).sample(np.random.default_rng(2), n)
    for rep in range(2):
        m, t = timed(lambda: BLMLQuick(engine='gpu', dtype=np.float32).fit(X, make_modes(FC_TRUE, ['sinc'] * d, None)))
        rows.append(dict(n=n, d=d, rep=rep, fit_s=t, sweeps=m.n_sweeps_, B=m.n_bins_))
        print(f'n={n} d={d} rep {rep}: {t:.2f}s, {m.n_sweeps_} sweeps, B={m.n_bins_}', flush=True)
json.dump(rows, open(os.path.join(OUT, 'speed.json'), 'w'), indent=1)
print('done', flush=True)
