"""Warm-started BLML fits for streaming and sliding windows: speed and agreement with a cold fit (GPU).
Run: ./colab_job.sh start p2stream p2_stream_bench.py

For each setting, fit cold on an initial window, then repeatedly add a batch of 24 points (one day;
'stream') or add 24 and drop the 24 oldest ('slide'). Each step is fitted twice -- cold, and warm from
the previous step's fit -- and we record sweeps, time, the objective, and the largest relative
difference of the two fitted densities on held-out points.
"""
import os, sys, json, time, subprocess

OUT = os.environ.get('BLML_OUT_DIR', 'outputs_p2stream')
CODE = os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('BLML_DATA_DIR')
os.makedirs(OUT, exist_ok=True)
if subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda13x', 'numba'], check=True)
sys.path.insert(0, CODE)
import numpy as np
import cupy as cp
import blml_trivial as bt
from blml_trivial import make_modes
from blml_trivial import BLMLQuick
from p1_truth import Truth
import caiso_data

bt.DIRICHLET_LEGACY = False
STEPS = 20

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def timed_fit(X, fc, kinds, periods, init=None):
    cp.cuda.Device().synchronize(); t0 = time.time()
    m = BLMLQuick(engine='gpu', dtype=np.float64, seed=0).fit(X, make_modes(fc, kinds, periods), init=init)
    cp.cuda.Device().synchronize()
    return m, time.time() - t0

def run(name, X, Xeval, fc, kinds, periods, n0, mode):
    batch = 120 if mode == 'stream' else 24     # stream: grow past the 10% re-bin threshold
    res = []
    lo = 0
    prev, _ = timed_fit(X[lo:n0], fc, kinds, periods)
    hi = n0
    for step in range(STEPS):
        hi += batch
        if mode == 'slide':
            lo += batch
        Xw = X[lo:hi]
        cold, tc = timed_fit(Xw, fc, kinds, periods)
        warm, tw = timed_fit(Xw, fc, kinds, periods, init=prev)
        # Cold and warm use different lattices (h depends on n): compare densities, not coefficients.
        pc, pw = cold.predict(Xeval), warm.predict(Xeval)
        keep = pc > 1e-3 * pc.max()
        r = dict(step=step, n=len(Xw), B=warm.n_bins_, new_bins=warm.n_new_bins_, rebinned=bool(warm.rebinned_),
                 sweeps_cold=cold.n_sweeps_, sweeps_warm=warm.n_sweeps_, t_cold=tc, t_warm=tw,
                 conv_cold=bool(cold.converged_), conv_warm=bool(warm.converged_),
                 ll_cold=float(np.log(np.maximum(pc, 1e-300)).mean()), ll_warm=float(np.log(np.maximum(pw, 1e-300)).mean()),
                 max_rel_diff=float(np.max(np.abs(pw[keep] - pc[keep]) / pc[keep])))
        res.append(r)
        log(f"{name} {mode} step {step}: n={r['n']} B={r['B']} new={r['new_bins']} rebin={r['rebinned']} sweeps cold/warm "
            f"{r['sweeps_cold']}/{r['sweeps_warm']} time {tc:.2f}/{tw:.2f}s  maxreldiff {r['max_rel_diff']:.1e}")
        prev = warm
    return res

out = {}
# 1) synthetic T1, d = 3 and 6
for d, n0 in [(3, 20000), (6, 20000)]:
    tr = Truth('T1', d); rng = np.random.default_rng(d)
    X = tr.sample(rng, n0 + 120 * STEPS); Xe = tr.sample(rng, 5000)
    for mode in ('stream', 'slide'):
        out[f'T1_d{d}_{mode}'] = run(f'T1 d={d}', X, Xe, [0.4] * d, ['sinc'] * d, [None] * d, n0, mode)
# 2) CAISO (price, hour, lag24) in time order, cutoffs as selected on the day split
cols = caiso_data.load(DATA)
ok = np.isfinite(cols['lag24'])
X = np.column_stack([cols['price'], cols['hour'], cols['lag24']])[ok]
fc = [0.093, 0.084, 0.092]
n0 = len(X) - 120 * STEPS
Xe = X[np.random.default_rng(0).choice(n0, 3000, replace=False)]
for mode in ('stream', 'slide'):
    out[f'caiso_hl_{mode}'] = run('CAISO hl', X, Xe, fc, ['sinc', 'dirichlet', 'sinc'], [None, 24.0, None], n0 - 0, mode)
with open(os.path.join(OUT, 'stream_bench.json'), 'w') as f:
    json.dump(out, f, default=float)
log('ALL DONE')
