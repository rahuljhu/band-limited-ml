"""(1) Warm-started fused fit (dep=0) vs cold, on growing and sliding CAISO windows; (2) the five-model
day split on hour + lag24 for end-to-end timings. ./colab_job.sh start p2warm p2_warmcheck.py"""
import os, sys, json, time, subprocess
OUT = os.environ.get('BLML_OUT_DIR', 'outputs_warm'); CODE = os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT, exist_ok=True)
for mod, pkg in [('cupy', 'cupy-cuda13x'), ('ngboost', 'ngboost'), ('tabpfn', 'tabpfn')]:
    if subprocess.run([sys.executable, '-c', f'import {mod}'], capture_output=True).returncode:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=False)
sys.path.insert(0, CODE)
import numpy as np, cupy as cp, blml_trivial as bt, caiso_data
from blml_trivial import make_modes
from blml_trivial import BLMLQuick
bt.DIRICHLET_LEGACY = False
D = os.environ.get('BLML_DATA_DIR')
cols = caiso_data.load(D); ok = np.isfinite(cols['lag24'])
X = np.column_stack([cols['price'], cols['hour'], cols['lag24']])[ok]
fc, kinds, per = [0.0931, 0.084, 0.092], ['sinc', 'dirichlet', 'sinc'], [None, 24.0, None]
Xe = X[np.random.default_rng(0).choice(len(X) - 2000, 3000, replace=False)]
def tm(f):
    cp.cuda.Device().synchronize(); t0 = time.time(); r = f(); cp.cuda.Device().synchronize(); return r, time.time() - t0
fit = lambda Xw, init=None: BLMLQuick(engine='gpu', seed=0).fit(Xw, make_modes(fc, kinds, per, dep=0), init=init)
fit(X[:2000])
res = {}
for mode, batch in (('stream', 120), ('slide', 24)):
    lo, hi = 0, 9000
    prev = fit(X[lo:hi]); rows = []
    for step in range(12):
        hi += batch; lo += batch if mode == 'slide' else 0
        cold, tc = tm(lambda: fit(X[lo:hi]))
        warm, tw = tm(lambda: fit(X[lo:hi], prev))
        rc, rw = cold.predict_ratio(Xe), warm.predict_ratio(Xe)
        rows.append(dict(t_cold=tc, t_warm=tw, sweeps_cold=(cold.n_sweeps_, cold.marg_sweeps_),
                         sweeps_warm=(warm.n_sweeps_, warm.marg_sweeps_), rebinned=bool(warm.rebinned_),
                         conv=(bool(warm.converged_), bool(warm.marg_converged_)),
                         ll_diff=float(abs(np.log(rc).mean() - np.log(rw).mean()))))
        prev = warm
    a = lambda k: float(np.mean([r[k] for r in rows]))
    res[mode] = dict(t_cold=a('t_cold'), t_warm=a('t_warm'), max_ll_diff=max(r['ll_diff'] for r in rows),
                     all_converged=all(all(r['conv']) for r in rows), rebins=sum(r['rebinned'] for r in rows),
                     sweeps_cold=rows[-1]['sweeps_cold'], sweeps_warm=rows[-1]['sweeps_warm'])
    print(mode, res[mode], flush=True)
json.dump(res, open(os.path.join(OUT, 'warmcheck.json'), 'w'), default=float)
os.environ.update(BLML_P2_RUNS='daysplit', BLML_P2_COVSETS='hl')
import p2_experiments
p2_experiments.main()
