"""Fused joint+marginal fit (dep=0) vs separate joint and marginal fits (GPU): agreement and time."""
import os, sys, json, time, subprocess
OUT = os.environ.get('BLML_OUT_DIR', 'outputs_fusetest'); CODE = os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT, exist_ok=True)
if subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda13x', 'numba'], check=True)
sys.path.insert(0, CODE)
import numpy as np, cupy as cp, blml_trivial as bt, caiso_data
from blml_trivial import make_modes
from blml_trivial import BLMLQuick
bt.DIRICHLET_LEGACY = False
cols = caiso_data.load(os.environ.get('BLML_DATA_DIR')); ok = np.isfinite(cols['lag24'])
X = np.column_stack([cols['price'], cols['hour'], cols['lag24']])[ok]
tr, te = X[:10000], X[10000:]
fc, kinds, per = [0.0931, 0.084, 0.092], ['sinc', 'dirichlet', 'sinc'], [None, 24.0, None]
def tm(f):
    cp.cuda.Device().synchronize(); t0 = time.time(); r = f(); cp.cuda.Device().synchronize(); return r, time.time() - t0
tm(lambda: BLMLQuick(engine='gpu').fit(tr[:2000], make_modes(fc, kinds, per, dep=0)))       # warm-up / compile
res = {}
for rep in range(2):
    J, tj = tm(lambda: BLMLQuick(engine='gpu', seed=0).fit(tr, make_modes(fc, kinds, per)))
    Mg, tm_ = tm(lambda: BLMLQuick(engine='gpu', seed=0).fit(tr[:, 1:], make_modes(fc[1:], kinds[1:], per[1:])))
    F, tf = tm(lambda: BLMLQuick(engine='gpu', seed=0).fit(tr, make_modes(fc, kinds, per, dep=0)))
sep = J.predict(te) / Mg.predict(te[:, 1:]); fus = F.predict_ratio(te)
rj = np.abs(F.predict(te) / J.predict(te) - 1); rm = np.abs(F.predict_marginal(te[:, 1:]) / Mg.predict(te[:, 1:]) - 1)
res = dict(t_joint=tj, t_marginal=tm_, t_separate=tj + tm_, t_fused=tf, speedup=(tj + tm_) / tf,
           joint_max_rel=float(rj.max()), marg_max_rel=float(rm.max()), ratio_max_rel=float(np.max(np.abs(fus / sep - 1))),
           ll_sep=float(np.log(sep).mean()), ll_fused=float(np.log(fus).mean()),
           sweeps_joint=J.n_sweeps_, sweeps_marg=Mg.n_sweeps_, sweeps_fused=(F.n_sweeps_, F.marg_sweeps_),
           converged=(bool(F.converged_), bool(F.marg_converged_)))
res['passed'] = bool(res['converged'][0] and res['converged'][1] and abs(res['ll_sep'] - res['ll_fused']) < 1e-3)
print(res, flush=True); json.dump(res, open(os.path.join(OUT, 'fusetest.json'), 'w'), default=float)
