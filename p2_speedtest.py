"""Speed and correctness of the fused BLML fit (GPU). ./colab_job.sh start p2speedtest p2_speedtest.py
  1. dep=0 (bitmask) fused fit vs separate joint and marginal fits: agreement, time.
  2. dep=[0, 1] (two dependent dims) vs a separate marginal of the other dims: agreement.
  3. Block sizes 128..4096 for the fused fit: time, sweeps, agreement with block 256.
  4. Full cutoff selection (selector.select_auto) at the default and at the fastest block size.
"""
import os, sys, json, time, subprocess
OUT = os.environ.get('BLML_OUT_DIR', 'outputs_speedtest'); CODE = os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT, exist_ok=True)
if subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda13x', 'numba'], check=True)
sys.path.insert(0, CODE)
import numpy as np, cupy as cp, blml_trivial as bt, caiso_data, blml_gpu
from blml_trivial import make_modes, modes_fc
from blml_trivial import BLMLQuick
import selector
bt.DIRICHLET_LEGACY = False
D = os.environ.get('BLML_DATA_DIR')
cols = caiso_data.load(D); ts = caiso_data.timestamps(D)
tr, te = caiso_data.split_days(ts); ok = np.isfinite(cols['lag24']); tr, te = tr[ok[tr]], te[ok[te]]
X = np.column_stack([cols['price'], cols['hour'], cols['lag24']])
Xtr, Xte = X[tr], X[te]
fc, kinds, per = [0.0931, 0.084, 0.092], ['sinc', 'dirichlet', 'sinc'], [None, 24.0, None]
res = {}
def tm(f):
    cp.cuda.Device().synchronize(); t0 = time.time(); r = f(); cp.cuda.Device().synchronize(); return r, time.time() - t0
def log(k, v):
    res[k] = v; print(k, v, flush=True)
tm(lambda: BLMLQuick(engine='gpu').fit(Xtr[:2000], make_modes(fc, kinds, per, dep=0)))      # compile / warm up

# 1. fused (dep=0) vs separate
J, tj = tm(lambda: BLMLQuick(engine='gpu', seed=0).fit(Xtr, make_modes(fc, kinds, per)))
Mg, tmg = tm(lambda: BLMLQuick(engine='gpu', seed=0).fit(Xtr[:, 1:], make_modes(fc[1:], kinds[1:], per[1:])))
F, tf = tm(lambda: BLMLQuick(engine='gpu', seed=0).fit(Xtr, make_modes(fc, kinds, per, dep=0)))
sep = J.predict(Xte) / Mg.predict(Xte[:, 1:]); fus = F.predict_ratio(Xte)
log('1_fused_vs_separate', dict(t_separate=tj + tmg, t_fused=tf, ratio_max_rel=float(np.max(np.abs(fus / sep - 1))),
                                ll_sep=float(np.log(sep).mean()), ll_fused=float(np.log(fus).mean())))

# 2. two dependent dims: marginal of dim 2 alone
F2 = BLMLQuick(engine='gpu', seed=0).fit(Xtr, make_modes(fc, kinds, per, dep=[0, 1]))
M2 = BLMLQuick(engine='gpu', seed=0).fit(Xtr[:, 2:], make_modes(fc[2:], kinds[2:], per[2:]))
log('2_two_dep', dict(marg_max_rel=float(np.max(np.abs(F2.predict_marginal(Xte[:, 2:]) / M2.predict(Xte[:, 2:]) - 1))),
                      joint_max_rel=float(np.max(np.abs(F2.predict(Xte) / J.predict(Xte) - 1))),
                      converged=(bool(F2.converged_), bool(F2.marg_converged_))))

# 3. block sizes
bs_res = {}
for bs in (128, 256, 512, 1024, 2048, 4096):
    times = []
    for rep in range(2):
        m, t = tm(lambda: BLMLQuick(engine='gpu', seed=0, block_size=bs).fit(Xtr, make_modes(fc, kinds, per, dep=0)))
        times.append(t)
    bs_res[bs] = dict(t=min(times), sweeps=(m.n_sweeps_, m.marg_sweeps_), safeguard=bool(m.safeguard_used_),
                      ratio_max_rel=float(np.max(np.abs(m.predict_ratio(Xte) / fus - 1))))
    print('block', bs, bs_res[bs], flush=True)
res['3_block_sizes'] = bs_res
best_bs = min(bs_res, key=lambda b: bs_res[b]['t'])

# 4. full selection at the default and the fastest block size
day = ts[tr].astype('datetime64[D]'); days = np.unique(day)
isva = np.isin(day, np.random.default_rng(0).choice(days, int(0.2 * len(days)), replace=False))
sel = {}
for bs in sorted({256, best_bs}):
    from blml_gpu import _BLMLQuickCUDA as _G                         # the GPU class's block_size default
    d = list(_G.__init__.__defaults__); d[1] = bs; _G.__init__.__defaults__ = tuple(d)
    (sm, ll), t = tm(lambda: selector.select(Xtr[~isva], make_modes(np.nan, kinds, per, dep=0), search='auto', engine='gpu', X_val=Xtr[isva]))
    sel[bs] = dict(t=t, fc=[float(v) for v in modes_fc(sm)[0]], val_ll=float(ll))
    print('select block', bs, sel[bs], flush=True)
res['4_selection'] = sel
json.dump(res, open(os.path.join(OUT, 'speedtest.json'), 'w'), default=float)
print('DONE', flush=True)
