"""T4 check of BLMLQuick(engine='gpu') complex / rank-K against the CPU solver.
Part A: 1-D far field f = FC sinc^2(FC x) (real amplitude) and a 2-D defocus-like mixture; raw samples.
Part B: paper-2 day split (hl, hll) conditional = joint / marginal at the paper's fc.
Writes outputs/gpu_check.json; CPU numbers for the same data are computed by the same script with ENGINE=cpu."""
import os, sys, json, time, subprocess
ENGINE = os.environ.get('ENGINE', 'gpu')
if ENGINE == 'gpu' and subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda12x'], check=False)
CODE = os.environ.get('BLML_CODE_DIR', '.'); sys.path.insert(0, CODE)
import numpy as np, blml_trivial as bt, caiso_data
bt.DIRICHLET_LEGACY = False
OUT = os.environ.get('BLML_OUT_DIR', 'outputs'); os.makedirs(OUT, exist_ok=True)
def est(dtype):
    if ENGINE == 'gpu':
        from blml_trivial import BLMLQuick
        return BLMLQuick(engine='gpu', dtype=dtype, seed=0, tol=1e-4)
    return bt.BLMLQuick(dtype=np.float64, seed=0, tol=1e-4)
fitk = lambda m, X, fc, kinds, periods, rank=1, complex_g=False: m.fit(X, bt.make_modes(fc, kinds, periods, rank=rank), complex_g=complex_g)
res = {'engine': ENGINE}
# ---- Part A
FC = 0.328; rng = np.random.default_rng(0)
xs = np.linspace(-80, 80, 640001); p = FC * np.sinc(FC * xs) ** 2; p /= p.sum()
Xa = rng.choice(xs, size=60000, p=p)[:, None]; Xa = Xa[np.abs(Xa[:, 0]) <= 40]
Xtr, Xte = Xa[:40000], Xa[40000:]
for dt in (['float32', 'float64'] if ENGINE == 'gpu' else ['float64']):
    for lab, kw in [('real', {}), ('complex', dict(complex_g=True)), ('real_rank2', dict(rank=2)), ('complex_rank2', dict(complex_g=True, rank=2))]:
        t0 = time.time(); m = fitk(est(dt), Xtr, [FC], ['sinc'], [None], **kw); s = time.time() - t0
        ll = float(np.mean(np.log(np.maximum(m.predict(Xte), 1e-300))))
        res[f'A_{lab}_{dt}'] = dict(test_ll=ll, s=s, conv=bool(m.converged_), sweeps=int(m.n_sweeps_))
        print('A', lab, dt, res[f'A_{lab}_{dt}'], flush=True)
# ---- Part B
DATA = os.environ.get('BLML_DATA_DIR', 'data')
cols = caiso_data.load(DATA); ts = caiso_data.timestamps(DATA); tr, te = caiso_data.split_days(ts)
COV = {'hl': ['hour', 'lag24'], 'hll': ['hour', 'lag24', 'lag168']}
FCS = {'hl': [0.09307852223091491, 0.11876631814960742, 0.09198188948200109],
       'hll': [0.08741551964164404, 0.15860400221279208, 0.12478630077329025, 0.03038756516897867]}
for cs, covs in COV.items():
    y = cols['price']; ok = np.all([np.isfinite(cols[c]) for c in covs], axis=0); trc, tec = tr[ok[tr]], te[ok[te]]
    periods = [None] + [caiso_data.PERIODS.get(c) for c in covs]; kinds = ['sinc'] + ['dirichlet' if q else 'sinc' for q in periods[1:]]
    X = lambda idx: np.stack([y[idx]] + [cols[c][idx] for c in covs], axis=1)
    fc = FCS[cs]
    for dt in (['float32', 'float64'] if ENGINE == 'gpu' else ['float64']):
        for lab, kw in [('real', {}), ('complex', dict(complex_g=True)), ('complex_rank2', dict(complex_g=True, rank=2))]:
            t0 = time.time()
            mj = fitk(est(dt), X(trc), fc, kinds, periods, **kw)
            mm = fitk(est(dt), X(trc)[:, 1:], fc[1:], kinds[1:], periods[1:], **kw)
            s = time.time() - t0
            lp = np.log(np.maximum(mj.predict(X(tec)), 1e-300)) - np.log(np.maximum(mm.predict(X(tec)[:, 1:]), 1e-300))
            res[f'B_{cs}_{lab}_{dt}'] = dict(test_ll=float(np.mean(lp)), s=s, conv=[bool(mj.converged_), bool(mm.converged_)],
                                            sweeps=[int(mj.n_sweeps_), int(mm.n_sweeps_)])
            print('B', cs, lab, dt, res[f'B_{cs}_{lab}_{dt}'], flush=True)
json.dump(res, open(os.path.join(OUT, f'gpu_check_{ENGINE}.json'), 'w'), indent=1)
