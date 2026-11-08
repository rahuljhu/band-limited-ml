"""Speed test: real vs complex vs complex rank-2 BLML, n = 1000 and 10000, three data sets.
Reports wall time (after a warm-up fit), sweeps, time per sweep, converged flag."""
import os, sys, json, time, subprocess
ENGINE = os.environ.get('ENGINE', 'gpu')
if ENGINE == 'gpu' and subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda12x'], check=False)
CODE = os.environ.get('BLML_CODE_DIR', '.'); sys.path.insert(0, CODE)
import numpy as np, blml_trivial as bt, caiso_data
bt.DIRICHLET_LEGACY = False
OUT = os.environ.get('BLML_OUT_DIR', 'outputs'); os.makedirs(OUT, exist_ok=True)
DT = os.environ.get('DTYPE', 'float32')
def est():
    if ENGINE == 'gpu':
        from blml_trivial import BLMLQuick
        return BLMLQuick(engine='gpu', dtype=DT, seed=0, tol=1e-4)
    return bt.BLMLQuick(dtype=np.float64, seed=0, tol=1e-4)
fitk = lambda X, fc, kinds, periods, rank=1, complex_g=False: est().fit(X, bt.make_modes(fc, kinds, periods, rank=rank), complex_g=complex_g)
rng = np.random.default_rng(0); FC = 0.328
xs = np.linspace(-80, 80, 640001); p1 = FC * np.sinc(FC * xs) ** 2; p1 /= p1.sum()
D = os.environ.get('BLML_DATA_DIR', 'data'); cols = caiso_data.load(D)
covs = ['hour', 'lag24']; ok = np.all([np.isfinite(cols[c]) for c in covs], axis=0)
Xc = np.stack([cols['price'][ok]] + [cols[c][ok] for c in covs], 1)
pc = [None] + [caiso_data.PERIODS.get(c) for c in covs]; kc = ['sinc'] + ['dirichlet' if q else 'sinc' for q in pc[1:]]
def data(name, n):
    if name == '1d_farfield':
        return rng.choice(xs, size=n, p=p1)[:, None], [FC], ['sinc'], [None]
    if name == '2d_gauss_mix':
        X = np.r_[rng.normal(0, 1, (n // 2, 2)), rng.normal(2.5, 0.7, (n - n // 2, 2))]
        return X, [1.2, 1.2], ['sinc', 'sinc'], [None, None]
    idx = rng.permutation(len(Xc))[:n]
    return Xc[idx], [0.09307852223091491, 0.11876631814960742, 0.09198188948200109], kc, pc
res = []
w = data('2d_gauss_mix', 500); fitk(w[0], w[1], w[2], w[3], complex_g=True)      # warm-up (JIT/compile)
for n in [int(v) for v in os.environ.get("SPEED_N", "1000").split(",")]:
    for name in ['1d_farfield', '2d_gauss_mix', 'caiso_hl']:
        X, fc, kinds, periods = data(name, n)
        for lab, kw in [('real', {}), ('complex', dict(complex_g=True)), ('complex_rank2', dict(complex_g=True, rank=2))]:
            t0 = time.time(); m = fitk(X, fc, kinds, periods, **kw); s = time.time() - t0
            r = dict(engine=ENGINE, dtype=DT, n=n, data=name, model=lab, B=int(m.n_bins_), s=s, sweeps=int(m.n_sweeps_),
                     s_per_sweep=s / max(1, m.n_sweeps_), conv=bool(m.converged_))
            res.append(r); print(json.dumps(r), flush=True)
json.dump(res, open(os.path.join(OUT, f'speed_{ENGINE}_{DT}.json'), 'w'), indent=1)
