"""T4: general rank-K GPU solver vs CPU, n = 1000 (ENGINE=gpu|cpu). Same data, same models."""
import os, sys, json, time, subprocess
ENGINE = os.environ.get('ENGINE', 'gpu')
if ENGINE == 'gpu' and subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda12x'], check=False)
sys.path.insert(0, os.environ.get('BLML_CODE_DIR', '.'))
import numpy as np, blml_trivial as bt
OUT = os.environ.get('BLML_OUT_DIR', 'outputs'); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0); n = 1000
X2 = np.r_[rng.normal(0, 1, (n // 2, 2)), rng.normal(2.5, 0.7, (n - n // 2, 2))]
FC = 0.328; xs = np.linspace(-80, 80, 640001); p1 = FC * np.sinc(FC * xs) ** 2; p1 /= p1.sum()
X1 = np.random.default_rng(1).choice(xs, size=n, p=p1)[:, None]
q2 = np.random.default_rng(2).normal(1, 1.5, (2000, 2)); q1 = np.linspace(-40, 40, 2000)[:, None]
cases = [('1d complex', X1, [FC], q1, dict(complex_g=True)),
         ('2d real', X2, [1.2, 1.2], q2, {}),
         ('2d complex', X2, [1.2, 1.2], q2, dict(complex_g=True)),
         ('2d complex rank2', X2, [1.2, 1.2], q2, dict(complex_g=True, rank=2)),
         ('2d real modes fc 1.2 / 0.6', X2, [1.2, 1.2], q2, dict(fc2=[[1.2, 1.2], [0.6, 0.6]]))]
res = []
for dt in (['float32', 'float64'] if ENGINE == 'gpu' else ['float64']):
    for lab, X, fc, q, kw in cases:
        if ENGINE == 'gpu':
            from blml_trivial import BLMLQuick; m = BLMLQuick(engine='gpu', dtype=dt, seed=0, tol=1e-4)
        else:
            m = bt.BLMLQuick(dtype=np.float64, seed=0, tol=1e-4)
        modes = bt.make_modes(kw.get('fc2', fc), d=X.shape[1], rank=kw.get('rank', 1))
        t0 = time.time(); m.fit(X, modes, complex_g=kw.get('complex_g', False)); s = time.time() - t0
        r = dict(engine=ENGINE, dtype=dt, case=lab, s=s, sweeps=int(m.n_sweeps_), conv=bool(m.converged_),
                 mean_log_f=float(np.mean(np.log(np.maximum(m.predict(q), 1e-300)))))
        res.append(r); print(json.dumps(r), flush=True)
json.dump(res, open(os.path.join(OUT, f'gpu_modes_{ENGINE}.json'), 'w'), indent=1)
