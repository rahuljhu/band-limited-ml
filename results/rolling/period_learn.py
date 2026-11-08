"""Paper 5: learning a period by streaming. Ten-day record with a daily cycle, z = sin(2 pi t / 24) + 0.4 eps,
t uniform on [0, 240) h; the conditional p(z | t) with t a Dirichlet column of trainable period (a period is
learned only as a conditioning column: the joint's marginal of t mod P carries -log P). 4e4 points, batches of
2000, 3 epochs, starts 22 h and 26 h, 3 seeds each. Writes results/rolling/period_learn.json."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')
import numpy as np
import blml_trivial as bt
from blml_rolling import BLMLQuickStreaming
RS = dict(rebin_tol=float(os.environ.get('BLML_REBIN_TOL', 1.0)))   # 1: the paper's runs (library default 15 since the paper)
if os.environ.get('BLML_REBIN_SWEEPS') is not None: RS['rebin_sweeps'] = int(os.environ['BLML_REBIN_SWEEPS'])   # 2: the paper's runs
FT = {} if os.environ.get('BLML_STREAM_TOL') is None else dict(tol=float(os.environ['BLML_STREAM_TOL']))   # stop at the solver's test (epochs = cap)
bt.DIRICHLET_LEGACY = False
out = []
for seed in range(3):
    r = np.random.default_rng(100 + seed); t = r.uniform(0, 240, 40000)
    X = np.c_[np.sin(2 * np.pi * t / 24) + 0.4 * r.normal(size=40000), t]
    for P0 in (22.0, 26.0):
        tmpl = [{0: dict(kind='sinc', dep=True), 1: dict(kind='dirichlet', fc=0.2, period=dict(value=P0, trainable=True))}]
        t0 = time.time()
        s = BLMLQuickStreaming(search='auto', burn_in=4000, seed=seed, dtype=np.float64, **RS).fit(X, tmpl, epochs=int(os.environ.get('BLML_STREAM_EPOCHS', 3)), batch=2000, **FT)
        path = [float(h['periods'][0]) for h in s.history_ if h.get('periods')]
        out.append(dict(seed=seed, P0=P0, P=path[-1], path=path, s=time.time() - t0, epochs=s.epochs_))
        print(f'seed {seed} start {P0}: period {path[-1]:.2f} ({time.time() - t0:.0f}s)', flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'period_learn.json'), 'w'))
