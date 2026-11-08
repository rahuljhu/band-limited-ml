"""Paper 5 ablation (CPU): gradient variants of streaming BLML on the 2-D benchmark of bench_cpu_stream.py
(n = 2e4, batches of 2000, burn-in 5000, 2 epochs, float32), one-shot selection as the reference. Variants:
fixed-e gradient (no response, no smoothing, every batch), + one-sweep response, + smoothing log(f + eps),
+ pooling over 3 batches (the default). Writes results/bench/ablation.json."""
import os, sys, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np
warnings.filterwarnings('ignore')
import blml_trivial as bt
from blml_rolling import BLMLQuickStreaming
from selector import select
RS = dict(rebin_tol=float(os.environ.get('BLML_REBIN_TOL', 1.0)))   # 1: the paper's runs (library default 15 since the paper)
if os.environ.get('BLML_REBIN_SWEEPS') is not None: RS['rebin_sweeps'] = int(os.environ['BLML_REBIN_SWEEPS'])   # 2: the paper's runs
FT = {} if os.environ.get('BLML_STREAM_TOL') is None else dict(tol=float(os.environ['BLML_STREAM_TOL']))   # stop at the solver's test (epochs = cap)
N = 20000
rng = np.random.default_rng(0)            # the truth and sampler of bench_cpu_stream.py
W = np.array([0.5, 0.3, 0.2]); A = np.array([[1.0, 1.5], [1.5, 1.0], [1.2, 1.2]]); MU = np.array([[0.0, 0.0], [2.5, 1.0], [-1.5, 2.0]])


def sinc2_sample(n):
    out = np.empty(0)
    while len(out) < n:
        u = np.tan(np.pi * (rng.random(2 * n) - 0.5)) / np.pi
        out = np.r_[out, u[rng.random(2 * n) < np.sinc(u) ** 2 * (1 + (np.pi * u) ** 2) / 2]]
    return out[:n]


def draw(n):
    k = rng.choice(3, n, p=W)
    return np.c_[sinc2_sample(n), sinc2_sample(n)] / A[k] + MU[k]


Xtr, Xte = draw(N), draw(2000)
tmpl = bt.make_modes(np.nan, ['sinc', 'sinc'])
tll = lambda m: round(float(np.mean(np.log(np.maximum(m.predict(Xte), 1e-300)))), 4)
rows = []
for name, resp, eps, every in (('fixed-e gradient', False, 0.0, 1), ('+ one-sweep response', True, 0.0, 1),
                               ('+ smoothing', True, 1e-3, 1), ('+ pooled over 3 batches (default)', True, 1e-3, 3)):
    t = time.time()
    s = BLMLQuickStreaming(search='auto', burn_in=5000, dtype=np.float32, grad_eps=eps, param_every=every, implicit_grad=resp, **RS).fit(Xtr, tmpl, epochs=int(os.environ.get('BLML_STREAM_EPOCHS', 2)), **FT, batch=2000)
    rows.append(dict(variant=name, s=round(time.time() - t, 1), ll=tll(s), fc=np.round(bt.modes_fc(s.modes_)[0], 3).tolist()))
    print(rows[-1], flush=True)
t = time.time(); sel, _ = select(Xtr, tmpl, search='auto'); m = bt.BLMLQuick(dtype=np.float32).fit(Xtr, sel)
rows.insert(0, dict(variant='one-shot selection (reference)', s=round(time.time() - t, 1), ll=tll(m), fc=np.round(bt.modes_fc(sel)[0], 3).tolist()))
print(rows[0], flush=True)
os.makedirs('results/bench', exist_ok=True)
json.dump(rows, open('results/bench/ablation.json', 'w'), indent=1)
