"""CPU: BLMLQuickStreaming (touched-bin updates, analytic Adam, batches of 2000, burn-in 5000, 2 epochs) vs one-shot select(search='auto') + fit,
2-D band-limited truth of bench_2d.py, 2000 held out, float32. Usage: bench_cpu_stream.py N [N ...]"""
import os, sys, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np
warnings.filterwarnings('ignore')
EPOCHS = int(os.environ.get('BLML_STREAM_EPOCHS', 2))
BURN_IN = int(os.environ.get('BLML_STREAM_BURN_IN', 5000))   # the paper's burn-in, pinned (the estimator's default is 2000 since 8501fbc)
import blml_trivial as bt
from selector import select
from blml_rolling import BLMLQuickStreaming
RS = dict(rebin_tol=float(os.environ.get('BLML_REBIN_TOL', 1.0)))   # 1: the paper's runs (library default 15 since the paper)
if os.environ.get('BLML_REBIN_SWEEPS') is not None: RS['rebin_sweeps'] = int(os.environ['BLML_REBIN_SWEEPS'])   # 2: the paper's runs
FT = {} if os.environ.get('BLML_STREAM_TOL') is None else dict(tol=float(os.environ['BLML_STREAM_TOL']))   # stop at the solver's test (epochs = cap)

W = np.array([0.5, 0.3, 0.2]); A = np.array([[1.0, 1.5], [1.5, 1.0], [1.2, 1.2]]); MU = np.array([[0.0, 0.0], [2.5, 1.0], [-1.5, 2.0]])
tmpl = bt.make_modes(np.nan, ['sinc', 'sinc'])
for N in [int(a) for a in sys.argv[1:]] or [10000]:
    rng = np.random.default_rng(0)

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
    tll = lambda m: round(float(np.mean(np.log(np.maximum(m.predict(Xte), 1e-300)))), 4)
    t = time.time()
    s = BLMLQuickStreaming(search='auto', burn_in=BURN_IN, dtype=np.float32, **RS).fit(Xtr, tmpl, epochs=EPOCHS, batch=2000, **FT)
    t_s = time.time() - t
    t = time.time(); sel, _ = select(Xtr, tmpl, search='auto'); m = bt.BLMLQuick(dtype=np.float32).fit(Xtr, sel); t_o = time.time() - t
    print('RESULT ' + json.dumps(dict(N=N, stream_s=round(t_s, 1), epochs=s.epochs_, burn_in_s=round(s.history_[0]['s'], 1), stream_ll=tll(s), stream_fc=np.round(bt.modes_fc(s.modes_)[0], 3).tolist(),
                                      oneshot_s=round(t_o, 1), oneshot_ll=tll(m), oneshot_fc=np.round(bt.modes_fc(sel)[0], 3).tolist(),
                                      speedup=round(t_o / t_s, 2))), flush=True)
