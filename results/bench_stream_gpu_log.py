"""bench_stream_gpu.py with progress: for each N, streaming (per-batch times as they happen) then one-shot.
Usage: bench_stream_gpu_log.py N [N ...]   (prints RESULT lines as bench_stream_gpu.py)"""
import os, sys, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np
warnings.filterwarnings('ignore')
import blml_trivial as bt
import blml_rolling as R
from blml_trivial import BLMLQuick
from selector import select
RS = dict(rebin_tol=float(os.environ.get('BLML_REBIN_TOL', 1.0)))   # 1: the paper's runs (library default 15 since the paper)
if os.environ.get('BLML_REBIN_SWEEPS') is not None: RS['rebin_sweeps'] = int(os.environ['BLML_REBIN_SWEEPS'])   # 2: the paper's runs
FT = {} if os.environ.get('BLML_STREAM_TOL') is None else dict(tol=float(os.environ['BLML_STREAM_TOL']))   # stop at the solver's test (epochs = cap)

BS, BURN, DT = 4000, 10000, np.float32
EPOCHS = int(os.environ.get('BLML_STREAM_EPOCHS', 2))
SEARCH = os.environ.get('BURN_SEARCH', 'auto')            # the burn-in's selector search
ONESHOT = not os.environ.get('NO_ONESHOT')
W = np.array([0.5, 0.3, 0.2]); A = np.array([[1.0, 1.5], [1.5, 1.0], [1.2, 1.2]]); MU = np.array([[0.0, 0.0], [2.5, 1.0], [-1.5, 2.0]])
_step = R._StreamingBase._step
def step(self, X, wt, revisit, **kw):                     # log every batch as it finishes
    t = time.time(); _step(self, X, wt, revisit, **kw)
    print(f'    batch {len(self.history_)} revisit={revisit} bins={len(self.lj.e)} {time.time() - t:.2f}s', flush=True)
R._StreamingBase._step = step

for N in map(int, sys.argv[1:]):
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
    tmpl = bt.make_modes(np.nan, ['sinc', 'sinc'])
    BLMLQuick(engine='gpu', dtype=DT).fit(Xtr[:200], bt.make_modes([1.5, 1.5], ['sinc', 'sinc']))      # CUDA compile, not timed
    print(f'N={N} start {time.strftime("%H:%M:%S")}', flush=True)
    t = time.time()
    s = R.BLMLQuickStreaming(engine='gpu', search=SEARCH, dtype=DT, burn_in=BURN, **RS).fit(Xtr, tmpl, epochs=EPOCHS, batch=BS, **FT)
    tot = time.time() - t; burn = s.history_[0]['s']
    print('RESULT ' + json.dumps(dict(method=f'stream-{SEARCH}', N=N, epochs=s.epochs_, total_s=round(tot, 1), burn_in_s=round(burn, 1),
                                      bins=int(len(s.lj.e)), ll=tll(s), fc=np.round(bt.modes_fc(s.modes_)[0], 3).tolist())), flush=True)
    if not ONESHOT:
        continue
    t = time.time(); sel, _ = select(Xtr, tmpl, search='auto', engine='gpu'); t_sel = time.time() - t
    t = time.time(); m = BLMLQuick(engine='gpu', dtype=DT).fit(Xtr, sel); t_fit = time.time() - t
    print('RESULT ' + json.dumps(dict(method='one-shot', N=N, total_s=round(t_sel + t_fit, 1), select_s=round(t_sel, 1),
                                      fit_s=round(t_fit, 1), ll=tll(m), fc=np.round(bt.modes_fc(sel)[0], 3).tolist())), flush=True)
print('BENCH DONE', flush=True)
