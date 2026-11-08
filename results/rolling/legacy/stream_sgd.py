"""BLMLQuickStreaming (k sweeps per batch, prequential fc steps) vs one-shot select + fit, same data as
stream_vs_quick.py. Usage: stream_sgd.py [N] [batch] [k] [fc_update adam|test] [lr or step] [burn_in]"""
import sys, time, warnings, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_rolling import BLMLQuickStreaming
warnings.filterwarnings('ignore')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
BS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
K = int(sys.argv[3]) if len(sys.argv) > 3 else 1
UPD = sys.argv[4] if len(sys.argv) > 4 else 'adam'
LR = float(sys.argv[5]) if len(sys.argv) > 5 else (0.05 if UPD == 'adam' else 1.05)
BURN = int(sys.argv[6]) if len(sys.argv) > 6 else 1000
rng = np.random.default_rng(0); M = N + 5000
hr = rng.integers(0, 24, M).astype(float)
z = np.sin(2 * np.pi * hr / 24) + 0.5 * rng.normal(size=M)
X = np.c_[z, hr, rng.normal(size=M)]
X = X[rng.permutation(M)]; Xs, Xte = X[:N], X[N:]
tmpl = bt.make_modes(np.nan, ['sinc', 'dirichlet', 'sinc'], [None, 24.0, None], dep=0)
tll = lambda s: float(np.mean(np.log(np.maximum(s.predict_ratio(Xte), 1e-300))))
kw = dict(lr=LR) if UPD == 'adam' else dict(step=LR)
s = BLMLQuickStreaming(tmpl, k=K, fc_update=UPD, burn_in=BURN, search='auto', dtype=np.float64, **kw); T = 0.0
for j in range(N // BS):
    t = time.time(); s.partial_fit(Xs[j * BS:(j + 1) * BS]); dt = time.time() - t; T += dt
    if not s.history_:
        continue
    h = s.history_[-1]
    print(f'batch {j + 1:3d} n={h["n"]:6d} {dt:6.2f}s sweeps={h["sweeps"]:3d} preq={h.get("preq", float("nan")):.4f} '
          f'moved={h.get("moved", "burn-in")} rebin={h.get("rebinned", "-")} test ll {tll(s):.4f} fc {np.round(h["fc"][0], 3)}', flush=True)
print(f'stream pass {T:6.1f}s  test ll {tll(s):.4f}', flush=True)
t = time.time(); s.epoch(None); print(f'+ epoch to tol {time.time() - t:6.1f}s sweeps={s.model_.n_sweeps_} test ll {tll(s):.4f}', flush=True)
