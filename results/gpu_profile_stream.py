"""GPU: where does BLMLQuickStreaming's time go? 40K points of the bench_2d truth, batches of 4000, burn-in 10000,
analytic Adam, under cProfile (CUDA synchronises at every host transfer, so host-side tottime is meaningful).
Prints the run's time and LL, then the top functions by total and own time."""
import os, sys, time, cProfile, pstats, io, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np
warnings.filterwarnings('ignore')
import blml_trivial as bt
from blml_trivial import BLMLQuick
from blml_rolling import BLMLQuickStreaming

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
rng = np.random.default_rng(0)
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
BLMLQuick(engine='gpu', dtype=np.float32).fit(Xtr[:200], bt.make_modes([1.5, 1.5], ['sinc', 'sinc']))     # CUDA compile
s = BLMLQuickStreaming(engine='gpu', search='auto', dtype=np.float32, burn_in=10000)
pr = cProfile.Profile(); t = time.time(); pr.enable()
s.fit(Xtr, bt.make_modes(np.nan, ['sinc', 'sinc']), epochs=2, batch=4000)
pr.disable(); T = time.time() - t
print(f'PROFILE N={N} total {T:.1f}s burn-in {s.history_[0]["s"]:.1f}s ll {np.mean(np.log(np.maximum(s.predict(Xte), 1e-300))):.4f} '
      f'fc {np.round(bt.modes_fc(s.modes_)[0], 3)}', flush=True)
for key in ('cumulative', 'tottime'):
    buf = io.StringIO(); pstats.Stats(pr, stream=buf).sort_stats(key).print_stats(22)
    print(f'PROFILE --- by {key}'); print('\n'.join('PROFILE ' + l for l in buf.getvalue().splitlines()[6:32]), flush=True)
