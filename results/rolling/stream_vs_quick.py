"""Streaming vs one-shot BLMLQuick (CPU), same data and selector.
One-shot: select(search) on the first 80% of the N points (validation = the newest 20%), then a fit on
all N. Streaming: the points arrive in batches; after each batch, BLMLRolling.update on everything so far
(validation = the newest 20%; batch 1: full select). Epoch 1 = the pass over the batches; later epochs
= update on all N again (warm BLMLQuick + fc refinement) until no fc moves. Test LL on held-out points.
Usage: stream_vs_quick.py [N] [n_batches]"""
import sys, time, warnings, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_rolling import BLMLRolling
from selector import select
warnings.filterwarnings('ignore')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 10
rng = np.random.default_rng(0); M = N + 5000
hr = rng.integers(0, 24, M).astype(float)
z = np.sin(2 * np.pi * hr / 24) + 0.5 * rng.normal(size=M)
X = np.c_[z, hr, rng.normal(size=M)]
X = X[rng.permutation(M)]; Xs, Xte = X[:N], X[N:]          # stationary stream, held-out test
tmpl = bt.make_modes(np.nan, ['sinc', 'dirichlet', 'sinc'], [None, 24.0, None], dep=0)
kw = dict(engine='cpu', search='auto', dtype=np.float64)
split = lambda D: (D[:int(0.8 * len(D))], D[int(0.8 * len(D)):])
tll = lambda m: float(np.mean(np.log(np.maximum(m.predict_ratio(Xte), 1e-300))))

t = time.time(); a, b = split(Xs)
q = BLMLRolling(**kw).fit(a, tmpl, b); t_q = time.time() - t
print(f'one-shot  N={N}: {t_q:6.1f}s  test ll {tll(q.model_):.4f}  fc {np.round(bt.modes_fc(q.modes_)[0], 3)}', flush=True)

r = BLMLRolling(**kw); r.template = tmpl; bs = N // NB; t_tot = 0.0
t = time.time()
for j in range(1, NB + 1):
    tj = time.time(); a, b = split(Xs[:j * bs]); (r.update)(a, b)
    h = r.history_[-1]
    print(f'  epoch 1 batch {j:2d} n={j * bs:6d}: {time.time() - tj:6.1f}s fits={h.get("n_fits", "sel")} '
          f'moved={h.get("moved", "-")} fc {np.round(h["fc"][0], 3)}', flush=True)
t_ep = time.time() - t; t_tot += t_ep
print(f'epoch 1: {t_ep:6.1f}s  test ll {tll(r.model_):.4f}  fc {np.round(bt.modes_fc(r.modes_)[0], 3)}', flush=True)
for ep in range(2, 11):
    t = time.time(); a, b = split(Xs); r.update(a, b); t_ep = time.time() - t; t_tot += t_ep
    h = r.history_[-1]
    print(f'epoch {ep}: {t_ep:6.1f}s fits={h["n_fits"]} moved={h["moved"]} sweeps(final)={r.model_.n_sweeps_} '
          f'test ll {tll(r.model_):.4f}  fc {np.round(h["fc"][0], 3)}', flush=True)
    if not h['moved']:
        break
print(f'streaming total {t_tot:6.1f}s vs one-shot {t_q:6.1f}s', flush=True)
