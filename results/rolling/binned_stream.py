"""BLMLQuickStreaming on counts recorded on a fixed lattice (h in the modes; 2-D, a mixture), batches of photon
counts, vs one-shot select_counts + a fit on the total counts. Usage: binned_stream.py [n_total] [n_batches] [engine]"""
import sys, time, warnings, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_rolling import BLMLQuickStreaming
from selector import select_counts
warnings.filterwarnings('ignore')
NT = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 8
ENG = sys.argv[3] if len(sys.argv) > 3 else 'cpu'
rng = np.random.default_rng(0); h = 0.1                      # detector pixel
def draw(n):
    c = rng.random(n) < 0.6
    return np.where(c[:, None], rng.normal(size=(n, 2)) * [0.6, 0.4], rng.normal(size=(n, 2)) * 0.3 + [1.2, -0.8])
def bin_counts(P):
    Zi = np.round(P / h).astype(np.int64); u, inv = np.unique(Zi, axis=0, return_inverse=True)
    return u * h, np.bincount(np.ravel(inv)).astype(float)
Pte = draw(5000); Zte, wte = bin_counts(Pte)
tmpl = bt.make_modes(np.nan, ['sinc', 'sinc'])
tll = lambda m: float(wte @ np.log(np.maximum(m.predict(Zte), 1e-300)) / wte.sum())
P = draw(NT); Z, w = bin_counts(P)
t = time.time(); wtr = rng.binomial(w.astype(np.int64), 0.8).astype(float)
sel, _ = select_counts(Z, wtr, w - wtr, h, tmpl, engine=ENG)
est = bt.BLMLQuick(engine=ENG, dtype=np.float64)
one = est.fit(Z, bt.with_lattice(sel, h), sample_weight=w); t_one = time.time() - t
print(f'one-shot: {t_one:6.1f}s test ll {tll(one):.4f} fc {np.round(bt.modes_fc(sel)[0], 3)}', flush=True)
s = BLMLQuickStreaming(engine=ENG, dtype=np.float64); T = 0.0; bs = NT // NB
for j in range(NB):
    Zb, wb = bin_counts(P[j * bs:(j + 1) * bs])
    t = time.time(); s.partial_fit(Zb, bt.with_lattice(tmpl, h), sample_weight=wb); dt = time.time() - t; T += dt
    hh = s.history_[-1]
    print(f'  batch {j + 1} counts={hh["n"]:7.0f} {dt:5.2f}s preq={hh.get("preq", float("nan")):.4f} '
          f'test ll {tll(s) if s.lj is not None else float("nan"):.4f} fc {np.round(hh.get("fc", [np.nan]), 3)}', flush=True)
print(f'streaming: {T:6.1f}s test ll {tll(s):.4f}', flush=True)
