"""GPU only: BLMLQuickStreaming (burn-in 10000, batches of 4000) vs one-shot select(search='auto') + fit on the
same N, 2-D band-limited truth of bench_2d.py, 2000 held out, float32.
Usage: bench_stream_gpu.py N [fc_update,...]"""
import os, sys, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..'); sys.path.insert(0, '.')
import numpy as np
EPOCHS = int(os.environ.get('BLML_STREAM_EPOCHS', 2))
warnings.filterwarnings('ignore')
import blml_trivial as bt
from blml_trivial import BLMLQuick
from selector import select
from blml_rolling import BLMLQuickStreaming

N = int(sys.argv[1]); UPDS = (sys.argv[2] if len(sys.argv) > 2 else 'adam').split(',')
BS, BURN, DT = 4000, 10000, np.float32
rng = np.random.default_rng(0)
W = np.array([0.5, 0.3, 0.2]); A = np.array([[1.0, 1.5], [1.5, 1.0], [1.2, 1.2]])
MU = np.array([[0.0, 0.0], [2.5, 1.0], [-1.5, 2.0]])


def sinc2_sample(n):
    out = np.empty(0)
    while len(out) < n:
        u = np.tan(np.pi * (rng.random(2 * n) - 0.5)) / np.pi
        out = np.r_[out, u[rng.random(2 * n) < np.sinc(u) ** 2 * (1 + (np.pi * u) ** 2) / 2]]
    return out[:n]


def draw(n):
    k = rng.choice(3, n, p=W)
    return np.c_[sinc2_sample(n), sinc2_sample(n)] / A[k] + MU[k]


def true_pdf(X):
    return sum(W[k] * np.prod(A[k] * np.sinc(A[k] * (X - MU[k])) ** 2, 1) for k in range(3))


Xtr, Xte = draw(N), draw(2000)
tll = lambda m: round(float(np.mean(np.log(np.maximum(m.predict(Xte), 1e-300)))), 4)
tmpl = bt.make_modes(np.nan, ['sinc', 'sinc'])
BLMLQuick(engine='gpu', dtype=DT).fit(Xtr[:200], bt.make_modes([1.5, 1.5], ['sinc', 'sinc']))      # CUDA compile, not timed
print(f'N={N} true ll {np.mean(np.log(true_pdf(Xte))):.4f}', flush=True)

for upd in UPDS:
    t = time.time()
    s = BLMLQuickStreaming(engine='gpu', search='auto', dtype=DT, burn_in=BURN, fc_update=upd).fit(
        Xtr, tmpl, epochs=EPOCHS, batch=BS)
    burn = s.history_[0]['s']
    r = dict(method=f'stream-{upd}', N=N, epochs=EPOCHS, total_s=round(time.time() - t, 1), burn_in_s=round(burn, 1),
             stream_s=round(time.time() - t - burn, 1), bins=int(len(s.lj.e)), ll=tll(s),
             fc=np.round(bt.modes_fc(s.modes_)[0], 3).tolist())
    print('RESULT ' + json.dumps(r), flush=True)

if os.environ.get('BENCH_NO_ONESHOT'):
    sys.exit(0)
t = time.time(); sel, _ = select(Xtr, tmpl, search='auto', engine='gpu'); t_sel = time.time() - t
t = time.time(); m = BLMLQuick(engine='gpu', dtype=DT).fit(Xtr, sel); t_fit = time.time() - t
r = dict(method='one-shot', N=N, total_s=round(t_sel + t_fit, 1), select_s=round(t_sel, 1), fit_s=round(t_fit, 1),
         sweeps=int(m.n_sweeps_), bins=int(m.n_bins_), ll=tll(m), fc=np.round(bt.modes_fc(sel)[0], 3).tolist())
print('RESULT ' + json.dumps(r), flush=True)
