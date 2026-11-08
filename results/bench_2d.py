"""2-D joint density, band-limited truth: BLMLQuick engine='cpu' vs engine='gpu', side by side.
Truth: mixture of 3 products a sinc^2(a (x - mu)) b sinc^2(b (y - nu)) (each band-limited to |nu| <= a, b).
n = 10000 train + 2000 held out. Per engine: select(search='auto') + final fit, and one fit at a fixed fc.
Usage: bench_2d.py [engines, e.g. cpu,gpu] [dtype float32|float64]"""
import os, sys, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..'); sys.path.insert(0, '.')
import numpy as np
warnings.filterwarnings('ignore')
import blml_trivial as bt
from selector import select

ENGINES = (sys.argv[1] if len(sys.argv) > 1 else 'cpu,gpu').split(',')
DT = np.dtype(sys.argv[2] if len(sys.argv) > 2 else 'float32')
rng = np.random.default_rng(0)
W = np.array([0.5, 0.3, 0.2]); A = np.array([[1.0, 1.5], [1.5, 1.0], [1.2, 1.2]])
MU = np.array([[0.0, 0.0], [2.5, 1.0], [-1.5, 2.0]])


def sinc2_sample(n):
    """u ~ sinc^2(u) = (sin(pi u)/(pi u))^2 by rejection from a Cauchy(0, 1/pi): sinc^2 <= 2 / (1 + pi^2 u^2)."""
    out = np.empty(0)
    while len(out) < n:
        u = np.tan(np.pi * (rng.random(2 * n) - 0.5)) / np.pi
        acc = rng.random(2 * n) < np.sinc(u) ** 2 * (1 + (np.pi * u) ** 2) / 2
        out = np.r_[out, u[acc]]
    return out[:n]


def draw(n):
    k = rng.choice(3, n, p=W)
    return np.c_[sinc2_sample(n), sinc2_sample(n)] / A[k] + MU[k]


def true_pdf(X):
    return sum(W[k] * np.prod(A[k] * np.sinc(A[k] * (X - MU[k])) ** 2, 1) for k in range(3))


Xtr, Xte = draw(10000), draw(2000)
ll_true = float(np.mean(np.log(true_pdf(Xte))))
print(f'true held-out ll {ll_true:.4f} (n_train 10000, n_test 2000, dtype {DT.name})', flush=True)
tmpl = bt.make_modes(np.nan, ['sinc', 'sinc'])
FIX = bt.make_modes([1.5, 1.5], ['sinc', 'sinc'])          # fixed fc: the largest true band
rows = []
for eng in ([] if os.environ.get('BENCH_STREAM_ONLY') else ENGINES):
    if eng == 'gpu':
        import functools; from blml_trivial import BLMLQuick; Est = functools.partial(BLMLQuick, engine='gpu')
        t = time.time(); Est(dtype=DT).fit(Xtr[:200], FIX); t_compile = time.time() - t
    else:
        Est = bt.BLMLQuick; t = time.time(); Est(dtype=DT).fit(Xtr[:200], FIX); t_compile = time.time() - t
    t = time.time(); m = Est(dtype=DT).fit(Xtr, FIX); t_fix = time.time() - t
    ll_fix = float(np.mean(np.log(np.maximum(m.predict(Xte), 1e-300))))
    t = time.time(); sel, _ = select(Xtr, tmpl, search='auto', engine=eng); t_sel = time.time() - t
    t = time.time(); mf = Est(dtype=DT).fit(Xtr, sel); t_fit = time.time() - t
    ll_sel = float(np.mean(np.log(np.maximum(mf.predict(Xte), 1e-300))))
    r = dict(engine=eng, warmup_s=round(t_compile, 2), fixed_fc_fit_s=round(t_fix, 2), fixed_fc_ll=round(ll_fix, 4),
             fixed_fc_bins=int(m.n_bins_), fixed_fc_sweeps=int(m.n_sweeps_), select_s=round(t_sel, 2),
             final_fit_s=round(t_fit, 2), selected_fc=np.round(bt.modes_fc(sel)[0], 3).tolist(), selected_ll=round(ll_sel, 4),
             selected_bins=int(mf.n_bins_))
    rows.append(r); print('RESULT ' + json.dumps(r), flush=True)

# ---- streaming: BLMLQuickStreaming (burn-in select(search='auto') on the first batch(es), Adam fc steps,
# touched-bin updates, 2 epochs); same data in batches, test ll on the same held-out points
from blml_rolling import BLMLQuickStreaming
for eng in ENGINES:
    if eng == 'gpu':
        from blml_trivial import BLMLQuick
        BLMLQuick(engine='gpu', dtype=DT).fit(Xtr[:200], FIX)          # CUDA compile, not timed
    for bs in (1000, 2000):
        t = time.time()
        s = BLMLQuickStreaming(engine=eng, search='auto', dtype=DT, fc_update=os.environ.get('STREAM_UPD', 'adam')).fit(
            Xtr, tmpl, epochs=int(os.environ.get('BLML_STREAM_EPOCHS', 2)), batch=bs)
        t_all = time.time() - t
        ll_s = float(np.mean(np.log(np.maximum(s.predict(Xte), 1e-300))))
        r = dict(engine=eng, stream_batch=bs, total_s=round(t_all, 2), burn_in_s=round(s.history_[0]['s'], 2),
                 ll=round(ll_s, 4), fc=np.round(bt.modes_fc(s.modes_)[0], 3).tolist(), fc_update=s.fc_update)
        print('STREAM ' + json.dumps(r), flush=True)
