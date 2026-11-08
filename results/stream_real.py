"""Paper 2 (streaming BLML), real data: particles in arrival order. BLMLQuickStreaming (touched-bin updates, analytic
Adam, burn-in BLML_BURN (5000) with the selector's 'auto' search, batches of BLML_BATCH (2000), BLML_EPOCHS passes (2;
BLML_TOL: until the fit and the cut-off are stationary), float32, CPU -- the paper's CPU protocol) against one-shot
select(search='auto') + fit on the same points. Data: detected-particle positions (row, column) from
results/dslit_export.py (~/data_repo/dslit_gpu): single-mode photons (Aspden et al. 2016), multimode photons, electrons
(Bach et al. 2013). Test: the same random 20% as paper 3 (rng 0); training points in arrival order.
Usage: python results/stream_real.py <set> [<set> ...] -> results/stream_real/<set>.json"""
import os, sys, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np
warnings.filterwarnings('ignore')
import blml_trivial as bt
from selector import select
from blml_rolling import BLMLQuickStreaming
D = os.path.expanduser('~/data_repo/dslit_gpu/'); OUT = 'results/stream_real'; os.makedirs(OUT, exist_ok=True)
BURN, BATCH = int(os.environ.get('BLML_BURN', 5000)), int(os.environ.get('BLML_BATCH', 2000))   # the paper's CPU protocol
EPOCHS = int(os.environ.get('BLML_EPOCHS', 2)); TOL = None if os.environ.get('BLML_TOL') is None else float(os.environ['BLML_TOL'])
TAG = os.environ.get('BLML_TAG', '')
tmpl = bt.make_modes(np.nan, ['sinc', 'sinc'])
for name in sys.argv[1:] or ['electron', 'photon']:
    pts = np.load(D + f'dslit_{name}_points.npy'); X = pts[:, 1:3].astype(float)
    te = np.random.default_rng(0).random(len(pts)) >= 0.80
    Xtr, Xte = X[~te], X[te]                                     # arrival order kept (the points are time-ordered)
    tll = lambda m: float(np.mean(np.log(np.maximum(m.predict(Xte), 1e-300))))
    t = time.time()
    s = BLMLQuickStreaming(search='auto', burn_in=BURN, dtype=np.float32).fit(Xtr, tmpl, epochs=EPOCHS, batch=BATCH, tol=TOL)
    t_s = time.time() - t
    t = time.time(); sel, _ = select(Xtr, tmpl, search='auto'); m = bt.BLMLQuick(dtype=np.float32).fit(Xtr, sel); t_o = time.time() - t
    ls, lo = np.log(np.maximum(s.predict(Xte), 1e-300)), np.log(np.maximum(m.predict(Xte), 1e-300))
    d = ls - lo
    r = dict(set=name, n_train=int(len(Xtr)), n_test=int(len(Xte)), stream_s=t_s, oneshot_s=t_o, speedup=t_o / t_s,
             burn_in_s=float(s.history_[0].get('s', np.nan)), epochs=s.epochs_, converged=s.converged_, stream_ll=float(ls.mean()), oneshot_ll=float(lo.mean()),
             diff=float(d.mean()), diff_se=float(d.std(ddof=1) / np.sqrt(len(d))),
             stream_fc=bt.modes_fc(s.modes_)[0].tolist(), oneshot_fc=bt.modes_fc(sel)[0].tolist(),
             fc_trace=[h['fc'][0] if isinstance(h.get('fc'), list) else None for h in s.history_ if 'fc' in h][:2000])
    json.dump(r, open(f'{OUT}/{name}{TAG}.json', 'w'), indent=1)
    print('RESULT', json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != 'fc_trace'}), flush=True)
