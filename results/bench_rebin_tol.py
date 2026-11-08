"""2-D sinc^2 benchmark: lattice rebuild at h-target factor 2^(1/4) (rebin_tol=1) vs 2 (rebin_tol=15), rebin_sweeps 0/2."""
import os, sys, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np
warnings.filterwarnings('ignore')
import blml_trivial as bt
from blml_rolling import BLMLQuickStreaming
W = np.array([0.5, 0.3, 0.2]); A = np.array([[1.0, 1.5], [1.5, 1.0], [1.2, 1.2]]); MU = np.array([[0.0, 0.0], [2.5, 1.0], [-1.5, 2.0]])
tmpl = bt.make_modes(np.nan, ['sinc', 'sinc'])
def data(N, seed):
    rng = np.random.default_rng(seed)
    def s2(n):
        out = np.empty(0)
        while len(out) < n:
            u = np.tan(np.pi * (rng.random(2 * n) - 0.5)) / np.pi
            out = np.r_[out, u[rng.random(2 * n) < np.sinc(u) ** 2 * (1 + (np.pi * u) ** 2) / 2]]
        return out[:n]
    def draw(n):
        k = rng.choice(3, n, p=W); return np.c_[s2(n), s2(n)] / A[k] + MU[k]
    return draw(N), draw(2000)
for N, seeds in [(20000, [0, 1, 2]), (40000, [0])]:
    for seed in seeds:
        Xtr, Xte = data(N, seed)
        for tol in [1.0, 15.0]:
            for sw in [0, 2]:
                t = time.time()
                s = BLMLQuickStreaming(search='auto', burn_in=5000, dtype=np.float32, rebin_tol=tol, rebin_sweeps=sw, seed=seed).fit(Xtr, tmpl, epochs=2, batch=2000)
                ll = float(np.mean(np.log(np.maximum(s.predict(Xte), 1e-300))))
                print('RESULT ' + json.dumps(dict(N=N, seed=seed, rebin_tol=tol, sweeps=sw, s=round(time.time() - t, 1), ll=round(ll, 4),
                      rebins=sum(1 for h in s.history_ if h.get('rebinned')), bins=len(s.lj.e) if hasattr(s, 'lj') else None,
                      fc=np.round(bt.modes_fc(s.modes_)[0], 3).tolist())), flush=True)
