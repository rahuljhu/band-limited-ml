"""p(price | hour) on CPU: BLML (auto cutoffs, joint/marginal ratio), MDN, TabPFN; day split.
LL, LL by price range, region-mass calibration per hour, curves for a figure."""
import os, json, time, numpy as np, caiso_data, p2_models as M, blml_trivial as bt
from blml_trivial import BLMLQuick, make_modes, modes_fc
from selector import select
bt.DIRICHLET_LEGACY = False
D = os.environ['BLML_DATA_DIR']; cols = caiso_data.load(D); ts = caiso_data.timestamps(D)
y, hr = cols['price'], cols['hour']; tr, te = caiso_data.split_days(ts)
hours = (ts[tr] - ts[tr].min()).astype('timedelta64[h]').astype(np.int64); unit = hours // 24
vu = np.random.default_rng(0).choice(np.unique(unit), int(round(0.2 * len(np.unique(unit)))), replace=False)
iv = np.isin(unit, vu); va, tr2 = tr[iv], tr[~iv]
OUT = 'results/p2cpu'; res = {}
# BLML: same selector (dep = price), then joint and marginal fits with the same cutoffs on train + val
kinds, per = ['sinc', 'dirichlet'], [None, 24.0]
X = lambda i: np.column_stack([y[i], hr[i]])
t0 = time.time()
sel, _ = select(X(tr2), make_modes(np.nan, kinds, per, dep=0), search='auto', seed=0, engine='cpu', X_val=X(va))
fc = [float(f) for f in modes_fc(sel)[0]]
Xf = np.concatenate([X(tr2), X(va)])
J = BLMLQuick(seed=0); J.fit(Xf, make_modes(fc, kinds, per))
Mg = BLMLQuick(seed=0); Mg.fit(Xf[:, 1:], make_modes(fc[1:], kinds[1:], per[1:]))
blml_pdf = lambda Yq, Hq: np.asarray(J.predict(np.column_stack([Yq, Hq]))).ravel() / np.maximum(np.asarray(Mg.predict(np.asarray(Hq)[:, None])).ravel(), 1e-300)
lp = {'BLML': np.log(np.maximum(blml_pdf(y[te], hr[te]), 1e-300))}
res['BLML'] = dict(fc=fc, s=time.time() - t0)
cov = lambda i: {'hour': hr[i]}
mods = {}
for name, f in [('MDN', M.MDN), ('TabPFN', M.TabPFN)]:
    t0 = time.time(); mods[name] = f().fit(cov(tr2), y[tr2], cov(va), y[va])
    lp[name] = mods[name].logpdf(cov(te), y[te]); res[name] = dict(s=time.time() - t0)
for k in lp:
    res[k]['ll'] = float(np.mean(lp[k]))
print('test LL', {k: round(v['ll'], 4) for k, v in res.items()}, '| BLML fc', fc, flush=True)
yt, ht = y[te], hr[te]; N = len(yt)
bins = [(-np.inf, 0), (0, 25), (25, 60), (60, 100), (100, 150), (150, np.inf)]
print(f"{'range':>12s} {'n':>5s} | BLML-MDN  BLML-TabPFN")
res['by_range'] = []
for lo, hi in bins:
    s = (yt >= lo) & (yt < hi)
    cb, cm, ct = [lp[k][s].sum() / N for k in ('BLML', 'MDN', 'TabPFN')]
    res['by_range'].append(dict(lo=lo, hi=hi, n=int(s.sum()), blml=cb, mdn=cm, tabpfn=ct))
    print(f"{f'[{lo},{hi})':>12s} {s.sum():5d} | {cb-cm:+.4f}  {cb-ct:+.4f}", flush=True)
# densities on a grid at the centre of each hour; region masses vs test fractions per hour
g = np.linspace(-80, 1150, 6000); H = np.arange(24) + 0.5
dens = {'BLML': np.stack([blml_pdf(g, np.full(len(g), h)) for h in H])}
for name in mods:
    dens[name] = mods[name].grid_pdf({'hour': H}, g)
regs = [(-80, 0), (0, 25), (25, 60), (60, 100), (100, 150), (150, 300), (300, 1150)]
tv = {k: [] for k in dens}; mass = {k: [] for k in dens}
for i, h in enumerate(range(24)):
    s = (ht >= h) & (ht < h + 1)
    emp = np.array([np.mean((yt[s] >= lo) & (yt[s] < hi)) for lo, hi in regs])
    for k, d in dens.items():
        m = np.array([np.trapezoid(d[i][(g >= lo) & (g <= hi)], g[(g >= lo) & (g <= hi)]) for lo, hi in regs])
        tv[k].append(0.5 * np.abs(m / m.sum() - emp).sum()); mass[k].append(float(np.trapezoid(d[i], g)))
res['tv_mean'] = {k: float(np.mean(v)) for k, v in tv.items()}
res['grid_mass_range'] = {k: [float(min(v)), float(max(v))] for k, v in mass.items()}
print('mean TV over hours', {k: round(v, 4) for k, v in res['tv_mean'].items()}, '| grid mass', res['grid_mass_range'], flush=True)
json.dump(res, open(f'{OUT}/cond_hour.json', 'w'), indent=1, default=float)
np.savez(f'{OUT}/cond_hour_curves.npz', g=g, yt=yt, ht=ht, **dens, **{'lp_' + k: v for k, v in lp.items()})
