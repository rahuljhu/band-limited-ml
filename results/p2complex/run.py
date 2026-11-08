"""Paper-2 day-split experiment with complex / rank-K BLML (CPU).
Same data, split, validation days and conditional form as p2_experiments.py daysplit / BLML-auto:
p(y | x) = f_joint(y, x) / f_marg(x), two BLMLQuick fits (ratio), y untransformed.
Cut-offs: start from the real BLML-auto selection (results/p2day7), then pick a shared multiplier
alpha on fc by validation conditional LL (fit on train days, score on validation days), refit on
train + validation, score on test days. Usage: run.py <covset hl|hll> <variant> ; variant in
real, complex, real2 (= complex), complex2, complex3."""
import os, sys, json, time
os.environ.setdefault('BLML_OUT_DIR', 'results/p2complex/_tmp')
sys.path.insert(0, '.')
import numpy as np
import p2_experiments as E, caiso_data, blml_trivial as bt
covset, variant = sys.argv[1], sys.argv[2]
CFG = {'real': dict(), 'complex': dict(complex_g=True), 'real2': dict(rank=2),
       'complex2': dict(complex_g=True, rank=2), 'complex3': dict(complex_g=True, rank=3)}[variant]
ALPHAS = [float(a) for a in os.environ.get('P2C_ALPHAS', '0.7,1.0,1.4').split(',')]
cols = caiso_data.load(E.DATA); ts = caiso_data.timestamps(E.DATA)
tr, te = caiso_data.split_days(ts)
covs = E.COVSETS[covset]
y = cols['price']
ok = np.all([np.isfinite(cols[c]) for c in covs], axis=0)       # lags: first week has none
tr, te = tr[ok[tr]], te[ok[te]]
# validation days exactly as in run_one (daysplit): random 20% of the training days, seed 0
hours = (ts[tr] - ts[tr].min()).astype('timedelta64[h]').astype(np.int64); unit = hours // 24
va_units = np.random.default_rng(0).choice(np.unique(unit), max(1, int(round(0.2 * len(np.unique(unit))))), replace=False)
is_va = np.isin(unit, va_units); va, tr2 = tr[is_va], tr[~is_va]
periods = [None] + [caiso_data.PERIODS.get(c) for c in covs]
kinds = ['sinc'] + ['dirichlet' if p else 'sinc' for p in periods[1:]]
X = lambda idx: np.stack([y[idx]] + [cols[c][idx] for c in covs], axis=1)
fc0 = np.array(json.load(open(f'results/p2day7/p2/daysplit_{covset}_BLML-auto.json'))['fc'])
def fit_cond(Xfit, fc):
    t0 = time.time()
    K, cg = CFG.get('rank', 1), CFG.get('complex_g', False)
    mj = bt.BLMLQuick(dtype=np.float64, seed=0).fit(Xfit, bt.make_modes(list(fc), kinds, periods, rank=K), complex_g=cg)
    mm = bt.BLMLQuick(dtype=np.float64, seed=0).fit(Xfit[:, 1:], bt.make_modes(list(fc[1:]), kinds[1:], periods[1:], rank=K), complex_g=cg)
    return mj, mm, time.time() - t0
def cll(mj, mm, Xq):
    return np.log(np.maximum(mj.predict(Xq), 1e-300)) - np.log(np.maximum(mm.predict(Xq[:, 1:]), 1e-300))
rec = dict(covset=covset, variant=variant, fc0=fc0.tolist(), n_tr2=len(tr2), n_va=len(va), n_te=len(te), val={})
for a in ALPHAS:
    mj, mm, s = fit_cond(X(tr2), fc0 * a); v = float(np.mean(cll(mj, mm, X(va))))
    rec['val'][a] = dict(ll=v, s=s, conv=[bool(mj.converged_), bool(mm.converged_)])
    print(f'{covset} {variant} alpha {a}: val LL {v:.4f} ({s:.0f}s, conv {mj.converged_},{mm.converged_})', flush=True)
a = max(rec['val'], key=lambda k: rec['val'][k]['ll'])
mj, mm, s = fit_cond(X(np.r_[tr2, va]), fc0 * a)
lp = cll(mj, mm, X(te)); rec.update(alpha=a, fc=(fc0 * a).tolist(), test_ll=float(np.mean(lp)), final_s=s, per_obs=np.round(lp, 5).tolist())
print(f'{covset} {variant}: alpha {a} TEST LL {rec["test_ll"]:.4f}  (paper real BLML-auto: '
      f'{-json.load(open(f"results/p2day7/p2/daysplit_{covset}_BLML-auto.json"))["scores"]["nll"]:.4f})', flush=True)
json.dump(rec, open(f'results/p2complex/{covset}_{variant}.json', 'w'))
