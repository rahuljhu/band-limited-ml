"""Paper 2 experiments (GPU). Run on Colab via:  ./colab_job.sh start p2 p2_experiments.py

p(price | calendar) on CAISO for three covariate sets, evaluated on
  * rolling-origin folds: train on everything before month m, test on month m
    (2024-06 .. 2024-11) -- the primary, forecasting-style protocol;
  * the random 80/20 split used so far -- secondary (interpolation).
Every model is tuned on the same validation rows: 20% of 28-day training blocks
(rolling folds) or of training days (random split). One JSON per
(run, covariate set, model) with summary scores and per-observation arrays
(log density, CRPS, PIT, predictive mean) for significance tests and the
battery backtest, which is computed afterwards from these files.
"""
import os, sys, json, time, subprocess, traceback

OUT = os.environ.get('BLML_OUT_DIR', 'outputs_p2')
CODE = os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('BLML_DATA_DIR')
os.makedirs(os.path.join(OUT, 'p2'), exist_ok=True)

def pip(*pkgs):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *pkgs], check=False)

if __name__ == '__main__' and os.environ.get('BLML_ENGINE', 'gpu') == 'gpu':   # Colab: install GPU / baseline packages
    if subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
        pip('cupy-cuda13x', 'numba')
    for mod, pkg in [('xgboostlss', 'xgboostlss'), ('tabpfn', 'tabpfn'), ('zuko', 'zuko')]:   # MDN: torch, GMM: sklearn (preinstalled); NSF: zuko
        if subprocess.run([sys.executable, '-c', f'import {mod}'], capture_output=True).returncode:
            pip(pkg)
sys.path.insert(0, CODE)

import numpy as np
import blml_trivial as bt
import caiso_data
import p2_models as M

bt.DIRICHLET_LEGACY = False
TAUS = np.r_[0.01, 0.025, np.round(np.arange(0.05, 0.951, 0.05), 3), 0.975, 0.99]
YGRID = np.arange(-100.0, 1200.5, 1.0)
COVSETS = {'2d': ['hour'], '3d': ['hour', 'doy'], '4d': ['hour', 'dow', 'moy'],
           # day-ahead forecasting sets: yesterday's and last week's price at the same hour
           'hl': ['hour', 'lag24'], 'hdl': ['hour', 'doy', 'lag24'], 'hdll': ['hour', 'doy', 'lag24', 'lag168'],
           'hll': ['hour', 'lag24', 'lag168']}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def path(name):
    return os.path.join(OUT, 'p2', name + '.json')

def save(name, obj):
    tmp = path(name) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, default=float)
    os.replace(tmp, path(name))

def pinball(Q, y, taus):
    u = y[:, None] - Q
    return np.mean(np.maximum(taus * u, (taus - 1) * u), axis=0)

# Log-likelihood only (fast iteration): no predictive grid, no CRPS or quantiles, and
# quantile-only models are skipped. Set FULL_SCORES = True for the full set of proper scores.
FULL_SCORES = False

def scores_ll(lp):
    lp = np.asarray(lp, float)
    return {'nll': float(-np.mean(lp))}, {'logpdf': np.round(lp, 4).tolist()}

def scores(y, lp, pdf, Q):
    """Proper scores; NLL only for density models."""
    out, per = {}, {}
    if lp is not None:
        out['nll'] = float(-np.mean(lp)); per['logpdf'] = lp
    if pdf is not None:
        dy = np.gradient(YGRID)
        F = np.cumsum(pdf * dy[None, :], axis=1)
        out['grid_mass_mean'] = float(F[:, -1].mean())
        F = F / np.maximum(F[:, -1:], 1e-300)
        crps = ((F - (YGRID[None, :] >= y[:, None])) ** 2 * dy[None, :]).sum(1)
        out['crps'] = float(crps.mean()); per['crps'] = crps
        per['pit'] = np.array([np.interp(v, YGRID, Fi) for v, Fi in zip(y, F)])
        per['mean'] = (pdf * YGRID[None, :] * dy[None, :]).sum(1) / np.maximum((pdf * dy[None, :]).sum(1), 1e-300)
    pb = pinball(Q, y, TAUS)
    crps_q = 2 * np.trapezoid(pb, TAUS) / (TAUS[-1] - TAUS[0])     # quantile-grid CRPS (all models)
    u = y[:, None] - Q
    per['crps_q'] = 2 * np.trapezoid(np.maximum(TAUS * u, (TAUS - 1) * u), TAUS, axis=1) / (TAUS[-1] - TAUS[0])
    i05, i50, i95 = [int(np.argmin(np.abs(TAUS - t))) for t in (0.05, 0.5, 0.95)]
    out.update(crps_q=float(crps_q), pinball05=float(pb[i05]), pinball50=float(pb[i50]),
               pinball95=float(pb[i95]), cover90=float(np.mean((y >= Q[:, i05]) & (y <= Q[:, i95]))),
               mae_median=float(np.mean(np.abs(y - Q[:, i50]))))
    if 'mean' not in per:
        per['mean'] = Q.mean(1)
    # Median and the full quantile grid: BLML densities have 1/y^2 tails, so their
    # mean does not exist; point decisions (battery backtest) use the median.
    per['median'] = Q[:, i50]
    per['q'] = Q
    return out, {k: np.round(np.asarray(v, float), 4).tolist() for k, v in per.items()}

def make_models(covset):
    # BLML and the assumption-light baselines (conditional KDE, k-NN); structured references: TabPFN
    # (pretrained, a different model class), MDN, GMM and XGBoostLSS (trained on these data only).
    models = [
        ('BLML-auto', lambda: M.BLML(mode='auto')),
        ('BLML-stream', lambda: M.BLMLStream()),
        # rank 2 from the rank-1 streaming fit's cut-offs (price, hour, lag24): one mode sharper, one broader in price
        ('BLML-stream-rank2', lambda: M.BLMLStream(rank=2, fc_init=[[0.13, 0.119, 0.091], [0.04, 0.119, 0.091]])),
        ('BLML-stream-rank1', lambda: M.BLMLStream(rank=1, fc_init=[[0.0856, 0.119, 0.0912]])),
        # the same, the hour cut-off held at 0.119 (one daily harmonic) in every mode
        ('BLML-stream-rank2-hourfixed', lambda: M.BLMLStream(rank=2, fc_init=[[0.13, 0.119, 0.091], [0.04, 0.119, 0.091]], freeze=('hour',))),
        # rank 3, mode k with k-1 fixed daily harmonics (hour cut-offs 0.0417, 0.125, 0.2083); price and lag24 learned
        ('BLML-stream-rank3-harm', lambda: M.BLMLStream(rank=3, fc_init=[[0.0856, 0.0417, 0.0912], [0.0856, 0.125, 0.0912],
                                                                         [0.0856, 0.2083, 0.0912]], freeze=('hour',))),
        ('BLML-stream-rank1-hourfixed', lambda: M.BLMLStream(rank=1, fc_init=[[0.0856, 0.119, 0.0912]], freeze=('hour',))),
        ('BLML-auto-complex', lambda: M.BLML(mode='auto', complex_g=True)),
        ('BLML-auto-complex2', lambda: M.BLML(mode='auto', complex_g=True, rank=2)),
        ('TabPFN', lambda: M.TabPFN()),
        ('MDN', lambda: M.MDN()),
        ('NSF', lambda: M.NSF()),
        ('GMM', lambda: M.CondGMM()),
        ('NGBoost-Normal', lambda: M.NGBoost('Normal')),
        ('XGBoostLSS', lambda: M.XGBoostLSS()),
        ('CondKDE', lambda: M.CondKDE()),
        ('CondKNN', lambda: M.CondKNN()),
    ]
    return models

def run_one(run, covset, name, factory, cols, tr, te, ts):
    nm = f'{run}_{covset}_{name}'
    if os.path.exists(path(nm)):
        return
    covs = COVSETS[covset]
    y = cols['price']
    ok = np.all([np.isfinite(cols[c]) for c in covs], axis=0)     # lags: first week has none
    tr, te = tr[ok[tr]], te[ok[te]]
    # Validation mimics the test, and is shared by every model. Rolling folds test
    # a whole unseen month, so validation holds out a random 20% of contiguous
    # 28-day blocks of the training period; neighboring days are correlated, so
    # holding out single days (or rows) still rewards memorizing the calendar.
    hours = (ts[tr] - ts[tr].min()).astype('timedelta64[h]').astype(np.int64)
    # Random-split run (stationarity assumed): validation is a random 20% of the
    # training points, the same protocol as the test split. Rolling folds:
    # contiguous 28-day blocks.
    # Day-split run: validation holds out a random 20% of the training *days*, like the test.
    # Forward-chained rolling folds ('fwd2024-06', ...): validation is the last 28 days before
    # the test month, so tuning sees the same task as the test -- forecasting a new month.
    if run.startswith('fwd'):
        unit = (hours >= hours.max() - 24 * 28 + 1).astype(int)
        va_units = np.array([1])
    else:
        unit = None
    unit = (unit if unit is not None else np.arange(len(tr)) if run == 'random' else hours // 24 if run == 'daysplit'
            else hours // (24 * 28))
    units = np.unique(unit)
    if not run.startswith('fwd'):
        va_units = np.random.default_rng(0).choice(units, max(1, int(round(0.2 * len(units)))), replace=False)
    is_va = np.isin(unit, va_units)
    va, tr2 = tr[is_va], tr[~is_va]
    sub = lambda idx: {c: cols[c][idx] for c in covs}
    extra = lambda idx: {'hour': cols['hour'][idx], 'moy': cols['moy'][idx]}
    rec = dict(run=run, covset=covset, model=name, n_train=len(tr), n_test=len(te),
               test_times=[str(t) for t in ts[te]], y=np.round(y[te], 5).tolist())
    try:
        m = factory()
        if hasattr(m, 'epoch_eval') and os.environ.get('BLML_EPOCH_TEST'):    # test LL after every streaming epoch
            m.epoch_eval = (sub(te), y[te])
        if not FULL_SCORES and isinstance(m, M.QuantileModel):
            return                     # no likelihood
        if isinstance(m, M.Climatology) and covset == '2d':
            m.use_month = False        # hour-only set: no month information
        t0 = time.time()
        if not FULL_SCORES:            # training (tuning + final fit) and inference timed separately
            if isinstance(m, M.Climatology):
                m.fit(sub(tr2), y[tr2], sub(va), y[va], extra(tr2), extra(va)); rec['fit_s'] = time.time() - t0
                lp = m.logpdf(sub(te), y[te], extra(te))
            else:
                m.fit(sub(tr2), y[tr2], sub(va), y[va]); rec['fit_s'] = time.time() - t0
                lp = m.logpdf(sub(te), y[te])
            rec['predict_s'] = time.time() - t0 - rec['fit_s']
            if os.environ.get('BLML_POINT') and hasattr(m, 'median') and not isinstance(m, M.Climatology):   # point = median of p(y | x): optimal for |error|
                t1 = time.time(); yhat = np.asarray(m.median(sub(te)), float); rec['point_s'] = time.time() - t1
                err = np.abs(yhat - y[te])
                rec['point'] = np.round(yhat, 3).tolist()
                rec['mae'], rec['medae'] = float(err.mean()), float(np.median(err))
                rec.update({f'hit{t}': float(np.mean(err <= t)) for t in (2, 5, 10)})
                if hasattr(m, 'ess_frac') or hasattr(getattr(m, 'base', None), 'ess_frac'):
                    rec['ess_frac'] = float(getattr(m, 'ess_frac', getattr(getattr(m, 'base', None), 'ess_frac', np.nan)))
                el = np.abs(cols['lag24'][te] - y[te]); rec['mae_lag24'], rec['medae_lag24'] = float(np.nanmean(el)), float(np.nanmedian(el))
        elif isinstance(m, M.Climatology):
            m.fit(sub(tr2), y[tr2], sub(va), y[va], extra(tr2), extra(va))
            lp = m.logpdf(sub(te), y[te], extra(te)); pdf = m.grid_pdf(sub(te), YGRID, extra(te))
            Q = M.cdf_quantiles(pdf, YGRID, TAUS)
        elif isinstance(m, M.QuantileModel):
            m.fit(sub(tr2), y[tr2], sub(va), y[va], taus=TAUS)
            lp, pdf, Q = None, None, m.quantiles(sub(te), TAUS)
        else:
            m.fit(sub(tr2), y[tr2], sub(va), y[va])
            lp = m.logpdf(sub(te), y[te]); pdf = m.grid_pdf(sub(te), YGRID)
            Q = M.cdf_quantiles(pdf, YGRID, TAUS)
        rec['fit_predict_s'] = time.time() - t0
        rec['scores'], rec['per_obs'] = scores(y[te], lp, pdf, Q) if FULL_SCORES else scores_ll(lp)
        for attr in ('fc', 'fc_modes', 'epoch_ll', 'fc_epochs', 'K', 'kappa', 'hy', 'k', 'select_s', 'final_fit_s', 'tabpfn_version'):
            if hasattr(m, attr):
                rec[attr] = getattr(m, attr)
        log(f"{nm}: " + ' '.join(f'{k}={v:.4f}' for k, v in rec['scores'].items()
                                 if k in ('nll', 'crps', 'crps_q', 'cover90'))
            + (f" MAE={rec['mae']:.2f} MedAE={rec['medae']:.2f} hit5={rec['hit5']:.3f}" if 'mae' in rec else '') + f" ({rec['fit_predict_s']:.0f}s)")
    except Exception as e:
        rec['error'] = f'{type(e).__name__}: {e}'
        rec['traceback'] = traceback.format_exc()[-3000:]
        log(f'{nm}: FAILED {rec["error"]}')
    save(nm, rec)

def main():
    cols = caiso_data.load(DATA)
    ts = caiso_data.timestamps(DATA)
    runs = [(name, tr, te) for name, tr, te in caiso_data.rolling_folds(DATA)]
    tr, te = caiso_data.split(len(ts))
    runs.append(('random', tr, te))
    tr, te = caiso_data.split_days(ts)
    runs.append(('daysplit', tr, te))
    for c in ('lag24', 'lag168'):                  # baselines standardize non-periodic covariates
        v = cols[c][np.isfinite(cols[c])]
        M.SCALES[c] = (float(np.median(v)), float(np.std(v)))
    if os.environ.get('BLML_P2_COVSETS'):
        for k in [k for k in COVSETS if k not in os.environ['BLML_P2_COVSETS'].split(',')]:
            del COVSETS[k]
    want = os.environ.get('BLML_P2_RUNS')          # e.g. 'daysplit' or 'rolling,daysplit'
    if want:
        keep = want.split(',')
        if 'rollfwd' in keep:
            runs += [('fwd' + name, tr, te) for name, tr, te in caiso_data.rolling_folds(DATA)]
        runs = [r for r in runs if r[0] in keep or ('rolling' in keep and r[0][:2] == '20')
                or ('rollfwd' in keep and r[0][:3] == 'fwd')]
    # BLML and the fast baselines first for every run; the slow ones afterwards.
    models = make_models(None)
    # Default comparison: BLML against TabPFN, MDN, GMM and XGBoostLSS; BLML_P2_MODELS selects a subset.
    only = os.environ.get('BLML_P2_MODELS') or 'BLML-auto,TabPFN,MDN,GMM,XGBoostLSS,CondKDE,CondKNN'
    models = [m for m in models if m[0] in only.split(',')]
    fast = [m for m in models if m[0] not in ('NGBoost-Normal', 'NGBoost-Laplace', 'TabPFN')]
    slow = [m for m in models if m[0] in ('NGBoost-Normal', 'NGBoost-Laplace', 'TabPFN')]
    for group in (fast, slow):
        for run, tr, te in runs:
            for covset in COVSETS:
                for name, factory in group:
                    run_one(run, covset, name, factory, cols, tr, te, ts)
    log('ALL DONE')


if __name__ == '__main__':
    main()
