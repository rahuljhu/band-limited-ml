"""Paper 2: day-ahead forecasting with daily refits (GPU).  ./colab_job.sh start p2daily p2_daily.py

For each test month 2024-06..11 and each test day D in it, every model is refitted on the data
before D -- all of it ('expanding') or the last 120 days ('sliding') -- and predicts the 24 hours
of D. Validation is always the last 28 days before D (forward-chained), fitting on the rest.
Every model is tuned on the first day of each month and refitted daily with those settings
(GMM: one EM run from the current mixture; MDN: continued training from the current weights;
XGBoostLSS: the month's number of rounds; TabPFN: same loaded model, new context). BLML (on z = price - lag24):
  * BLML-daily: full cutoff selection (stages 1-4) on the first day of the month; on later days
    stage 4 only (per-dimension sqrt(2) steps from yesterday's cutoffs, a move accepted only if
    its validation gain exceeds one standard error), every fit warm-started from the previous one.
Conditional densities are joint / marginal BLML fits (same cutoffs), O(B) per point.
One JSON per (window, month, covariate set, model), in the format of p2_experiments.py.
Env: BLML_P2_MODELS, BLML_P2_COVSETS, BLML_P2_RUNS (windows: expanding,sliding).
"""
import os, sys, json, time, traceback, subprocess

OUT = os.environ.get('BLML_OUT_DIR', 'outputs_p2daily')
CODE = os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('BLML_DATA_DIR')
os.makedirs(os.path.join(OUT, 'p2'), exist_ok=True)
def pip(*pkgs):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *pkgs], check=False)
if subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
    pip('cupy-cuda13x', 'numba')
for mod, pkg in [('xgboostlss', 'xgboostlss'), ('tabpfn', 'tabpfn')]:   # MDN: torch, GMM: sklearn (preinstalled)
    if subprocess.run([sys.executable, '-c', f'import {mod}'], capture_output=True).returncode:
        pip(pkg)
sys.path.insert(0, CODE)
import numpy as np
import blml_trivial as bt
from blml_trivial import make_modes
import caiso_data
import p2_models as M
from p2_experiments import scores, scores_ll, FULL_SCORES, YGRID, TAUS, COVSETS
from blml_rolling import BLMLRolling
from blml_trivial import modes_fc

bt.DIRICHLET_LEGACY = False
WINDOWS = {'expanding': None, 'sliding': 120}
VAL_DAYS = 28
MONTHS = ['2024-06', '2024-07', '2024-08', '2024-09', '2024-10', '2024-11']

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class BLMLDaily:
    """BLML on z = y - lag24 given the covariates, with warm-started daily updates (blml_rolling.BLMLRolling)."""
    ZGRID = np.arange(-1300.0, 1300.5, 1.0)

    def __init__(self, refine=True, z_se=1.0):
        self.refine, self.z_se = refine, z_se
        self.r = self.fc = None

    def _X(self, cov, y):
        return np.column_stack([np.asarray(y) - cov['lag24']] + [cov[c] for c in self.covs])

    def fit(self, cov_tr, y_tr, cov_va, y_va):
        self.covs = list(cov_tr)
        periods = [None] + [M.PERIODS.get(c) for c in self.covs]
        kinds = ['sinc'] + ['dirichlet' if p else 'sinc' for p in periods[1:]]
        Xa, Xb = self._X(cov_tr, y_tr), self._X(cov_va, y_va)
        if self.r is None:                                  # month start: full selection (stages 1-4)
            self.r = BLMLRolling(engine='gpu', search='auto', refine=self.refine, z_se=self.z_se).fit(
                Xa, make_modes(np.nan, kinds, periods, dep=0), Xb)
            self.n_fits = 0
        else:                                               # stage 4 from yesterday's cutoffs, warm fits
            self.r.update(Xa, Xb); self.n_fits = self.r.history_[-1]['n_fits'] - 1   # validation fits, as before
        self.fc = modes_fc(self.r.modes_)[0]
        self.final = self.r.model_                          # dep=0 fit on train + validation
        return self

    def logpdf(self, cov, y):
        X = self._X(cov, y)     # conditional = joint / marginal of the covariates, O(B) per point
        return np.log(np.maximum(self.final.predict_ratio(X), 1e-300))

    def grid_pdf(self, cov, ygrid):
        Xrest = np.stack([np.asarray(cov[c], float) for c in self.covs], axis=1)
        m, G = len(Xrest), len(self.ZGRID)
        pts = np.column_stack([np.tile(self.ZGRID, m), np.repeat(Xrest, G, axis=0)])
        pz = self.final.predict(pts).reshape(m, G) / np.maximum(self.final.predict_marginal(Xrest), 1e-300)[:, None]
        return np.stack([np.interp(ygrid - l, self.ZGRID, row, left=0.0, right=0.0) for l, row in zip(cov['lag24'], pz)])


FACTORIES = {
    'BLML-daily': lambda: BLMLDaily(refine=True),
    'TabPFN': lambda: M.TabPFN(), 'MDN': lambda: M.MDN(), 'GMM': lambda: M.CondGMM(),
    'XGBoostLSS': lambda: M.XGBoostLSS(),
}

def main():
    cols = caiso_data.load(DATA); ts = caiso_data.timestamps(DATA)
    for c in ('lag24', 'lag168'):
        v = cols[c][np.isfinite(cols[c])]
        M.SCALES[c] = (float(np.median(v)), float(np.std(v)))
    y = cols['price']; day = ts.astype('datetime64[D]')
    models = [m for m in (os.environ.get('BLML_P2_MODELS') or ','.join(FACTORIES)).split(',')]
    covsets = (os.environ.get('BLML_P2_COVSETS') or 'hl,hll').split(',')
    windows = (os.environ.get('BLML_P2_RUNS') or 'expanding,sliding').split(',')
    for window in windows:
        for month in MONTHS:
            days = np.unique(day[ts.astype('datetime64[M]') == np.datetime64(month, 'M')])
            for cs in covsets:
                covs = COVSETS[cs]
                ok = np.all([np.isfinite(cols[c]) for c in covs], axis=0)
                for name in models:
                    nm = f'{window}{month}_{cs}_{name}'
                    p = os.path.join(OUT, 'p2', nm + '.json')
                    if os.path.exists(p):
                        continue
                    t0 = time.time(); m = None
                    lps, pdfs, Qs, te_all, extra = [], [], [], [], []
                    try:
                        for D in days:
                            lo = None if WINDOWS[window] is None else D - np.timedelta64(WINDOWS[window], 'D')
                            tr = np.where(ok & (day < D) & ((day >= lo) if lo is not None else True))[0]
                            te = np.where(ok & (day == D))[0]
                            va_mask = day[tr] >= D - np.timedelta64(VAL_DAYS, 'D')
                            va, tr2 = tr[va_mask], tr[~va_mask]
                            sub = lambda idx: {c: cols[c][idx] for c in covs}
                            # One object per month for every model: tuned on the first day of the
                            # month (fit), then refitted daily with those settings (update), warm-started
                            # where the model allows. BLML refines its cutoffs locally each day.
                            first = m is None
                            m = m or FACTORIES[name]()
                            if not FULL_SCORES:
                                if isinstance(m, M.QuantileModel):
                                    raise RuntimeError('quantile-only model: no likelihood (FULL_SCORES is off)')
                                (m.fit if first or isinstance(m, BLMLDaily) else m.update)(sub(tr2), y[tr2], sub(va), y[va])
                                lp, pdf, Q = m.logpdf(sub(te), y[te]), None, None
                            elif isinstance(m, M.QuantileModel):
                                m.fit(sub(tr2), y[tr2], sub(va), y[va], taus=TAUS)
                                lp, pdf, Q = None, None, m.quantiles(sub(te), TAUS)
                            else:
                                m.fit(sub(tr2), y[tr2], sub(va), y[va])
                                lp = m.logpdf(sub(te), y[te]); pdf = m.grid_pdf(sub(te), YGRID)
                                Q = M.cdf_quantiles(pdf, YGRID, TAUS)
                            lps.append(lp); pdfs.append(pdf); Qs.append(Q); te_all.append(te)
                            if isinstance(m, BLMLDaily):
                                extra.append(dict(day=str(D), fc=[float(v) for v in m.fc], n_fits=m.n_fits))
                        te = np.concatenate(te_all)
                        lp = None if lps[0] is None else np.concatenate(lps)
                        pdf = None if pdfs[0] is None else np.concatenate(pdfs)
                        rec = dict(run=f'{window}{month}', covset=cs, model=name, n_test=len(te),
                                   test_times=[str(t) for t in ts[te]], y=np.round(y[te], 5).tolist(),
                                   fit_predict_s=time.time() - t0)
                        rec['scores'], rec['per_obs'] = (scores(y[te], lp, pdf, np.concatenate(Qs)) if FULL_SCORES
                                                         else scores_ll(lp))
                        if extra:
                            rec['daily'] = extra
                        log(f"{nm}: " + ' '.join(f'{k}={v:.4f}' for k, v in rec['scores'].items()
                                                 if k in ('nll', 'crps', 'cover90')) + f" ({rec['fit_predict_s']:.0f}s)")
                    except Exception as e:
                        rec = dict(run=f'{window}{month}', covset=cs, model=name,
                                   error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc()[-3000:])
                        log(f'{nm}: FAILED {rec["error"]}')
                    with open(p + '.tmp', 'w') as f:
                        json.dump(rec, f, default=float)
                    os.replace(p + '.tmp', p)
    log('ALL DONE')

if __name__ == '__main__':
    main()
