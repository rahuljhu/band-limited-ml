"""XGBoostLSS on GPU, day split, hour + lag24: Gaussian mixtures (M=2, 4) and a single Gaussian.
Target standardized for training; densities mapped back to $ with the Jacobian 1/sd.
./colab_job.sh start p2xgblss p2_xgblss.py T4"""
import os, sys, json, time, subprocess, traceback
OUT = os.environ.get('BLML_OUT_DIR', 'outputs_xgblss'); CODE = os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT, exist_ok=True)
if subprocess.run([sys.executable, '-c', 'import xgboostlss'], capture_output=True).returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'xgboostlss'], check=False)
sys.path.insert(0, CODE)
import numpy as np, xgboost as xgb, caiso_data
import p2_models as M
from xgboostlss.model import XGBoostLSS
from xgboostlss.distributions.Gaussian import Gaussian
from xgboostlss.distributions.Mixture import Mixture
D = os.environ.get('BLML_DATA_DIR')
cols = caiso_data.load(D); ts = caiso_data.timestamps(D)
tr, te = caiso_data.split_days(ts); ok = np.isfinite(cols['lag24']); tr, te = tr[ok[tr]], te[ok[te]]
y = cols['price']
v = cols['lag24'][ok]; M.SCALES['lag24'] = (float(np.median(v)), float(np.std(v)))
F = lambda i: M.features({'hour': cols['hour'][i], 'lag24': cols['lag24'][i]})
day = ts[tr].astype('datetime64[D]'); days = np.unique(day)
isva = np.isin(day, np.random.default_rng(0).choice(days, int(0.2 * len(days)), replace=False))
tr2, va = tr[~isva], tr[isva]
mu, sd = float(np.median(y[tr2])), float(np.std(y[tr2]))
z = lambda i: (y[i] - mu) / sd
dtr, dva, dte = xgb.DMatrix(F(tr2), label=z(tr2)), xgb.DMatrix(F(va), label=z(va)), xgb.DMatrix(F(te))
print(f'train {len(tr2)} val {len(va)} test {len(te)}; mu {mu:.2f} sd {sd:.2f}', flush=True)

def mixture_ll(P, yt):
    c = list(P.columns); get = lambda pre: np.stack([P[k].values for k in c if k.startswith(pre)], 1)
    loc, scale = get('loc'), get('scale')
    w = get('mix_prob') if any(k.startswith('mix_prob') for k in c) else np.ones_like(loc)
    zt = ((yt - mu) / sd)[:, None]
    dens = (w * np.exp(-0.5 * ((zt - loc) / scale) ** 2) / (scale * np.sqrt(2 * np.pi))).sum(1) / sd
    return np.log(np.maximum(dens, 1e-300)), c

res = {}
for name, dist in [('Gaussian', Gaussian()), ('Mix2', Mixture(Gaussian(), M=2)), ('Mix4', Mixture(Gaussian(), M=4))]:
    try:
        model = XGBoostLSS(dist)
        params = dict(eta=0.03, max_depth=4, device='cuda', tree_method='hist')
        hist = {}
        t0 = time.time()
        model.train(params, dtr, num_boost_round=2000, evals=[(dtr, 'train'), (dva, 'val')],
                    early_stopping_rounds=50, verbose_eval=False, evals_result=hist)
        t_fit = time.time() - t0
        t0 = time.time(); P = model.predict(dte, pred_type='parameters'); t_pred = time.time() - t0
        lp, c = mixture_ll(P, y[te])
        vh = next(iter(hist.get('val', {}).values()), [])
        res[name] = dict(ll=float(lp.mean()), t_fit=t_fit, t_pred=t_pred, columns=c,
                         best_iteration=int(getattr(model.booster, 'best_iteration', -1)),
                         val_curve=[float(vh[i]) for i in range(0, len(vh), max(1, len(vh) // 10))])
    except Exception as e:
        res[name] = dict(error=f'{type(e).__name__}: {e}', tb=traceback.format_exc()[-2000:])
    print(name, res[name], flush=True)
json.dump(res, open(os.path.join(OUT, 'xgblss.json'), 'w'), default=float)
print('DONE', flush=True)
