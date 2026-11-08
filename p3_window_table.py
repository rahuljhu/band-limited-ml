"""Paper 3, Table III: every method on a double-slit record's window, the periodic two-slit kernel the reference. The
window, split (seed 0) and the periodic kernel's selected (hi, lo, N) are those of p3_periodic_main.py
(results/p3_periodic_main/<set>_s0.json); on the same training+validation counts we fit
  periodic        the periodic two-slit kernel at its selected harmonics and background (refitted)
  periodic_pub    the periodic two-slit kernel at the published slits (nearest harmonics: hi = round(L(d+a)/2),
                  lo = round(L(d-a)/2), d = 1/P of the record's Fresnel fit, a/d published), background N in {0,1,2} by CV
  band            the periodic plain band (lo = 0), hi by CV, the selected N
  kde, akde       Gaussian KDE (h by CV) and Abramson's adaptive KDE ((h, alpha) by CV), flat background by EM
  fresnel, fraunhofer   the two-slit models (free slit ratio, amplitude ratio, phase, visibility; Fresnel: defocus),
                  flat background, refitted on the window from the record's fits
  hist            the add-1/2 histogram
and score each by the test log-likelihood per particle and its paired difference to the periodic kernel (+- SE).
  BLML_DATA_DIR=~/data_repo/dslit_gpu python p3_window_table.py <set>   -> results/p3_window_table/<set>.json"""
import os, sys, json, time, numpy as np
os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_SOLVER', 'lbfgs'); os.environ.setdefault('BLML_COLLAPSE', '1')
name = sys.argv[1]; sys.argv = sys.argv[:1]; sys.path.insert(0, '.')
import dslit_job as dj
from blml_binned import BLMLBinned
from p3_buildup import setup
OUT = 'results/p3_window_table'; os.makedirs(OUT, exist_ok=True); fn = f'{OUT}/{name}.json'
if os.path.exists(fn):
    sys.exit(0)
PUB = dict(photon=100 / 500, ghost=100 / 500, photon_mm=100 / 500, electron=62 / 272)      # as p3_slit_main.py
R = json.load(open(f'results/p3_periodic_main/{name}_s0.json')); a0, a1 = R['window']; L = a1 - a0
fr = json.load(open(f'results/dslit_1d/{name}/flat/{name}_fresnel.json'))
x = setup(name)[0]; keep = (x >= a0) & (x < a1); x = x[keep] - a0; assert len(x) == R['n_kept']
u = np.random.default_rng(0).random(len(x)); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))   # as p3_periodic_main
fold = np.random.default_rng(1000).integers(0, 5, len(x))
cnt = lambda msk: np.bincount(x[msk], minlength=L).astype(float)
TV, TE = cnt(part < 2), cnt(part == 2)
FOLDS = [(cnt((part < 2) & (fold != k)), cnt((part < 2) & (fold == k))) for k in range(5)]
fold_of = {id(T): k for k, (T, _) in enumerate(FOLDS)}; nV = sum(V.sum() for _, V in FOLDS)
dj.BFIX = np.array([np.nan]); bflat = np.ones((1, L)) / L; t0 = time.time()
def pfit(T, hi, lo, N):
    fc = (2 * hi + 0.5) / L; w = 1.0 if lo == 0 else 1.0 - (lo - 0.5) / (hi + 0.25)
    sig = {0: dict(kind='dirichlet', fc=fc, period=float(L), **({} if w >= 1 else dict(fill=w)))}
    bg = {0: dict(kind='dirichlet', fc=(N + 0.5) * 2 / L, period=float(L))}
    m = BLMLBinned(tol=1e-6, max_sweeps=5000, ll_tol=1e-10, engine=dj.ENGINE, solver=dj.SOLVER).fit(T, [sig, bg], complex_g=True)
    P_ = np.maximum(dj._np(m.density_), 1e-300); return P_ / P_.sum()
res = dict(set=name, window=[a0, a1], L=L, n_test=int(TE.sum())); LOG = {}
def score(k, Q, **ex):
    Q = Q / Q.sum(); LOG[k] = np.log(np.maximum(Q, 1e-300)); res[k] = dict(test=dj.ll(Q, TE), **ex)
    print(f'{time.time() - t0:.0f}s', k, res[k], flush=True)
p = R['periodic']; score('periodic', pfit(TV, p['hi'], p['lo'], p['N']), hi=p['hi'], lo=p['lo'], N=p['N'])
d, r = 1 / fr['params']['P'], PUB[name]; hp, lp = int(round(L * d * (1 + r) / 2)), int(round(L * d * (1 - r) / 2))
cvp = {N: sum(float((V * np.log(pfit(T, hp, lp, N))).sum()) for T, V in FOLDS) / nV for N in (0, 1, 2)}
Np = max(cvp, key=cvp.get); score('periodic_pub', pfit(TV, hp, lp, Np), hi=hp, lo=lp, N=Np)
his = sorted({hh for hh in range(1, int(2 * R['phys'] * L / 2) + 2) if hh >= 0.8 * R['phys'] * L / 2 - 0.5})
(hb,), _ = dj.cv_select(FOLDS, lambda c, T: pfit(T, c[0], 0, p['N']), [(hh,) for hh in his])
score('band', pfit(TV, hb, 0, p['N']), hi=hb, N=p['N'])
score('kde', dj.em(TV, bflat, dj.kde_q(R['kde']['h']))[0], h=R['kde']['h'])
HG = [(h, a) for h in (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24) for a in (0.25, 0.5)]
(h, a), _ = dj.cv_select(FOLDS, lambda c, T: dj.em(T, bflat, dj.akde_q(*c))[0], sorted(HG, key=lambda c: (-c[0], c[1])))
score('akde', dj.em(TV, bflat, dj.akde_q(h, a))[0], h=h, alpha=a)
th0 = None
for nm in ('fraunhofer', 'fresnel'):
    if th0 is None:
        th0 = np.r_[np.array(json.load(open(f'results/dslit_1d/{name}/flat/{name}_{nm}.json'))['theta'])[:9], -1.0]; th0[0] -= a0
    th = dj.ref_fit(TV, bflat, 0, nm == 'fresnel', th0); th0 = th
    score(nm, dj.ref_P(th, bflat, 0, nm == 'fresnel')[0], params=dj.ref_desc(th, nm == 'fresnel'))
score('hist', TV + 0.5)
for k in LOG:
    dd = np.repeat(LOG[k] - LOG['periodic'], TE.astype(int)); res[k].update(diff=float(dd.mean()), se=float(dd.std(ddof=1) / np.sqrt(len(dd))))
res['s'] = time.time() - t0; json.dump(res, open(fn, 'w'), indent=1); print('done', fn, f"{res['s']:.0f}s")
