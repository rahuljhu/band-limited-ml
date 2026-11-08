"""Paper 3: the pseudo-true slit geometry of a double-slit record's truth (the record's Fresnel fit on its crop): the
(fc, fill, N) of the two-slit kernel with the background mode of N harmonics that maximises the expected
log-likelihood sum_x P(x) log Q(x) under the truth, i.e. the n -> infinity target of the fit on that crop (the
windowed aperture A * sinc_L, the normaliser and the background mode included). Fitted to the expected counts
n_eff P (n_eff = 10^8), on the grid of p3_slit_main.py.
  python p3_pseudo_true.py <set>   -> results/p3_pseudo_true/<set>.json"""
import os, sys, json, numpy as np
os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_SOLVER', 'lbfgs')
name = sys.argv[1]; sys.argv = sys.argv[:1]; sys.path.insert(0, '.')
import dslit_job as dj
OUT = 'results/p3_pseudo_true'; os.makedirs(OUT, exist_ok=True)
P = np.load(os.path.join(dj.DATA, f'dslit_{name}_fresnel_P.npy')).astype(float); P /= P.sum(); W = len(P)
fr = json.load(open(f'results/dslit_1d/{name}/flat/{name}_fresnel.json'))['params']
phys = (1 + fr['a_over_d']) / fr['P']; r = fr['a_over_d']
T = 1e8 * P; dj.BFIX = np.array([np.nan])
FILLS = [1.0, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.36, 0.33, 0.3, 0.25]
G = [float(f) for f in dj.FC_GRID if 0.8 * phys <= f <= 2.0 * phys]
res = dict(set=name, fill_sharp=2 * r / (1 + r), fc_sharp=phys, grid=[])
for N in (0, 1, 2):
    for w in FILLS:
        prev = None
        for f in G:
            Q, prev, beta = dj.bg_fit(T, [f], (N,), True, 1, init=prev, max_sweeps=5000, tol=1e-7, fill=w)
            res['grid'].append(dict(N=N, fill=w, fc=f, ell=float((P * np.log(np.maximum(Q, 1e-300))).sum()), beta=beta))
    Q, _, beta = dj.bg_fit(T, [phys], (N,), True, 1, max_sweeps=5000, tol=1e-7, fill=2 * r / (1 + r))
    res[f'sharp_N{N}'] = dict(ell=float((P * np.log(np.maximum(Q, 1e-300))).sum()), beta=beta)
negH = float((P * np.log(P)).sum())
best = max(res['grid'], key=lambda g: g['ell']); res['best'] = dict(best, kl=negH - best['ell'], ratio=best['fc'] / phys)
for N in (0, 1, 2):
    b = max((g for g in res['grid'] if g['N'] == N), key=lambda g: g['ell'])
    res[f'best_N{N}'] = dict(b, kl=negH - b['ell'], ratio=b['fc'] / phys)
    res[f'sharp_N{N}']['kl'] = negH - res[f'sharp_N{N}']['ell']
json.dump(res, open(f'{OUT}/{name}.json', 'w'), indent=1)
print(name, 'sharp fill %.2f' % res['fill_sharp'], {k: (v['N'], v['fill'], round(v['ratio'], 3), '%.1e' % v['kl']) for k, v in res.items() if k.startswith('best')},
      {k: '%.1e' % v['kl'] for k, v in res.items() if k.startswith('sharp_N')})
