"""Paper 7, detector lines: maximum-likelihood BLML (real, the positive root) against the empirical normalisation
(BLMLEmpirical ridge, window) on the double-slit lines of paper 3 (counts along the fringes, results/dslit_1d), with the
same EM background (a flat floor over the line); fc on the 2^(1/8) grid of dslit_job.FC_GRID, lam on psfkit.RIDGE_LAM
extended to 3; refit on train + validation; test LL per particle, paired. Split 64/16/20 (rng 0).
Usage: python results/p7_lines.py [set ...] -> results/p7_lines/<set>.json"""
import os, sys, json, time, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_binned import BLMLBinned, BLMLEmpirical, _np
D = os.path.expanduser('~/data_repo/dslit_gpu/'); OUT = 'results/p7_lines'; os.makedirs(OUT, exist_ok=True)
GRID = 0.008 * 2.0 ** (np.arange(0, 49) / 8.0); GRID = GRID[GRID <= 0.5]
LAMS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0]
ll = lambda P, N: float((N * np.log(np.maximum(P, 1e-300))).sum() / N.sum())


def em(N, fit_q, iters=200, tol=1e-9):
    B = N.size; pi, r, prev = 0.1, np.full(N.shape, 0.1), -np.inf
    for _ in range(iters):
        q = fit_q(N * (1 - r)); P = pi / B + (1 - pi) * q
        r = (pi / B) / P; pi = float((N * r).sum() / N.sum()); P = pi / B + (1 - pi) * q
        L = ll(P, N)
        if L - prev < tol:
            break
        prev = L
    return P / P.sum(), pi


def q_of(kind, fc, lam=None):
    def f(w):
        m = (BLMLBinned(tol=1e-7).fit(w, bt.make_modes([fc], ['sinc'])) if kind == 'ml' else
             BLMLEmpirical(reg='ridge', lam=lam, normalise='window').fit(w, bt.make_modes([fc], ['sinc'])))
        d = np.maximum(_np(m.density_), 1e-300); return d / d.sum()
    return f


for name in sys.argv[1:] or ['photon', 'electron']:
    pts = np.load(D + f'dslit_{name}_points.npy'); SH = np.load(D + f'dslit_{name}_b.npy').shape[-2:]
    axis = int(np.load(D + f'dslit_{name}_axis.npy'))
    x = np.clip(np.floor(pts[:, 1 + axis] + 0.5).astype(int), 0, SH[axis] - 1)
    u = np.random.default_rng(0).random(len(pts)); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))
    TR, VA, TE = [np.bincount(x[part == k], minlength=SH[axis]).astype(float) for k in (0, 1, 2)]
    G = GRID[GRID >= 2.0 / len(TR)]; res = dict(set=name, per_bin=float((TR + VA).sum() / len(TR)), n_test=int(TE.sum()))
    t = time.time(); v = [ll(em(TR, q_of('ml', f))[0], VA) for f in G]; fc = float(G[int(np.argmax(v))])
    P_ml, pi = em(TR + VA, q_of('ml', fc)); res['ml'] = dict(fc=fc, val=max(v), test=ll(P_ml, TE), pi=pi, s=time.time() - t)
    t = time.time(); best = None
    for lam in LAMS:
        v = [ll(em(TR, q_of('emp', f, lam))[0], VA) for f in G]; k = int(np.argmax(v))
        if best is None or v[k] > best[2]:
            best = (float(G[k]), lam, v[k])
    fc, lam, vb = best; P_e, pi = em(TR + VA, q_of('emp', fc, lam))
    res['empirical'] = dict(fc=fc, lam=lam, lam_at_edge=lam in (LAMS[0], LAMS[-1]), val=vb, test=ll(P_e, TE), pi=pi, s=time.time() - t)
    d = (np.log(P_e) - np.log(P_ml))[x[part == 2]]
    res['emp_minus_ml'] = dict(mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))))
    json.dump(res, open(f'{OUT}/{name}.json', 'w'), indent=1)
    print(name, f"{res['per_bin']:.1f}/bin", {m: round(res[m]['test'], 5) for m in ('ml', 'empirical')},
          f"emp-ml {res['emp_minus_ml']['mean']:+.5f}+-{res['emp_minus_ml']['se']:.5f}", f"fc ml {res['ml']['fc']:.4f} emp {fc:.4f} lam {lam}", flush=True)
