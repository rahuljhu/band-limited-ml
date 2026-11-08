"""Paper 3 / 4, double slit in the plane: the coherence test by rank. Complex BLML with K = 1 or 2 signal modes (one
shared sinc kernel) plus the background mode (dirichlet, periodic over the crop, N = (1, 1)), f = sum_k |g_k|^2 +
|g_bg|^2, on the 2-d crop (results/dslit_export.py). On the line a sum of squares is no richer than one field; in the
plane a partially coherent beam has rank > 1. Signal fc per axis on a 3 x 3 grid around the 1-d band (validation
log-likelihood, float32 fits), the same grid for both ranks; the final fits at each rank's best grid point on train +
validation (float64), scored on test (paired). Split 64/16/20 (rng 0). Every fit is cached (resumable).
Usage: python results/dslit_rank2d.py <set> [--time]   -> results/dslit_rank2d/<set>.json"""
import os, sys, json, time, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_binned import BLMLBinned, _np
D = os.path.expanduser('~/data_repo/dslit_gpu/'); OUT = 'results/dslit_rank2d'; os.makedirs(OUT, exist_ok=True)
name = sys.argv[1]; ONLY_TIME = '--time' in sys.argv
pts = np.load(D + f'dslit_{name}_points.npy'); SH = np.load(D + f'dslit_{name}_b.npy').shape[-2:]
axis = int(np.load(D + f'dslit_{name}_axis.npy'))
ij = [np.clip(np.floor(pts[:, k] + 0.5).astype(int), 0, SH[k - 1] - 1) for k in (1, 2)]
u = np.random.default_rng(0).random(len(pts)); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))
def img(s):
    C = np.zeros(SH); np.add.at(C, (ij[0][s], ij[1][s]), 1); return C
TR, VA, TE = img(part == 0), img(part == 1), img(part == 2)
ll = lambda P, N: float((N * np.log(np.maximum(P / P.sum(), 1e-300))).sum() / N.sum())
fc1d = json.load(open(f'results/dslit_1d/{name}/bg1/{name}_complex_bg.json'))['fc'][0]   # the band along the fringes
ALONG = [fc1d / 2 ** 0.5, fc1d, fc1d * 2 ** 0.5]; ACROSS = [0.03, 0.06, 0.12]


def fit(N, fcs, K, dtype, sweeps):
    fa = [0.0, 0.0]; fa[axis], fa[1 - axis] = fcs
    sig = bt.make_modes(fa, ['sinc', 'sinc'])[0]
    bg = bt.make_modes([1.5 * 2 / SH[0], 1.5 * 2 / SH[1]], ['dirichlet', 'dirichlet'], periods=[float(SH[0]), float(SH[1])])[0]
    m = BLMLBinned(tol=1e-5, max_sweeps=sweeps, ll_tol=1e-10, dtype=dtype).fit(N, [sig] * K + [bg], complex_g=True)
    E, U = _np(m.e_hat_), _np(m.Se_); Q = np.array([float(np.real(np.sum(np.conj(E[k]) * U[k]))) for k in range(len(E))])
    return np.maximum(_np(m.density_).astype(float), 1e-300), m, Q / Q.sum()


cache_f = f'{OUT}/{name}_cache.json'; cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
if ONLY_TIME:
    t = time.time(); P, m, q = fit(TR, (fc1d, 0.06), 2, np.float32, 3000)
    print(f'{name}: one rank-2 fit {time.time() - t:.0f}s, {m.n_sweeps_} sweeps, stop {m.stop_reason_}'); sys.exit()
for K in (1, 2):
    for fa in ALONG:
        for fc_ in ACROSS:
            key = f'{K}|{fa:.5f}|{fc_:.5f}'
            if key in cache:
                continue
            t = time.time(); P, m, q = fit(TR, (fa, fc_), K, np.float32, 3000)
            cache[key] = dict(val=ll(P, VA), shares=q.tolist(), stop=m.stop_reason_, s=time.time() - t)
            json.dump(cache, open(cache_f, 'w'), indent=1)
            print(f'{name} K={K} fc=({fa:.4f}, {fc_:.3f}): val {cache[key]["val"]:.5f} shares {np.round(q, 3).tolist()} '
                  f'{cache[key]["stop"]} {cache[key]["s"]:.0f}s', flush=True)
res = dict(set=name, n_test=int(TE.sum()), fc1d=fc1d, grid=cache)
logP = {}
for K in (1, 2):
    best = max((k for k in cache if k.startswith(f'{K}|')), key=lambda k: cache[k]['val'])
    fa, fc_ = map(float, best.split('|')[1:])
    P, m, q = fit(TR + VA, (fa, fc_), K, np.float64, 5000)
    logP[K] = np.log(P / P.sum())[ij[0][part == 2], ij[1][part == 2]]
    res[f'K{K}'] = dict(fc=[fa, fc_], val=cache[best]['val'], test=float(logP[K].mean()), shares=q.tolist(), stop=m.stop_reason_)
d = logP[2] - logP[1]
res['rank2_minus_rank1'] = dict(test=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))),
                                val=res['K2']['val'] - res['K1']['val'])
json.dump(res, open(f'{OUT}/{name}.json', 'w'), indent=1)
print(name, json.dumps({k: v for k, v in res.items() if k != 'grid'}), flush=True)
