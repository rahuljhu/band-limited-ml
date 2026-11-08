"""Paper 3 learning curve, v5 pipeline (run_bead_v5.py: ball kernel, EM background, fc by the selector on validation):
thin bead k's train+val counts by q to n photons (binomial thinning, as the split), keep the full test counts.
Methods: real / complex BLML (v5), KT histogram, Gaussian KDE (pk.kde, h on validation from a grid extended below
pk.KDE_H so the pixel limit is inside it), KDE + the same EM uniform background.
Usage: python results/psf/lc_v5.py <k> <n> <seed>  -> results/psf/v5/lc/b<k>_<n>_<seed>.json"""
import sys, os, json, time, numpy as np
ARGS = sys.argv[1:]                                  # the exec'd v5 prefix reads k from sys.argv[1]
src = open('results/psf/run_bead_v5.py').read()
exec(src[:src.index('res = dict(')])                  # imports, Z, cnt, tr/va/te split, em_bg, certificate
n_t, seed = int(float(ARGS[1])), int(ARGS[2])
q = min(1.0, n_t / (tr + va).sum()); tr, va = pk.thin(tr, q, 100 + seed), pk.thin(va, q, 200 + seed)
KH = [0.2, 0.3, 0.4] + list(pk.KDE_H)
def kde_bg(N, h, iters=200, tol=1e-9):
    N = np.asarray(N).ravel().astype(float); B = len(Z); pi = 0.1; prev = -np.inf; r = np.full(B, pi)
    for _ in range(iters):
        q_ = pk.kde(N * (1 - r), h, cnt.shape); P = pi / B + (1 - pi) * q_
        r = (pi / B) / P; pi = float((N * r).sum() / N.sum()); P = pi / B + (1 - pi) * q_; ll = pk.score(P, N)
        if ll - prev < tol: break
        prev = ll
    return P / P.sum()
res = dict(bead=k, n=int((tr + va).sum()), n_target=n_t, seed=seed, photons_full=int(cnt.sum()))
res['histogram'] = dict(test_ll=pk.score(pk.histogram(tr + va), te))
h = max(KH, key=lambda h_: pk.score(pk.kde(tr, h_, cnt.shape), va)); res['KDE'] = dict(h=h, test_ll=pk.score(pk.kde(tr + va, h, cnt.shape), te))
h = max(KH, key=lambda h_: pk.score(kde_bg(tr, h_), va)); res['KDE+bg'] = dict(h=h, test_ll=pk.score(kde_bg(tr + va, h), te))
for kname, kinds in [('ball', ['ball', 'ball'])]:
    for model, cg in [('real', False), ('complex', True)]:
        lab = f'{model}_{kname}'; t0 = time.time()
        def obj(fcv):
            if np.any(np.asarray(fcv) > 1.0 + 1e-9): return -np.inf
            P, _, _ = em_bg(tr, fcv, kinds, cg); return pk.score(P, va)
        fc, vll = sel._search(obj, 2, kinds, [None, None], fc0=0.1, tag=f'b{k}:{lab}', factors=FACTORS)
        P, pi, m = em_bg(tr + va, fc, kinds, cg)
        res[lab] = dict(fc=np.asarray(fc).tolist(), test_ll=pk.score(P, te), pi=pi, s=time.time() - t0)
os.makedirs('results/psf/v5/lc', exist_ok=True)
json.dump(res, open(f'results/psf/v5/lc/b{k}_{n_t}_{seed}.json', 'w'), indent=1)
print(k, n_t, seed, {m: round(d['test_ll'], 5) for m, d in res.items() if isinstance(d, dict)}, flush=True)
