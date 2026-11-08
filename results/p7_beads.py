"""Paper 7, data binned by the instrument: maximum-likelihood BLML (int g^2 = 1; real, the positive root) against the
empirical normalisation (BLMLEmpirical: ridge, normalised on the window, sum_b g_b^2 = 1) on fluorescent-bead pixel
counts (paper 3's beads, 31 x 31 windows, photon-calibrated), as the photons per pixel fall. Both on the binned engine
with the separable sinc kernel and the same EM background (a uniform floor over the window: P = pi / B + (1 - pi) q,
q refitted on the counts N (1 - r)); fc per axis on psfkit.FC_GRID by the same coordinate search for both (the shared
best, then two passes per axis), lam on psfkit.RIDGE_LAM extended to 3 for the empirical fit; refit on train + validation; scored on
the bead's full test counts (fixed), paired over test photons. Context: the add-1/2 histogram, the Gaussian KDE with the
same floor. Train + validation thinned binomially to n photons (as results/psf/lc_v5.py).
Usage: python results/p7_beads.py <bead> <n|full> <seed>  -> results/p7_beads/b<bead>_<n>_<seed>.json"""
import os, sys, json, time, numpy as np
sys.path.insert(0, 'results/psf'); sys.path.insert(0, '.')
import psfkit as pk, blml_trivial as bt
from blml_binned import BLMLBinned, BLMLEmpirical, _np
k, n_arg, seed = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
OUT = 'results/p7_beads'; os.makedirs(OUT, exist_ok=True)
cnt, pos = pk.bead_counts(k, 15); tr, va, te = pk.split(cnt); B = cnt.size
if n_arg != 'full':
    q = min(1.0, float(n_arg) / (tr + va).sum()); tr, va = pk.thin(tr, q, 100 + seed), pk.thin(va, q, 200 + seed)


def em(N, fit_q, iters=60, tol=1e-7):
    N = np.asarray(N, float); pi, r, prev = 0.1, np.full(N.shape, 0.1), -np.inf
    for _ in range(iters):
        qd = fit_q(N * (1 - r)); P = pi / B + (1 - pi) * qd
        r = (pi / B) / P; pi = float((N * r).sum() / N.sum()); P = pi / B + (1 - pi) * qd
        L = pk.score(P.ravel(), N.ravel())
        if L - prev < tol:
            break
        prev = L
    return P / P.sum(), pi


def q_ml(fc):
    def f(w):
        m = BLMLBinned(tol=1e-6).fit(w, bt.make_modes(list(fc), ['sinc', 'sinc']))
        d = np.maximum(_np(m.density_), 1e-300); return d / d.sum()
    return f


def q_emp(fc, lam):
    def f(w):
        m = BLMLEmpirical(reg='ridge', lam=lam, normalise='window').fit(w, bt.make_modes(list(fc), ['sinc', 'sinc']))
        d = np.maximum(_np(m.density_), 1e-300); return d / d.sum()
    return f


def coord_search(make):
    """The shared best on the grid, then two passes per axis (the same search for every method)."""
    G = list(pk.FC_GRID); val = lambda fc: pk.score(em(tr, make(fc))[0].ravel(), va.ravel())
    f0 = max(G, key=lambda g: val((g, g))); fc = [f0, f0]
    for _ in range(2):
        for a in range(2):
            fc[a] = max(G, key=lambda g: val(tuple(fc[:a] + [g] + fc[a + 1:])))
    return tuple(fc), val(tuple(fc))


res = dict(bead=k, n=int((tr + va).sum()), per_pixel=float((tr + va).sum() / B), seed=seed, n_test=int(te.sum()))
t = time.time(); fc, v = coord_search(q_ml); P_ml, pi = em(tr + va, q_ml(fc))
res['ml'] = dict(fc=list(fc), val=v, test=pk.score(P_ml.ravel(), te.ravel()), pi=pi, s=time.time() - t)
t = time.time(); best = None
LAMS = list(pk.RIDGE_LAM) + [0.3, 1.0, 3.0] + ([10.0, 30.0, 100.0] if os.environ.get('P7_LAM_WIDE') == '1' else [])   # extended upward where 0.1, then 3, was selected
for lam in LAMS:
    fc, v = coord_search(lambda f: q_emp(f, lam))
    if best is None or v > best[2]:
        best = (fc, lam, v)
fc, lam, v = best; P_e, pi = em(tr + va, q_emp(fc, lam))
res['empirical'] = dict(fc=list(fc), lam=lam, lam_at_edge=bool(lam in (LAMS[0], LAMS[-1])), fc_at_edge=bool(set(fc) & {pk.FC_GRID[0], pk.FC_GRID[-1]}), val=v, test=pk.score(P_e.ravel(), te.ravel()), pi=pi, s=time.time() - t)
res['histogram'] = dict(test=pk.score(pk.histogram(tr + va), te))
h = max(pk.KDE_H, key=lambda h_: pk.score(em(tr, lambda w: (lambda d: d / d.sum())(np.maximum(pk.kde(w, h_, cnt.shape).reshape(cnt.shape), 1e-300)))[0].ravel(), va.ravel()))
P_k, _ = em(tr + va, lambda w: (lambda d: d / d.sum())(np.maximum(pk.kde(w, h, cnt.shape).reshape(cnt.shape), 1e-300)))
res['kde'] = dict(h=h, test=pk.score(P_k.ravel(), te.ravel()))
w = te.ravel().astype(float); d = np.log(P_e.ravel()) - np.log(P_ml.ravel())       # per test photon, at its pixel
mu = float((w * d).sum() / w.sum()); sd = np.sqrt(float((w * (d - mu) ** 2).sum() / w.sum()))
res['emp_minus_ml'] = dict(mean=mu, se=sd / np.sqrt(w.sum()))
json.dump(res, open(f'{OUT}/b{k}_{n_arg}_{seed}.json', 'w'), indent=1)
print(k, n_arg, seed, f"per px {res['per_pixel']:.1f}", {m: round(res[m]['test'], 5) for m in ('ml', 'empirical', 'histogram', 'kde')},
      f"emp-ml {mu:+.5f}+-{res['emp_minus_ml']['se']:.5f}", f"fc ml {np.round(res['ml']['fc'], 3).tolist()} emp {np.round(res['empirical']['fc'], 3).tolist()} lam {lam}",
      f"{res['ml']['s'] + res['empirical']['s']:.0f}s", flush=True)
