"""Pixel-level bead fit: BLML on camera pixels (centres + photon counts), no jitter.
Photons are split train/val/test by multinomial thinning of each pixel's count.
All models are scored on held-out pixel counts: LL = sum_b N_b log P_b / sum_b N_b, with
P_b = (mass of pixel b) / (mass of the ROI), pixel mass from a 6x6 midpoint rule.
Usage: python fit_bead_px.py <bead index>"""
import sys, json, time, numpy as np
import blml_trivial as bt
from blml_trivial import make_modes
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
bt.DIRICHLET_LEGACY = False
D = 'results/psf/'; NA, LAM, PX = 1.3, 0.515, 0.065
FC_PHYS = 2 * NA / LAM * PX
k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
S = np.load(D + 'stack_photons.npy', mmap_mode='r'); im = np.asarray(S[5])
y0, x0 = np.load(D + 'beads_iso.npy')[k]; H = 15
roi = im[y0 - H:y0 + H + 1, x0 - H:x0 + H + 1].astype(float)
bg = np.median(np.concatenate([roi[0], roi[-1], roi[:, 0], roi[:, -1]]))
cnt = np.round(np.maximum(roi - bg, 0)).astype(np.int64)
rng = np.random.default_rng(0)
tr = rng.binomial(cnt, 0.64); rest = cnt - tr; va = rng.binomial(rest, 0.16 / 0.36); te = rest - va
ax = np.arange(-H, H + 1.0); PXX, PXY = np.meshgrid(ax, ax); Z = np.column_stack([PXX.ravel(), PXY.ravel()])
sub = (np.arange(6) + 0.5) / 6 - 0.5
SQ = (Z[:, None, None, :] + np.stack(np.meshgrid(sub, sub), -1)[None]).reshape(-1, 2)   # 36 points per pixel
def pix_prob(dens_fn):
    m = np.maximum(dens_fn(SQ).reshape(len(Z), -1).mean(1), 1e-300)
    return m / m.sum()
pll = lambda p, N: float((N.ravel() * np.log(p)).sum() / N.sum())
def blml(fc, N):
    return bt.BLMLQuick(dtype=np.float64).fit(Z, bt.with_lattice(make_modes(fc, None, None, d=Z.shape[1]), 1.0), sample_weight=N.ravel())
res = dict(bead=[int(y0), int(x0)], photons=int(cnt.sum()), n_train=int(tr.sum()), n_val=int(va.sum()),
           n_test=int(te.sum()), bins=int((cnt > 0).sum()), fc_phys=FC_PHYS)
t0 = time.time()
grid = np.round(np.geomspace(0.12, 0.5, 14), 4); nfit = 0
best = max(((pll(pix_prob(blml([f, f], tr).predict), va), f) for f in grid)); nfit += len(grid)
fc = [best[1], best[1]]
for _ in range(2):                                    # per-axis coordinate search around the shared optimum
    for a in range(2):
        sc = []
        for f in grid:
            c = list(fc); c[a] = f; sc.append((pll(pix_prob(blml(c, tr).predict), va), f)); nfit += 1
        fc[a] = max(sc)[1]
tv = tr + va
bl = blml(fc, tv); bp = blml([FC_PHYS] * 2, tv)
res['BLML'] = dict(fc=[float(f) for f in fc], ll=pll(pix_prob(bl.predict), te), s=time.time() - t0, fits=nfit,
                   val_ll=float(max(sc)[0]), at_grid_edge=bool(max(fc) == grid[-1] or min(fc) == grid[0]))
res['BLML_phys'] = dict(fc=[FC_PHYS] * 2, ll=pll(pix_prob(bp.predict), te))
# baselines need points: jitter photons uniformly within pixels (as in fit_bead.py), at most 200k
def points(N, m=200000):
    yy, xx = np.nonzero(N); P = np.column_stack([np.repeat(xx, N[yy, xx]), np.repeat(yy, N[yy, xx])]).astype(float) - H
    P = P[rng.permutation(len(P))[:m]]; return P + rng.random(P.shape) - 0.5
Ptr, Ptv = points(tr), points(tv)
g1 = GaussianMixture(1, covariance_type='full', random_state=0).fit(Ptv)
res['Gaussian'] = dict(ll=pll(pix_prob(lambda q: np.exp(g1.score_samples(q))), te))
sc = []
for K in [2, 3, 5, 8, 12]:
    g = GaussianMixture(K, covariance_type='full', random_state=0, max_iter=300).fit(Ptr)
    sc.append((pll(pix_prob(lambda q: np.exp(g.score_samples(q))), va), K))
gm = GaussianMixture(max(sc)[1], covariance_type='full', random_state=0, max_iter=300).fit(Ptv)
res['GMM'] = dict(K=max(sc)[1], ll=pll(pix_prob(lambda q: np.exp(gm.score_samples(q))), te))
kd = gaussian_kde(Ptv[:20000].T)
res['KDE'] = dict(ll=pll(pix_prob(lambda q: kd(q.T)), te), note='Scott rule, 20k jittered points')
res['empirical_train_hist'] = dict(ll=pll(np.maximum(tv.ravel(), 0.5) / np.maximum(tv, 0.5).sum(), te),
                                   note='train+val pixel histogram (+0.5 floor), the no-model reference')
print(json.dumps({m: (v['ll'] if isinstance(v, dict) and 'll' in v else v) for m, v in res.items()}, indent=0))
print('BLML fc', res['BLML']['fc'], 'phys', round(FC_PHYS, 3), f"{nfit} fits in {res['BLML']['s']:.1f}s, B={bl.n_bins_}")
json.dump(res, open(D + f'bead{k}_px.json', 'w'), indent=1)
