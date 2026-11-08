"""Bead fit on raw photon counts, every method fits the TOTAL density (bead + background), no EM.
Background is estimated afterwards from the spectrum of each fitted pixel-probability image P:
    F(nu) = sum_b P_b exp(-2 pi i nu.(z_b - c))  =  (1 - pi) O(nu) + pi W(nu),
W = DFT of the uniform window (exact Dirichlet product, width 1/L), O = smooth bead OTF near 0,
O = 1 + a|nu| + b|nu|^2 + c|nu|^3; linear least squares on |nu| <= NU_MAX gives pi.
Bead OTF = (F - pi W)/(1 - pi); bead image = (P - pi/B)/(1 - pi).
Counts: round(raw / 1.438) (photon-transfer gain fix), per-pixel binomial thinning 64/16/20.
Usage: python fit_bead_total.py <bead index>"""
import sys, json, time, numpy as np
from scipy.ndimage import gaussian_filter
import blml_trivial as bt
from blml_trivial import make_modes
bt.DIRICHLET_LEGACY = False
D = 'results/psf/'; NA, LAM, PX = 1.3, 0.515, 0.065; FC_PHYS = 2 * NA / LAM * PX; GAIN_CORR = 1.438
k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
S = np.load(D + 'stack_photons.npy', mmap_mode='r'); im = np.asarray(S[5])
y0, x0 = np.load(D + 'beads_iso.npy')[k]; H = 15; L = 2 * H + 1
cnt = np.round(np.maximum(im[y0 - H:y0 + H + 1, x0 - H:x0 + H + 1].astype(float) / GAIN_CORR, 0)).astype(np.int64)
rng = np.random.default_rng(0)
tr = rng.binomial(cnt, 0.64); rest = cnt - tr; va = rng.binomial(rest, 0.16 / 0.36); te = rest - va
tv = tr + va
ax = np.arange(-H, H + 1.0); X, Y = np.meshgrid(ax, ax); Z = np.column_stack([X.ravel(), Y.ravel()]); B = len(Z)
pll = lambda P, N: float((N.ravel() * np.log(np.maximum(P.ravel(), 1e-300))).sum() / N.sum())
norm = lambda v: v / v.sum()

def blml_P(fc, N):
    m = bt.BLMLQuick(dtype=np.float64).fit(Z, bt.with_lattice(make_modes(fc, None, None, d=Z.shape[1]), 1.0), sample_weight=N.ravel().astype(float))
    blml_P.model = m; return norm(m.predict(Z))
def gmm_P(K, N, iters=300):
    w = N.ravel().astype(float); g = np.random.default_rng(1)
    c = (w[:, None] * Z).sum(0) / w.sum(); mu = c + g.normal(0, 3.0, (K, 2)) * (K > 1)
    C = np.repeat(np.eye(2)[None] * 9.0, K, 0); a = np.full(K, 1.0 / K)
    for _ in range(iters):
        dens = np.empty((K, B))
        for j in range(K):
            d = Z - mu[j]; Ci = np.linalg.inv(C[j])
            dens[j] = a[j] * np.exp(-0.5 * np.einsum('bi,ij,bj->b', d, Ci, d)) / (2 * np.pi * np.sqrt(np.linalg.det(C[j])))
        R = dens / np.maximum(dens.sum(0), 1e-300); Wk = R * w; a = Wk.sum(1) / w.sum()
        for j in range(K):
            s = max(Wk[j].sum(), 1e-12); mu[j] = (Wk[j][:, None] * Z).sum(0) / s
            d = Z - mu[j]; C[j] = (Wk[j][:, None, None] * d[:, :, None] * d[:, None, :]).sum(0) / s + 0.05 * np.eye(2)
    return norm(dens.sum(0))
kde_P = lambda h, N: norm(np.maximum(gaussian_filter(N.astype(float), h, mode='reflect').ravel(), 1e-12))
hist_P = lambda N: norm(np.maximum(N.ravel().astype(float), 0.5))

res = dict(bead=[int(y0), int(x0)], photons=int(cnt.sum()), n_test=int(te.sum()), gain_corr=GAIN_CORR, fc_phys=FC_PHYS)
P = {}
t0 = time.time(); grid = np.round(np.geomspace(0.12, 0.5, 14), 4); cache = {}
vl = lambda fc: cache.setdefault(tuple(fc), pll(blml_P(list(fc), tr), va))
fc = [max((vl((f, f)), f) for f in grid)[1]] * 2
for _ in range(2):
    for a_ in range(2):
        fc[a_] = max((vl(tuple(fc[:a_] + [f] + fc[a_ + 1:])), f) for f in grid)[1]
P['BLML'] = blml_P(fc, tv); mB = blml_P.model
res['BLML'] = dict(fc=[float(f) for f in fc], s=time.time() - t0, fits=len(cache), at_grid_edge=bool(max(fc) == grid[-1]))
P['BLML@phys'] = blml_P([FC_PHYS] * 2, tv); mBp = blml_P.model; res['BLML@phys'] = dict(fc=[FC_PHYS] * 2)
K = max((pll(gmm_P(K, tr), va), K) for K in [1, 2, 3, 5, 8, 12])[1]; P['GMM'] = gmm_P(K, tv); res['GMM'] = dict(K=K)
P['Gaussian'] = gmm_P(1, tv); res['Gaussian'] = {}
h = max((pll(kde_P(h, tr), va), h) for h in [0.5, 0.75, 1, 1.5, 2, 3])[1]; P['KDE'] = kde_P(h, tv); res['KDE'] = dict(h=h)
P['histogram'] = hist_P(tv); res['histogram'] = {}

# spectral background estimate
c = (cnt.ravel()[:, None] * Z).sum(0) / cnt.sum()
NU_MAX = 4.0 / L; nu1 = np.linspace(-NU_MAX, NU_MAX, 41); NX, NY = np.meshgrid(nu1, nu1); NR = np.hypot(NX, NY); band = NR <= NU_MAX
def F_of(Pimg, NX, NY):
    ph = np.exp(-2j * np.pi * ((Z[:, 0] - c[0])[:, None, None] * NX + (Z[:, 1] - c[1])[:, None, None] * NY))
    return np.tensordot(Pimg.ravel(), ph, 1)
Wfull = F_of(np.full(B, 1.0 / B), NX, NY)                      # exact window transform (incl. centroid phase)
def spectral_pi(Pimg):
    F = F_of(Pimg, NX, NY); y = (F - Wfull)[band]
    A = np.column_stack([(1 - Wfull)[band], NR[band], NR[band] ** 2, NR[band] ** 3]).astype(complex)
    sol = np.linalg.lstsq(np.vstack([A.real, A.imag]), np.concatenate([y.real, y.imag]), rcond=None)[0]
    return float(1 - sol[0])
for m in P:
    res[m]['ll'] = pll(P[m], te); res[m]['pi_spectral'] = spectral_pi(P[m])
res['pi_spectral_raw_counts'] = spectral_pi(norm(cnt.ravel().astype(float)))
res['pi_border_median'] = float(np.median(np.r_[cnt[0], cnt[-1], cnt[:, 0], cnt[:, -1]]) * B / cnt.sum())
print(f"{'model':10s} {'test LL':>9s} {'pi (spectral)':>14s}")
for m in P: print(f"{m:10s} {res[m]['ll']:9.5f} {res[m]['pi_spectral']:14.4f}")
print('pi spectral from raw counts %.4f | border-median %.4f | EM (BLML+bg run) 0.067' % (res['pi_spectral_raw_counts'], res['pi_border_median']))
print('BLML fc', np.round(fc, 3), 'phys %.3f' % FC_PHYS, '| %d fits %.0fs' % (len(cache), res['BLML']['s']), '| GMM K', K, '| KDE h', h)
json.dump(res, open(D + f'bead{k}_total.json', 'w'), indent=1)
gr = np.linspace(-H - 0.5, H + 0.5, 241); GX, GY = np.meshgrid(gr, gr); Q = np.column_stack([GX.ravel(), GY.ravel()])
nu = np.linspace(-0.3, 0.3, 121); NXg, NYg = np.meshgrid(nu, nu)
def ftg(mm):
    F = np.zeros(NXg.shape, complex)
    for i in range(0, len(mm.Z_), 200):
        zz = mm.Z_[i:i + 200] - c
        F += np.tensordot(mm.e_hat_[i:i + 200], np.exp(-2j * np.pi * (zz[:, 0, None, None] * NXg + zz[:, 1, None, None] * NYg)), 1)
    return np.where((np.abs(NXg) <= mm._fc_arr[0] / 2) & (np.abs(NYg) <= mm._fc_arr[1] / 2), F / mm.n_samples_, 0)
np.savez(D + f'bead{k}_total.npz', cnt=cnt, tv=tv, te=te, center=c, gr=gr, d_blml=mB.predict(Q).reshape(GX.shape),
         nu=nu, ftg=ftg(mB), **{'P_' + m.replace('@', '_'): v for m, v in P.items()})
