"""Fit one isolated bead image (photons as points) with BLML, a 2-D Gaussian, a GMM and a KDE.
Physical cutoff of the intensity: 2 NA / lambda = 2*1.3/0.515 um = 5.05 /um = 0.328 cycles/pixel (65 nm px).
Usage: python fit_bead.py <bead index in beads_iso.npy>"""
import sys, json, time, numpy as np
import blml_trivial as bt
from blml_trivial import BLMLQuick, make_modes, modes_fc
from selector import select
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
bt.DIRICHLET_LEGACY = False
D = 'results/psf/'; NA, LAM, PX = 1.3, 0.515, 0.065
FC_PHYS = 2 * NA / LAM * PX                                    # cycles / pixel (intensity band limit)
k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
S = np.load(D + 'stack_photons.npy', mmap_mode='r'); im = np.asarray(S[5])
y0, x0 = np.load(D + 'beads_iso.npy')[k]; H = 15
roi = im[y0 - H:y0 + H + 1, x0 - H:x0 + H + 1].astype(float)
border = np.concatenate([roi[0], roi[-1], roi[:, 0], roi[:, -1]]); bg = np.median(border)
cnt = np.round(np.maximum(roi - bg, 0)).astype(int)
rng = np.random.default_rng(0)
yy, xx = np.nonzero(cnt); reps = cnt[yy, xx]
P = np.column_stack([np.repeat(xx, reps), np.repeat(yy, reps)]).astype(float) - H        # pixel coords, centred
P += rng.random(P.shape) - 0.5                                 # uniform within the pixel
N0 = len(P); P = P[rng.permutation(N0)[:200000]]; N = len(P)
ntr, nva = int(0.64 * N), int(0.16 * N)
tr, va, te = P[:ntr], P[ntr:ntr + nva], P[ntr + nva:]
res = dict(bead=[int(y0), int(x0)], photons_total=int(N0), used=N, bg=float(bg), fc_phys=FC_PHYS)
ll = lambda d: float(np.mean(np.log(np.maximum(d, 1e-300))))
t0 = time.time()
sel_m, _ = select(tr, make_modes(np.nan, ['sinc', 'sinc'], d=2), search='auto', engine='cpu', X_val=va)
fc = [float(f) for f in modes_fc(sel_m)[0]]
bl = BLMLQuick(seed=0); bl.fit(np.vstack([tr, va]), make_modes(fc, ['sinc', 'sinc']))
res['BLML'] = dict(fc=fc, ll=ll(bl.predict(te)), s=time.time() - t0)
bp = BLMLQuick(seed=0); bp.fit(np.vstack([tr, va]), make_modes([FC_PHYS] * 2, ['sinc', 'sinc']))
res['BLML_phys'] = dict(fc=[FC_PHYS] * 2, ll=ll(bp.predict(te)))
g1 = GaussianMixture(1, covariance_type='full', random_state=0).fit(np.vstack([tr, va]))
res['Gaussian'] = dict(ll=float(g1.score(te)))
best = None
for K in [2, 3, 5, 8, 12]:
    g = GaussianMixture(K, covariance_type='full', random_state=0, max_iter=300).fit(tr)
    s = g.score(va)
    if best is None or s > best[0]:
        best = (s, K)
gm = GaussianMixture(best[1], covariance_type='full', random_state=0, max_iter=300).fit(np.vstack([tr, va]))
res['GMM'] = dict(K=best[1], ll=float(gm.score(te)))
kd = gaussian_kde(np.vstack([tr, va])[:20000].T)                 # Scott's rule, 20k subsample
res['KDE'] = dict(ll=ll(kd(te.T)), note='Scott rule, 20k-point subsample')
print(json.dumps({k: (v['ll'] if isinstance(v, dict) and 'll' in v else v) for k, v in res.items()}, indent=0), flush=True)
print('BLML selected fc (cycles/px):', [round(f, 3) for f in fc], ' physical:', round(FC_PHYS, 3), flush=True)
# densities on a fine grid, and Fourier transforms of g-hat and f-hat
gr = np.linspace(-H - 0.5, H + 0.5, 241); GX, GY = np.meshgrid(gr, gr); Q = np.column_stack([GX.ravel(), GY.ravel()])
dens = {'BLML': bl.predict(Q), 'BLML_phys': bp.predict(Q), 'Gaussian': np.exp(g1.score_samples(Q)),
        'GMM': np.exp(gm.score_samples(Q))}
nu = np.linspace(-0.3, 0.3, 121)
def ghat_ft(m):
    Z, e = m.Z_, m.e_hat_.astype(float) / m.n_samples_
    NX, NY = np.meshgrid(nu, nu); F = np.zeros(NX.shape, complex)
    for i in range(0, len(Z), 4000):
        z = Z[i:i + 4000]
        F += np.tensordot(e[i:i + 4000], np.exp(-2j * np.pi * (z[:, 0, None, None] * NX + z[:, 1, None, None] * NY)), 1)
    inband = (np.abs(NX) <= m._fc_arr[0] / 2) & (np.abs(NY) <= m._fc_arr[1] / 2)
    return np.where(inband, F, 0)
ftg = ghat_ft(bl)
f_img = dens['BLML'].reshape(GX.shape); dx = gr[1] - gr[0]
ftf = np.fft.fftshift(np.abs(np.fft.fft2(np.fft.ifftshift(f_img)))) * dx * dx
fq = np.fft.fftshift(np.fft.fftfreq(len(gr), d=dx))
np.savez(D + f'bead{k}.npz', P=P, te=te, gr=gr, nu=nu, ftg=ftg, ftf=ftf, fq=fq, center=g1.means_[0],
         **{'d_' + k2: v.reshape(GX.shape) for k2, v in dens.items()}, cnt=cnt)
json.dump(res, open(D + f'bead{k}.json', 'w'), indent=1)
