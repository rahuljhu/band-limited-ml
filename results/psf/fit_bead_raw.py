"""Bead fit on raw photon counts (no background subtraction, no clipping).
Counts: N_b = round(raw_b / GAIN_CORR), raw = (ADU-100)*0.51 as stored; GAIN_CORR = 1.438 from the
photon-transfer fit on background pixels (var = 1.438 mean + 0.48), so N_b is ~Poisson and per-pixel
binomial thinning gives independent train / val / test (64/16/20).
Every model is  P_b = pi/B + (1 - pi) q_b  over the B ROI pixels (q = bead component, normalised over
the ROI); pi and the component are fitted by EM on the training counts, hyper-parameters on val.
Models: BLML+bg (BLMLQuick on counts in the M-step), BLML without bg (ablation), Gaussian+bg, GMM(K)+bg,
KDE (Gaussian smoothing of the count image; bg is just smoothed), raw histogram.
Usage: python fit_bead_raw.py <bead index>"""
import sys, json, time, numpy as np
from scipy.ndimage import gaussian_filter
import blml_trivial as bt
from blml_trivial import make_modes
bt.DIRICHLET_LEGACY = False
D = 'results/psf/'; NA, LAM, PX = 1.3, 0.515, 0.065; FC_PHYS = 2 * NA / LAM * PX; GAIN_CORR = 1.438
k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
S = np.load(D + 'stack_photons.npy', mmap_mode='r'); im = np.asarray(S[5])
y0, x0 = np.load(D + 'beads_iso.npy')[k]; H = 15
cnt = np.round(np.maximum(im[y0 - H:y0 + H + 1, x0 - H:x0 + H + 1].astype(float) / GAIN_CORR, 0)).astype(np.int64)
rng = np.random.default_rng(0)
tr = rng.binomial(cnt, 0.64); rest = cnt - tr; va = rng.binomial(rest, 0.16 / 0.36); te = rest - va
ax = np.arange(-H, H + 1.0); X, Y = np.meshgrid(ax, ax); Z = np.column_stack([X.ravel(), Y.ravel()]); B = len(Z)
pll = lambda P, N: float((N.ravel() * np.log(np.maximum(P, 1e-300))).sum() / N.sum())

def em(N, comp_fit, pi0=0.5, iters=60, tol=1e-7):
    """EM for P = pi/B + (1-pi) q. comp_fit(weights) -> q (normalised over the ROI)."""
    N = N.ravel().astype(float); pi = pi0; r = np.full(B, pi0); prev = -np.inf
    for _ in range(iters):
        q = comp_fit(N * (1 - r))
        P = pi / B + (1 - pi) * q; r = (pi / B) / P; pi = float((N * r).sum() / N.sum())
        ll = pll(pi / B + (1 - pi) * q, N)
        if ll - prev < tol: break
        prev = ll
    return pi, q

def blml_q(fc, kinds=('sinc', 'sinc')):
    def f(w):
        m = bt.BLMLQuick(dtype=np.float64).fit(Z, bt.with_lattice(make_modes(fc, list(kinds), None), 1.0), sample_weight=np.where(w > 1e-6, w, 0))
        g2 = m.predict(Z); f.model = m; return g2 / g2.sum()
    return f

def gmm_q(K, iters=200):
    """Weighted EM for a K-component full-covariance GMM on pixel centres (component mass ~ density at centre)."""
    state = {}
    def f(w):
        if 'mu' not in state:
            c = (w[:, None] * Z).sum(0) / w.sum(); state['mu'] = c + np.random.default_rng(1).normal(0, 1.0, (K, 2)) * (K > 1)
            state['C'] = np.repeat(np.eye(2)[None] * 4.0, K, 0); state['a'] = np.full(K, 1.0 / K)
        mu, C, a = state['mu'], state['C'], state['a']
        for _ in range(iters):
            dens = np.empty((K, B))
            for j in range(K):
                d = Z - mu[j]; Ci = np.linalg.inv(C[j])
                dens[j] = a[j] * np.exp(-0.5 * np.einsum('bi,ij,bj->b', d, Ci, d)) / (2 * np.pi * np.sqrt(np.linalg.det(C[j])))
            R = dens / np.maximum(dens.sum(0), 1e-300); Wk = R * w
            a = Wk.sum(1) / w.sum()
            for j in range(K):
                s = max(Wk[j].sum(), 1e-12); mu[j] = (Wk[j][:, None] * Z).sum(0) / s
                d = Z - mu[j]; C[j] = (Wk[j][:, None, None] * d[:, :, None] * d[:, None, :]).sum(0) / s + 0.05 * np.eye(2)
        state.update(mu=mu, C=C, a=a); q = dens.sum(0); return q / q.sum()
    return f

def kde_q(h):
    return lambda w: (lambda s: s / s.sum())(np.maximum(gaussian_filter(w.reshape(2 * H + 1, -1), h, mode='reflect').ravel(), 1e-12))

res = dict(bead=[int(y0), int(x0)], photons=int(cnt.sum()), n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(te.sum()),
           gain_corr=GAIN_CORR, fc_phys=FC_PHYS)
pi0 = float(np.median(np.r_[cnt[0], cnt[-1], cnt[:, 0], cnt[:, -1]]) * B / cnt.sum())
# --- BLML + background: fc on val (shared grid, then per-axis), refit on train+val
t0 = time.time(); grid = np.round(np.geomspace(0.12, 0.6, 15), 4); cache = {}
def val_ll(fc):
    key = tuple(fc)
    if key not in cache:
        pi, q = em(tr, blml_q(list(fc)), pi0); cache[key] = pll(pi / B + (1 - pi) * q, va)
    return cache[key]
best = max((val_ll((f, f)), f) for f in grid); fc = [best[1], best[1]]
for _ in range(2):
    for a in range(2):
        fc[a] = max((val_ll(tuple(fc[:a] + [f] + fc[a + 1:])), f) for f in grid)[1]
tv = tr + va
out = {}
for name, fcs in [('BLML+bg', fc), ('BLML+bg@phys', [FC_PHYS] * 2)]:
    qf = blml_q(fcs); pi, q = em(tv, qf, pi0)
    res[name] = dict(fc=[float(x) for x in fcs], pi=pi, ll=pll(pi / B + (1 - pi) * q, te)); out[name] = (qf.model, pi, q)
res['BLML+bg'].update(s=time.time() - t0, fits=len(cache), val_ll=float(val_ll(tuple(fc))), at_grid_edge=bool(max(fc) == grid[-1]))
# ablation: BLML on total counts (no background term), fc on val
sc = max((pll(blml_q([f, f])(tr.ravel().astype(float)), va), f) for f in grid)
res['BLML (no bg term)'] = dict(fc=[sc[1]] * 2, ll=pll(blml_q([sc[1]] * 2)(tv.ravel().astype(float)), te))
# baselines
for name, mk, hp in [('Gaussian+bg', gmm_q, [1]), ('GMM+bg', gmm_q, [2, 3, 5, 8]), ('KDE', kde_q, [0.5, 0.75, 1, 1.5, 2, 3])]:
    if name == 'KDE':
        sc = [(pll(kde_q(h)(tr.ravel().astype(float)), va), h) for h in hp]; h = max(sc)[1]
        res[name] = dict(h=h, ll=pll(kde_q(h)(tv.ravel().astype(float)), te)); continue
    sc = []
    for K in hp:
        pi, q = em(tr, mk(K), pi0); sc.append((pll(pi / B + (1 - pi) * q, va), K))
    K = max(sc)[1]; qf = mk(K); pi, q = em(tv, qf, pi0)
    res[name] = dict(K=K, pi=pi, ll=pll(pi / B + (1 - pi) * q, te)); out[name] = (None, pi, q)
res['raw histogram'] = dict(ll=pll(np.maximum(tv.ravel(), 0.5) / np.maximum(tv, 0.5).sum(), te))
print(json.dumps({m: (v['ll'] if isinstance(v, dict) and 'll' in v else v) for m, v in res.items()}, indent=0))
print('BLML+bg fc', np.round(res['BLML+bg']['fc'], 3), 'pi %.3f' % res['BLML+bg']['pi'], '| phys %.3f' % FC_PHYS,
      '| %d EM-fits for selection in %.0fs' % (len(cache), res['BLML+bg']['s']))
json.dump(res, open(D + f'bead{k}_raw.json', 'w'), indent=1)
# save what the figure needs
m, pi, q = out['BLML+bg']; mp, pip, qp = out['BLML+bg@phys']
gr = np.linspace(-H - 0.5, H + 0.5, 241); GX, GY = np.meshgrid(gr, gr); Q = np.column_stack([GX.ravel(), GY.ravel()])
c = (q[:, None] * Z).sum(0)
nu = np.linspace(-0.3, 0.3, 121)
def ftg(mm):
    NX, NY = np.meshgrid(nu, nu); F = np.zeros(NX.shape, complex)
    for i in range(0, len(mm.Z_), 200):
        z = mm.Z_[i:i + 200] - c
        F += np.tensordot(mm.e_hat_[i:i + 200], np.exp(-2j * np.pi * (z[:, 0, None, None] * NX + z[:, 1, None, None] * NY)), 1)
    return np.where((np.abs(NX) <= mm._fc_arr[0] / 2) & (np.abs(NY) <= mm._fc_arr[1] / 2), F / mm.n_samples_, 0)
np.savez(D + f'bead{k}_raw.npz', cnt=cnt, tv=tv, te=te, gr=gr, center=c, pi=pi, q=q, pi_phys=pip, q_phys=qp,
         q_gmm=out['GMM+bg'][2], pi_gmm=out['GMM+bg'][1], q_gauss=out['Gaussian+bg'][2], pi_gauss=out['Gaussian+bg'][1],
         d_blml=m.predict(Q).reshape(GX.shape), d_phys=mp.predict(Q).reshape(GX.shape), nu=nu, ftg=ftg(m), ftg_phys=ftg(mp))
