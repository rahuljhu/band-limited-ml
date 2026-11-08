"""Paper 3, single-photon double slit (Glasgow heralded far-field frames, cleaned by results/dslit_prep.py): held-out
log-likelihood of BLML and baselines, raw and background-corrected, and the field spectra.
Split: each photon to train / validation / test with probabilities 0.64 / 0.16 / 0.20 (i.i.d. photons, so the test image is
a fresh exposure). Every model's pixel probabilities P_b are normalised over the 512 x 512 frame.
Models
  Fraunhofer   the physics reference, fitted by Poisson ML on train+val: in the fringe direction y (rows) the two-slit
               far field I(y) = sinc^2((y - y0) / W) [1 + V cos(2 pi (y - y0) / P - phi)], in x a Gaussian beam
               profile, plus a uniform floor pi (slit width a and separation d enter through W = P d / a)
  BLML real    positive root (the default solve), sinc x sinc kernel, EM uniform background (P_b = pi/B + (1 - pi) q_b)
  BLML signed  real g with its signs fixed to those of the Fraunhofer field u(y) = sinc((y - y0)/W) cos(pi (y - y0)/P)
               (the coherent symmetric-slit field is real up to a global phase), same kernel and background
  histogram    add-1/2 (KT);  KDE  anisotropic Gaussian (h_y, h_x on validation), and KDE + the EM background
  fc (per axis) and h by validation log-likelihood on grids; refit on train+val.
Scores, nats per test photon
  raw          sum_b N_b log P_b / N
  background-corrected   sum_b N_b (1 - r_b) log q_b / sum_b N_b (1 - r_b): the signal photons' log-likelihood under the
               model's signal density q, with one common responsibility r_b = (pi/B) / P_b from the Fraunhofer fit (so every
               model is scored on the same expected signal photons); for histogram / KDE, q = P.
Spectra: g along y through the beam centre (BLML: S e / n on the lattice), its |FT| against the aperture (two slits of
width a at +-d/2, in units of d: xi / d = nu P), and |FT f| against the aperture autocorrelation.
Usage: python results/dslit_fit.py <set> -> results/dslit/fit_<set>.json, .npz"""
import sys, os, json, time, numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_binned import BLMLBinned, _np
name = sys.argv[1]; SH = (512, 512); B = SH[0] * SH[1]
Ph = np.load(os.path.expanduser(f'~/data_repo/glasgow_dslit/{name}_clean.npz'))['photons']
iy, ix = (np.clip(np.floor(Ph[:, j] + 0.5).astype(int), 0, 511) for j in (1, 2))
u = np.random.default_rng(0).random(len(Ph)); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))
def img(s):
    C = np.zeros(SH); np.add.at(C, (iy[s], ix[s]), 1); return C
TR, VA, TE = img(part == 0), img(part == 1), img(part == 2); TV = TR + VA
def ll(P, N):
    P = P / P.sum(); return float((N * np.log(np.maximum(P, 1e-300))).sum() / N.sum())
Y, X = np.meshgrid(np.arange(512.0), np.arange(512.0), indexing='ij')

# ---------------- Fraunhofer reference
def fr_parts(th):
    y0, x0, P, W, V, phi, sx = th[:7]
    s = Y - y0
    return np.sinc(s / W) ** 2 * (1 + V * np.cos(2 * np.pi * s / P - phi)) * np.exp(-0.5 * ((X - x0) / sx) ** 2)
def fr_P(th):
    sig = fr_parts(th); sig = sig / sig.sum(); pi = 1 / (1 + np.exp(-th[7])); return pi / B + (1 - pi) * sig, pi
def fr_fit(N):
    ry, rx = N.sum(1), N.sum(0); y0 = float((ry * np.arange(512)).sum() / ry.sum()); x0 = float(np.argmax(gaussian_filter(rx, 3)))
    best = None
    for P0 in (45.0, 50.0, 55.0, 60.0):
        th0 = np.array([y0, x0, P0, 5 * P0, 0.9, 0.0, 8.0, -1.0])
        r = minimize(lambda t: -ll(fr_P(t)[0], N) if 1 < t[2] and 1 < t[3] and 0 <= t[4] <= 1 and t[6] > 0.5 else 1e9, th0,
                     method='Nelder-Mead', options=dict(maxiter=20000, maxfev=20000, xatol=1e-4, fatol=1e-9))
        if best is None or r.fun < best.fun: best = r
    return best.x
t0 = time.time(); th = fr_fit(TV); Pfr, pi_fr = fr_P(th)
r_common = (pi_fr / B) / Pfr
print(f'Fraunhofer: y0 {th[0]:.1f} x0 {th[1]:.1f} P {th[2]:.2f} W {th[3]:.1f} (d/a = {th[3] / th[2]:.2f}) V {th[4]:.3f} '
      f'phi {th[5]:.3f} sx {th[6]:.2f} pi {pi_fr:.3f}  {time.time() - t0:.0f}s', flush=True)
fcy_phys = 2 * (0.5 / th[2] + 0.5 / th[3])            # f band = 2 x the field band (d + a) / (2 lambda f), in cycles / px

# ---------------- BLML with the EM background
def em_blml(N, fc, signs=None, iters=100, tol=1e-8):
    pi, prev, m = 0.1, -np.inf, None; r = np.full(SH, pi)
    for _ in range(iters):
        w = N * (1 - r)
        m = BLMLBinned(tol=1e-5, max_sweeps=3000, ll_tol=1e-10, workers=4).fit(w, bt.make_modes(list(fc), ['sinc', 'sinc']),
                                                                               signs=signs, init=m)
        q = np.maximum(_np(m.density_), 1e-300); q = q / q.sum(); P = pi / B + (1 - pi) * q
        r = (pi / B) / P; pi = float((N * r).sum() / N.sum()); P = pi / B + (1 - pi) * q; L = ll(P, N)
        if L - prev < tol: break
        prev = L
    return P, q, pi, m
field = np.sinc((Y - th[0]) / th[3]) * np.cos(np.pi * (Y - th[0]) / th[2])
sg = np.where(field >= 0, 1.0, -1.0)
FY = [0.012, 0.016, 0.02, 0.024, 0.03, 0.036, 0.044, 0.052, 0.064]; FX = [0.03, 0.04, 0.05, 0.065, 0.08]
res = dict(set=name, n=dict(train=int(TR.sum()), val=int(VA.sum()), test=int(TE.sum())),
           fraunhofer=dict(theta=th.tolist(), pi=pi_fr, d_over_a=float(th[3] / th[2]), fcy_phys=fcy_phys))
out = {'Fraunhofer': (Pfr, (Pfr - pi_fr / B) / (1 - pi_fr))}
for lab, signs in (('BLML real', None), ('BLML signed', sg)):
    t0 = time.time(); grid = {}
    for fy in FY:
        for fx in FX:
            grid[(fy, fx)] = ll(em_blml(TR, (fy, fx), signs)[0], VA)
    fc = max(grid, key=grid.get); P, q, pi, m = em_blml(TV, fc, signs)
    out[lab] = (P, q); res[lab] = dict(fc=list(fc), pi=pi, val_grid={f'{a},{b}': v for (a, b), v in grid.items()})
    np.save(f'results/dslit/g_{name}_{lab.split()[1]}.npy', _np(m.Se_)[0] / m.n_)
    print(f'{lab}: fc {fc} pi {pi:.3f} {time.time() - t0:.0f}s', flush=True)
out['histogram'] = ((TV + 0.5),) * 2
HG = [(a, b) for a in (0.5, 1, 1.5, 2, 3, 4, 6) for b in (0.5, 1, 1.5, 2, 3, 4)]
kde = lambda N, h: np.maximum(gaussian_filter(N, h, mode='constant'), 1e-12)
h = max(HG, key=lambda h_: ll(kde(TR, h_), VA)); out['KDE'] = (kde(TV, h),) * 2; res['KDE'] = dict(h=list(h))
def kde_bg(N, h, iters=200, tol=1e-9):
    pi, prev = 0.1, -np.inf; r = np.full(SH, pi)
    for _ in range(iters):
        q = kde(N * (1 - r), h); q = q / q.sum(); P = pi / B + (1 - pi) * q
        r = (pi / B) / P; pi = float((N * r).sum() / N.sum()); P = pi / B + (1 - pi) * q; L = ll(P, N)
        if L - prev < tol: break
        prev = L
    return P, q
h = max(HG, key=lambda h_: ll(kde_bg(TR, h_)[0], VA)); out['KDE + background'] = kde_bg(TV, h); res['KDE + background'] = dict(h=list(h))
w = TE * (1 - r_common)
for lab, (P, q) in out.items():
    res.setdefault(lab, {}).update(test_ll=ll(P, TE), test_ll_bgcorr=float((w * np.log(np.maximum(q / q.sum(), 1e-300))).sum() / w.sum()))
    print(f"{lab:18s} test LL {res[lab]['test_ll']:.5f}   background-corrected {res[lab]['test_ll_bgcorr']:.5f}", flush=True)
json.dump(res, open(f'results/dslit/fit_{name}.json', 'w'), indent=1)
np.savez_compressed(f'results/dslit/fit_{name}.npz', **{k.replace(' ', '_'): v[1] for k, v in out.items()}, field=field)
