"""Paper 3, bead k: real vs complex BLML, each with the EM background (uniform over the window),
cut-off selected by the selector's search on validation counts (no grid; fc <= 1/px), for the
separable sinc x sinc kernel and the radial (ball, circular pupil) kernel.
Records fc vs 2NA/lambda, background fraction pi, global-optimality certificate of the final
M-step fit, test log-likelihood (sanity check only) and the fitted coefficients (pupil figure).
Usage: python run_bead_v5.py <k>"""
from blml_trivial import with_lattice
import sys, json, time, numpy as np
sys.path.insert(0, 'results/psf'); sys.path.insert(0, '.')
import psfkit as pk, blml_trivial as bt, selector as sel
from blml_trivial import make_modes
k = int(sys.argv[1]); H = 15; Z = pk.lattice(H)
cnt, pos = pk.bead_counts(k, H); tr, va, te = pk.split(cnt)
FACTORS = (np.sqrt(2.0), 2.0 ** 0.25, 2.0 ** 0.125)
def em_bg(N, fc, kinds, complex_g, iters=60, tol=1e-7, warm=True):
    """P_b = pi/B + (1 - pi) q_b, q = BLML density at the pixel centres normalised over the window."""
    N = np.asarray(N).ravel().astype(float); B = len(Z); pi = 0.1; r = np.full(B, pi); prev = -np.inf; m = None
    for _ in range(iters):
        w = N * (1 - r)
        m = bt.BLMLQuick(dtype=np.float64, tol=1e-5).fit(Z, with_lattice(make_modes(list(fc), kinds, None), 1.0), sample_weight=np.where(w > 1e-6, w, 0), complex_g=complex_g, init=m if warm else None)
        q = np.maximum(m.predict(Z), 1e-300); q = q / q.sum(); P = pi / B + (1 - pi) * q
        r = (pi / B) / P; pi = float((N * r).sum() / N.sum()); P = pi / B + (1 - pi) * q
        ll = pk.score(P, N)
        if ll - prev < tol: break
        prev = ll
    return P / P.sum(), pi, m
def certificate(m, kinds, fc):
    C = m.e_hat_ if m.e_hat_.ndim == 2 else m.e_hat_[:, None]; D = (np.abs(C) ** 2).sum(1) / m.w_
    S = m._fit_kernel(m.Z_, m.Z_)
    return float(np.linalg.eigvalsh(np.sqrt(D)[:, None] * S * np.sqrt(D)[None])[-1] / m.n_samples_)
res = dict(bead=k, pos=pos, photons=int(cnt.sum()), fc_phys=pk.FC_PHYS)
coef = {}
for kname, kinds in [('ball', ['ball', 'ball'])]:      # circular pupil: one fc
    for model, cg in [('real', False), ('complex', True)]:
        lab = f'{model}_{kname}'; t0 = time.time()
        def obj(fcv):
            if np.any(np.asarray(fcv) > 1.0 + 1e-9): return -np.inf
            P, _, _ = em_bg(tr, fcv, kinds, cg); return pk.score(P, va)
        fc, vll = sel._search(obj, 2, kinds, [None, None], fc0=0.1, tag=f'b{k}:{lab}', factors=FACTORS)
        P, pi, m = em_bg(tr + va, fc, kinds, cg)
        res[lab] = dict(fc=np.asarray(fc).tolist(), val_ll=vll, test_ll=pk.score(P, te), pi=pi,
                        cert=certificate(m, kinds, fc), conv=bool(m.converged_), s=time.time() - t0)
        coef[lab + '_e'] = m.e_hat_; coef[lab + '_Z'] = m.Z_; coef[lab + '_w'] = m.w_
        print(k, lab, {kk: (np.round(v, 4).tolist() if isinstance(v, (list, np.ndarray)) else v) for kk, v in res[lab].items()}, flush=True)
res['histogram_test_ll'] = pk.score(pk.histogram(tr + va), te)
json.dump(res, open(f'results/psf/v5/bead{k}.json', 'w'), indent=1)
np.savez_compressed(f'results/psf/v5/bead{k}_coef.npz', **coef)
