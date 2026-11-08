"""Crop test on the 53 beads (plane 5): window-normalised BLMLEmpirical (paper 7, ridge, lam on validation) vs
BLMLBinned real / complex (normalised on the plane) vs histogram (+a) and Gaussian blur (sigma on validation),
at 31 and 51 px windows. Split 70/10/20 by binomial thinning; score = test log-probability per photon with every
model normalised over the window. Output results/binned_local/bead_windows.json"""
import sys, json, time, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_binned import BLMLBinned, BLMLEmpirical
from scipy.ndimage import gaussian_filter
FC = 2 * 1.3 / 0.515 * 0.065
F = np.load('/private/tmp/claude-501/-Users-ra/e6cd45ce-b6f4-42cc-be9a-45c946facadf/scratchpad/gpu_data/psf_plane5.npy').astype(np.int64)
pos = np.load('results/psf/beads_iso.npy'); modes = bt.make_modes([FC, FC], ['ball', 'ball'])
sc = lambda P, M: float((M * np.log(np.maximum(P / P.sum(), 1e-300))).sum() / M.sum())
LAMS = [1e-3, 1e-2, 1e-1, 1, 10, 100]
rows = []
for k, (y0, x0) in enumerate(pos):
    r0 = np.random.default_rng(100 + k)
    for H in (15, 25):
        if not (H <= y0 < 2048 - H and H <= x0 < 2048 - H):
            continue
        C = F[y0 - H:y0 + H + 1, x0 - H:x0 + H + 1]; tr = r0.binomial(C, 0.7); rest = C - tr; va = r0.binomial(rest, 1 / 3); te = rest - va
        r = dict(bead=k, window=2 * H + 1, photons=int(C.sum())); t = time.time()
        e = BLMLEmpirical.select(tr, va, modes, LAMS, reg='ridge'); r['empirical'] = sc(e.density_, te); r['emp_lam'] = e.lam
        r['emp_plane_over_window'] = e.plane_norm_ / e.window_norm_
        for lab, cg in (('blml_real', False), ('blml_complex', True)):
            m = BLMLBinned(tol=1e-5, max_sweeps=20000).fit(tr, modes, complex_g=cg); r[lab] = sc(m.density_, te)
            r[lab + '_mass_in_window'] = float(m.density_.sum()); r[lab + '_conv'] = bool(m.converged_)
        a = max([0.01, 0.1, 0.3, 1, 3], key=lambda a: sc(tr + a, va)); r['histogram'] = sc(tr + a, te)
        sg = max([0.5, 0.7, 1.0, 1.4, 2.0], key=lambda s: sc(gaussian_filter(tr.astype(float), s, mode='nearest') + 1e-3, va))
        r['gauss_blur'] = sc(gaussian_filter(tr.astype(float), sg, mode='nearest') + 1e-3, te); r['sigma'] = sg
        r['time_s'] = time.time() - t; rows.append(r)
        print(json.dumps({q: (round(v, 5) if isinstance(v, float) else v) for q, v in r.items()}), flush=True)
    json.dump(rows, open('results/binned_local/bead_windows.json', 'w'), indent=1)
