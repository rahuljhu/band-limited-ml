"""Re-run of the paper-7 estimator in bead_windows.py with the lam grid extended downwards (same splits)."""
import sys, json, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_binned import BLMLEmpirical
FC = 2 * 1.3 / 0.515 * 0.065
F = np.load('/private/tmp/claude-501/-Users-ra/e6cd45ce-b6f4-42cc-be9a-45c946facadf/scratchpad/gpu_data/psf_plane5.npy').astype(np.int64)
pos = np.load('results/psf/beads_iso.npy'); modes = bt.make_modes([FC, FC], ['ball', 'ball'])
sc = lambda P, M: float((M * np.log(np.maximum(P / P.sum(), 1e-300))).sum() / M.sum())
LAMS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
R = json.load(open('results/binned_local/bead_windows.json')); idx = {(r['bead'], r['window']): r for r in R}
for k, (y0, x0) in enumerate(pos):
    r0 = np.random.default_rng(100 + k)
    for H in (15, 25):
        if not (H <= y0 < 2048 - H and H <= x0 < 2048 - H):
            continue
        C = F[y0 - H:y0 + H + 1, x0 - H:x0 + H + 1]; tr = r0.binomial(C, 0.7); rest = C - tr; va = r0.binomial(rest, 1 / 3); te = rest - va
        r = idx[(k, 2 * H + 1)]
        for reg in ('ridge', 'l2'):
            e = BLMLEmpirical.select(tr, va, modes, LAMS, reg=reg, max_iter=5000)
            r[f'emp_{reg}'] = sc(e.density_, te); r[f'emp_{reg}_lam'] = e.lam; r[f'emp_{reg}_path'] = e.path_
    print(k, flush=True)
json.dump(R, open('results/binned_local/bead_windows.json', 'w'), indent=1)
