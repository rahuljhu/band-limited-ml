"""Focus plane: background as a second band-limited mode instead of EM. Complex rank 1 + uniform floor by EM
(psfkit.blml_complex_bg) vs complex rank 2 with per-mode kernels: mode 0 the pupil band (ball, fc = 2NA/lambda),
mode 1 a ball of small fc_bg (the background; fc_bg -> 0 is the uniform floor), weights from the likelihood, no EM.
Also complex rank 1 without any background. Thinning split of psfkit (train+val fitted, test scored).
Usage: python results/psf/bead_bgmode.py [bead ...]; output results/psf/bgmode.json"""
import sys, json, time, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'results/psf')
import psfkit as pk, blml_trivial as bt
H = 15; Z = pk.lattice(H); FC = pk.FC_PHYS; KB = ['ball', 'ball']; FB = [0.03, 0.06, 0.12]
def rank2(N, fb):
    modes = bt.with_lattice(bt.make_modes([[FC, FC], [fb, fb]], KB), 1.0)
    m = bt.BLMLQuick(dtype=np.float64, tol=1e-5).fit(Z, modes, sample_weight=np.asarray(N).ravel().astype(float), complex_g=True)
    P = m.predict(Z)
    return P / P.sum(), m
rows = []
for k in (map(int, sys.argv[1:]) if len(sys.argv) > 1 else range(53)):
    cnt, _ = pk.bead_counts(k, H); tr, va, te = pk.split(cnt); tv = tr + va
    r = dict(bead=k); T = {}
    t = time.time(); P, pi = pk.blml_complex_bg(Z, tv, (FC, FC), kinds=KB); T['r1_em'] = time.time() - t
    r['r1_em'] = pk.score(P, te); r['pi_em'] = pi; r['em_iters'] = pk.blml_complex_bg.n_iter
    t = time.time(); P, _ = pk.blml_complex(Z, tv, (FC, FC), kinds=KB); T['r1'] = time.time() - t; r['r1'] = pk.score(P, te)
    for fb in FB:
        t = time.time(); P, m = rank2(tv, fb); T[f'r2_{fb}'] = time.time() - t
        r[f'r2_{fb}'] = pk.score(P, te); r[f'conv_{fb}'] = [bool(m.converged_), int(m.n_sweeps_), float(m.residual_)]
    r['time_s'] = T; rows.append(r); print(json.dumps(r), flush=True)
    json.dump(rows, open('results/psf/bgmode.json', 'w'))
