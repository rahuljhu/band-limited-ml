"""Bead k, no background model: real rank 1, complex rank 1 (= real rank 2), complex rank 2 (= real rank 4).
Selected fc (validation, psfkit.select_fc) and fc = 2NA/lambda. Reports fc, test LL (reference only), certificate."""
from blml_trivial import with_lattice
import sys, json, time, numpy as np
sys.path.insert(0, 'results/psf'); import psfkit as pk, blml_trivial as bt
k = int(sys.argv[1]) if len(sys.argv) > 1 else 0; H = 15; Z = pk.lattice(H)
cnt, pos = pk.bead_counts(k, H); tr, va, te = pk.split(cnt); tv = tr + va
def fitter(kw):
    def f(Z, N, fc):
        modes = bt.make_modes(list(np.broadcast_to(fc, (2,))), d=2, rank=kw.get('rank', 1))
        m = bt.BLMLQuick(dtype=np.float64, tol=1e-5).fit(Z, with_lattice(modes, 1.0), sample_weight=np.asarray(N).ravel().astype(float), complex_g=kw.get('complex_g', False))
        P = m.predict(Z); return P / P.sum(), m
    return f
def cert(m, fc):
    C = m.e_hat_ if m.e_hat_.ndim == 2 else m.e_hat_[:, None]; D = (np.abs(C) ** 2).sum(1) / m.w_
    S = pk._K(m.Z_, fc); return float(np.linalg.eigvalsh(np.sqrt(D)[:, None] * S * np.sqrt(D)[None])[-1] / m.n_samples_)
res = dict(bead=k, photons=int(cnt.sum()))
for lab, kw in [('real_r1', {}), ('complex_r1', dict(complex_g=True)), ('complex_r2', dict(complex_g=True, rank=2))]:
    t0 = time.time(); fc, nf = pk.select_fc(Z, tr, va, fitter(kw))
    for tag, f in [('sel', fc), ('phys', (pk.FC_PHYS, pk.FC_PHYS))]:
        P, m = fitter(kw)(Z, tv, f)
        res[f'{lab}_{tag}'] = dict(fc=list(f), ll=pk.score(P, te), cert=cert(m, f), conv=bool(m.converged_))
    res[f'{lab}_s'] = time.time() - t0
    print(lab, {t: res[f'{lab}_{t}'] for t in ['sel', 'phys']}, f'{res[lab + "_s"]:.0f}s', flush=True)
res['histogram_ll'] = pk.score(pk.histogram(tv), te)
json.dump(res, open(f'results/psf/v4/bead{k}_rank.json', 'w'), indent=1)
