"""Paper 3, Sec. 4.1 on the whole frame: the same truths as sec41_table.py (far field of a slit / disk pupil, in focus
or with 2 rad of defocus, fixed fc = 2NA/lambda), but photons are drawn from the whole frame and every estimator is fitted
to the whole frame (BLMLBinned, the full-lattice FFT engine), so no window cuts the pattern. A window truncates the
density, and a truncated density is not band-limited: BLML pays for the cut, the histogram does not.
KL(true pixel probabilities || estimate) over the frame, and over the central window (both renormalised there) for
comparison with the windowed table. Frame: 1-D 2049 px, 2-D 255^2 px.
Run from the repo root: OMP_NUM_THREADS=4 nice .venv/bin/python results/complex/sec41_fullframe.py"""
import sys, json, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'results/psf')
import blml_trivial as bt, psfkit as pk
from blml_binned import BLMLBinned
from sec41_table_truth import truth_1d, truth_2d
FC = pk.FC_PHYS
OUT = 'results/complex/sec41_fullframe.json'


def kl(p, q):
    q = np.maximum(q, 1e-300); q = q / q.sum(); m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


def central(a, H):
    if a.ndim == 1:
        c = len(a) // 2; return a[c - H:c + H + 1]
    c = a.shape[0] // 2; return a[c - H:c + H + 1, c - H:c + H + 1]


rows = []
for dim, case, F in [(1, 'focus', 1024), (1, 'defocus', 1024), (2, 'focus', 127), (2, 'defocus', 127)]:
    if dim == 1:
        h = truth_1d(case, np.arange(-F, F + 1.0)); modes = bt.make_modes([FC], ['sinc'])
    else:
        h = truth_2d(case, F).reshape(2 * F + 1, 2 * F + 1); modes = bt.make_modes([FC, FC], ['ball', 'ball'])
    p = np.abs(h) ** 2; p /= p.sum()
    for n in [10 ** 4, 10 ** 5]:
        for seed in range(3):
            N = np.random.default_rng(seed).multinomial(n, p.ravel()).reshape(p.shape).astype(float)
            fit = lambda **kw: np.asarray(BLMLBinned(engine='cpu', tol=1e-6, max_sweeps=20000).fit(N, modes, **kw).density_)
            est = dict(histogram=N + 0.5, real=fit(), complex=fit(complex_g=True))
            if case == 'focus':
                est['oracle_signs'] = fit(signs=np.sign(h.real + 1e-12))
            r = dict(dim=dim, case=case, frame=2 * F + 1, n=n, seed=seed)
            for k, q in est.items():
                r[k] = kl(p, q)
                for H in (15, 30, 150):
                    if H < F:
                        pc = central(p, H); r[f'{k}_w{2 * H + 1}'] = kl(pc / pc.sum(), central(q, H))
            rows.append(r); print(json.dumps(r), flush=True)
            json.dump(rows, open(OUT, 'w'))
