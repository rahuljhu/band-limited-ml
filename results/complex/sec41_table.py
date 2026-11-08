"""Paper 3, Sec. 4.1: where complex BLML helps, ties or loses. Fixed fc = 2NA/lambda (the true band).
KL(true pixel probabilities || estimate) for histogram, real BLML, complex BLML, oracle-signed real BLML.
Truth: far field sampled at pixel centres (1-D slit pupil, 2-D disk pupil), in focus or defocused (2 rad)."""
import sys, json, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'results/psf')
import blml_trivial as bt, psfkit as pk
FC = pk.FC_PHYS; RHO = FC / 2
def truth_1d(case, z):
    nu = np.linspace(-RHO, RHO, 2001); dnu = nu[1] - nu[0]
    ph = 2.0 * (2 * (nu / RHO) ** 2 - 1) if case == 'defocus' else 0 * nu
    return (np.exp(1j * ph)[None] * np.exp(2j * np.pi * z[:, None] * nu[None])).sum(1) * dnu
def truth_2d(case, H, OS=8, NF=2048):
    k = np.fft.fftfreq(NF, d=1.0 / OS); KX, KY = np.meshgrid(k, k, indexing='ij'); r = np.hypot(KX, KY) / RHO
    ph = 2.0 * (2 * r ** 2 - 1) if case == 'defocus' else 0 * r
    h = np.fft.fftshift(np.fft.ifft2(np.where(r <= 1, np.exp(1j * ph), 0)))[::OS, ::OS]; c = h.shape[0] // 2
    return h[c - H:c + H + 1, c - H:c + H + 1].ravel()
rows = []
for dim, case, H in [(1, 'focus', 15), (1, 'focus', 150), (1, 'defocus', 15), (1, 'defocus', 150),
                     (2, 'focus', 15), (2, 'focus', 30), (2, 'defocus', 15), (2, 'defocus', 30)]:
    if dim == 1:
        Z = np.arange(-H, H + 1.0)[:, None]; h = truth_1d(case, Z[:, 0]); kinds = None; fc = [FC]
    else:
        g = np.arange(-H, H + 1.0); I, J = np.meshgrid(g, g, indexing='ij'); Z = np.c_[I.ravel(), J.ravel()]
        h = truth_2d(case, H); kinds = ['ball', 'ball']; fc = [FC, FC]
    p = np.abs(h) ** 2; p /= p.sum()
    for n in [10 ** 4, 10 ** 5]:
        for seed in range(3):
            N = np.random.default_rng(seed).multinomial(n, p)
            kl = lambda q: float((p * np.log(p / (np.maximum(q, 1e-300) / np.maximum(q, 1e-300).sum()))).sum())
            fit = lambda **kw: bt.BLMLQuick(dtype=np.float64, tol=1e-5).fit(Z, bt.with_lattice(bt.make_modes(fc, kinds), 1.0), sample_weight=N, **kw).predict(Z)
            r = dict(dim=dim, case=case, window=2 * H + 1, n=n, seed=seed, histogram=kl(N + 0.5), real=kl(fit()),
                     complex=kl(fit(complex_g=True)))
            if case == 'focus':
                r['oracle_signs'] = kl(fit(signs=np.sign(h.real + 1e-12)))
            rows.append(r); print(json.dumps(r), flush=True)
            json.dump(rows, open('results/complex/sec41.json', 'w'))
# LaTeX table: mean over seeds
import collections
agg = collections.defaultdict(list)
for r in rows:
    agg[(r['dim'], r['case'], r['window'], r['n'])].append(r)
L = [r'\begin{tabular}{llrrcccc}', r'\toprule',
     r'dim & truth & window & $n$ & histogram & real BLML & complex BLML & real, true signs \\', r'\midrule']
for (dim, case, w, n), rs in sorted(agg.items()):
    m = lambda key: np.mean([x[key] for x in rs]) if key in rs[0] else None
    vals = {k: m(k) for k in ['histogram', 'real', 'complex', 'oracle_signs']}
    best = min(v for v in vals.values() if v is not None)
    fmt = lambda v: '--' if v is None else (r'\textbf{%.4f}' % v if v == best else '%.4f' % v)
    win = f'{w}' if dim == 1 else f'{w}$^2$'
    L.append(f"{dim}-D & {'in focus' if case == 'focus' else 'defocus'} & {win} & $10^{int(np.log10(n))}$ & "
             + ' & '.join(fmt(vals[k]) for k in ['histogram', 'real', 'complex', 'oracle_signs']) + r' \\')
L += [r'\bottomrule', r'\end{tabular}']
open('papers/paper3/tables/sim_complex.tex', 'w').write('\n'.join(L) + '\n')
