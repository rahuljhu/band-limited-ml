"""Paper 3, Sec. 3.4: rounding complex BLML to a real square root. In-focus truths of sec41 (real h with sign changes),
fixed fc = 2NA/lambda. Complex fit -> remove the global phase (theta = arg sum_b w_b g_b^2 / 2) -> signs of Re -> real
BLML refit in that orthant. Reports sign agreement with the truth (occupied bins, photon-weighted, up to a global sign),
the truth intensity at mismatched bins, and KL of complex / rounded refit / true-sign refit."""
import sys, json, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'results/psf')
import blml_trivial as bt
exec(open('results/complex/sec41_table.py').read().split('rows = []')[0])
rows = [] if '--table' not in sys.argv else json.load(open('results/complex/sec44.json'))
for dim, H in ([] if '--table' in sys.argv else [(1, 15), (1, 150), (2, 15), (2, 30)]):
    if dim == 1:
        Z = np.arange(-H, H + 1.0)[:, None]; h = truth_1d('focus', Z[:, 0]); kinds = None; fc = [FC]
    else:
        g = np.arange(-H, H + 1.0); I, J = np.meshgrid(g, g, indexing='ij'); Z = np.c_[I.ravel(), J.ravel()]
        h = truth_2d('focus', H); kinds = ['ball', 'ball']; fc = [FC, FC]
    h = h.real; p = h ** 2; p /= p.sum()
    for n in [10 ** 4, 10 ** 5]:
        for seed in range(3):
            N = np.random.default_rng(seed).multinomial(n, p)
            kl = lambda q: float((p * np.log(p / (np.maximum(q, 1e-300) / np.maximum(q, 1e-300).sum()))).sum())
            mk = lambda: bt.with_lattice(bt.make_modes(fc, kinds), 1.0)
            mc = bt.BLMLQuick(dtype=np.float64, tol=1e-5).fit(Z, mk(), sample_weight=N, complex_g=True)
            gz = mc.sqrt_density(Z)
            th = 0.5 * np.angle((N * gz ** 2).sum()); sg = np.sign((np.exp(-1j * th) * gz).real + 1e-300)
            occ = N > 0; st = np.sign(h + 1e-12)
            agree = (sg[occ] == st[occ]).mean(); agree = max(agree, 1 - agree)
            flip = 1.0 if (sg[occ] == st[occ]).mean() >= 0.5 else -1.0
            mis = occ & (flip * sg != st)
            fr = lambda s: bt.BLMLQuick(dtype=np.float64, tol=1e-5).fit(Z, mk(), sample_weight=N, signs=s).predict(Z)
            r = dict(dim=dim, window=2 * H + 1, n=n, seed=seed, agree_bins=float(agree),
                     agree_photons=float(N[~mis & occ].sum() / N[occ].sum()), n_mis=int(mis.sum()),
                     mis_rel_intensity=float(p[mis].max() / p.max()) if mis.any() else 0.0,
                     complex=kl(mc.predict(Z)), rounded=kl(fr(sg)), oracle=kl(fr(st)))
            rows.append(r); print(json.dumps(r), flush=True)
            json.dump(rows, open('results/complex/sec44.json', 'w'))
# LaTeX table: mean over seeds (rerun with --table to rebuild from sec44.json)
import collections
agg = collections.defaultdict(list)
for r in rows:
    agg[(r['dim'], r['window'], r['n'])].append(r)
L = [r'\begin{tabular}{lrrccccc}', r'\toprule',
     r'dim & window & $n$ & \shortstack{signs agree\\(occupied bins)} & \shortstack{signs agree\\(photons)} & complex & rounded refit & true signs \\',
     r'\midrule']
for (dim, w, n), rs in sorted(agg.items()):
    m = lambda k: np.mean([x[k] for x in rs])
    win = f'{w}' if dim == 1 else f'{w}$^2$'
    L.append(f"{dim}-D & {win} & $10^{int(np.log10(n))}$ & {100 * min(x['agree_bins'] for x in rs):.1f}--{100 * max(x['agree_bins'] for x in rs):.1f}\\,\\% & "
             f"{100 * min(x['agree_photons'] for x in rs):.2f}\\,\\% & {m('complex'):.4f} & {m('rounded'):.4f} & {m('oracle'):.4f} \\\\")
L += [r'\bottomrule', r'\end{tabular}']
open('papers/paper3/tables/round.tex', 'w').write('\n'.join(L) + '\n')
