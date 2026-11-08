"""Paper 3, the cropped benchmark (p3_cbench.py + p3_cbench_phase.py, flat background N = 0: results/p3_cbench_n0):
figure figs/cbench.pdf and table tables/cbench.tex. Panels against n, the median over the 6 data sets per point
(background 0 and 0.1, three each): (a) KL from the window's truth (geometric mean over the cases); (b) slit
separation, relative error; (c) fringe contrast, absolute error; (d) slit phase about the fitted envelope centre
(equal and unequal slits); (e) amplitude ratio (unequal slits: the coherence prior); (f) the share of data sets in
which the line kernel's selected fill is the plain band (fill 1). Methods: the periodic two-slit kernel, the line
two-slit kernel, the KDE; for the partially coherent truth the periodic three-mode model where it applies.
Run from the repo root:  python results/p3_fig_cbench.py [dir]"""
import sys, json, glob, collections, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
PP, D = 'papers/submit/pra_complex', sys.argv[1] if len(sys.argv) > 1 else 'results/p3_cbench_n0'
R = collections.defaultdict(list)
for f in glob.glob(f'{D}/*.json'):
    d = json.load(open(f))
    if 'phase' in d:
        for e, s in d['phase'].items():
            if e != 'truth':
                d[e].update(s)
        R[(d['case'], d['n'])].append(d)
CASES = [('sym', 'equal slits'), ('asym', 'unequal, phase'), ('partial', 'partially coherent')]
NS = sorted({n for _, n in R})
M = [('periodic', 'periodic two-slit kernel', 'C3', 's', '-'), ('line', 'line two-slit kernel', 'C0', 'o', '--'),
     ('kde', 'KDE', '0.4', 'x', ':'), ('periodic_three', 'periodic, three modes', 'C2', 'd', '-.')]
med = lambda c, n, e, m: np.median([d[e][m] for d in R[(c, n)] if e in d and m in d[e]]) if any(
    e in d and m in d[e] for d in R[(c, n)]) else np.nan
gm = lambda v: (lambda v: np.exp(np.mean(np.log(v))) if v else np.nan)([x for x in v if np.isfinite(x) and x > 0])
fig, ax = plt.subplots(2, 3, figsize=(7.0, 4.4))
def panel(a, metric, cases, ylab, title, skip=(), log=True):
    for e, nm, col, mk, ls in M:
        if e in skip:
            continue
        y = [gm([med(c, n, e, metric) for c in cases]) for n in NS]
        if not np.all(np.isnan(y)):
            (a.loglog if log else a.semilogx)(NS, y, color=col, marker=mk, ls=ls, lw=1.1, ms=3.5, label=nm)
    a.set_title(title, fontsize=8, loc='left'); a.set_xlabel('photons $n$'); a.set_ylabel(ylab, fontsize=7.5)
allc = [c for c, _ in CASES]
panel(ax[0, 0], 'kl', allc, 'KL from the truth', '(a) intensity on the window')
panel(ax[0, 1], 'd_err', allc, 'relative error', '(b) slit separation', skip=('kde',))
panel(ax[0, 2], 'C_err', allc, 'absolute error', '(c) fringe contrast')
panel(ax[1, 0], 'phi_env_err', ['sym', 'asym'], 'error (rad)', '(d) slit phase, no axis given', skip=('kde', 'periodic_three'))
panel(ax[1, 1], 'rho_err', ['asym'], 'absolute error', '(e) amplitude ratio (coherent)', skip=('kde', 'periodic_three'))
a = ax[1, 2]
for (c, lab), shade, mk in zip(CASES, ('0.0', '0.35', '0.65'), ('o', 's', '^')):
    a.semilogx(NS, [np.mean([d['line']['fill'] == 1.0 for d in R[(c, n)]]) for n in NS], color=shade, marker=mk, ms=3.5, lw=1, label=lab)
a.set_ylim(-0.05, 1.05); a.set_xlabel('photons $n$'); a.set_ylabel('share of data sets', fontsize=7.5)
a.set_title('(f) line kernel picks fill 1', fontsize=8, loc='left'); a.legend(fontsize=6, frameon=False, loc='upper left')
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))
fig.savefig(f'{PP}/figs/cbench.pdf'); fig.savefig(f'{PP}/figs/cbench.png', dpi=200)
# table: medians per case and n
cols = [('periodic', 'kl', 'KL per.', '{:.1e}'), ('line', 'kl', 'KL line', '{:.1e}'), ('kde', 'kl', 'KL KDE', '{:.1e}'),
        ('periodic', 'd_err', '$d$ per.', '{:.3f}'), ('line', 'd_err', '$d$ line', '{:.3f}'),
        ('periodic', 'C_err', '$C$ per.', '{:.3f}'), ('kde', 'C_err', '$C$ KDE', '{:.3f}'),
        ('periodic', 'phi_env_err', r'$\phi$ per.', '{:.2f}'), ('periodic', 'rho_err', r'$\rho$ per.', '{:.2f}'),
        ('periodic_three', 'V_hat', r'$\hat V$ three', '{:.2f}')]
L = [r'\begin{tabular}{ll' + 'r' * len(cols) + '}', r'\toprule', 'case & $n$ & ' + ' & '.join(c[2] for c in cols) + r' \\', r'\midrule']
for c, lab in CASES:
    for i, n in enumerate(NS):
        v = [med(c, n, e, m) for e, m, _, _ in cols]
        L.append(f"{lab if i == 0 else ''} & $10^{int(np.log10(n))}$ & " + ' & '.join('--' if not np.isfinite(x) else f.format(x) for x, (_, _, _, f) in zip(v, cols)) + r' \\')
    L.append(r'\addlinespace')
L = L[:-1] + [r'\bottomrule', r'\end{tabular}']
open(f'{PP}/tables/cbench.tex', 'w').write('\n'.join(L) + '\n')
print('done', sum(len(v) for v in R.values()), 'data sets')
