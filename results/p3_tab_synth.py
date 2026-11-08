"""Paper 3, Table II: complex BLML against the other methods on the synthetic truths of Fig. 1, at n = 1e4 and 1e5.
Uncropped (p3_ladder.py, results/p3_ladder): KDE, complex BLML with the band, with the two-slit kernel (shape and fill
by CV) and the full physical model (V = 1 for the amplitude ratio). On the window of seven fringes (p3_cbench.py +
p3_cbench_phase.py, results/p3_cbench_tap): KDE and the periodic two-slit kernel. Medians over the six data sets of a
truth, then the geometric mean over the truths a column uses: KL (all three truths; on the window, the KL on the
window), slit separation and width (relative; equal and unequal slits), relative slit phase (rad) and amplitude ratio
given V = 1 (unequal slits), fringe contrast (absolute; unequal slits and partially coherent). Run from the repo root."""
import json, glob, collections, numpy as np
PP = 'papers/submit/pra_complex'
A, B = collections.defaultdict(list), collections.defaultdict(list)
for f in glob.glob('results/p3_ladder/*.json'):
    d = json.load(open(f)); A[(d['case'], d['n'])].append(d)
for f in glob.glob('results/p3_cbench_tap/*.json'):
    d = json.load(open(f))
    for e, s in d['phase'].items():
        if e != 'truth':
            d[e].update(s)
    B[(d['case'], d['n'])].append(d)
gm = lambda v: (lambda v: float(np.exp(np.mean(np.log(v)))) if v else np.nan)([x for x in v if np.isfinite(x) and x > 0])
def med(R, c, n, e, m, f=lambda x: x):
    v = [f(d[e][m]) for d in R[(c, n)] if e in d and m in d[e]]; return np.median(v) if v else np.nan
ALL, CO, AS, GC = ['sym', 'asym', 'partial'], ['sym', 'asym'], ['asym'], ['asym', 'partial']
rho_l = lambda d: d                                                     # the ladder's rho_err is already |rho - 0.6|
COLS = ['KL', 'separation', 'width', 'phase (rad)', 'contrast', r'$\rho$ ($V=1$)']
UN = [('KDE', 'kde'), ('complex BLML, band', 'band'), ('complex BLML, two-slit kernel', 'shape'), ('full physical model', 'model_p')]
WI = [('KDE', 'kde'), ('complex BLML, periodic two-slit kernel', 'periodic')]
def row_un(e, n):
    return [gm([med(A, c, n, e, 'kl') for c in ALL]), gm([med(A, c, n, e, 'sep_err') for c in ALL]) if e != 'kde' else np.nan,
            gm([med(A, c, n, e, 'width_err') for c in CO]) if e != 'kde' else np.nan, med(A, 'asym', n, e, 'phase_err'),
            gm([med(A, c, n, e, 'gamma_err') for c in GC]), med(A, 'asym', n, e, 'rho_err')]
def row_wi(e, n):
    f = e != 'kde'
    return [gm([med(B, c, n, e, 'kl') for c in ALL]), gm([med(B, c, n, e, 'd_err') for c in ALL]) if f else np.nan, np.inf if f else np.nan,
            med(B, 'asym', n, e, 'phi_rel_err') if f else np.nan, gm([med(B, c, n, e, 'C_err') for c in GC]),
            med(B, 'asym', n, e, 'rho_err') if f else np.nan]
def fmt(x, j):
    if x == np.inf:
        return 'lost'
    if not np.isfinite(x):
        return '--'
    return f'{x:.1e}' if j == 0 else ((f'{100 * x:.0f}\\%' if x >= 0.1 else f'{100 * x:.2g}\\%') if j in (1, 2) else f'{x:.2g}')
T = [r'\begin{tabular}{ll' + 'r' * len(COLS) + '}', r'\toprule', r'& $n$ & ' + ' & '.join(COLS) + r' \\', r'\midrule',
     r'\multicolumn{%d}{l}{\emph{uncropped, 2048 pixels}} \\' % (len(COLS) + 2)]
for nm, e in UN:
    for i, n in enumerate((10 ** 4, 10 ** 5)):
        T.append(f"{nm if i == 0 else ''} & $10^{int(np.log10(n))}$ & " + ' & '.join(fmt(x, j) for j, x in enumerate(row_un(e, n))) + r' \\')
T += [r'\addlinespace', r'\multicolumn{%d}{l}{\emph{window of seven fringes, 412 pixels}} \\' % (len(COLS) + 2)]
for nm, e in WI:
    for i, n in enumerate((10 ** 4, 10 ** 5)):
        T.append(f"{nm if i == 0 else ''} & $10^{int(np.log10(n))}$ & " + ' & '.join(fmt(x, j) for j, x in enumerate(row_wi(e, n))) + r' \\')
T += [r'\bottomrule', r'\end{tabular}']
open(f'{PP}/tables/synth_compare.tex', 'w').write('\n'.join(T) + '\n'); print('\n'.join(T))
