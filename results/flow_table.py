"""Neural spline flow against BLML (cut-off chosen from the data) and the best data-driven KDE on the synthetic truths
(flow: results/flow/syn; BLML one-shot and KDEs: results/p2syn5/syn for T1, T3 and results/synns/syn for L, B).
Writes papers/submit/jmlr_blml/tables/flow.tex. Run from the repo root."""
import json, glob, numpy as np
med = lambda v: float(np.median(v))
K = ('kde_scott', 'kde_lscv', 'kde_lcv', 'kde_adaptive')
F = {(r['truth'], r['d'], r['n']): med([x['rel_ise'] for x in r['reps']])
     for r in (json.load(open(f)) for f in glob.glob('results/flow/syn/*.json'))}
B = {}
for f in glob.glob('results/p2syn5/syn/*.json') + glob.glob('results/synns/syn/*.json'):
    r = json.load(open(f)); R = r['reps']; one = 'blml_auto' if 'blml_auto' in R[0] else 'blml_oneshot'
    B[(r['truth'], r['d'], r['n'])] = (med([x[one]['rel_ise'] for x in R]), min(med([x[k]['rel_ise'] for x in R]) for k in K))
TITLE = {'T1': 'band-limited, separable (T1)', 'T3': 'correlated normal (T3)', 'L': 'Laplace (kink)', 'B': 'bounded (edges)'}
rows = []
for t in ('T1', 'T3', 'L', 'B'):
    rows.append(f'\\multicolumn{{5}}{{l}}{{\\textit{{{TITLE[t]}}}}} \\\\')
    for key in sorted(k for k in F if k[0] == t and k in B):
        fl, (bl, kd) = F[key], B[key]; best = min(fl, bl, kd)
        f = lambda v: f'\\textbf{{{v:.2g}}}' if v == best else f'{v:.2g}'
        rows.append(f'{key[1]} & $10^{{{int(np.log10(key[2]))}}}$ & {f(bl)} & {f(fl)} & {f(kd)} \\\\')
tab = ('\\begin{tabular}{rrrrr}\n\\toprule\n$d$ & $n$ & BLML & neural spline flow & best KDE \\\\\n\\midrule\n'
       + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')
open('papers/submit/jmlr_blml/tables/flow.tex', 'w').write(tab); print(tab)
