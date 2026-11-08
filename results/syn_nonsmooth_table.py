"""Kinks and edges (syn_nonsmooth_job.py, results/synns/syn): table and numbers for papers 1 and 2.
Writes papers/submit/{csda_highdim,jcgs_streaming}/tables/nonsmooth.tex and numbers_nonsmooth.tex. Run from the repo root."""
import json, glob, numpy as np
med = lambda v: float(np.median(v))
KDES = {'kde_scott': 'Scott', 'kde_lscv': 'LSCV', 'kde_lcv': 'LCV', 'kde_adaptive': 'adaptive'}
rows, wins, cells, conv, runs = [], {'L': 0, 'B': 0}, {'L': 0, 'B': 0}, 0, 0
for kind, title in (('L', 'Laplace (kink)'), ('B', 'bounded (edges)')):
    rows.append(f'\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}} \\\\')
    for d in (2, 4, 8):
        for n in (1000, 10000):
            R = json.load(open(f'results/synns/syn/{kind}_d{d}_n{n}.json'))['reps']
            one, st = med([x['blml_oneshot']['rel_ise'] for x in R]), med([x['blml_stream']['rel_ise'] for x in R])
            kd = {k: med([x[k]['rel_ise'] for x in R]) for k in KDES}; b = min(kd, key=kd.get)
            best = min(one, st, kd[b]); f = lambda v: f'\\textbf{{{v:.3g}}}' if v == best else f'{v:.3g}'
            rows.append(f'{d} & $10^{{{int(np.log10(n))}}}$ & {f(one)} & {f(st)} & {f(kd[b])} ({KDES[b]}) \\\\')
            cells[kind] += 1; wins[kind] += min(one, st) < kd[b]
            conv += sum(x['blml_stream']['converged'] for x in R); runs += len(R)
tab = ('\\begin{tabular}{rrrrl}\n\\toprule\n$d$ & $n$ & BLML, one-shot & BLML, streaming & best data-driven KDE \\\\\n\\midrule\n'
       + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')
num = dict(NsWinsL=wins['L'], NsWinsB=wins['B'], NsCells=cells['L'], NsConv=conv, NsRuns=runs)
for P in ('papers/submit/csda_highdim', 'papers/submit/jcgs_streaming'):
    open(f'{P}/tables/nonsmooth.tex', 'w').write(tab)
    open(f'{P}/numbers_nonsmooth.tex', 'w').write(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in num.items()))
print(tab, num)
