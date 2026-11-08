"""Table I of paper 3 from results/complex/sec41_fullframe.json: mean KL over three data sets, whole-frame fits;
KL over the frame. Writes papers/submit/pra_complex/tables/sim_complex.tex."""
import json, collections, numpy as np
R = json.load(open('results/complex/sec41_fullframe.json')); agg = collections.defaultdict(list)
for r in R:
    agg[(r['dim'], r['case'], r['n'])].append(r)
K = ['histogram', 'real', 'complex', 'oracle_signs']
L = [r'\begin{tabular}{llrrcccc}', r'\toprule',
     r'dim & truth & frame & $n$ & histogram & real BLML & complex BLML & real, true signs \\', r'\midrule']
for (dim, case, n), rs in sorted(agg.items()):
    v = {k: (np.mean([x[k] for x in rs]) if k in rs[0] else None) for k in K}; best = min(x for x in v.values() if x is not None)
    f = lambda x: '--' if x is None else (r'\textbf{%.4f}' % x if x == best else '%.4f' % x)
    fr = rs[0]['frame']; fr = f'{fr}' if dim == 1 else f'{fr}$^2$'
    L.append(f"{dim}-D & {'in focus' if case == 'focus' else 'defocus'} & {fr} & $10^{int(np.log10(n))}$ & " + ' & '.join(f(v[k]) for k in K) + r' \\')
L += [r'\bottomrule', r'\end{tabular}']
open('papers/submit/pra_complex/tables/sim_complex.tex', 'w').write('\n'.join(L) + '\n')
