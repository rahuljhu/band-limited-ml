"""Paper-3 table: complex BLML vs exhaustive orthant search (small n)."""
import json, numpy as np
R = json.load(open('results/complex/bqp_exhaustive_partial.json'))
L = [r'\begin{tabular}{rrcccc}', r'\toprule',
     r'$n$ & data sets & \shortstack{complex = best\\orthant} & \shortstack{complex $\hat g$\\real} & \shortstack{same signs as\\best orthant} & \shortstack{positive orthant\\optimal} \\', r'\midrule']
for n in sorted({r['n'] for r in R}):
    rs = [r for r in R if r['n'] == n]; k = len(rs)
    eq = sum(abs(r['complex'] - r['best_orthant']) < 1e-5 for r in rs)
    real = sum(r['complex_realness'] < 1e-6 for r in rs)
    same = sum(bool(r['complex_matches_best_signs']) for r in rs)
    pos = sum(abs(r['all_positive'] - r['best_orthant']) < 1e-5 for r in rs)
    gap = max(r['best_orthant'] - r['all_positive'] for r in rs)
    L.append(f'{n} & {k} & {eq}/{k} & {real}/{k} & {same}/{k} & {pos}/{k} (max gap {gap:.2f}) \\\\')
L += [r'\bottomrule', r'\end{tabular}']
open('papers/paper3/tables/orthant.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
