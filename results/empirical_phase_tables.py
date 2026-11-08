"""papers/paper7/tables/phase.tex and phase_conv.tex from results/empirical_phase_table.json / empirical_phase_conv.json."""
import json, os
T = json.load(open('results/empirical_phase_table.json')); C = json.load(open('results/empirical_phase_conv.json'))
os.makedirs('papers/paper7/tables', exist_ok=True)
rows = [('Truth', 'truth', None), ('BLML, real $g \\ge 0$', 'BLML real', None), ('BLML, complex', 'BLML complex', None),
        ('Empirical, $+\\sqrt h$', '+sqrt h', None), ('Empirical, signed', 'signed', 'outer'),
        ('Empirical, complex', 'complex', 'outer'), ('Empirical, real rank 2', 'real rank 2', 'outer')]
cols = list(T)
val = lambda r, k: r[k] if isinstance(r[k], float) else r[k]['test']
L = ['\\begin{table}[t]\\centering\\small',
     '\\caption{Held-out log-likelihood per count (higher is better) on sign-changing synthetic truths; in brackets the '
     'outer steps of the alternation, and the difference to the truth.}\\label{tab:phase}',
     '\\begin{tabular}{l' + 'rr' * len(cols) + '}\\toprule',
     ' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{c}}}' for c in cols) + ' \\\\',
     ' & ' + ' & '.join(['score', 'vs truth'] * len(cols)) + ' \\\\ \\midrule']
for lab, key, extra in rows:
    cells = []
    for c in cols:
        r = T[c]; v = val(r, key); o = f" ({r[key]['outer']})" if extra and r[key]['outer'] else ''
        cells += [f'{v:.5f}{o}', '--' if key == 'truth' else f"{v - T[c]['truth']:+.5f}"]
    L.append(f'{lab} & ' + ' & '.join(cells) + ' \\\\')
L += ['\\bottomrule\\end{tabular}\\end{table}']
open('papers/paper7/tables/phase.tex', 'w').write('\n'.join(L) + '\n')
L = ['\\begin{table}[t]\\centering\\small',
     '\\caption{Convergence of the alternation (complex rank 1, ridge $\\lambda = 10^{-3}$, 1-d truth): objective, its '
     'last decrease, the gradient norm of the reduced objective, and $\\min_b \\|g_b\\|/\\sqrt{h_b}$ over occupied bins '
     '(no bin near the kink at $g_b = 0$).}\\label{tab:phase-conv}',
     '\\begin{tabular}{rrrrr}\\toprule outer steps & $J$ & last decrease & $\\|\\nabla J\\|$ & $\\min \\|g_b\\|/\\sqrt{h_b}$ \\\\ \\midrule']
for r in C:
    L.append(f"{r['outer']:,} & {r['J']:.10f} & {r['dJ']:.1e} & {r['grad']:.1e} & {r['min_ratio']:.2f} \\\\")
L += ['\\bottomrule\\end{tabular}\\end{table}']
open('papers/paper7/tables/phase_conv.tex', 'w').write('\n'.join(L) + '\n')
