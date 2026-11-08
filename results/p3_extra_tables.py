"""Paper 3: build-up and independence tables and numbers. Inputs: results/p3_buildup/<set>.json (p3_buildup.py),
results/p3indep/p3_indep/<set>.json (p3_independence.py on Colab). Writes papers/submit/pra_complex/tables/buildup.tex,
tables/indep.tex and numbers_extra.tex. Run from the repo root."""
import json, os, numpy as np
PP = 'papers/submit/pra_complex'; SETS = [('photon', 'single-mode photons'), ('ghost', 'ghost diffraction'), ('electron', 'electrons'),
                                          ('photon_mm', 'multimode photons')]
phys = {s: (lambda fr: (1 + fr['a_over_d']) / fr['P'])(json.load(open(f'results/dslit_1d/{s}/flat/{s}_fraunhofer.json'))['params']) for s, _ in SETS}
L = [r'\begin{tabular}{lrrrrr}', r'\toprule', r'record & $n$ & BLML & KDE & histogram & $f_c$ / physical \\', r'\midrule']
for s, lab in SETS:
    f = f'results/p3_buildup/{s}.json'
    if not os.path.exists(f):
        continue
    rows = [r for r in json.load(open(f))['rows'] if r['n'] >= 100]
    for i, r in enumerate(rows):
        b = max(r['blml'], r['kde'], r['hist']); cell = lambda v: (r'\textbf{%.4f}' % v) if v == b else '%.4f' % v
        L.append(f"{lab if i == 0 else ''} & {r['n']} & {cell(r['blml'])} & {cell(r['kde'])} & {cell(r['hist'])} & {r['fc'] / phys[s]:.2f} \\\\")
    L.append(r'\addlinespace')
L = L[:-1] + [r'\bottomrule', r'\end{tabular}']
open(f'{PP}/tables/buildup.tex', 'w').write('\n'.join(L) + '\n')
num = {}; bounds = []
T = [r'\begin{tabular}{llrrrrr}', r'\toprule', r'record & pair & pairs & gain & null mean $\pm$ sd & $p$ \\', r'\midrule']
for s, lab in [x for x in SETS if x[0] != 'ghost']:
    f = f'results/p3indep/p3_indep/{s}.json'
    if not os.path.exists(f):
        continue
    rows = json.load(open(f))['rows']; ps = [r['p'] for r in rows]
    tag = dict(photon='Ph', electron='El', photon_mm='Mm')[s]
    num[f'Ind{tag}Bound'] = f"{max(r['gain'] - r['null_mean'] + 1.645 * r['null_sd'] for r in rows):.2f}"
    num[f'Ind{tag}Pmin'] = f'{min(ps):.2f}'; num[f'Ind{tag}Pmax'] = f'{max(ps):.2f}'
    for i, r in enumerate(rows):
        bounds.append(r['gain'] - r['null_mean'] + 1.645 * r['null_sd'])
        pair = 'same frame' if r['same_frame'] else f"{r['lag']} frame{'s' if r['lag'] > 1 else ''} apart"
        T.append(f"{lab if i == 0 else ''} & {pair} & {r['pairs']} & {r['gain']:.4f} & {r['null_mean']:.4f} $\\pm$ {r['null_sd']:.4f} & {r['p']:.2f} \\\\")
    T.append(r'\addlinespace')
T = T[:-1] + [r'\bottomrule', r'\end{tabular}']
open(f'{PP}/tables/indep.tex', 'w').write('\n'.join(T) + '\n')
num['IndBound'] = f'{max(bounds):.2f}' if bounds else '--'
for k in ('IndElPmin', 'IndElPmax', 'IndMmPmin', 'IndMmPmax', 'IndPhPmin', 'IndPhPmax', 'IndPhBound', 'IndElBound', 'IndMmBound'):
    num.setdefault(k, '--')
open(f'{PP}/numbers_extra.tex', 'w').write(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in num.items()))
print(num)
