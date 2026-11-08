"""Paper 3, Sec. 6 table from results/complex/zstack/bead*.json: mean held-out log-probability per photon by test
plane, and the number of beads on which joint BLML scores higher; median fit times."""
import json, glob, numpy as np
R = [json.load(open(f)) for f in sorted(glob.glob('results/complex/zstack/bead*.json'))]
M = [('blml_stack', 'joint BLML'), ('ml_zernike', 'Zernike ML'), ('ml_pixel', 'pixel ML'), ('gs', 'GS'),
     ('blml_focus', 'one plane'), ('neighbour_hist', 'histogram')]
L = [r'\begin{tabular}{l' + 'r' * len(M) + '}', r'\toprule', 'test plane & ' + ' & '.join(n for _, n in M) + r' \\',
     r'\midrule']
for p in ['2', '4', '6', '8', 'all']:
    ps = ['2', '4', '6', '8'] if p == 'all' else [p]
    X = np.array([[np.mean([r['planes'][q][m] for q in ps]) for m, _ in M] for r in R])
    lab = {'2': r'2 ($-1$)', '4': r'4 (between)', '6': r'6 (between)', '8': r'8 ($+1$)', 'all': 'mean'}[p]
    cells = [f'{X[:, 0].mean():.4f}'] + [f'{X[:, j].mean():.4f} ({(X[:, 0] > X[:, j]).sum()})' for j in range(1, len(M))]
    L.append((r'\midrule ' if p == 'all' else '') + lab + ' & ' + ' & '.join(cells) + r' \\')
t = {m: np.median([r['time_s'][m] for r in R]) for m in ['blml_stack', 'ml_zernike', 'ml_pixel', 'gs']}
L.append(r'\midrule median time (s) & ' + ' & '.join(f'{t[m]:.1f}' if m in t else '--' for m, _ in M) + r' \\')
L += [r'\bottomrule', r'\end{tabular}']
open('papers/paper3/tables/zstack.tex', 'w').write('\n'.join(L) + '\n')
X = np.array([[np.mean([r['planes'][q][m] for q in '2468']) for m, _ in M] for r in R])
for j in (1, 5):
    d = X[:, 0] - X[:, j]; print(M[j][0], f'{d.mean():.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}')
print(len(R), 'beads')
