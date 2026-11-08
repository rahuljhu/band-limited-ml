"""Paper 3, Table III: every method on the records' windows, the periodic two-slit kernel the reference
(p3_window_table.py, results/p3_window_table): its test log-likelihood per particle, and each other method's paired
difference to it (+- SE) on the paper's split. -> papers/submit/pra_complex/tables/dslit_window.tex. Run from the
repo root."""
import json
PP = 'papers/submit/pra_complex'
SETS = [('photon', 'photons, single mode'), ('ghost', 'ghost diffraction'), ('electron', 'electrons'), ('photon_mm', 'photons, multimode')]
ROWS = [('periodic_pub', 'periodic two-slit kernel, published slits'), ('band', 'periodic band (no aperture)'),
        ('kde', 'KDE'), ('akde', 'adaptive KDE'), ('fresnel', 'Fresnel model'), ('fraunhofer', 'Fraunhofer model'), ('hist', 'histogram')]
R = {s: json.load(open(f'results/p3_window_table/{s}.json')) for s, _ in SETS}
T = [r'\begin{tabular}{l' + 'r' * len(SETS) + '}', r'\toprule', ' & ' + ' & '.join(l for _, l in SETS) + r' \\',
     'window (pixels) / test particles & ' + ' & '.join(f"{R[s]['L']} / {R[s]['n_test']:,}" for s, _ in SETS) + r' \\', r'\midrule',
     'periodic two-slit kernel, test LL & ' + ' & '.join(f"${R[s]['periodic']['test']:.4f}$" for s, _ in SETS) + r' \\', r'\addlinespace',
     r'\multicolumn{%d}{l}{\emph{difference to the periodic two-slit kernel ($\pm$ SE)}} \\' % (len(SETS) + 1)]
for k, lab in ROWS:
    same = lambda s: k in ('periodic_pub', 'band') and R[s][k].get('lo', 0) == R[s]['periodic']['lo'] and R[s][k]['hi'] == R[s]['periodic']['hi'] and R[s][k]['N'] == R[s]['periodic']['N']
    T.append(lab + ' & ' + ' & '.join('same harmonics' if same(s) else f"${R[s][k]['diff']:+.4f}$ ({R[s][k]['se']:.4f})".replace('(0.0000)', '($<$0.0001)').replace('$-0.0000$', '$0.0000$') for s, _ in SETS) + r' \\')
T += [r'\bottomrule', r'\end{tabular}']
open(f'{PP}/tables/dslit_window.tex', 'w').write('\n'.join(T) + '\n'); print('\n'.join(T))
