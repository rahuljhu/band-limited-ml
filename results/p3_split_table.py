"""Paper 3, the double-slit prediction table over splits: for each record, complex BLML (background mode N = 1, 5-fold
CV, one-SE rule, L-BFGS) minus the KDE in test log-likelihood per particle over six random splits (results/
p3_split_check.json from results/p3_split_check.py), against five replicas of the record (its Fresnel fit as the
truth, results/p3synth_lbfgs/p3_synth/<set>_base_*.json), with the selected band relative to the physical one and the
replicas' KL from the truth. -> papers/submit/pra_complex/tables/dslit_splits.tex. Run from the repo root."""
import json, glob, numpy as np
S = json.load(open('results/p3_split_check.json'))
SETS = [('photon', 'photons, single mode'), ('ghost', 'photons, ghost diffraction'), ('electron', 'electrons'),
        ('photon_mm', 'photons, multimode')]
L = [r'\begin{tabular}{lcccccc}', r'\toprule',
     r' & \multicolumn{3}{c}{record, six splits} & \multicolumn{3}{c}{five replicas} \\',
     r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
     r'record & BLML $-$ KDE & range & $f_c$/physical & BLML $-$ KDE & KL, BLML & KL, KDE \\', r'\midrule']
for s, lab in SETS:
    fr = json.load(open(f'results/dslit_1d/{s}/flat/{s}_fraunhofer.json'))['params']; phys = (1 + fr['a_over_d']) / fr['P']
    d = np.array([r['diff'] for r in S[s]]); fc = np.array([r['fc'] for r in S[s]]) / phys
    D = [json.load(open(f)) for f in sorted(glob.glob(f'results/p3synth_lbfgs/p3_synth/{s}_base_*.json'))]
    g = np.array([x['blml_bg1']['test'] - x['kde']['test'] for x in D])
    kb, kk = np.mean([x['blml_bg1']['kl'] for x in D]), np.mean([x['kde']['kl'] for x in D])
    rng_fc = f'{fc.min():.2f}' if fc.min() == fc.max() else f'{fc.min():.2f}--{fc.max():.2f}'
    L.append(f'{lab} & ${d.mean():+.4f}$ & ${d.min():+.4f}$ to ${d.max():+.4f}$ & {rng_fc} & ${g.mean():+.4f}\\pm{g.std(ddof=1):.4f}$ & '
             f'{kb:.4f} & {kk:.4f} \\\\')
L += [r'\bottomrule', r'\end{tabular}']
open('papers/submit/pra_complex/tables/dslit_splits.tex', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
