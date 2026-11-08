"""Paper 3, table tables/dslit_cropfree.tex: replicas of the records drawn on the record's crop and on a window widened
by 1024 pixels on each side (crop-free), fitted with the line two-slit kernel (p3_slit_main.py; N in {0,1,2} by CV) and,
for the multimode record, the line three-mode model (p3_partial3_real.py) at 1 to 16 times the record's size. Columns:
the selected fill (published: photons 0.33, multimode 0.31), held-out log-likelihood of the fit and of the published
geometry minus the KDE's, the KL from the truth (fit / KDE), and the three-mode coherent share V_hat (Fresnel gamma
0.15). Run from the repo root."""
import json, glob, numpy as np
PP = 'papers/submit/pra_complex'
L = [r'\begin{tabular}{llrrrrr}', r'\toprule',
     r'replicas of & window & fill & fit $-$ KDE & published $-$ KDE & KL fit / KDE & $\hat V$ \\', r'\midrule']
rng = lambda v, f: f"{f.format(min(v))} to {f.format(max(v))}" if len(set(np.round(v, 4))) > 1 else f.format(v[0])
for tag, suf in (('crop', '_s0'), ('crop-free', '_w1024_s0')):
    R = [json.load(open(f)) for f in sorted(glob.glob(f'results/p3_slit_main_n012/photon_rep?{suf}.json'))]
    fl = [r['est']['fill'] for r in R]; e = [r['est_minus_kde']['diff'] for r in R]; p = [r['pub_minus_kde']['diff'] for r in R]
    kf = np.median([r['est']['kl'] for r in R]); kk = np.median([r['kde']['kl'] for r in R])
    L.append(f"photons, single mode & {tag} & {rng(fl, '{:.2f}')} & {rng(e, '{:+.4f}')} & {rng(p, '{:+.4f}')} & {kf:.4f} / {kk:.4f} & -- \\\\")
for m, lab in ((1, r'$1\times$'), (4, r'$4\times$'), (16, r'$16\times$')):
    for tag, suf in (('crop', ''), ('crop-free', '_w1024')):
        f = f'results/p3_partial3_real/photon_mm_rep0' + ('' if m == 1 else f'_x{m}') + f'{suf}_s0.json'
        try:
            r = json.load(open(f))
        except FileNotFoundError:
            continue
        t = r['three']
        L.append(f"photons, multimode, {lab} & {tag} & {t['fill']:.2f} & -- & -- & {t['kl']:.1e} & {t['V_hat']:.2f} \\\\")
L += [r'\bottomrule', r'\end{tabular}']
open(f'{PP}/tables/dslit_cropfree.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
