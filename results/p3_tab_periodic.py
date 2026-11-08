"""Paper 3, table tables/dslit_periodic.tex: the periodic two-slit kernel on the double-slit records' trough-aligned
windows (p3_periodic_main.py, p3_periodic_metrics.py): test log-likelihood minus the KDE's (mean and range over six
splits; mean +- sd over five replicas), KL from the truth on the replicas (periodic / KDE), and what the fit
recovers with a flat background: slit separation d (cycles/px, x 100), fringe contrast C, against the Fresnel fit.
Run from the repo root."""
import json, glob, numpy as np
PP = 'papers/submit/pra_complex'
SETS = [('photon', 'photons, single mode'), ('ghost', 'ghost diffraction'), ('photon_mm', 'photons, multimode'), ('electron', 'electrons')]
L = [r'\begin{tabular}{lrrrrrrr}', r'\toprule',
     r' & window & \multicolumn{2}{c}{periodic $-$ KDE} & KL on replicas & \multicolumn{3}{c}{$100\,d$ / contrast $C$} \\',
     r' & $L$ (fringes) & splits & replicas & periodic / KDE & splits & replicas & Fresnel \\', r'\midrule']
for s, lab in SETS:
    rs = [json.load(open(f)) for f in glob.glob(f'results/p3_periodic_main/{s}_*.json')]
    rs = [r for r in rs if r['set'] == s]; sp = [r for r in rs if r['rep'] is None]; rp = [r for r in rs if r['rep'] is not None]
    g = lambda R, k: np.array([r[k]['diff'] for r in R])
    kl = lambda e: np.mean([r[e]['kl'] for r in rp])
    m = json.load(open(f'results/p3_periodic_metrics/{s}.json')); ref = m['ref']
    q = lambda grp, k: np.array([x['flat'][k] for x in m['runs'] if (x['rep'] is None) == (grp == 'splits')])
    a, b = g(sp, 'periodic_minus_kde'), g(rp, 'periodic_minus_kde')
    L.append(f"{lab} & {rs[0]['L']} ({rs[0]['L'] / (1 / ref['d_nominal']):.0f}) & {a.mean():+.4f} ({a.min():+.4f}, {a.max():+.4f}) & "
             f"{b.mean():+.4f} $\\pm$ {b.std(ddof=1):.4f} & {kl('periodic'):.1e} / {kl('kde'):.1e} & "
             f"{100 * q('splits', 'd').mean():.2f} / {q('splits', 'C').mean():.3f} & {100 * q('reps', 'd').mean():.2f} / {q('reps', 'C').mean():.3f} & "
             f"{100 * ref['d_nominal']:.2f} / {ref['C']:.3f} \\\\")
L += [r'\bottomrule', r'\end{tabular}']
open(f'{PP}/tables/dslit_periodic.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
