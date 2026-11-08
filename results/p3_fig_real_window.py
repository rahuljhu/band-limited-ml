"""Paper 3, figure figs/real_window.pdf: the aperture on the window's harmonics for the four double-slit records
(p3_periodic_metrics.py): |a_k| of the periodic two-slit fit on split 0 (flat background) against the record's
Fresnel fit on the same window, both of unit norm, twin chosen to match; the fringe harmonic marked. Run from the repo root."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
PP = 'papers/submit/pra_complex'
SETS = [('photon', 'photons, single mode'), ('ghost', 'ghost diffraction'), ('electron', 'electrons'), ('photon_mm', 'photons, multimode')]   # as Fig. 4
fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=True)
for a, (s, lab) in zip(ax, SETS):
    r = json.load(open(f'results/p3_periodic_metrics/{s}.json')); ks = np.array(r['ref']['ks'])
    A = lambda c: np.array([complex(*v) for v in c]); ref = A(r['ref']['coef']); run = next(x for x in r['runs'] if x['seed'] == 0 and x['rep'] is None)
    fit = A(run['flat_coef']); tw = np.conj(fit[::-1])                       # the twin: a_k -> conj(a_{-k})
    if np.sum(np.abs(np.abs(tw) - np.abs(ref))) < np.sum(np.abs(np.abs(fit) - np.abs(ref))):
        fit = tw
    a.bar(ks, np.abs(ref), width=0.8, color='0.8', label='Fresnel fit')
    a.plot(ks, np.abs(fit), 'o', color='C3', ms=3, label='periodic kernel')
    a.set_xlim(-12.5, 12.5); a.set_title(f'{lab} ($L={r["ref"]["L"]}$)', fontsize=7.5, loc='left'); a.set_xlabel('harmonic $k$ of the window')
ax[0].set_ylabel(r'$|a_k|$'); ax[0].legend(fontsize=6, frameon=False, loc='upper left')
fig.tight_layout(pad=0.3); fig.savefig(f'{PP}/figs/real_window.pdf'); fig.savefig(f'{PP}/figs/real_window.png', dpi=200)
