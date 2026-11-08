"""Paper 3, Table II and the phase figure from results/complex/phase_frame2 (p3_phase_frame.py): rms pupil phase error
(rad, up to the ambiguities) of HIO, complex BLML from one start, complex BLML polished from the HIO and
symmetry-breaking starts, and complex BLML from the truth, per aberration and photon count (three seeds each).
-> papers/submit/pra_complex/tables/phase_hio.tex, figs/phase_hio.pdf, results/complex/phase_frame2_summary.json"""
import json, glob, collections, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = collections.defaultdict(list)
for f in glob.glob('results/complex/phase_frame2/*_[0-9].json'):
    d = json.load(open(f)); R[(d['case'], d['n'])].append(d)
CASES = ['coma', 'astig', 'defocus', 'mixed']; LAB = dict(coma='coma', astig='astigmatism', defocus='defocus', mixed='mixed (4 modes)')
M = ['hio', 'blml_1', 'blml', 'truth']; NS = sorted({k[1] for k in R})
fmt = lambda v: '--' if v is None else f'{v:.3f}'
L = [r'\begin{tabular}{lr cccc c}', r'\toprule',
     r'aberration & photons & HIO & BLML, one start & BLML, from HIO & BLML, from truth & BLML better \\', r'\midrule']
summ = {}
for c in CASES:
    for n in NS:
        rs = R.get((c, n), [])
        if not rs:
            continue
        med = {m: float(np.median([r[m]['phase_err'] for r in rs])) for m in M}
        wins = sum(r['blml']['phase_err'] < r['hio']['phase_err'] for r in rs)
        summ[f'{c}_{n}'] = dict(med, wins=wins, seeds=len(rs), all={m: [r[m]['phase_err'] for r in rs] for m in M},
                                ll_blml_minus_hio=[(r['blml']['ll'] - r['hio']['ll']) * n for r in rs])
        best = min(med['hio'], med['blml'])
        cell = lambda m: (r'\textbf{' + fmt(med[m]) + '}') if m in ('hio', 'blml') and med[m] == best else fmt(med[m])
        L.append(f"{LAB[c]} & $10^{int(np.log10(n))}$ & " + ' & '.join(cell(m) for m in M) + f" & {wins}/{len(rs)} \\\\")
    L.append(r'\addlinespace')
L = L[:-1] + [r'\bottomrule', r'\end{tabular}']
open('papers/submit/pra_complex/tables/phase_hio.tex', 'w').write('\n'.join(L) + '\n')
json.dump(summ, open('results/complex/phase_frame2_summary.json', 'w'), indent=1)
# figure: error against photons, one panel per aberration; maps for one case
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=True)
for a, c in zip(ax, CASES):
    for m, col, mk, lab in (('hio', 'C1', 's', 'HIO'), ('blml', 'C0', 'o', 'complex BLML'), ('truth', '0.5', '', 'BLML from truth')):
        xs, ys = [], []
        for n in NS:
            for r in R.get((c, n), []):
                xs.append(n * (1.08 if m == 'hio' else 0.93)); ys.append(r[m]['phase_err'])
        if m == 'truth':
            meds = [np.median([r['truth']['phase_err'] for r in R[(c, n)]]) for n in NS if R.get((c, n))]
            a.plot([n for n in NS if R.get((c, n))], meds, color=col, ls=':', lw=1, label=lab)
        else:
            a.scatter(xs, ys, s=10, color=col, marker=mk, label=lab, alpha=0.8)
    a.set_xscale('log'); a.set_yscale('log'); a.set_title(LAB[c], fontsize=8); a.set_xlabel('photons')
ax[0].set_ylabel('pupil phase error (rad)'); ax[0].legend(frameon=False, fontsize=6.5, loc='lower left')
fig.tight_layout(pad=0.3); fig.savefig('papers/submit/pra_complex/figs/phase_hio.pdf'); fig.savefig('papers/submit/pra_complex/figs/phase_hio.png', dpi=200)
print(json.dumps({k: {m: round(v[m], 3) for m in M} | dict(wins=v['wins']) for k, v in summ.items()}, indent=0))
