"""Paper 3, the synthetic two-slit benchmark (p3_dbench.py, results/dbench/p3_dbench/*.json): figure and tables.
Figure (figs/dbench.pdf): (a-d) KL from the truth against n, one panel per case; (e) aperture error, (f) slit-width
error, (g) relative slit phase error (asymmetric slits), each against n, the median over the data sets; (h) the share
of data sets whose CV picked the true fill. Methods: complex BLML (band), complex BLML (two-slit kernel), real BLML,
KDE, the Fresnel model, HIO given the true support. Tables (tables/dbench_kl.tex, tables/dbench_field.tex): medians
over the 6 data sets per cell (background 0 and 0.1, three replicas). Run from the repo root."""
import json, glob, collections, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
PP = 'papers/submit/pra_complex'
R = collections.defaultdict(list)
for f in glob.glob('results/dbench/p3_dbench/*.json'):
    if f.endswith(('_hio.json', '_s0.json')) or '_sharp_' in f:   # HIO-only, sharp-slit and N = 0 runs: below
        continue
    d = json.load(open(f)); R[(d['case'], d['n'])].append(d)
# the synthetic protocol: a flat background (N = 0; DB_FLAT_ONLY pass, *_s0.json) for complex BLML, and for the
# partially coherent truths the three-mode two-slit model; the N = 1 fits of the main pass are kept as *_n1
for (c, n), ds in R.items():
    for d in ds:
        f0 = f"results/dbench/p3_dbench/{c}_bg{d['bg']:g}_n{n}_{d['rep']}_s0.json"
        s = json.load(open(f0))
        d['blml_n1'], d['blml_slits_n1'] = d['blml'], d['blml_slits']
        d['blml'], d['blml_slits'] = s['blml0'], (s['blml_three'] if c.startswith('partial') else s['blml_slits0'])
        d['blml_slits_two'], d['blml_three'] = s['blml_slits0'], s['blml_three']
CASES = [('sym', 'equal slits'), ('asym', 'unequal, phase'), ('defocus', 'defocus'), ('partial', 'partially coherent')]
NS = sorted({n for _, n in R})
M = [('blml', 'complex BLML, band', 'C0', 'o', '-'), ('blml_slits', 'complex BLML, two-slit kernel', 'C3', 's', '-'),
     ('blml_real', 'real BLML (positive root)', 'C1', 'v', '--'), ('kde', 'KDE', '0.45', 'x', ':'),
     ('fresnel', 'Fresnel model', 'C2', 'd', '-.'), ('hio_true_support', 'HIO, true support', 'C4', '^', ':')]
med = lambda case, n, e, m: np.median([d[e][m] for d in R[(case, n)] if e in d and m in d[e]]) if any(
    e in d and m in d[e] for d in R[(case, n)]) else np.nan
M = [('blml_slits', 'complex BLML, two-slit kernel', 'C3', 's', '-'), ('blml', 'complex BLML, band', 'C0', 'o', '-'),
     ('fresnel', 'Fresnel model', 'C2', 'd', '-.'), ('hio_true_support', 'HIO, true support', 'C4', '^', ':'),
     ('kde', 'KDE', '0.4', 'x', '--'), ('blml_real', 'real BLML (positive root)', 'C1', 'v', (0, (1, 1)))]
fig, ax = plt.subplots(2, 3, figsize=(7.0, 4.4))
coh = [c for c, _ in CASES if c != 'partial']
def gm(vals):
    v = [x for x in vals if np.isfinite(x) and x > 0]; return np.exp(np.mean(np.log(v))) if v else np.nan
def panel(a, metric, cases, ylab, title, skip=()):
    for e, nm, col, mk, ls in M:
        if e in skip:
            continue
        y = [gm([med(c, n, e, metric) for c in cases]) for n in NS]
        if not np.all(np.isnan(y)):
            a.loglog(NS, y, color=col, marker=mk, ls=ls, lw=1.1, ms=3.5, label=nm)
    a.set_title(title, fontsize=8, loc='left'); a.set_xlabel('photons $n$'); a.set_ylabel(ylab, fontsize=7.5)
panel(ax[0, 0], 'kl', [c for c, _ in CASES], 'KL from the truth', '(a) intensity')
panel(ax[0, 1], 'aperture_err', coh, r'$\|\hat A-A\|$', '(b) aperture', skip=('kde',))
panel(ax[0, 2], 'centre_err', [c for c, _ in CASES], 'relative error', '(c) slit position', skip=('kde',))
panel(ax[1, 0], 'width_err', [c for c, _ in CASES], 'relative error', '(d) slit width', skip=('kde',))
panel(ax[1, 1], 'phase_err', ['asym'], 'error (rad)', '(e) relative slit phase', skip=('kde',))
a = ax[1, 2]
for (c, lab), shade, mk in zip(CASES, ('0.0', '0.3', '0.55', '0.75'), ('o', 's', 'd', '^')):
    a.semilogx(NS, [np.mean([d['blml_slits']['fill'] == 0.45 for d in R[(c, n)]]) for n in NS], color=shade, marker=mk, ms=3.5, lw=1, label=lab)
a.set_ylim(-0.05, 1.05); a.set_xlabel('photons $n$'); a.set_ylabel('share of data sets', fontsize=7.5)
a.set_title('(f) true slit fill selected', fontsize=8, loc='left'); a.legend(fontsize=6, frameon=False, loc='lower right')
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.92))
fig.savefig(f'{PP}/figs/dbench.pdf'); fig.savefig(f'{PP}/figs/dbench.png', dpi=200)
# tables
cols = [('blml', 'band'), ('blml_slits', 'two-slit'), ('blml_real', 'real'), ('kde', 'KDE'), ('fraunhofer', 'Fraunhofer'),
        ('fresnel', 'Fresnel'), ('hio_true_support', 'HIO, support'), ('hio_band', 'HIO, band')]
def table(metric, name, cases, fmt):
    L = [r'\begin{tabular}{ll' + 'r' * len(cols) + '}', r'\toprule',
         'case & $n$ & ' + ' & '.join(c[1] for c in cols) + r' \\', r'\midrule']
    for c, lab in [x for x in CASES if x[0] in cases]:
        for i, n in enumerate(NS):
            v = [med(c, n, e, metric) for e, _ in cols]; ok = [x for x in v if np.isfinite(x)]; b = min(ok) if ok else None
            cell = lambda x: '--' if not np.isfinite(x) else (r'\textbf{' + fmt(x) + '}' if x == b else fmt(x))
            L.append(f"{lab if i == 0 else ''} & $10^{int(np.log10(n))}$ & " + ' & '.join(cell(x) for x in v) + r' \\')
        L.append(r'\addlinespace')
    L = L[:-1] + [r'\bottomrule', r'\end{tabular}']
    open(f'{PP}/tables/{name}.tex', 'w').write('\n'.join(L) + '\n')
table('kl', 'dbench_kl', [c for c, _ in CASES], lambda x: f'{x:.1e}')
table('aperture_err', 'dbench_aperture', coh, lambda x: f'{x:.3f}')
table('centre_err', 'dbench_centre', [c for c, _ in CASES], lambda x: f'{x:.3f}')
table('width_err', 'dbench_width', [c for c, _ in CASES], lambda x: f'{x:.2f}')
table('phase_err', 'dbench_phase', ['asym'], lambda x: f'{x:.4f}')
table('defocus_err', 'dbench_defocus', ['defocus'], lambda x: f'{x:.2f}')
fills = {f'{c}_{n}': sorted(d['blml_slits']['fill'] for d in R[(c, n)]) for c, _ in CASES for n in NS}
json.dump(fills, open('results/dbench/fills.json', 'w'), indent=0)
print('done', len(sum(R.values(), [])), 'data sets')
# HIO robustness (p3_dbench.py with DB_HIO_ONLY=1: *_hio.json): the aperture error with 400 or 2000 iterations and the
# training histogram's or the KDE's amplitude as input, against the two-slit kernel, coherent truths
H = collections.defaultdict(list)
for f in glob.glob('results/dbench/p3_dbench/*_hio.json'):
    d = json.load(open(f)); H[(d['case'], d['n'])].append(d)
if H:
    V = [('hio_true_support', 'HIO, 400 it.'), ('hio_true_support_2000', 'HIO, 2000 it.'), ('hio_true_support_kde', 'HIO on KDE, 400 it.'),
         ('hio_true_support_kde_2000', 'HIO on KDE, 2000 it.')]
    L = [r'\begin{tabular}{llrrrrr}', r'\toprule', 'case & $n$ & ' + ' & '.join(v[1] for v in V) + r' & two-slit kernel \\', r'\midrule']
    for c, lab in [x for x in CASES if x[0] in coh]:
        for i, n in enumerate(sorted({n for _, n in H})):
            v = [np.median([d[k]['aperture_err'] for d in H[(c, n)]]) for k, _ in V] + [med(c, n, 'blml_slits', 'aperture_err')]
            L.append(f"{lab if i == 0 else ''} & $10^{int(np.log10(n))}$ & " + ' & '.join(f'{x:.3f}' for x in v) + r' \\')
        L.append(r'\addlinespace')
    L = L[:-1] + [r'\bottomrule', r'\end{tabular}']
    open(f'{PP}/tables/dbench_hio.tex', 'w').write('\n'.join(L) + '\n')
# sharp (rectangular) slits on a 1536-pixel window (cases <case>_sharp): KL, aperture and width errors, the fill chosen
S = collections.defaultdict(list)
for f in glob.glob('results/dbench/p3_dbench/*_sharp_*.json'):
    if not f.endswith('_hio.json'):
        d = json.load(open(f)); S[(d['case'], d['n'])].append(d)
if S:
    ms = lambda c, n, e, m: np.median([d[e][m] for d in S[(c, n)] if e in d and m in d[e]]) if any(e in d and m in d[e] for d in S[(c, n)]) else np.nan
    NSs = sorted({n for _, n in S})
    def stable(metric, name, cases, fmt):
        L = [r'\begin{tabular}{ll' + 'r' * len(cols) + '}', r'\toprule', 'case & $n$ & ' + ' & '.join(c[1] for c in cols) + r' \\', r'\midrule']
        for c, lab in cases:
            for i, n in enumerate(NSs):
                v = [ms(c, n, e, metric) for e, _ in cols]; ok = [x for x in v if np.isfinite(x)]; b = min(ok) if ok else None
                L.append(f"{lab if i == 0 else ''} & $10^{int(np.log10(n))}$ & " + ' & '.join('--' if not np.isfinite(x) else (r'\textbf{' + fmt(x) + '}' if x == b else fmt(x)) for x in v) + r' \\')
            L.append(r'\addlinespace')
        L = L[:-1] + [r'\bottomrule', r'\end{tabular}']
        open(f'{PP}/tables/{name}.tex', 'w').write('\n'.join(L) + '\n')
    SC = [('sym_sharp', 'equal slits'), ('asym_sharp', 'unequal, phase'), ('partial_sharp', 'partially coherent')]
    stable('kl', 'dbench_sharp_kl', SC, lambda x: f'{x:.4f}')
    stable('aperture_err', 'dbench_sharp_aperture', SC[:2], lambda x: f'{x:.3f}')
    stable('width_err', 'dbench_sharp_width', SC, lambda x: f'{x:.2f}')
    json.dump({f'{c}_{n}': sorted(d['blml_slits']['fill'] for d in S[(c, n)]) for c, _ in SC for n in NSs},
              open('results/dbench/fills_sharp.json', 'w'), indent=0)
