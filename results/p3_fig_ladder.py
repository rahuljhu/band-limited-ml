"""Paper 3, Fig. 1 and the particles-needed table: the prior ladder on a crop-free line (p3_ladder.py,
results/p3_ladder). Panels against n, the median over the six data sets per point (background 0 and 0.1, three each):
(a) KL from the truth (geometric mean over the cases); (b) slit separation (the aperture's position is a gauge: g e^{2 pi i s t} has the same |g|^2); (c) slit width;
(d) relative slit phase (unequal slits); (e) fringe contrast gamma; (f) amplitude ratio (unequal slits) and (g) visibility
(partially coherent), each with the prior that identifies it (V = 1, resp. rho = 1) and with the three-mode model,
which leaves the other free (then only gamma is identified); (h) the particles each
rung needs for a target accuracy. Table tables/ladder_n.tex: n at which the median error first falls below the
target (log-interpolated; > 10^6 if never; -- if the rung has no such estimate). Run from the repo root."""
import json, glob, collections, numpy as np, matplotlib, matplotlib.ticker
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
PP = 'papers/submit/pra_complex'
R = collections.defaultdict(list)
for f in glob.glob('results/p3_ladder/*.json'):
    d = json.load(open(f)); R[(d['case'], d['n'])].append(d)
NS = sorted({n for _, n in R}); CASES = ['sym', 'asym', 'partial']
RUNGS = [('kde', 'KDE (smoothness)', '0.45', 'x', ':'), ('band', 'band (aperture extent)', 'C0', 'o', '-'),
         ('shape', 'two-slit shape', 'C3', 's', '-'), ('known', 'known geometry', 'C1', 'v', '-'),
         ('three', 'known geometry, three modes', 'C2', 'd', '-.'), ('model', 'full physical model', 'k', '^', '--'), ('model_p', r'full model ($V=1$ for $\rho$)', 'k', '^', '-')]
def med(c, n, e, m):
    v = [d[e][m] for d in R[(c, n)] if e in d and m in d[e]]; return np.median(v) if v else np.nan
gm = lambda v: (lambda v: np.exp(np.mean(np.log(v))) if v else np.nan)([x for x in v if np.isfinite(x) and x > 0])
def curve(e, m, cases):
    return np.array([gm([med(c, n, e, m) for c in cases]) for n in NS])
fig, ax = plt.subplots(2, 4, figsize=(7.0, 3.9)); ax = ax.ravel()
PAN = [('kl', CASES, 'KL from the truth', '(a) intensity', ()), ('sep_err', CASES, 'relative error', '(b) slit separation', ('kde',)),
       ('width_err', ['sym', 'asym'], 'relative error', '(c) slit width', ('kde',)), ('phase_err', ['asym'], 'error (rad)', '(d) relative phase', ('kde',)),
       ('gamma_err', ['asym', 'partial'], 'absolute error', r'(e) contrast $\gamma$', ()), ]
for a, (m, cases, yl, title, skip) in zip(ax, PAN):
    for e, nm, col, mk, ls in RUNGS:
        if e in skip or e == 'model_p':
            continue
        y = curve(e, m, cases)
        if np.isfinite(y).any():
            a.loglog(NS, y, color=col, marker=mk, ls=ls, lw=1.0, ms=3, label=nm)
    a.set_title(title, fontsize=8, loc='left'); a.set_xlabel('particles $n$'); a.set_ylabel(yl, fontsize=7)
ax[0].loglog(NS, 3.0 / np.array(NS), color='0.7', lw=0.8, ls='--')        # the 1/n slope (a correctly specified model)
# (f), (g): the coherence parameters, with and without the identifying prior (the curves without stay flat)
for a, cs, key, tv, title, ls_ in ((ax[5], 'asym', 'rho_fit', 0.6, r'(f) amplitude ratio $\rho$', (('known', 'rho', 'known geometry, $V=1$'), ('three', 'rho', 'three modes, $V$ free'))),
                                  (ax[6], 'partial', 'V_fit', 0.6, r'(g) visibility $V$', (('three', 'V_hat', r'three modes, $\rho$ free'),))):
    for e, k_, nm, col, mk, ls in [(e, k_, nm, 'C1' if e == 'known' else 'C2', 'v' if e == 'known' else 'd', '-' if e == 'known' else '-.') for e, k_, nm in ls_] + \
            [('model_p', key, 'full model + prior', 'k', '^', '-')]:
        fold = (lambda r: min(r, 1 / r)) if k_ in ('rho', 'rho_fit') else (lambda r: r)              # the twin swaps the slits
        y = [np.median([abs(fold(d[e][k_]) - tv) for d in R[(cs, n)] if e in d]) if R[(cs, n)] else np.nan for n in NS]
        a.loglog(NS, np.maximum(y, 1e-5), color=col, marker=mk, ls=ls, lw=1.0, ms=3, mfc='none' if e == 'model' else col, label=nm)
    a.set_title(title, fontsize=8, loc='left'); a.set_xlabel('particles $n$'); a.set_ylabel('absolute error', fontsize=7)
    a.legend(fontsize=5.0, frameon=False, loc='best')
# (h) particles needed
TARGETS = [('kl', CASES, 1e-3, r'KL $\le10^{-3}$'), ('sep_err', CASES, 1e-3, r'separation err.\ $\le0.1\%$'),
           ('width_err', ['sym', 'asym'], 0.05, r'width err.\ $\le5\%$'), ('phase_err', ['asym'], 0.01, r'phase err.\ $\le0.01$'),
           ('gamma_err', ['asym', 'partial'], 0.005, r'$\gamma$ err.\ $\le0.005$'), ('rho_err', ['asym'], 0.01, r'$\rho$ err.\ $\le0.01$')]
def needed(e, m, cases, t):
    y = curve(e, m, cases); ok = np.where(np.isfinite(y) & (y <= t))[0]
    if not np.isfinite(y).any():
        return None
    if not len(ok):
        return np.inf
    i = ok[0]
    if i == 0:
        return NS[0]
    x0, x1, y0, y1 = np.log(NS[i - 1]), np.log(NS[i]), np.log(y[i - 1]), np.log(y[i])
    return float(np.exp(x0 + (np.log(t) - y0) * (x1 - x0) / (y1 - y0))) if y0 != y1 else NS[i]
TAB = {e: [needed(e, m, cs, t) if not (e in ('kde',) and m not in ('kl', 'gamma_err')) else None for m, cs, t, _ in TARGETS] for e, *_ in RUNGS}
a = ax[7]; xs = np.arange(len(TARGETS)); wd = 0.8 / (len(RUNGS) - 1)
for j, (e, nm, col, *_ ) in enumerate([r for r in RUNGS if r[0] != 'model']):
    v = [x if x is not None else np.nan for x in TAB[e]]
    a.bar(xs + (j - (len(RUNGS) - 1) / 2 + 0.5) * wd, [min(x, 3e6) if np.isfinite(x) else (3e6 if x == np.inf else np.nan) for x in v],
          width=wd, color=col, hatch=['', '//'][0])
a.set_yscale('log'); a.set_ylim(5e2, 4e6); a.set_xticks(xs); a.set_xticklabels(['KL ≤ 1e-3', 'separation ≤ 0.1%', 'width ≤ 5%', 'phase ≤ 0.01', 'γ ≤ 0.005', 'ρ ≤ 0.01'], rotation=60, fontsize=5.5, ha='right')
a.set_title('(h) particles needed', fontsize=8, loc='left'); a.set_ylabel('$n$', fontsize=7)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.91))
for a_ in ax:                                         # minor labels only where the axis spans under a decade
    if a_.get_yscale() == 'log' and a_.get_ylim()[1] / a_.get_ylim()[0] > 10:
        a_.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.savefig(f'{PP}/figs/ladder.pdf'); fig.savefig(f'{PP}/figs/ladder.png', dpi=200)
fmt = lambda x: '--' if x is None else (r'$>10^6$' if x == np.inf else (f'${x / 10 ** int(np.log10(x)):.1f}\\times10^{int(np.log10(x))}$'))
L = [r'\begin{tabular}{l' + 'r' * len(TARGETS) + '}', r'\toprule', 'prior & ' + ' & '.join(t[3] for t in TARGETS) + r' \\', r'\midrule']
for e, nm, *_ in [r for r in RUNGS if r[0] != 'model']:
    L.append(nm + ' & ' + ' & '.join(fmt(x) for x in TAB[e]) + r' \\')
L += [r'\bottomrule', r'\end{tabular}']
open(f'{PP}/tables/ladder_n.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L)); print(sum(len(v) for v in R.values()), 'data sets')
