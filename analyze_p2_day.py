"""Paper 2: synthetic selector study, CAISO hourly day split and the joint charge/discharge pairs.
Test log-likelihood only (no p-values).

    .venv/bin/python analyze_p2_day.py
Reads results/p2syn5/syn/*.json, results/p2lag, p2day7, p2day7b, p2day7c (day split) and results/p2pairs2, p2pairs3, p2pairskde
(pairs); writes papers/paper5/{tables,figs}, papers/paper5/numbers.tex and a frozen summary
results/p2_daysplit_summary.json.
"""
import os, json, glob, collections
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'papers', 'paper5')
for s in ('tables', 'figs'):
    os.makedirs(os.path.join(OUT, s), exist_ok=True)
BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5, 'legend.fontsize': 7,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 1.5, 'lines.markersize': 4, 'pdf.fonttype': 42})
nums, summary = {}, {}
def num(k, v):
    nums[k] = v

# ------------------------------------------------------------------ synthetic selector study
KDES = {'kde_scott': "Scott's rule", 'kde_lscv': 'LSCV', 'kde_lcv': 'likelihood CV', 'kde_adaptive': 'adaptive'}
syn = {}
for f in glob.glob(os.path.join(HERE, 'results', 'p2syn5', 'syn', '*.json')):
    r = json.load(open(f))
    if r['truth'] == 'T4':           # T4 = T1 with stretched axes; scale-equivariant methods give identical numbers
        continue
    syn[(r['truth'], r['d'], r['n'])] = r
def med(r, k):
    return float(np.median([x[k]['rel_ise'] for x in r['reps']]))
rows, ratio_kde, ratio_or, ratio_sh = [], [], [], []
for (T, d, n) in sorted(syn):
    r = syn[(T, d, n)]
    a, sh = med(r, 'blml_auto'), med(r, 'blml_shared')
    o = med(r, 'blml_oracle') if 'blml_oracle' in r['reps'][0] else None
    kd = {k: med(r, k) for k in KDES}
    bk = min(kd, key=kd.get)
    fc = np.median([x['blml_auto']['fc'] for x in r['reps']], 0)
    rows.append((T, d, n, a, o, sh, kd[bk], bk, float(fc.mean())))
    ratio_kde.append(a / kd[bk]); ratio_sh.append(a / sh)
    if o: ratio_or.append(a / o)
summary['synthetic'] = [dict(truth=T, d=d, n=n, auto=a, oracle=o, shared=sh, best_kde=b, best_kde_name=bn, fc_auto_mean=fc)
                        for T, d, n, a, o, sh, b, bn, fc in rows]
num('SynCells', len(rows))
num('SynWinsKDE', sum(x < 1 for x in ratio_kde))
num('SynKDERatioRange', f'{1/max(ratio_kde):.1f}--{1/min(ratio_kde):.0f}')
num('SynOracleMedian', f'{np.median(ratio_or):.2f}')
num('SynOracleRange', f'{min(ratio_or):.2f}--{max(ratio_or):.1f}')
num('SynSharedWins', sum(x < 1 for x in ratio_sh))
with open(os.path.join(OUT, 'tables', 'synth.tex'), 'w') as f:
    f.write('\\begin{tabular}{llr rrrr l r}\n\\toprule\n'
            'truth & $d$ & $n$ & auto & oracle & shared & best KDE & (rule) & $\\bar f_c$ \\\\\n\\midrule\n')
    last = None
    for T, d, n, a, o, sh, b, bn, fc in rows:
        if last and last != T:
            f.write('\\midrule\n')
        last = T
        e = lambda v: '--' if v is None else f'{v:.2e}'.replace('e-0', 'e-').replace('e+00', '')
        f.write(f'{T} & {d} & {n:,} & \\textbf{{{e(a)}}} & {e(o)} & {e(sh)} & {e(b)} & {KDES[bn]} & {fc:.2f} \\\\\n'
                .replace(',', '{,}'))
    f.write('\\bottomrule\n\\end{tabular}\n')

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
for ax, T in zip(axes, ['T1', 'T2', 'T3']):
    for d, col in [(2, BLUE), (4, ORANGE), (8, AQUA)]:
        ns = sorted(n for (t, dd, n) in syn if t == T and dd == d)
        R = [syn[(T, d, n)] for n in ns]
        ax.plot(ns, [min(med(r, k) for k in KDES) / med(r, 'blml_auto') for r in R], marker='o', color=col, label=f'$d={d}$, best KDE')
        if T != 'T3':
            ax.plot(ns, [med(r, 'blml_oracle') / med(r, 'blml_auto') for r in R], marker='s', ls=(0, (3, 2)),
                    mfc='white', color=col, label=f'$d={d}$, oracle')
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('$n$')
    ax.set_title({'T1': 'separable (T1)', 'T2': 'non-separable (T2)', 'T3': 'correlated normal (T3)'}[T])
axes[0].set_ylabel('error of competitor / error of BLML-auto')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.08), fontsize=6.5)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(os.path.join(OUT, 'figs', 'synth_ratio.pdf'), bbox_inches='tight'); plt.close(fig)

# ------------------------------------------------------------------ CAISO day split (test LL only)
# Latest run per model with the current code (later directories override earlier ones):
# KDE/k-NN (hour, lag 24 h only), XGBoostLSS and TabPFN, BLML, MDN and GMM.
DAY_DIRS = ['p2lag', 'p2day7', 'p2day7b', 'p2day7c', 'p2cpu', 'p2stream']   # p2cpu: KDE, k-NN with both lags, run on CPU
# p2stream: streaming BLML (BLML-stream, 6 epochs, batches of 42 whole days, seed 0), run on a laptop CPU
COVS = ['hl', 'hll']
SHORT = {'2d': 'hour', '3d': 'hour, doy', 'hl': 'hour, lag 24\\,h', 'hll': 'hour, lags 24, 168\\,h'}
LIGHT = ['BLML-stream', 'CondKDE', 'CondKNN']                     # assumption-light (nonparametric); one-shot BLML: paper 5
STRUCT = ['GMM', 'XGBoostLSS', 'MDN', 'TabPFN']                   # structured / parametric references
LAB = {'BLML-stream': 'BLML (this paper)', 'BLML-auto': 'BLML, one-shot selection',
       'BLML-joint-stream': 'BLML, joint 6-D (this paper)', 'CondKDE': 'conditional KDE', 'CondKNN': 'conditional $k$-NN',
       'GMM': 'Gaussian mixture', 'XGBoostLSS': 'XGBoostLSS (Gaussian)', 'MDN': 'mixture density network',
       'TabPFN': 'TabPFN$^*$', 'BLML-joint': 'BLML, joint 6-D (this paper)', 'KDE-joint': 'conditional KDE, joint 6-D', 'KNN-joint': 'conditional $k$-NN, joint 6-D',
       'GMM-chain': 'Gaussian mixture, chain', 'XGBoostLSS-chain': 'XGBoostLSS, chain', 'MDN-chain': 'mixture density network, chain',
       'TabPFN-chain': 'TabPFN$^*$, chain'}
R = {}
for j in DAY_DIRS:
    for f in glob.glob(os.path.join(HERE, 'results', j, '**', 'daysplit_*.json'), recursive=True):
        r = json.load(open(f))
        if r.get('run') == 'daysplit' and 'scores' in r and 'error' not in r and r['covset'] in COVS \
                and r['model'] in LIGHT + STRUCT:
            R[(r['covset'], r['model'])] = r
ll = {k: -r['scores']['nll'] for k, r in R.items()}
fit_s = {k: r.get('fit_s') for k, r in R.items()}
summary['caiso_daysplit'] = {f'{cs}|{m}': dict(ll=v, fit_s=fit_s[(cs, m)]) for (cs, m), v in ll.items()}
ref = R.get(('hl', 'BLML-stream'))
if ref:
    num('DayNtest', f"{len(ref['y']):,}".replace(',', '{,}'))
    num('DayNdays', len({t[:10] for t in ref['test_times']}))

def block(f, models, cols, getv, best):
    for m in models:
        cells = []
        for c in cols:
            v = getv(c, m)
            cells.append('--' if v is None else (f'\\textbf{{{v:.3f}}}' if abs(v - best[c]) < 1e-9 else f'{v:.3f}'))
        f.write(LAB.get(m, m) + ' & ' + ' & '.join(cells) + ' \\\\\n')

with open(os.path.join(OUT, 'tables', 'day_main.tex'), 'w') as f:
    f.write('\\begin{tabular}{l' + 'r' * len(COVS) + '}\n\\toprule\n')
    f.write('model & ' + ' & '.join(SHORT[c] for c in COVS) + ' \\\\\n\\midrule\n')
    for grp, title in [(LIGHT, 'assumption-light (nonparametric)'), (STRUCT, 'structured models (reference)')]:
        f.write(f'\\multicolumn{{{1 + len(COVS)}}}{{l}}{{\\emph{{{title}}}}} \\\\\n')
        best = {c: max([ll[(c, m)] for m in grp if (c, m) in ll] or [np.nan]) for c in COVS}
        block(f, grp, COVS, lambda c, m: ll.get((c, m)), best)
        if grp is LIGHT:
            f.write('\\midrule\n')
    f.write('\\bottomrule\n\\end{tabular}\n')
if ('hll', 'BLML-stream') in R:                    # last week's price: BLML's learned cut-off on that column
    num('DayStreamFcLagWeek', f"{R[('hll', 'BLML-stream')]['fc'][-1]:.3f}")
for m in LIGHT + STRUCT:                          # change from adding last week's price
    if ('hl', m) in ll and ('hll', m) in ll:
        num('DayAddWeek' + m.replace('-', ''), f"{ll[('hll', m)] - ll[('hl', m)]:+.3f}")
for c in COVS:
    cc = c.replace('2d', 'Two').replace('3d', 'Three').replace('hll', 'HLL').replace('hl', 'HL')
    if (c, 'BLML-auto') in ll:
        num('DayBLML' + cc, f"{ll[(c, 'BLML-auto')]:.3f}")
    if (c, 'BLML-stream') in ll:
        num('DayStream' + cc, f"{ll[(c, 'BLML-stream')]:.3f}")
        for m in ('CondKNN', 'CondKDE', 'MDN', 'GMM', 'TabPFN'):          # streaming BLML minus each method
            if (c, m) in ll:
                num(f'DayStreamVs{m}{cc}', f"{ll[(c, 'BLML-stream')] - ll[(c, m)]:.2f}")
                num(f'DayStreamVs{m}{cc}Abs', f"{abs(ll[(c, 'BLML-stream')] - ll[(c, m)]):.2f}")

# ------------------------------------------------------------------ joint charge/discharge pairs (test LL only)
PAIR_DIRS = ['p2pairs2', 'p2pairs3', 'p2pairskde', 'p2pairscpu', 'p2pairsstream']   # p2pairscpu: k-NN, run on CPU
PL = ['BLML-joint-stream', 'KDE-joint', 'KNN-joint']   # one-shot BLML-joint archived (selection results moved to paper 5)
PS = ['GMM-chain', 'XGBoostLSS-chain', 'MDN-chain', 'TabPFN-chain']
PR = {}
for j in PAIR_DIRS:
    for f in glob.glob(os.path.join(HERE, 'results', j, 'pairs', '*.json')):
        r = json.load(open(f))
        if 'll' in r and r.get('model') in PL + PS:
            PR[r['model']] = r
summary['caiso_pairs'] = {m: dict(ll=r['ll'], ll_p1=r['ll_p1'], ll_p2=r['ll_p2'], fit_predict_s=r['fit_predict_s']) for m, r in PR.items()}
if 'BLML-joint-stream' in PR:
    r = PR['BLML-joint-stream']
    num('PairNtrain', f"{r['n_train']:,}".replace(',', '{,}')); num('PairNtest', f"{r['n_test']:,}".replace(',', '{,}'))
    num('PairNdays', len(set(r['day1'])))
    for m, r2 in PR.items():
        if m == 'BLML-joint':
            continue
        if m == 'BLML-joint-stream':
            num('PairBLMLJoint', f"{r2['ll']:.3f}"); continue
        num('Pair' + m.replace('-', '').replace('chain', 'Chain').replace('joint', 'Joint').replace('stream', 'Stream'), f"{r2['ll']:.3f}")
with open(os.path.join(OUT, 'tables', 'pairs_main.tex'), 'w') as f:
    f.write('\\begin{tabular}{l r}\n\\toprule\nmodel & test LL (nats/pair) \\\\\n\\midrule\n')
    for grp, title in [(PL, 'assumption-light (nonparametric), joint fit in 6-D'),
                       (PS, 'structured models (reference), $p(p_1\\mid c)\\,p(p_2\\mid p_1,c)$')]:
        f.write(f'\\multicolumn{{2}}{{l}}{{\\emph{{{title}}}}} \\\\\n')
        best = max([PR[m]['ll'] for m in grp if m in PR] or [np.nan])
        for m in grp:
            if m in PR:
                v = PR[m]['ll']
                cell = f'\\textbf{{{v:.3f}}}' if abs(v - best) < 1e-9 else f'{v:.3f}'
                f.write(f"{LAB[m]} & {cell} \\\\\n")
        if grp is PL:
            f.write('\\midrule\n')
    f.write('\\bottomrule\n\\end{tabular}\n')

with open(os.path.join(OUT, 'numbers.tex'), 'w') as f:
    f.write('% generated by analyze_p2_day.py\n')
    for k, v in nums.items():
        f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
with open(os.path.join(HERE, 'results', 'p2_daysplit_summary.json'), 'w') as f:
    json.dump(summary, f, indent=1, default=float)
print(json.dumps(nums, indent=0))
