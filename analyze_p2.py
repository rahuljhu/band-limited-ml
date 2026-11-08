"""Paper 2 tables, figures, battery backtest and number macros from results/p2/p2/*.json.

    .venv/bin/python analyze_p2.py      # writes papers/paper5/{figs,tables,numbers.tex}
"""
import os, json, glob, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('P2_RES', os.path.join(HERE, 'results', 'p2', 'p2'))
OUT = os.path.join(HERE, 'papers', 'paper5')
os.makedirs(os.path.join(OUT, 'figs'), exist_ok=True)
os.makedirs(os.path.join(OUT, 'tables'), exist_ok=True)

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5, 'legend.fontsize': 7,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 1.5, 'lines.markersize': 4, 'pdf.fonttype': 42,
})
COVLAB = {'2d': 'hour', '3d': 'hour, day of year', '4d': 'hour, weekday, month'}
MAIN = ['BLML', 'CondKDE', 'CondKNN', 'GMM', 'Climatology', 'Normal(hour×month)', 'Student-t(hour×month)',
        'MDN', 'NGBoost-Normal', 'NGBoost-Laplace', 'CatBoost-Normal', 'TabPFN', 'LightGBM-quantile', 'QuantileForest']
ABL = ['BLML', 'BLML-volume', 'BLML-shared', 'BLML-hillclimb', 'BLML-inv_sigma', 'BLML-normalized_objective']
ABLLAB = {'BLML-volume': r'shared stage doubles $\\prod f_c$ per step', 'BLML': 'shared + hill climb (proposed; ratio objective)', 'BLML-shared': 'shared doubling only',
          'BLML-hillclimb': 'hill climb from coarsest', 'BLML-inv_sigma': r'rule $f_c=1/\sigma$',
          'BLML-normalized_objective': 'shared + hill climb, normalized objective'}
TEXNAME = {'Normal(hour×month)': r'Normal (hour$\times$month)', 'Student-t(hour×month)': r'Student-$t$ (hour$\times$month)'}

recs = [json.load(open(f)) for f in glob.glob(os.path.join(RES, '*.json'))]
recs = [r for r in recs if not r['run'].startswith('smoke')]
by = {}
for r in recs:
    by[(r['run'], r['covset'], r['model'])] = r
runs = sorted({r['run'] for r in recs})
folds = [x for x in runs if x != 'random']
models = sorted({r['model'] for r in recs})
nums = {}

def pooled(covset, model, runs_, key):
    """Concatenate a per-observation array over runs (None if any run is missing)."""
    out = []
    for run in runs_:
        r = by.get((run, covset, model))
        if r is None or 'per_obs' not in r or key not in r['per_obs']:
            return None
        out.append(np.asarray(r['per_obs'][key]))
    return np.concatenate(out) if out else None

def score_of(covset, model, runs_, metric):
    if metric == 'nll':
        lp = pooled(covset, model, runs_, 'logpdf')
        return None if lp is None else -lp.mean()
    if metric == 'crps':
        v = pooled(covset, model, runs_, 'crps')
        if v is None:
            v = pooled(covset, model, runs_, 'crps_q')
        return None if v is None else v.mean()
    if metric in ('cover90', 'pinball05', 'pinball50', 'pinball95'):
        vals, ns = [], []
        for run in runs_:
            r = by.get((run, covset, model))
            if r is None or 'scores' not in r:
                return None
            vals.append(r['scores'][metric]); ns.append(r['n_test'])
        return float(np.average(vals, weights=ns))

def dm_test(loss_a, loss_b, lag=24):
    """Diebold-Mariano on d = loss_a - loss_b with Newey-West variance (Bartlett, lag)."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    T = len(d); dm = d - d.mean()
    v = dm @ dm / T
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (dm[k:] @ dm[:-k]) / T
    stat = d.mean() / math.sqrt(max(v, 1e-300) / T)
    p = math.erfc(abs(stat) / math.sqrt(2))
    return stat, p

def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

def main_table(runs_, fname, caption_runs):
    metrics = [('nll', 'NLL'), ('crps', 'CRPS'), ('pinball05', r'PL$_{0.05}$'), ('pinball95', r'PL$_{0.95}$'), ('cover90', 'Cov.$_{90}$')]
    lines = [r'\begin{tabular}{l' + 'r' * len(metrics) * 1 + '}', r'\toprule']
    for cs in ['2d', '3d', '4d']:
        lines.append(r'\multicolumn{' + str(len(metrics) + 1) + r'}{l}{\textit{p(price $\mid$ ' + COVLAB[cs] + r')}} \\')
        lines.append('Model & ' + ' & '.join(m[1] for m in metrics) + r' \\')
        lines.append(r'\midrule')
        table = {}
        for mdl in MAIN:
            table[mdl] = {k: score_of(cs, mdl, runs_, k) for k, _ in metrics}
        best = {}
        for k, _ in metrics:
            vals = [(v[k], m) for m, v in table.items() if v[k] is not None]
            if not vals:
                continue
            best[k] = (min(vals, key=lambda t: abs(t[0] - 0.9)) if k == 'cover90' else min(vals))[1]
        ref_nll = pooled(cs, 'BLML', runs_, 'logpdf')
        ref_crps = pooled(cs, 'BLML', runs_, 'crps')
        for mdl in MAIN:
            v = table[mdl]
            if all(x is None for x in v.values()):
                continue
            cells = [TEXNAME.get(mdl, mdl)]
            for k, _ in metrics:
                if v[k] is None:
                    cells.append('--'); continue
                s = f'{v[k]:.3f}' if k != 'cover90' else f'{v[k]:.2f}'
                if mdl != 'BLML' and k in ('nll', 'crps'):
                    other = pooled(cs, mdl, runs_, 'logpdf') if k == 'nll' else (pooled(cs, mdl, runs_, 'crps') if pooled(cs, mdl, runs_, 'crps') is not None else pooled(cs, mdl, runs_, 'crps_q'))
                    ref = (-ref_nll if k == 'nll' else ref_crps) if ref_nll is not None else None
                    if other is not None and ref is not None and len(other) == len(ref):
                        la, lb = (-ref_nll, -other) if k == 'nll' else (ref, other)
                        _, p = dm_test(la, lb)
                        s += r'$^{' + stars(p) + '}$' if stars(p) else ''
                if best.get(k) == mdl:
                    s = r'\textbf{' + s + '}'
                cells.append(s)
            lines.append(' & '.join(cells) + r' \\')
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    open(os.path.join(OUT, 'tables', fname), 'w').write('\n'.join(lines))

if folds:
    main_table(folds, 'rolling.tex', 'rolling')
if 'random' in runs:
    main_table(['random'], 'random.tex', 'random')

# per-fold NLL figure (4-D and 3-D), BLML vs strongest baselines
def fold_fig(cs):
    cand = [m for m in MAIN if m != 'BLML' and score_of(cs, m, folds, 'nll') is not None]
    if not folds or score_of(cs, 'BLML', folds, 'nll') is None or not cand:
        return
    top = sorted(cand, key=lambda m: score_of(cs, m, folds, 'nll'))[:2]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(folds))
    for mdl, col, mk in [('BLML', BLUE, 'o')] + list(zip(top, [ORANGE, AQUA], ['s', '^'])):
        y = [score_of(cs, mdl, [f], 'nll') for f in folds]
        ax.plot(x, y, color=col, marker=mk, label=TEXNAME.get(mdl, mdl))
    ax.set_xticks(x); ax.set_xticklabels([f[5:] + '/' + f[2:4] for f in folds])
    ax.set_xlabel('test month (2024)'); ax.set_ylabel('test NLL (nats)'); ax.set_title(f'p(price | {COVLAB[cs]})')
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'figs', f'folds_{cs}.pdf'), bbox_inches='tight'); plt.close(fig)
for cs in ['2d', '3d', '4d']:
    fold_fig(cs)

# selector ablation
def ablation_table():
    lines = [r'\begin{tabular}{lrrrrrr}', r'\toprule',
             r' & \multicolumn{3}{c}{rolling-origin NLL} & \multicolumn{3}{c}{random-split NLL} \\',
             r'Selector & hour & +doy & +wkday,month & hour & +doy & +wkday,month \\', r'\midrule']
    for mdl in ABL:
        cells = [ABLLAB[mdl]]
        for runs_ in (folds, ['random']):
            for cs in ['2d', '3d', '4d']:
                v = score_of(cs, mdl, runs_, 'nll') if runs_ else None
                cells.append('--' if v is None else f'{v:.3f}')
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(OUT, 'tables', 'ablation.tex'), 'w').write('\n'.join(lines))
ablation_table()

# selected cutoffs per fold
def fc_table():
    lines = [r'\begin{tabular}{llll}', r'\toprule', r'Run & hour & hour, doy & hour, weekday, month \\', r'\midrule']
    for run in runs:
        cells = [run]
        for cs in ['2d', '3d', '4d']:
            r = by.get((run, cs, 'BLML'))
            cells.append('--' if r is None or 'fc' not in r else '(' + ', '.join(f'{f:.3g}' for f in r['fc']) + ')')
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(OUT, 'tables', 'fc.tex'), 'w').write('\n'.join(lines))
fc_table()

# PIT histograms (4-D, rolling)
def pit_fig(cs='4d'):
    cand = ['BLML', 'CondKDE', 'GMM', 'MDN', 'NGBoost-Normal', 'TabPFN']
    have = [m for m in cand if pooled(cs, m, folds, 'pit') is not None]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(1.2 * len(have) + 0.6, 1.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, have):
        pit = pooled(cs, m, folds, 'pit')
        ax.hist(pit, bins=10, range=(0, 1), density=True, color=BLUE if m == 'BLML' else '#8a8984',
                edgecolor='white', linewidth=1.0)
        ax.axhline(1, color=INK2, lw=0.8, ls=(0, (4, 2)))
        ax.set_title(TEXNAME.get(m, m), fontsize=7); ax.set_xticks([0, 0.5, 1])
    axes[0].set_ylabel('density')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'figs', f'pit_{cs}.pdf'), bbox_inches='tight'); plt.close(fig)
for cs in ['3d', '4d']:
    pit_fig(cs)

# ------------------------------------------------------------------ battery backtest
ETA, EMAX = 0.85, 4

def schedule(prices):
    """DP over SOC 0..EMAX (MWh), +-1 MWh/h, discharge revenue eta*p; start/end empty."""
    T = len(prices); NEG = -1e18
    V = np.full((T + 1, EMAX + 1), NEG); V[T, 0] = 0.0
    act = np.zeros((T, EMAX + 1), dtype=int)
    for t in range(T - 1, -1, -1):
        for s in range(EMAX + 1):
            best, ba = NEG, 0
            for a in (-1, 0, 1):              # -1 discharge, +1 charge
                s2 = s + a
                if 0 <= s2 <= EMAX and V[t + 1, s2] > NEG:
                    r = -prices[t] if a == 1 else (ETA * prices[t] if a == -1 else 0.0)
                    if r + V[t + 1, s2] > best:
                        best, ba = r + V[t + 1, s2], a
            V[t, s], act[t, s] = best, ba
    acts, s = [], 0
    for t in range(T):
        a = act[t, s]; acts.append(a); s += a
    return np.array(acts)

def realized(acts, prices):
    return float(np.sum(np.where(acts == 1, -prices, np.where(acts == -1, ETA * prices, 0.0))))

def battery():
    import sys
    sys.path.insert(0, HERE)
    import caiso_data
    ts_all = caiso_data.timestamps()
    price_all = caiso_data.load()['price']
    index = {str(t): i for i, t in enumerate(ts_all)}
    rows = {}
    any_run = folds
    if not any_run:
        return
    for cs in ['2d', '3d', '4d']:
        for mdl in MAIN + ['PerfectForesight', 'NaiveLastWeek']:
            total, days = 0.0, 0
            ok = True
            for run in any_run:
                base = by.get((run, cs, 'BLML')) or next((r for k, r in by.items() if k[0] == run and k[1] == cs), None)
                if base is None:
                    ok = False; break
                times = np.array(base['test_times']); y = np.array(base['y'])
                if mdl == 'PerfectForesight':
                    f = y
                elif mdl == 'NaiveLastWeek':
                    f = np.array([price_all[index[str(np.datetime64(t) - np.timedelta64(168, 'h'))]]
                                  if str(np.datetime64(t) - np.timedelta64(168, 'h')) in index else np.nan for t in times])
                    f = np.where(np.isnan(f), np.nanmean(f), f)
                else:
                    r = by.get((run, cs, mdl))
                    if r is None or 'per_obs' not in r:
                        ok = False; break
                    f = np.array(r['per_obs'].get('median', r['per_obs']['mean']))
                day = np.array([t[:10] for t in times])
                for dd in np.unique(day):
                    sel = day == dd
                    if sel.sum() < 20:
                        continue
                    total += realized(schedule(f[sel]), y[sel]); days += 1
            if ok and days:
                rows.setdefault(cs, {})[mdl] = total / days * 365 / 1000   # k$/MW-yr (1 MW, 4 MWh)
    if not rows:
        return
    lines = [r'\begin{tabular}{lrrr}', r'\toprule', r'Forecast & hour & hour, doy & hour, weekday, month \\', r'\midrule']
    order = ['PerfectForesight'] + MAIN + ['NaiveLastWeek']
    for mdl in order:
        if not any(mdl in rows.get(cs, {}) for cs in rows):
            continue
        lab = {'PerfectForesight': 'Perfect foresight (upper bound)', 'NaiveLastWeek': 'Naive: same hour last week'}.get(mdl, TEXNAME.get(mdl, mdl))
        cells = [lab] + [f"{rows[cs][mdl]:.1f}" if mdl in rows.get(cs, {}) else '--' for cs in ['2d', '3d', '4d']]
        lines.append(' & '.join(cells) + r' \\')
        if mdl == 'PerfectForesight':
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(OUT, 'tables', 'battery.tex'), 'w').write('\n'.join(lines))
    json.dump(rows, open(os.path.join(OUT, 'tables', 'battery.json'), 'w'), indent=1)
    return rows
bat = battery()

# ------------------------------------------------------------------ numbers
def put(k, v):
    nums[k] = v
for cs in ['2d', '3d', '4d']:
    for mdl in MAIN:
        v = score_of(cs, mdl, folds, 'nll') if folds else None
        if v is not None:
            key = 'nll' + cs.replace('2d', 'Two').replace('3d', 'Three').replace('4d', 'Four') + ''.join(c for c in mdl if c.isalpha())
            put(key, f'{v:.3f}')
with open(os.path.join(OUT, 'numbers.tex'), 'w') as f:
    f.write('% generated by analyze_p2.py -- do not edit\n')
    for k, v in sorted(nums.items()):
        f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
print('runs:', runs, '| models:', len(models), '| records:', len(recs))

# ------------------------------------------------------------------ validation-design ablation (June fold)
def validation_ablation():
    schemes = [('random rows', os.path.join(HERE, 'results', 'archive', 'p2_rowsplit_validation', 'p2')),
               ('random days', os.path.join(HERE, 'results', 'archive', 'p2_dayblock_validation', 'p2')),
               ('28-day blocks', RES)]
    mdls = ['BLML', 'BLML-inv_sigma', 'CondKDE', 'CondKNN', 'GMM', 'MDN', 'CatBoost-Normal']
    cell = {}
    for lab, d in schemes:
        for cs in ['2d', '3d']:
            for m in mdls:
                f = os.path.join(d, f'2024-06_{cs}_{m}.json')
                if os.path.exists(f):
                    r = json.load(open(f))
                    if 'scores' in r and 'nll' in r['scores']:
                        cell[(lab, cs, m)] = r['scores']['nll']
    if not cell:
        return
    lines = [r'\begin{tabular}{l' + 'r' * 6 + '}', r'\toprule',
             r' & \multicolumn{3}{c}{p(price $\mid$ hour)} & \multicolumn{3}{c}{p(price $\mid$ hour, doy)} \\',
             'Model & ' + ' & '.join(lab for lab, _ in schemes) * 1 + ' & ' + ' & '.join(lab for lab, _ in schemes) + r' \\', r'\midrule']
    for m in mdls:
        if not any((lab, cs, m) in cell for lab, _ in schemes for cs in ['2d', '3d']):
            continue
        cells = [TEXNAME.get(m, m) if m != 'BLML-inv_sigma' else r'BLML, rule $f_c=1/\sigma$ (untuned)']
        for cs in ['2d', '3d']:
            for lab, _ in schemes:
                v = cell.get((lab, cs, m))
                cells.append('--' if v is None else f'{v:.3f}')
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(OUT, 'tables', 'validation_design.tex'), 'w').write('\n'.join(lines))
validation_ablation()

# ------------------------------------------------------------------ paired log-likelihood tables (primary)
def ll_table(runs_, fname):
    """Mean test LL per model, per-point delta LL vs BLML (mean +- s.e.), paired
    t-test and Wilcoxon signed-rank p-values. Points are pooled over runs_."""
    from scipy import stats
    lines = [r'\begin{tabular}{lrrrrr}', r'\toprule']
    for cs in ['2d', '3d', '4d']:
        ref = pooled(cs, 'BLML', runs_, 'logpdf')
        if ref is None:
            continue
        lines += [r'\multicolumn{6}{l}{\textit{p(price $\mid$ ' + COVLAB[cs] + r'), ' + f'{len(ref):,}' + r' test points}} \\',
                  r'Model & test LL & $\Delta$LL vs BLML & $t$ & $p_t$ & $p_{\mathrm{Wilcoxon}}$ \\', r'\midrule']
        rows = []
        for mdl in MAIN:
            lp = pooled(cs, mdl, runs_, 'logpdf')
            if lp is None or len(lp) != len(ref):
                continue
            if mdl == 'BLML':
                rows.append((lp.mean(), f"BLML & {lp.mean():.3f} & -- & -- & -- & -- \\\\"))
                continue
            dlt = lp - ref                       # > 0: model better than BLML at that point
            t, pt = stats.ttest_rel(lp, ref)
            pw = stats.wilcoxon(dlt).pvalue
            fmt = lambda p: '<1e-4' if p < 1e-4 else f'{p:.3f}'
            rows.append((lp.mean(), f"{TEXNAME.get(mdl, mdl)} & {lp.mean():.3f} & {dlt.mean():+.3f} $\\pm$ {dlt.std(ddof=1)/np.sqrt(len(dlt)):.3f} & {t:.1f} & {fmt(pt)} & {fmt(pw)} \\\\"))
        lines += [r for _, r in sorted(rows, key=lambda x: -x[0])] + [r'\midrule']
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    open(os.path.join(OUT, 'tables', fname), 'w').write('\n'.join(lines))

if 'random' in runs:
    ll_table(['random'], 'll_random.tex')
if folds:
    ll_table(folds, 'll_rolling_pooled.tex')
