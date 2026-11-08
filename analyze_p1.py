"""Paper 1 figures, tables and number macros from results/p1*/ JSON files.

    .venv/bin/python analyze_p1.py      # writes papers/paper1/{figs/*.pdf, tables/*.tex, numbers.tex}

Works with partial results: anything missing becomes '??' in the macros.
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SYS = os.path.join(HERE, 'results', 'p1x')       # final systems rerun (+ T3)
R1 = os.path.join(HERE, 'results', 'p1')
OLD_SYS = os.path.join(HERE, 'results', 'archive', 'p1_systems_unsafeguarded')
OUT = os.path.join(HERE, 'papers', 'paper1')
os.makedirs(os.path.join(OUT, 'figs'), exist_ok=True)
os.makedirs(os.path.join(OUT, 'tables'), exist_ok=True)

# Validated categorical slots (all-pairs pass for the first three); grays for the rest.
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
STYLE = {   # name: (label, color, linestyle, marker, filled, zorder)
    'blml_fc1':     ('BLML, $f_c=f_c^{true}$', BLUE, '-', 'o', True, 10),
    'blml_fc2':     ('BLML, $f_c=2f_c^{true}$', BLUE, '--', 'o', False, 9),
    'kde_oracle':   ('KDE, oracle $h$', ORANGE, '-', 's', True, 8),
    'kde_lscv':     ('KDE, LSCV $h$', AQUA, '-', '^', True, 7),
    'kde_scott':    ('KDE, Scott $h$', '#8a8984', '-', 'v', False, 5),
    'kde6_oracle':  ('6th-order KDE, oracle $h$', '#52514e', ':', 'D', False, 5),
    'sinc_kde_fc2': ('sinc KDE (kernel band = band of $f$)', '#3f3e3b', '-.', 'P', False, 5),
    'kde_adaptive': ('Adaptive KDE', '#a3a29d', '--', 'X', False, 4),
    'knn':          ('$k$-NN', '#c3c2b7', ':', '*', False, 3),
}
T3_LABEL = {'blml_fc1': 'BLML, $f_c=2$', 'blml_fc2': 'BLML, $f_c=4$', 'sinc_kde_fc2': 'sinc KDE, $f_c=4$'}
T3_LABEL.update({'blml_fc0.35': 'BLML, $f_c=0.7$', 'blml_fc0.5': 'BLML, $f_c=1$'})
SIGNED = {'kde6_oracle', 'sinc_kde_fc1', 'sinc_kde_fc2'}     # can be negative: not densities
NOT_DENSITY = SIGNED | {'knn'}                             # k-NN: not normalized

def label(est, T):
    """T3 is not band-limited: name its cutoffs by value, never as a 'true' cutoff."""
    return T3_LABEL[est] if (T == "T3" and est in T3_LABEL) else STYLE[est][0]

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5, 'legend.fontsize': 7,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 1.5, 'lines.markersize': 4, 'pdf.fonttype': 42,
})

def load(path):
    return json.load(open(path)) if os.path.exists(path) else None

def sysres(name):
    return load(os.path.join(SYS, name + '.json')) or load(os.path.join(R1, name + '.json'))

TRUTH_TITLE = {'T1': 'separable (T1)', 'T2': 'non-separable (T2)', 'T3': 'correlated normal (T3)'}
nums = {}
def num(key, val):
    nums[key] = val

# ------------------------------------------------------------------ R1 rate study
def r1_table():
    """{truth: {d: {n: {est: [metric per rep]}}}}"""
    out = {}
    files = {}      # same (truth, d, n) can exist in several job folders (same seeds): keep one,
    for folder in ['p1', 'p1t2', 'p1x']:       # later folders override earlier ones
        for f in glob.glob(os.path.join(HERE, 'results', folder, 'r1', '*.json')):
            files[os.path.basename(f)] = f
    for f in files.values():
        r = load(f)
        T, d, n = r['truth'], r['d'], r['n']
        cell = out.setdefault(T, {}).setdefault(d, {}).setdefault(n, {})
        # d <= 2 used a grid whose ISE was recorded as an average over grid
        # points; scale back to the grid sum (see eval_sets in p1_experiments.py).
        grid_n = r.get('grid_points') or {1: 16001, 2: 641 ** 2}.get(d, 1)
        for rep in r['reps']:
            for est, m in rep.items():
                m = dict(m)
                m['ise'] *= grid_n; m['rel_ise'] *= grid_n
                cell.setdefault(est, []).append(m)
    # T3 at low cutoffs (separate jobs, same seeds): add only the extra BLML fits
    for folder in ['p1t3fc', 'p1t3fc8']:
        for f in glob.glob(os.path.join(HERE, 'results', folder, 'r1', '*.json')):
            r = load(f)
            T, d, n = r['truth'], r['d'], r['n']
            cell = out.get(T, {}).get(d, {}).get(n)
            if cell is None:
                continue
            grid_n = r.get('grid_points') or {1: 16001, 2: 641 ** 2}.get(d, 1)
            for est in r['reps'][0]:
                if est.startswith('blml_fc') and est not in cell:
                    cell[est] = []
                    for rep in r['reps']:
                        m = dict(rep[est]); m['ise'] *= grid_n; m['rel_ise'] *= grid_n
                        cell[est].append(m)
    return out

def slope_ci(ns, reps_by_n, metric='rel_ise', B=2000, seed=0):
    """OLS slope of log mean error vs log n, with a bootstrap CI over replicates."""
    rng = np.random.default_rng(seed)
    ln = np.log(ns)
    means = [np.mean([m[metric] for m in reps_by_n[n]]) for n in ns]
    if min(means) <= 0:                      # e.g. a KL estimate whose mean is not positive
        return np.nan, np.array([np.nan, np.nan])
    s = np.polyfit(ln, np.log(means), 1)[0]
    boots = []
    for _ in range(B):
        mb = [np.mean(rng.choice([m[metric] for m in reps_by_n[n]], len(reps_by_n[n]))) for n in ns]
        if min(mb) > 0:
            boots.append(np.polyfit(ln, np.log(mb), 1)[0])
    return s, (np.percentile(boots, [2.5, 97.5]) if boots else np.array([np.nan, np.nan]))

R = r1_table()
slopes = {}
slopes_last = {}
slopes_kl, slopes_kl_last = {}, {}   # same, for the KL divergence     # slope from the two largest n only (asymptotic-like)
for T in sorted(R):
    for d in sorted(R[T]):
        ns = sorted(R[T][d])
        if len(ns) < 3:
            continue
        for est in STYLE:
            if all(est in R[T][d][n] for n in ns):
                slopes[(T, d, est)] = slope_ci(ns, {n: R[T][d][n][est] for n in ns})
                slopes_last[(T, d, est)] = slope_ci(ns[-2:], {n: R[T][d][n][est] for n in ns[-2:]})
                slopes_kl[(T, d, est)] = slope_ci(ns, {n: R[T][d][n][est] for n in ns}, metric='kl')
                slopes_kl_last[(T, d, est)] = slope_ci(ns[-2:], {n: R[T][d][n][est] for n in ns[-2:]}, metric='kl')

def fig_rate(T, metric='rel_ise', ylabel='relative ISE', fname=None):
    if T not in R:
        return
    ds = sorted(R[T])
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), sharex=True)
    for ax, d in zip(axes.ravel(), ds):
        ns = sorted(R[T][d])
        for est, (lab, col, ls, mk, fill, z) in STYLE.items():
            lab = label(est, T)
            if metric == 'kl' and est in NOT_DENSITY:
                continue
            if not all(est in R[T][d][n] for n in ns):
                continue
            vals = [np.array([m[metric] for m in R[T][d][n][est]]) for n in ns]
            mean = np.array([v.mean() for v in vals]); se = np.array([v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0 for v in vals])
            ax.errorbar(ns, mean, yerr=se, color=col, ls=ls, marker=mk, mfc=col if fill else 'white',
                        mec=col, label=lab, zorder=z, capsize=0, elinewidth=0.8)
        ref = np.array([ns[0], ns[-1]], float)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(f'$d={d}$')
        if d in (ds[0], ds[3] if len(ds) > 3 else ds[0]):
            ax.set_ylabel(ylabel)
    for ax in axes[-1]:
        ax.set_xlabel('$n$')
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(OUT, 'figs', fname or f'rate_{T}_{metric}.pdf'), bbox_inches='tight')
    plt.close(fig)

for T in R:
    fig_rate(T)
    fig_rate(T, 'kl', 'KL divergence')

def fig_slopes(tab_all, tab_last, fname, ylab, truths=('T1', 'T2')):
    Ts = [T for T in truths if T in R]
    if not Ts:
        return
    fig, axes = plt.subplots(2, len(Ts), figsize=(7.0 if len(Ts) > 1 else 3.9, 5.0), sharex=True, sharey='row', squeeze=False)
    for row, (tab, rowlab) in enumerate([(tab_all, 'all $n$ (1e3 to 1e5)'), (tab_last, 'two largest $n$ (3e4, 1e5)')]):
        for ax, T in zip(axes[row], Ts):
            for est in ['blml_fc1', 'blml_fc2', 'kde_oracle', 'kde_lscv', 'kde6_oracle', 'sinc_kde_fc2']:
                if 'KL' in ylab and est in NOT_DENSITY:
                    continue
                lab, col, ls, mk, fill, z = STYLE[est]
                lab = label(est, T) if len(Ts) == 1 else lab
                ds = [d for d in sorted(R[T]) if (T, d, est) in tab and np.isfinite(tab[(T, d, est)][0])]
                if not ds:
                    continue
                sl = np.array([tab[(T, d, est)][0] for d in ds])
                lo = np.array([tab[(T, d, est)][1][0] for d in ds]); hi = np.array([tab[(T, d, est)][1][1] for d in ds])
                ax.errorbar(ds, sl, yerr=[sl - lo, hi - sl], color=col, ls=ls, marker=mk,
                            mfc=col if fill else 'white', mec=col, label=lab, zorder=z, capsize=2, elinewidth=0.8)
            dd = np.linspace(1, 8, 50)
            ax.plot(dd, -4 / (4 + dd), color=INK2, lw=0.8, ls=(0, (1, 2)), label='KDE theory $-4/(4+d)$')
            ax.axhline(-1, color=INK2, lw=0.8, ls=(0, (4, 2)), label='$-1$')
            if row == 0:
                ax.set_title(TRUTH_TITLE[T])
            else:
                ax.set_xlabel('dimension $d$')
        axes[row][0].set_ylabel(ylab + ', ' + rowlab, fontsize=7)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(OUT, 'figs', fname), bbox_inches='tight')
    plt.close(fig)
fig_slopes(slopes, slopes_last, 'slopes.pdf', 'ISE slope')
fig_slopes(slopes_kl, slopes_kl_last, 'slopes_kl.pdf', 'KL slope')
# T3 (normal): not band-limited, no true cutoff -> its own figures
fig_slopes(slopes, slopes_last, 'slopes_T3.pdf', 'ISE slope', truths=('T3',))
fig_slopes(slopes_kl, slopes_kl_last, 'slopes_kl_T3.pdf', 'KL slope', truths=('T3',))

def fmt_slope(T, d, est):
    if (T, d, est) not in slopes:
        return '??'
    s, (lo, hi) = slopes[(T, d, est)]
    return f'{s:.2f} [{lo:.2f}, {hi:.2f}]'

def slope_table():
    if not R:
        return
    ests = ['blml_fc1', 'blml_fc2', 'kde_oracle', 'kde_lscv', 'kde_scott', 'kde6_oracle', 'sinc_kde_fc2', 'kde_adaptive', 'knn']
    lines = [r'\begin{tabular}{l' + 'c' * 6 + '}', r'\toprule',
             'Estimator & ' + ' & '.join(f'$d={d}$' for d in [1, 2, 3, 4, 6, 8]) + r' \\', r'\midrule']
    for T in sorted(R):
        lines.append(r'\multicolumn{7}{l}{\textit{' + TRUTH_TITLE[T] + r'}} \\')
        for est in ests:
            row = [label(est, T)]
            for d in [1, 2, 3, 4, 6, 8]:
                if (T, d, est) in slopes:
                    s, (lo, hi) = slopes[(T, d, est)]
                    row.append(f'${s:.2f}$\\,{{\\scriptsize[{lo:.2f},{hi:.2f}]}}')
                else:
                    row.append('--')
            lines.append(' & '.join(row) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(OUT, 'tables', 'slopes.tex'), 'w').write('\n'.join(lines))
slope_table()

def err_table():
    """Relative ISE at the largest n for every estimator and d."""
    if not R:
        return
    lines = [r'\begin{tabular}{l' + 'c' * 6 + '}', r'\toprule',
             'Estimator & ' + ' & '.join(f'$d={d}$' for d in [1, 2, 3, 4, 6, 8]) + r' \\', r'\midrule']
    for T in sorted(R):
        lines.append(r'\multicolumn{7}{l}{\textit{' + T + r', largest $n$ per $d$}} \\')
        best = {}
        for d in [1, 2, 3, 4, 6, 8]:
            if d in R[T]:
                n = max(R[T][d]); cell = R[T][d][n]
                best[d] = min(np.mean([m['rel_ise'] for m in v]) for k, v in cell.items()
                              if (k in STYLE or k in T3_LABEL) and k not in NOT_DENSITY)
        ests = (['blml_fc0.35', 'blml_fc0.5'] if T == 'T3' else []) + list(STYLE)
        for est in ests:
            mark = '$^\\dagger$' if est in SIGNED else ('$^\\ddagger$' if est == 'knn' else '')
            row = [label(est, T) + mark]
            for d in [1, 2, 3, 4, 6, 8]:
                if d in R[T] and est in R[T][d][max(R[T][d])]:
                    v = np.mean([m['rel_ise'] for m in R[T][d][max(R[T][d])][est]])
                    s = f'{v:.2e}'.replace('e-0', 'e-')
                    row.append(r'\textbf{' + s + '}' if np.isclose(v, best[d]) else s)
                else:
                    row.append('--')
            lines.append(' & '.join(row) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(OUT, 'tables', 'err_largest_n.tex'), 'w').write('\n'.join(lines))
err_table()

for T in ['T1', 'T2']:
    for d in [1, 2, 3, 4, 6, 8]:
        for est in STYLE:
            if (T, d, est) in slopes:
                num(f'slope{T}d{d}{est.replace("_", "")}', f'{slopes[(T, d, est)][0]:.2f}')
b1 = [slopes[('T1', d, 'blml_fc1')][0] for d in [1, 2, 3, 4, 6, 8] if ('T1', d, 'blml_fc1') in slopes]
k1 = [slopes[('T1', d, 'kde_oracle')][0] for d in [1, 2, 3, 4, 6, 8] if ('T1', d, 'kde_oracle') in slopes]
num('SlopeSummary', f'from ${min(b1):.2f}$ to ${max(b1):.2f}$ across $d$' if b1 else '??')
num('SlopeSummaryKDE', f'oracle KDE: ${k1[0]:.2f}$ at $d=1$ to ${k1[-1]:.2f}$ at the largest $d$' if k1 else '??')

# ------------------------------------------------------------------ systems figures
A = sysres('expA_traces')
if A:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    cols = {'n3000_d1': '#8a8984', 'n3000_d2': BLUE, 'n10000_d3': ORANGE, 'n3000_d8': AQUA}
    for k, v in A.items():
        lab = f"$n={v['n']}$, $d={v['d']}$ ($B={v['B']}$)"
        sw = np.arange(1, len(v['rho']) + 1)
        axes[0].semilogy(sw, v['rho'], marker='o', color=cols.get(k, INK2), label=lab)
        obj = np.array(v['obj']); gap = obj - obj[-1]
        axes[1].semilogy(sw[:-1], np.maximum(gap[:-1], 1e-12), marker='o', color=cols.get(k, INK2), label=lab)
    axes[0].axhline(1e-4, color=INK2, lw=0.8, ls=(0, (4, 2)))
    axes[0].set_xlabel('sweep'); axes[0].set_ylabel(r'$\max_b|\rho_b|$')
    axes[1].set_xlabel('sweep'); axes[1].set_ylabel(r'$(\Phi-\Phi_{\mathrm{final}})/n$')
    axes[1].legend(frameon=False, loc='upper right')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'figs', 'convergence.pdf'), bbox_inches='tight'); plt.close(fig)

CE = sysres('expCE_dim')
if CE:
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.2))
    for nn, col, mk in [(3000, BLUE, 'o'), (30000, ORANGE, 's')]:
        rows = [r for r in CE if r.get('n', 3000) == nn]
        if not rows:
            continue
        d = [r['d'] for r in rows]; lab = f'$n={nn:,}$'
        axes[0].plot(d, [r['B_over_n'] for r in rows], marker=mk, color=col, label=lab)
        axes[1].plot(d, [r['sweeps'] for r in rows], marker=mk, color=col, label=lab)
        axes[2].plot(d, [r['time_s'] for r in rows], marker=mk, color=col, label=lab)
    axes[0].set_ylabel('$B/n$'); axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel('sweeps to tol'); axes[1].set_ylim(0, None)
    axes[2].set_ylabel('fit time (s)'); axes[2].set_ylim(0, None); axes[2].legend(frameon=False)
    for ax in axes:
        ax.set_xlabel('dimension $d$')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'figs', 'dim.pdf'), bbox_inches='tight'); plt.close(fig)
    sw = [r['sweeps'] for r in CE if r['d'] >= 2 and r.get('n', 3000) == 3000]
    num('SweepRange', f'{min(sw)}--{max(sw)}')
    num('BoverNdOne', f"{CE[0]['B_over_n']:.2f}")

Bt = sysres('expB_time_vs_n')
if Bt:
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    n = np.array(Bt['n'], float)
    ax.loglog(n, Bt['gpu_f32_s'], marker='o', color=BLUE, label='GPU float32')
    ax.loglog(n, Bt['gpu_f64_s'], marker='s', mfc='white', color=ORANGE, label='GPU float64')
    ref = Bt['gpu_f64_s'][-1] * (n / n[-1]) ** 2
    ax.loglog(n, ref, color=INK2, lw=0.8, ls=(0, (4, 2)), label=r'$\propto n^2$')
    ax.set_xlabel('$n$ ($d=3$, $B\\approx n$)'); ax.set_ylabel('fit time (s)'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'figs', 'time_vs_n.pdf'), bbox_inches='tight'); plt.close(fig)
    num('TimeNlarge', f"{Bt['gpu_f32_s'][-1]:.1f}")

M = sysres('expM_memory')
if M:
    lines = [r'\begin{tabular}{rrrrr}', r'\toprule', r'$d$ & $n$ & $B$ & BLML GPU peak & dense Gram (float32) \\', r'\midrule']
    for r in M:
        lines.append(f"{r['d']} & {r['n']:,} & {r['B']:,} & {r['gpu_bytes']/1e6:,.0f} MB & {r['dense_gram_bytes']/1e9:,.2f} GB \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(OUT, 'tables', 'memory.tex'), 'w').write('\n'.join(lines))
    big = [r for r in M if r['d'] == 8 and r['n'] == 100000]
    if big:
        num('MemPeakDEight', f"{big[0]['gpu_bytes']/1e6:.0f}\\,MB")
        num('MemDenseDEight', f"{big[0]['dense_gram_bytes']/1e9:.0f}\\,GB")

F = sysres('expF_block')
if F:
    lines = [r'\begin{tabular}{rrrc}', r'\toprule', r'block size $B_s$ & sweeps & fit time (s) & monotone $\Phi$ \\', r'\midrule']
    for bs, r in F.items():
        lines.append(f"{bs} & {r['sweeps']} & {r['time_s']:.2f} & {'yes' if r['obj_monotone'] else 'no'} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(OUT, 'tables', 'block.tex'), 'w').write('\n'.join(lines))

S = sysres('expS_cpu_vs_gpu')
if S:
    lines = [r'\begin{tabular}{lrrrrrr}', r'\toprule',
             r'Workload & $B$ & \multicolumn{2}{c}{fit (s)} & \multicolumn{2}{c}{predict (s)} & speedup \\',
             r' & & CPU & GPU & CPU & GPU & fit / predict \\', r'\midrule']
    for r in S:
        lines.append(f"{r['workload']} & {r['B']:,} & {r['cpu_fit_s']:.1f} & {r['gpu_fit_s']:.2f} & {r['cpu_pred_s']:.1f} & {r['gpu_pred_s']:.3f} & {r['fit_speedup']:.0f}$\\times$ / {r['pred_speedup']:.0f}$\\times$ \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(OUT, 'tables', 'speed.tex'), 'w').write('\n'.join(lines))
    fs = [r['fit_speedup'] for r in S]; ps = [r['pred_speedup'] for r in S]
    num('SpeedFitRange', f'{min(fs):.0f}--{max(fs):.0f}$\\times$')
    num('SpeedPredRange', f'{min(ps):.0f}--{max(ps):.0f}$\\times$')
    num('SpeedLLmaxdiff', f"{max(abs(r['test_ll_cpu'] - r['test_ll_gpu']) for r in S):.1e}")

D = sysres('expD_precision')
if D:
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule', r'setting & max rel.\ diff & \makecell{max rel.\ diff\\($\hat f>10^{-3}\max\hat f$)} & sweeps & converged \\', r'\midrule']
    lines[2] = r'setting & max rel.\ diff & max rel.\ diff where $\hat f>10^{-3}\max\hat f$ & sweeps & converged \\'
    for k, r in D.items():
        lab = k.replace('_tol', ', tol $=$ ').replace('float', 'float')
        lines.append(f"{lab} & {r['max_rel']:.1e} & {r['max_rel_where_f_gt_1em3max']:.1e} & {r['sweeps']} & {'yes' if r['converged'] else 'no'} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(OUT, 'tables', 'precision.tex'), 'w').write('\n'.join(lines))

Bc = sysres('expBin_check')
if Bc:
    lines = [r'\begin{tabular}{rrcc}', r'\toprule', r'$d$ & $n$ & rel.\ ISE, $h_k=0.1n^{-1/4}/f_c$ & rel.\ ISE, $h_k=0.025n^{-1/4}/f_c$ \\', r'\midrule']
    by = {}
    for r in Bc:
        by.setdefault((r['d'], r['n']), {})[r['bin_scale']] = r
    for (d, n), v in sorted(by.items()):
        a, b = v.get(0.1), v.get(0.025)
        lines.append(f"{d} & {n:,} & {a['rel_ise']:.2e} $\\pm$ {a['rel_ise_sd']:.1e} & {b['rel_ise']:.2e} $\\pm$ {b['rel_ise_sd']:.1e} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(OUT, 'tables', 'bincheck.tex'), 'w').write('\n'.join(lines))

G = sysres('expG_caiso3d')
if G:
    num('CaisoThreeDFit', f"{G['fit_s']:.2f}")
    num('CaisoThreeDB', f"{G['B']:,}")

old = load(os.path.join(OLD_SYS, 'expCE_dim.json'))
if old:
    num('UnsafeSweepsDOne', str(old[0]['sweeps']))


# ------------------------------------------------------------------ rescaled error: n * rel. ISE / asymptote of the unbiased sinc KDE
def fig_scaled():
    from p1_truth import f1
    t = np.arange(-400.0, 400.0, 0.05)
    I1 = float(np.sum(f1(t) ** 2) * 0.05)             # int f1^2, so int f^2 = I1^d for T1
    big = {}
    for f in glob.glob(os.path.join(HERE, 'results', 'p1*', 'big', '*.json')):
        r = load(f); big[(r['d'], r['n'])] = r
    ds = [d for d in (3, 4, 6, 8) if d in R.get('T1', {})]
    if not ds:
        return
    fig, axes = plt.subplots(1, len(ds), figsize=(7.0, 2.6), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], ds):
        A = (0.8 / I1) ** d - 1        # n * rel. ISE of the sinc KDE with kernel band = band of f (unbiased): exact
        for est in ['blml_fc1', 'blml_fc2', 'sinc_kde_fc2']:
            lab, col, ls, mk, fill, z = STYLE[est]
            ns = sorted(R['T1'][d])
            xs = list(ns); y = [np.mean([m['rel_ise'] for m in R['T1'][d][n][est]]) * n / A for n in ns]
            if est.startswith('blml'):
                for (dd, n), r in sorted(big.items()):
                    if dd == d and est in r['reps'][0]:
                        xs.append(n); y.append(np.mean([x[est]['rel_ise'] for x in r['reps']]) * n / A)
            ax.plot(xs, y, color=col, ls=ls, marker=mk, mfc=col if fill else 'white', mec=col, label=lab, zorder=z)
        ax.axhline(1, color=INK2, lw=0.8, ls=(0, (4, 2)), label='unbiased sinc KDE, exact: $(f_c^d-\\int f^2)/n$')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title(f'$d={d}$'); ax.set_xlabel('$n$')
    axes[0][0].set_ylabel('$n\\cdot$rel. ISE / sinc-KDE constant', fontsize=7)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(OUT, 'figs', 'scaled_error.pdf'), bbox_inches='tight')
    plt.close(fig)
fig_scaled()


# ------------------------------------------------------------------ cutoff sensitivity: error vs fc/fc_ref
def fig_fc_sweep():
    data = {}
    for f in sorted(glob.glob(os.path.join(HERE, 'results', 'p1fc*', 'fcsweep*', '*.json'))):
        r = load(f); k = (r['truth'], r['d'], r['n'])
        if k not in data:
            data[k] = r
        else:                    # merge a second sweep over other cutoffs (same seeds, same cell)
            for rep, extra in zip(data[k]['reps'], r['reps']):
                rep.update(extra)
            data[k]['mults'] = sorted(set(data[k]['mults']) | set(r['mults']))
    cols = {1: '#8a8984', 2: BLUE, 4: ORANGE, 8: AQUA}
    ns = [10000, 100000]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex='col', sharey='col')
    for c, T in enumerate(['T1', 'T3']):
        for r_, n in enumerate(ns):
            ax = axes[r_][c]
            for d in [1, 2, 4, 8]:
                if (T, d, n) not in data:
                    continue
                r = data[(T, d, n)]
                ms = [m for m in r['mults'] if T == 'T3' or m >= 0.5]
                y = np.array([np.mean([rep[f'{m:g}']['rel_ise'] for rep in r['reps']]) for m in ms])
                xs = np.array([2 * m for m in ms] if T == 'T3' else ms)
                ib = int(np.argmin(y))
                lab = f'$d={d}$'
                if T == 'T1':            # right branch: slope of log error against log cutoff above the truth
                    rb = np.array(ms) >= 1
                    sl = np.polyfit(np.log(xs[rb]), np.log(y[rb]), 1)[0]
                    lab += f', slope {sl:.1f}'
                    print(f'fc sweep {T} n={n} d={d}: best fc/fc_ref = {ms[ib]:g}; right-branch slope {sl:.2f}')
                else:
                    print(f'fc sweep {T} n={n} d={d}: best fc = {xs[ib]:g}')
                ax.plot(xs, y, marker='o', ms=3.5, color=cols[d], label=lab)
                ax.plot(xs[ib], y[ib], marker='o', ms=9, mfc='none', mec=cols[d], mew=1.6, zorder=5)
            if T == 'T1':
                ax.axvline(1, color=INK2, lw=0.8, ls=(0, (4, 2)))
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_title(f'{TRUTH_TITLE[T]}, $n={n:,}$', fontsize=8)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.legend(frameon=False, fontsize=6.5, loc='lower left' if T == 'T1' else 'upper left')
        if T == 'T3':
            axes[1][c].set_xlabel('cutoff $f_c$')
            axes[1][c].set_xticks([0.25, 0.5, 1, 2, 4]); axes[1][c].set_xticklabels(['0.25', '0.5', '1', '2', '4'])
        else:
            axes[1][c].set_xlabel('$f_c/f_c^{true}$ ($f_c^{true}=0.4$)')
            axes[1][c].set_xticks([0.5, 1, 2]); axes[1][c].set_xticklabels(['0.5', '1', '2'])
    for a in axes[:, 0]:
        a.set_ylabel('relative ISE')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figs', 'fc_sensitivity.pdf'), bbox_inches='tight')
    plt.close(fig)
fig_fc_sweep()


# ------------------------------------------------------------------ the truths in 2-D, for intuition
def fig_truths():
    from p1_truth import Truth
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6))
    for j, (T, L0, L) in enumerate([('T1', 10, 25), ('T2', 18, 25), ('T3', 3.5, 3.5)]):
        tr = Truth(T, 2)
        g = np.linspace(-L0, L0, 301); X1, X2 = np.meshgrid(g, g)
        f = tr.pdf(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)
        im = axes[0, j].pcolormesh(g, g, f, cmap='Blues', shading='auto', rasterized=True)
        fig.colorbar(im, ax=axes[0, j], shrink=0.85)
        g = np.linspace(-L, L, 301); X1, X2 = np.meshgrid(g, g)
        f = tr.pdf(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)
        lf = np.log10(np.maximum(f / f.max(), 1e-6))
        im = axes[1, j].pcolormesh(g, g, lf, cmap='Blues', vmin=-6, vmax=0, shading='auto', rasterized=True)
        fig.colorbar(im, ax=axes[1, j], shrink=0.85, ticks=[-6, -4, -2, 0])
        axes[0, j].set_title(TRUTH_TITLE[T])
        for ax in axes[:, j]:
            ax.set_aspect('equal'); ax.grid(False); ax.set_xlabel('$x_1$')
    axes[0, 0].set_ylabel('$f(x)$\n$x_2$'); axes[1, 0].set_ylabel('$\\log_{10} f/\\max f$\n$x_2$')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figs', 'truths_2d.pdf'), bbox_inches='tight', dpi=200)
    plt.close(fig)
fig_truths()


# ------------------------------------------------------------------ Fig. 1: what the solver costs, measured vs theory
def fig_complexity():
    if not (CE and Bt and M):
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
    ax = axes[0]
    for nn, col, mk in [(3000, BLUE, 'o'), (30000, ORANGE, 's')]:
        rows = [r for r in CE if r.get('n', 3000) == nn]
        ax.plot([r['d'] for r in rows], [r['B_over_n'] for r in rows], marker=mk, color=col, label=f'$n={nn:,}$')
    ax.set_xlabel('dimension $d$'); ax.set_ylabel('occupied bins $B/n$'); ax.set_ylim(0, 1.05)
    ax.set_title('(a) binning stops compressing'); ax.legend(frameon=False, loc='lower right')

    ax = axes[1]
    nn = np.logspace(3, 6, 50)
    ax.loglog(nn, 4 * nn ** 2, color=INK2, lw=1.0, ls=(0, (4, 2)), label='dense Gram, $4n^2$ B')
    for d, col, mk in [(3, BLUE, 'o'), (8, ORANGE, 's')]:
        rows = [r for r in M if r['d'] == d]
        ax.loglog([r['n'] for r in rows], [r['gpu_bytes'] for r in rows], marker=mk, color=col, label=f'BLML, $d={d}$')
    c = np.mean([r['gpu_bytes'] / r['n'] for r in M if r['n'] >= 10000])
    ax.loglog(nn, c * nn, color=BLUE, lw=0.8, ls=':', label='$O(n)$')
    ax.axhline(40e9, color=INK, lw=0.6); ax.text(1.5e5, 1.2e10, 'A100, 40 GB', fontsize=6.5)
    ax.set_xlabel('$n$'); ax.set_ylabel('GPU memory (bytes)'); ax.set_title('(b) memory')
    ax.legend(frameon=False, fontsize=6, loc='upper left')

    ax = axes[2]
    n3 = np.array(Bt['n'], float)
    ax.loglog(n3, Bt['gpu_f64_s'], marker='o', color=BLUE, label='$d=3$')
    pts = {}
    for n in sorted(R.get('T1', {}).get(8, {})):
        pts[n] = np.mean([m['fit_s'] for m in R['T1'][8][n]['blml_fc1']])
    for f in glob.glob(os.path.join(HERE, 'results', 'p1*', 'big', 'T1_d8_*.json')):
        r = load(f); pts[r['n']] = np.mean([rep['blml_fc1']['fit_s'] for rep in r['reps']])
    n8 = np.array(sorted(pts), float); t8 = np.array([pts[n] for n in sorted(pts)])
    ax.loglog(n8, t8, marker='s', color=ORANGE, label='$d=8$')
    ax.loglog(nn, t8[-1] * (nn / n8[-1]) ** 2, color=INK2, lw=1.0, ls=(0, (4, 2)), label=r'$O(n^2d)$ per sweep')
    ax.set_xlabel('$n$'); ax.set_ylabel('fit time (s)'); ax.set_title('(c) time')
    ax.legend(frameon=False, fontsize=6, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figs', 'complexity.pdf'), bbox_inches='tight')
    plt.close(fig)
    num('FitTimeBig', f'{t8[-1] / 60:.0f} minutes')
    print('fig1 d=8 fit times:', {int(k): round(float(v), 2) for k, v in zip(n8, t8)})
fig_complexity()

with open(os.path.join(OUT, 'numbers.tex'), 'w') as f:
    f.write('% generated by analyze_p1.py -- do not edit\n')
    for k in ['SlopeSummary', 'SlopeSummaryKDE', 'SweepRange', 'MemPeakDEight', 'MemDenseDEight',
              'SpeedFitRange', 'SpeedPredRange', 'SpeedLLmaxdiff', 'TimeNlarge', 'CaisoThreeDFit',
              'CaisoThreeDB', 'BoverNdOne', 'UnsafeSweepsDOne', 'FitTimeBig']:
        nums.setdefault(k, '??')
    for k, v in sorted(nums.items()):
        name = ''.join(ch for ch in k if ch.isalpha())
        digits = ''.join(ch for ch in k if ch.isdigit())
        name = name + ''.join('ABCDEFGHIJ'[int(c)] for c in digits)
        f.write(f'\\newcommand{{\\{name}}}{{{v}}}\n')
print('wrote', len(nums), 'numbers;', len(slopes), 'slopes;', 'R1 cells:', {T: sorted(R[T]) for T in R})


# ------------------------------------------------------------------ T3: the cutoff trades bias for variance
def fig_t3_cutoffs():
    if 'T3' not in R:
        return
    ds = [d for d in (1, 2, 4, 8) if d in R['T3']]
    fig, axes = plt.subplots(1, len(ds), figsize=(7.0, 2.6), squeeze=False)
    seq = plt.get_cmap('Blues')
    mults = [0.25, 0.35, 0.5, 0.7, 1.0, 2.0]
    for ax, d in zip(axes[0], ds):
        ns = sorted(R['T3'][d])
        for i, m in enumerate(mults):
            est = f'blml_fc{m:g}'
            if not all(est in R['T3'][d][n] for n in ns):
                continue
            y = [np.mean([x['rel_ise'] for x in R['T3'][d][n][est]]) for n in ns]
            ax.loglog(ns, y, marker='o', ms=3, color=seq(0.35 + 0.65 * i / (len(mults) - 1)), label=f'BLML $f_c={2 * m:g}$')
        for est, col, ls in [('kde6_oracle', ORANGE, '--'), ('kde_oracle', AQUA, ':')]:
            y = [np.mean([x['rel_ise'] for x in R['T3'][d][n][est]]) for n in ns]
            ax.loglog(ns, y, color=col, ls=ls, lw=1.2, label=STYLE[est][0])
        ax.set_title(f'$d={d}$'); ax.set_xlabel('$n$')
        for n in ns:
            row = {e: np.mean([x['rel_ise'] for x in R['T3'][d][n][e]]) for e in R['T3'][d][n] if e.startswith('blml') or e.startswith('kde')}
            b = min((v, e) for e, v in row.items() if e.startswith('blml'))
            k = min((v, e) for e, v in row.items() if e.startswith('kde'))
            print(f'T3 d={d} n={n}: best BLML {b[1]} {b[0]:.2e} | best KDE {k[1]} {k[0]:.2e} | ratio {b[0] / k[0]:.2f}')
    axes[0][0].set_ylabel('relative ISE')
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(OUT, 'figs', 'rate_T3_cutoffs.pdf'), bbox_inches='tight')
    plt.close(fig)
fig_t3_cutoffs()


# ------------------------------------------------------------------ main-text slope figure: asymptotic slopes + cost of 2 fc
def fig_slopes_asym():
    big = {}
    for f in glob.glob(os.path.join(HERE, 'results', 'p1*', 'big', 'T1_*.json')):
        r = load(f); big[(r['d'], r['n'])] = r
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    ests = ['blml_fc1', 'blml_fc2', 'kde_oracle', 'kde_lscv', 'kde6_oracle', 'sinc_kde_fc2']
    for ax, T in zip(axes[:2], ('T1', 'T2')):
        for est in ests:
            lab, col, ls, mk, fill, z = STYLE[est]
            ds, sl, lo, hi = [], [], [], []
            for d in sorted(R[T]):
                ns = sorted(R[T][d])
                reps = {n: R[T][d][n].get(est) for n in ns[-2:]}
                if T == 'T1' and est.startswith('blml') and (d, 1000000) in big:   # use the large-n runs
                    reps = {n: [rep[est] for rep in big[(d, n)]['reps']] for n in (300000, 1000000)}
                if any(v is None for v in reps.values()):
                    continue
                s, (a, b) = slope_ci(sorted(reps), reps)
                ds.append(d); sl.append(s); lo.append(a); hi.append(b)
            sl, lo, hi = map(np.array, (sl, lo, hi))
            ax.errorbar(ds, sl, yerr=[sl - lo, hi - sl], color=col, ls=ls, marker=mk, mfc=col if fill else 'white',
                        mec=col, label=lab, zorder=z, capsize=2, elinewidth=0.8)
        dd = np.linspace(1, 8, 50)
        ax.plot(dd, -4 / (4 + dd), color=INK2, lw=0.8, ls=(0, (1, 2)), label='KDE theory $-4/(4+d)$')
        ax.axhline(-1, color=INK2, lw=0.8, ls=(0, (4, 2)), label='$-1$')
        ax.set_title(TRUTH_TITLE[T]); ax.set_xlabel('dimension $d$')
    axes[0].set_ylabel('asymptotic ISE slope'); axes[1].sharey(axes[0])
    ax = axes[2]
    for T, col, mk in [('T1', BLUE, 'o'), ('T2', ORANGE, 's')]:
        ds = sorted(R[T]); n = 100000
        ratio = [np.mean([m['rel_ise'] for m in R[T][d][n]['blml_fc2']]) / np.mean([m['rel_ise'] for m in R[T][d][n]['blml_fc1']]) for d in ds]
        ax.semilogy(ds, ratio, marker=mk, color=col, label=f'{T}, $n=10^5$')
    ds = [d for d in (4, 6, 8) if (d, 1000000) in big]
    if ds:
        ratio = [np.mean([rep['blml_fc2']['rel_ise'] for rep in big[(d, 1000000)]['reps']]) /
                 np.mean([rep['blml_fc1']['rel_ise'] for rep in big[(d, 1000000)]['reps']]) for d in ds]
        ax.semilogy(ds, ratio, marker='o', mfc='white', color=BLUE, ls='--', label='T1, $n=10^6$')
    dd = np.linspace(1, 8, 50)
    ax.semilogy(dd, 2.0 ** dd, color=INK2, lw=0.8, ls=(0, (4, 2)), label='$2^d$')
    ax.set_xlabel('dimension $d$'); ax.set_ylabel('ISE at $2f_c^{true}$ / ISE at $f_c^{true}$')
    ax.set_title('cost of a doubled cutoff'); ax.legend(frameon=False, fontsize=6)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(os.path.join(OUT, 'figs', 'slopes_asym.pdf'), bbox_inches='tight')
    plt.close(fig)
fig_slopes_asym()
