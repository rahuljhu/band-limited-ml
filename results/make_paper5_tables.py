"""Paper 5: synthetic selector study tables, figure and numbers.
  results/p2syn5/syn/*.json     oracle (BLML at fc_true), Gaussian KDEs (paper 2's frozen GPU runs)
  results/synstream2/*.json     streaming BLML (2 epochs), same samples and evaluation, n <= 1e4
writes papers/paper2/tables/synth.tex, papers/paper2/figs/synth_stream.pdf, papers/paper2/numbers.tex.
Run from the repo root."""
import os, glob, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P5 = 'papers/paper2/'
TRUTHS, DS, NS = ['T1', 'T2', 'T3'], [2, 4, 8], [1000, 10000]     # streaming run up to n = 1e4
KDES = {'kde_scott': "Scott", 'kde_lscv': 'LSCV', 'kde_lcv': 'LCV', 'kde_adaptive': 'adaptive'}
med = lambda v: float(np.median(v)) if len(v) else np.nan
F, S = {}, {}
for f in glob.glob('results/p2syn5/syn/*.json'):
    r = json.load(open(f))
    if r['truth'] in TRUTHS:
        F[(r['truth'], r['d'], r['n'])] = r
SYN = 'results/synstream2'                        # the current streaming code; results/synstream: the first design
if len(glob.glob(SYN + '/T*_d*_n*.json')) < 18:
    print(f'WARNING: {SYN} incomplete, using results/synstream (first streaming design)'); SYN = 'results/synstream'
for f in glob.glob(SYN + '/T*_d*_n*.json'):
    r = json.load(open(f)); S[(r['truth'], r['d'], r['n'])] = r

rows, num = [], {}
cmp_kde, cmp_or, fcerr = [], [], []
FC_T = 0.4                                         # T1, T2: true cut-off in every dimension; T3 has none
for t in TRUTHS:
    for d in DS:
        for n in NS:
            if (t, d, n) not in F or (t, d, n) not in S:
                continue
            fr, sr = F[(t, d, n)]['reps'], S[(t, d, n)]['reps']
            orc = med([x['blml_oracle']['rel_ise'] for x in fr]) if 'blml_oracle' in fr[0] else np.nan
            kd = {k: med([x[k]['rel_ise'] for x in fr]) for k in KDES}
            kb = min(kd, key=kd.get)
            stv = med([x['stream']['rel_ise'] for x in sr])
            fe = med([np.sqrt(np.mean((np.array(x['stream']['fc']) / FC_T - 1) ** 2)) for x in sr]) if t != 'T3' else np.nan
            cmp_kde.append(kd[kb] / stv)
            if np.isfinite(orc):
                cmp_or.append(stv / orc); fcerr.append(fe)
            fmt = lambda v: '--' if not np.isfinite(v) else '$' + f'{v:.2e}'.replace('e-0', 'e-').replace('e-', '\\text{e-}') + '$'
            fep = '--' if not np.isfinite(fe) else f'{100 * fe:.0f}\\%'
            rows.append(f"{t} & {d} & $10^{{{int(np.log10(n))}}}$ & {fep} & {fmt(stv)} & {fmt(orc)} & "
                        f"{fmt(kd[kb])} ({KDES[kb]}) \\\\")
open(P5 + 'tables/synth.tex', 'w').write(
    "\\begin{tabular}{llrcccc}\n\\toprule\n"
    "& & & $f_c$ error & \\multicolumn{3}{c}{relative ISE} \\\\\n\\cmidrule(lr){5-7}\n"
    "truth & $d$ & $n$ & (rel.\\ rms) & streaming & oracle & best KDE \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
cmp_kde, cmp_or, fcerr = map(np.array, (cmp_kde, cmp_or, fcerr))
if len(cmp_kde):
    num.update(SynStreamCells=len(cmp_kde), SynStreamBeatsKDE=int((cmp_kde > 1).sum()),
               SynKDEOverStreamRange=f'{cmp_kde.min():.1f}--{cmp_kde.max():.0f}',
               SynStreamOverOracleMed=f'{np.median(cmp_or):.2f}',
               SynStreamOverOracleRange=f'{cmp_or.min():.2f}--{cmp_or.max():.2f}',
               SynFcErrMed=f'{100 * np.median(fcerr):.0f}', SynFcErrRange=f'{100 * fcerr.min():.0f}--{100 * fcerr.max():.0f}')

# figure: competitor error / streaming error (median over replicates), n <= 1e4
fig, axs = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True)
cols = {2: '#2a78d6', 4: '#eb6834', 8: '#1baf7a'}
for ax, t in zip(axs, TRUTHS):
    for d in DS:
        xs, os_, kd_, or_ = [], [], [], []
        for n in NS:
            if (t, d, n) in S and (t, d, n) in F:
                fr = F[(t, d, n)]['reps']; stv = med([x['stream']['rel_ise'] for x in S[(t, d, n)]['reps']])
                xs.append(n); os_.append(med([x['blml_auto']['rel_ise'] for x in fr]) / stv)
                kd_.append(min(med([x[k]['rel_ise'] for x in fr]) for k in KDES) / stv)
                or_.append(med([x['blml_oracle']['rel_ise'] for x in fr]) / stv if 'blml_oracle' in fr[0] else np.nan)
        if xs:
            ax.plot(xs, kd_, ':s', color=cols[d], label=f'best KDE, d={d}')
            if np.any(np.isfinite(or_)):
                ax.plot(xs, or_, '--o', color=cols[d], label=f'oracle, d={d}')
    ax.axhline(1, color='0.5', lw=0.8); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title(t)
    ax.set_xlabel('n'); ax.set_xticks([1000, 10000]); ax.set_xticklabels(['$10^3$', '$10^4$'])
axs[0].set_ylabel('error of competitor / error of streaming')
axs[0].legend(fontsize=6, frameon=False, ncol=1, loc='upper left', bbox_to_anchor=(1.0, 1.0))
fig.tight_layout(); fig.savefig(P5 + 'figs/synth_stream.pdf'); plt.close(fig)
print(num, f'{len(S)} streaming cells')

# figure: cut-off trajectories, 2-D T1/T2, n = 1e4 (results/rolling/fc_trace.py); reference lines: the true
# cut-off and the ISE-optimal common cut-off of replicate 0 (results/rolling/fc_surface.py)
TP, SP_ = 'results/synstream/fc_trace.json', 'results/synstream/fc_surface.json'
if os.path.exists(TP):
    TR = json.load(open(TP)); SU = json.load(open(SP_)) if os.path.exists(SP_) else {}
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.9), sharey=True)
    runs = {'default': ('#2a78d6', 'burn-in search, 2 passes'), 'start0.2': ('#eb6834', 'start 0.2, 5 passes'),
            'start0.8': ('#1baf7a', 'start 0.8, 5 passes')}
    for ax, t in zip(axs, ('T1', 'T2')):
        for run, (c, lab) in runs.items():          # geometric mean over replicates and both columns
            acc = {}
            for k, tr in TR.items():
                if k.startswith(f'{t}/{run}/'):
                    for s_ in tr:
                        acc.setdefault(s_['seen'], []).extend(np.log(s_['fc']))
            x = np.array(sorted(acc)); y = np.exp([np.mean(acc[v]) for v in x])
            ax.plot(x / 1e4, y, '-', color=c, lw=1.6, label=lab)
        ax.axhline(0.4, color='k', lw=0.8, ls=':', label='true cut-off' if t == 'T1' else None)
        if t in SU:
            iso = [r for r in SU[t] if r['fc'][0] == r['fc'][1]]
            best = min(iso, key=lambda r: r['rel_ise'])['fc'][0]
            ax.axhline(best, color='0.4', lw=0.8, ls='-.', label='ISE-optimal (rep. 0)' if t == 'T1' else None)
        ax.set_title(f'{t}, $d=2$, $n=10^4$'); ax.set_xlabel('points seen / $n$ (passes)'); ax.set_yscale('log')
        ax.set_yticks([0.2, 0.3, 0.4, 0.6, 0.8]); ax.set_yticklabels(['0.2', '0.3', '0.4', '0.6', '0.8'])
    axs[0].set_ylabel('cut-off $f_c$ (mean)'); axs[0].legend(fontsize=6.5, frameon=False, loc='lower right')
    fig.tight_layout(); fig.savefig(P5 + 'figs/fc_trace.pdf'); plt.close(fig)
    for run in runs:                                    # final relative rms fc error vs the truth (T1), median over reps
        e = [np.sqrt(np.mean((np.array(tr[-1]['fc']) / 0.4 - 1) ** 2)) for k, tr in TR.items() if k.startswith(f'T1/{run}/')]
        num['TraceTOneErr' + {'default': 'Search', 'start0.2': 'Low', 'start0.8': 'High'}[run]] = f'{100 * np.median(e):.0f}'

# CPU speed table (results/bench_cpu_stream.py, saved to results/bench/cpu_stream.json)
CP = 'results/bench/cpu_stream.json'
if os.path.exists(CP):
    rows = json.load(open(CP))['rows']
    lines = [f"${r['N'] // 1000}$K & {r['stream_s']:.1f} & {r['burn_in_s']:.1f} & {r['oneshot_s']:.1f} & "
             f"{r['speedup']:.1f}$\\times$ & {r['stream_ll']:.4f} & {r['oneshot_ll']:.4f} \\\\" for r in rows]
    open(P5 + 'tables/cpu.tex', 'w').write(
        "\\begin{tabular}{rrrrrrr}\n\\toprule\n& \\multicolumn{2}{c}{streaming (s)} & one-shot & & \\multicolumn{2}{c}{held-out LL} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){6-7}\n$n$ & total & burn-in & (s) & speed-up & streaming & one-shot \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    num.update(CpuSpeedupRange=f"{min(r['speedup'] for r in rows):.1f}--{max(r['speedup'] for r in rows):.1f}")

# GPU speed table (results/bench_stream_gpu_log.py, BURN_SEARCH=shared, T4, float32; results/bench/gpu_stream.json)
GP = 'results/bench/gpu_stream.json'
if os.path.exists(GP):
    rows = json.load(open(GP))['rows']
    by = {}
    for r in rows:
        by.setdefault(r['N'], {})['os' if r['method'] == 'one-shot' else 'st'] = r
    lines, sp = [], []
    for N in sorted(by):
        st, os_ = by[N]['st'], by[N]['os']; sp.append(os_['total_s'] / st['total_s'])
        lines.append(f"${N // 1000}$K & {st['total_s']:.1f} & {st['burn_in_s']:.1f} & {os_['total_s']:.1f} & {sp[-1]:.1f}$\\times$ & "
                     f"{st['ll']:.4f} & {os_['ll']:.4f} \\\\")
    open(P5 + 'tables/gpu.tex', 'w').write(
        "\\begin{tabular}{rrrrrrr}\n\\toprule\n& \\multicolumn{2}{c}{streaming (s)} & one-shot & & \\multicolumn{2}{c}{held-out LL} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){6-7}\n$n$ & total & burn-in & (s) & speed-up & streaming & one-shot \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    num.update(GpuSpeedupRange=f'{min(sp):.1f}--{max(sp):.1f}')

# gradient ablation (results/bench_ablation.py)
AB = 'results/bench/ablation.json'
if os.path.exists(AB):
    rows = json.load(open(AB))
    open(P5 + 'tables/ablation.tex', 'w').write(
        "\\begin{tabular}{lcc}\n\\toprule\nvariant ($n=2\\cdot10^4$, CPU, two passes) & time (s) & held-out log-lik. \\\\\n\\midrule\n"
        + "\n".join(f"{r['variant'].replace('+ ', '\\quad + ')} & {r['s']:.1f} & ${r['ll']:.4f}$ \\\\" for r in rows)
        + "\n\\bottomrule\n\\end{tabular}\n")

# period learning (results/rolling/period_learn.py)
PP = 'results/rolling/period_learn.json'
if os.path.exists(PP):
    R = json.load(open(PP)); P = np.array([r['P'] for r in R])
    num.update(PeriodRuns=len(R), PeriodRange=f'{P.min():.2f}--{P.max():.2f}', PeriodMean=f'{P.mean():.2f}',
               PeriodTime=f"{np.median([r['s'] for r in R]):.0f}")
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for r in R:
        ax.plot(np.arange(len(r['path'])), r['path'], color='#2a78d6' if r['P0'] < 24 else '#eb6834', lw=0.8, alpha=0.8)
    ax.axhline(24, color='k', lw=0.8, ls=':'); ax.set_xlabel('batch'); ax.set_ylabel('period (h)')
    fig.tight_layout(); fig.savefig(P5 + 'figs/period_path.pdf'); plt.close(fig)

open(P5 + 'numbers.tex', 'w').write('% generated by results/make_paper5_tables.py\n' +
                                    ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in num.items()))
print({k: v for k, v in num.items() if not k.startswith('Syn')})
