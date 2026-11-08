"""Paper 2 tables (streaming KDE section) from results/stream_kde.json: 1-d ISE slopes and n = 64k ISE (tables/kde_ise.tex), excess KL at
n = 64k for d = 1, 2 (tables/kde_kl.tex), bandwidth slopes; macros in tables/kde_numbers.tex."""
import json, numpy as np
R = json.load(open('results/stream_kde.json')); P = 'papers/paper2/tables/'
M = [('stream_kde', 'streaming KDE'), ('kde_oracle', 'KDE, oracle $h$'), ('kde_val', 'KDE, validated $h$'),
     ('kde_scott', 'KDE, Scott'), ('stream_blml', 'streaming BLML')]
e = lambda v: f'{v:.1e}'.replace('e-0', 'e-')
rows = []
for kind in ('T1', 'T2', 'T3'):
    rs = [r for r in R if r['kind'] == kind and r['d'] == 1]
    for r in rs:
        v = [r['ise_' + k][-1] for k, _ in M]; b = int(np.argmin(v))
        rows.append(f"{kind} & {r['seed']} & " + ' & '.join((r'\textbf{%s}' if i == b else '%s') % e(x) for i, x in enumerate(v)) + ' & '
                    + ' & '.join(f"${r['slopes']['ise_' + k]:+.2f}$" for k, _ in M) + r' \\')
    rows.append(r'\midrule')
open(P + 'kde_ise.tex', 'w').write(r'\begin{tabular}{ll rrrrr rrrrr}' '\n' r'\toprule' '\n'
    r' & & \multicolumn{5}{c}{ISE at $n=64{,}000$} & \multicolumn{5}{c}{slope of $\log$ ISE in $\log n$} \\' '\n'
    r'\cmidrule(lr){3-7}\cmidrule(lr){8-12}' '\n'
    r'truth & seed & str.\ KDE & oracle & valid. & Scott & str.\ BLML & str.\ KDE & oracle & valid. & Scott & str.\ BLML \\' '\n'
    r'\midrule' '\n' + '\n'.join(rows[:-1]) + '\n' r'\bottomrule' '\n' r'\end{tabular}' '\n')
rows = []
for d in (1, 2):
    for kind in ('T1', 'T2', 'T3'):
        for r in [r for r in R if r['kind'] == kind and r['d'] == d]:
            v = [r[k][-1] for k, _ in M]; b = int(np.argmin(v))
            rows.append(f"{kind} & {d} & {r['seed']} & " + ' & '.join((r'\textbf{%s}' if i == b else '%s') % f'{x:.5f}' for i, x in enumerate(v))
                        + f" & {r['h_stream'][-1][0] / r['h_oracle'][-1][0]:.2f}" + r' \\')
    rows.append(r'\midrule')
open(P + 'kde_kl.tex', 'w').write(r'\begin{tabular}{lll rrrrr r}' '\n' r'\toprule' '\n'
    r'truth & $d$ & seed & ' + ' & '.join(l for _, l in M) + r' & $h_{\rm str}/h_{\rm or}$ \\' '\n' r'\midrule' '\n'
    + '\n'.join(rows[:-1]) + '\n' r'\bottomrule' '\n' r'\end{tabular}' '\n')
one = [r for r in R if r['d'] == 1]
rng = lambda k, rs: (min(r['slopes'][k] for r in rs), max(r['slopes'][k] for r in rs))
hs = lambda k, rs: (min(r['h_slope'][k] for r in rs), max(r['h_slope'][k] for r in rs))
T12 = [r for r in one if r['kind'] in ('T1', 'T2')]; T1 = [r for r in R if r['kind'] == 'T1']
win = sum(min(r['ise_' + k][-1] for k in ('kde_oracle', 'kde_val', 'kde_scott')) > r['ise_stream_blml'][-1] for r in one)
mac = dict(KdeOracleSlopeLo=rng('ise_kde_oracle', T12)[0], KdeOracleSlopeHi=rng('ise_kde_oracle', T12)[1],
           BlmlSlopeLo=rng('ise_stream_blml', one)[0], BlmlSlopeHi=rng('ise_stream_blml', one)[1],
           HOracleSlopeLo=hs('h_oracle_ise', one)[0], HOracleSlopeHi=hs('h_oracle_ise', one)[1],
           HStreamTOneLo=hs('h_stream', T1)[0], HStreamTOneHi=hs('h_stream', T1)[1],
           StreamKdeTOneIseLo=rng('ise_stream_kde', [r for r in one if r['kind'] == 'T1'])[0],
           StreamKdeTOneIseHi=rng('ise_stream_kde', [r for r in one if r['kind'] == 'T1'])[1])
s = ''.join(f'\\newcommand{{\\{k}}}{{${v:+.2f}$}}\n' for k, v in mac.items()) + f'\\newcommand{{\\KdeBlmlIseWins}}{{{win}}}\n'
open(P + 'kde_numbers.tex', 'w').write(s); print(s)
