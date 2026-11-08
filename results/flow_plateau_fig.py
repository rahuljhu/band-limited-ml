"""Fixed flow architectures against n on T1, d = 4 (flow_job.py BLML_FLOW_PART=plateau -> results/flowplateau/plateau)
and BLML at the true cutoff (paper 1 runs, results/p1* r1 and results/p1big). Writes
papers/submit/jmlr_blml/figs/flow_sizes.pdf. Run from the repo root."""
import os, json, glob, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
src = open('analyze_p1.py').read()
exec(src[:src.index('# ------------------------------------------------------------------ R1 rate study')])
HERE = os.getcwd()                                   # analyze_p1 resolves results/ from its own folder
exec(src[src.index('def r1_table():'):src.index('def slope_ci(')])
R = r1_table()['T1'][4]
bl = {n: float(np.mean([m['rel_ise'] for m in R[n]['blml_fc1']])) for n in R}
for f in glob.glob('results/p1big/big/T1_d4_n*.json'):
    r = json.load(open(f)); bl[r['n']] = float(np.mean([x['blml_fc1']['rel_ise'] for x in r['reps']]))
fl = {}
for f in glob.glob('results/flowplateau/plateau/*.json'):
    r = json.load(open(f)); fl.setdefault(r['size'], {})[r['n']] = float(np.mean([x['rel_ise'] for x in r['reps']]))
LAB = {'tiny': 'flow, 1 layer of 8 (868 weights)', 'small': 'flow, 2 layers of 32 (8,504)', 'base': 'flow, 5 layers of 128 (145,100)'}
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for s, c in (('tiny', '#9ecae1'), ('small', '#4292c6'), ('base', '#08519c')):
    ns = sorted(fl[s]); ax.plot(ns, [fl[s][n] for n in ns], 'o-', color=c, label=LAB[s])
ns = sorted(bl); ax.plot(ns, [bl[n] for n in ns], 's-', color='#cb181d', label='BLML, true cutoff')
ax.plot([1e3, 1e6], [bl[1000], bl[1000] * 1e-3], ':', color='#cb181d', lw=1, label='$n^{-1}$')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('$n$'); ax.set_ylabel('relative ISE')
ax.legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig('papers/submit/jmlr_blml/figs/flow_sizes.pdf')
print({k: {n: round(v, 6) for n, v in sorted(d.items())} for k, d in fl.items()}, {n: round(v, 6) for n, v in sorted(bl.items())})
