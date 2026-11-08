"""Tables, numbers.tex and figures for paper 3 from results/psf/v3/{bead*.json, sim/, lc/}."""
import glob, json, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
V = 'results/psf/v3/'; OUT = 'papers/paper3/'; os.makedirs(OUT + 'figs', exist_ok=True); os.makedirs(OUT + 'tables', exist_ok=True)
M = ['BLML-W', 'BLML-W@phys', 'BLML', 'histogram', 'KDE', 'GMM+bg', 'Gaussian+bg']
LAB = {'BLML-W': 'BLML-W', 'BLML-W@phys': r'BLML-W at $2\mathrm{NA}/\lambda$', 'BLML': 'BLML (plain)', 'histogram': 'histogram',
       'KDE': 'KDE', 'GMM+bg': 'GMM+bg', 'Gaussian+bg': 'Gaussian+bg'}
COL = {'BLML-W': '#1f6fb4', 'BLML-W@phys': '#1f6fb4', 'BLML': '#9ec3e6', 'histogram': '0.35', 'KDE': '#6a9a3a', 'GMM+bg': '#e08a00', 'Gaussian+bg': '#b8322a'}
LS = {'BLML-W@phys': '--'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
num = {}
# ---------------- beads ----------------
R = [json.load(open(f)) for f in sorted(glob.glob(V + 'bead*.json'), key=lambda s: int(s.split('bead')[1].split('.')[0]))]
if R:
    n = np.array([r['photons'] for r in R]); ll = {m: np.array([r[m]['ll'] for r in R]) for m in M}
    ref = ll['histogram']
    rows = []
    for m in M:
        d = ll[m] - ref; dw = ll['BLML-W'] - ll[m]
        rows.append((m, np.median(d), np.mean(d), (d > 0).sum(), (dw > 0).sum() if m != 'BLML-W' else None))
    print(f'{len(R)} beads; photons {n.min():,}-{n.max():,} median {int(np.median(n)):,}')
    print(f"{'method':14s} {'med Δ vs hist':>14s} {'mean Δ':>9s} {'beats hist':>11s} {'BLML-W beats it':>16s}")
    for m, md, mn, w, bw in rows:
        print(f'{m:14s} {md:+14.5f} {mn:+9.5f} {w:5d}/{len(R)} {"" if bw is None else f"{bw:5d}/{len(R)}":>16s}')
    fcs = np.array([r['BLML-W']['fc'] for r in R]); print('BLML-W fc median', np.median(fcs, 0), 'at 0.5 cap:', (fcs >= 0.5 - 1e-9).sum(0))
    num.update(NBeads=len(R), PhotMin=f'{n.min():,}', PhotMax=f'{n.max():,}', PhotMed=f'{int(np.median(n)):,}',
               FcWxMed=f'{np.median(fcs[:, 0]):.2f}', FcWyMed=f'{np.median(fcs[:, 1]):.2f}')
    with open(OUT + 'tables/beads.tex', 'w') as f:
        f.write('\\begin{tabular}{lrrrr}\n\\toprule\nMethod & median $\\Delta$ & mean $\\Delta$ & beats histogram & BLML-W better \\\\\n\\midrule\n')
        for m, md, mn, w, bw in rows:
            if m == 'histogram': continue
            f.write(f"{LAB[m]} & {md:+.4f} & {mn:+.4f} & {w}/{len(R)} & {'--' if bw is None else f'{bw}/{len(R)}'} \\\\\n")
        f.write('\\bottomrule\n\\end{tabular}\n')
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8)); a = axs[0]; rng = np.random.default_rng(0)
    show = [m for m in M if m != 'histogram']
    for i, m in enumerate(show):
        d = ll[m] - ref; a.scatter(i + rng.uniform(-.18, .18, len(d)), d, s=10, color=COL[m], alpha=.8); a.hlines(np.median(d), i - .3, i + .3, color='k')
    a.axhline(0, color='0.6', lw=.8); a.set_xticks(range(len(show))); a.set_xticklabels([LAB[m].replace('$', '').replace('\\mathrm{NA}', 'NA').replace('\\lambda', 'λ').replace('\\', '') for m in show], rotation=20)
    a.set_yscale('symlog', linthresh=1e-3); a.set_ylabel('test LL − histogram (nats/photon)'); a.set_title('(a) per bead, relative to the pixel histogram')
    a = axs[1]
    for m in ['BLML-W', 'BLML-W@phys', 'BLML', 'KDE', 'GMM+bg']:
        a.scatter(n, ll[m] - ref, s=10, color=COL[m], marker='o' if m != 'BLML-W@phys' else 'x', label=LAB[m].replace('$', '').replace('\\mathrm{NA}', 'NA').replace('\\lambda', 'λ').replace('\\', ''))
    nn = np.geomspace(n.min(), n.max(), 50); a.plot(nn, 960 / (2 * 0.8 * nn), ':', color='0.4', lw=1, label='histogram excess (B−1)/2n')
    a.axhline(0, color='0.6', lw=.8); a.set_xscale('log'); a.set_yscale('symlog', linthresh=1e-3)
    a.set_xlabel('photons in window'); a.set_ylabel('test LL − histogram'); a.set_title('(b) against bead brightness'); a.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(OUT + 'figs/beads.pdf'); fig.savefig(OUT + 'figs/beads.png', dpi=160)
# ---------------- simulation ----------------
S = [json.load(open(f)) for f in glob.glob(V + 'sim/*.json')]
if S:
    cases = [c for c in ['airy', 'defocus', 'defocus+fp'] if any(s['case'] == c for s in S)]
    fig, axs = plt.subplots(1, len(cases), figsize=(4.2 * len(cases), 3.6), sharey=True); axs = np.atleast_1d(axs)
    for a, c in zip(axs, cases):
        ns = sorted({s['n'] for s in S if s['case'] == c})
        for m in M:
            y = [np.mean([s[m]['kl'] for s in S if s['case'] == c and s['n'] == nv]) for nv in ns]
            a.plot(ns, y, LS.get(m, '-'), marker='o', ms=3, color=COL[m], label=LAB[m].replace('$', '').replace('\\mathrm{NA}', 'NA').replace('\\lambda', 'λ').replace('\\', ''))
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('photons'); a.set_title({'airy': '(a) in focus', 'defocus': '(b) defocus 1 rad', 'defocus+fp': '(c) defocus + 3% pixel gain pattern'}[c])
        print('sim', c, {m: [round(np.mean([s[m]['kl'] for s in S if s['case'] == c and s['n'] == nv]), 5) for nv in ns] for m in ['BLML-W', 'BLML-W@phys', 'histogram', 'KDE', 'GMM+bg']}, 'n', ns)
    axs[0].set_ylabel('KL(truth ‖ estimate), nats'); axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(OUT + 'figs/sim.pdf'); fig.savefig(OUT + 'figs/sim.png', dpi=160)
# ---------------- learning curves ----------------
L = [json.load(open(f)) for f in glob.glob(V + 'lc/*.json')]
if L:
    beads = sorted({l['bead'] for l in L})
    fig, axs = plt.subplots(1, len(beads), figsize=(4.2 * len(beads), 3.6), sharey=False); axs = np.atleast_1d(axs)
    for a, b in zip(axs, beads):
        ns = sorted({l['n_target'] for l in L if l['bead'] == b})
        best = {nv: max(np.mean([l[m]['ll'] for l in L if l['bead'] == b and l['n_target'] == nv]) for m in M) for nv in ns}
        for m in M:
            y = [best[nv] - np.mean([l[m]['ll'] for l in L if l['bead'] == b and l['n_target'] == nv]) for nv in ns]
            a.plot(ns, np.maximum(y, 1e-5), LS.get(m, '-'), marker='o', ms=3, color=COL[m], label=LAB[m].replace('$', '').replace('\\mathrm{NA}', 'NA').replace('\\lambda', 'λ').replace('\\', ''))
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('training photons'); a.set_title(f'bead {b}')
    axs[0].set_ylabel('test LL below the best method'); axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(OUT + 'figs/lc.pdf'); fig.savefig(OUT + 'figs/lc.png', dpi=160)
num['PhysFc'] = '0.328'
with open(OUT + 'numbers.tex', 'w') as f:
    for k, v in num.items(): f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
print('numbers:', num)
