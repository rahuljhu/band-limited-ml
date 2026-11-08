"""Paper 3, double slit: table and figure from dslit_job.py outputs.
Usage: python results/dslit_analyze.py <out dir> <set> [<set> ...]   (e.g. results/dslit_cpu electron)
Table: test log-likelihood per particle, paired difference to the best method with its standard error (per test
particle: log P_A - log P_B at its pixel; SE = sd / sqrt(n_test)), background weight beta, fc along the fringes against
the physical band of f: (1 + a/d) / P cycles / px (the field's band is (d + a) / (2 lambda z) in the aperture's units;
in pixels (1 + a/d) / (2 P); f has twice it), from the Fraunhofer fit.
Figure per set: (a) the counts; (b) the fringe-axis profile of the test counts with the fitted P; (c) |FT g| along the
fringe axis (BLML positive / complex, the Fraunhofer field) against the aperture, in units of the slit separation
(xi / d = nu P: slits of width a/d at +-1/2); (d) |FT f| against the aperture autocorrelation (peaks at 0, +-1).
Writes papers/paper3/tables/dslit_<set>.tex, papers/paper3/figs/dslit_<set>.png/pdf, prints the table."""
import os, sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
D, SETS = sys.argv[1], sys.argv[2:]; DATA = os.path.expanduser('~/data_repo/dslit_gpu/')
LAB = dict(positive='BLML, positive root', complex='BLML, complex', signed='BLML, Fresnel signs', fresnel='Fresnel',
           fraunhofer='Fraunhofer', kde='KDE', hist='histogram')
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
for name in SETS:
    R = {p: json.load(open(f'{D}/{name}_{p}.json')) for p in LAB if os.path.exists(f'{D}/{name}_{p}.json')}
    pts = np.load(DATA + f'dslit_{name}_points.npy'); b = np.load(DATA + f'dslit_{name}_b.npy'); SH = b.shape[-2:]; axis = int(np.load(DATA + f'dslit_{name}_axis.npy'))
    ij = [np.clip(np.floor(pts[:, k] + 0.5).astype(int), 0, SH[k - 1] - 1) for k in (1, 2)]
    u = np.random.default_rng(0).random(len(pts)); te = u >= 0.80; tv = ~te
    LP = {p: np.log(np.maximum(np.load(f'{D}/{name}_{p}_P.npy').astype(float), 1e-300))[ij[0][te], ij[1][te]] for p in R if os.path.exists(f'{D}/{name}_{p}_P.npy')}
    best = max(LP, key=lambda p: LP[p].mean()); fr = R['fraunhofer']['params']; phys = (1 + fr['a_over_d']) / fr['P']
    rows = []
    for p in sorted(LP, key=lambda p: -LP[p].mean()):
        d = LP[p] - LP[best]; se = d.std(ddof=1) / np.sqrt(len(d))
        fc = R[p].get('fc'); fca = None if fc is None else fc[axis]
        be = float(np.sum(R[p].get('beta', [0.0])))                         # total background weight
        rows.append((LAB[p], LP[p].mean(), d.mean(), se, be, fca))
        print(f"{LAB[p]:24s} {LP[p].mean():.5f}  diff {d.mean():+.5f} +- {se:.5f}  beta {np.round(R[p].get('beta', [0.0]), 3)}  fc {fca if fca is None else round(fca, 4)}"
              f"{'' if fca is None else f'  ({fca / phys:.2f} x physical {phys:.4f})'}")
    with open(f'papers/paper3/tables/dslit_{name}.tex', 'w') as f:
        f.write('\\begin{tabular}{lrrrr}\n\\toprule\nmodel & test LL & $-$ best ($\\pm$ SE) & $\\beta$ & $f_c$ / physical \\\\\n\\midrule\n')
        for lab, m, dm, se, be, fca in rows:
            f.write(f"{lab} & {m:.4f} & {'--' if dm == 0 else f'{dm:+.4f} ({se:.4f})'} & {'--' if not be else f'{be:.2f}'} & "
                    f"{'--' if fca is None else f'{fca / phys:.2f}'} \\\\\n")
        f.write('\\bottomrule\n\\end{tabular}\n')
    # ---------------- figure
    C = np.zeros(SH); np.add.at(C, (ij[0][tv], ij[1][tv]), 1); Ct = np.zeros(SH); np.add.at(Ct, (ij[0][te], ij[1][te]), 1)
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
    ax[0].imshow(np.sqrt(C), cmap='gray', aspect='auto'); ax[0].set_title(f'(a) {name}: {len(pts):,} particles (sqrt counts)')
    prof = Ct.sum(1 - axis); s = np.arange(len(prof)); ax[1].plot(s, prof, color='0.6', lw=0.6, label='test counts')
    for p, c in (('fraunhofer', 'k'), ('positive', '#9ec3e6'), ('complex', '#1f6fb4')):
        if os.path.exists(f'{D}/{name}_{p}_P.npy'):
            P = np.load(f'{D}/{name}_{p}_P.npy').astype(float); ax[1].plot(s, P.sum(1 - axis) * Ct.sum(), color=c, lw=1, label=LAB[p])
    ax[1].legend(frameon=False, fontsize=6); ax[1].set_title('(b) along the fringes'); ax[1].set_xlabel('pixel')
    pad = 8 * len(prof); nu = np.fft.fftshift(np.fft.fftfreq(pad)); xi = nu * fr['P']
    for p, c in (('fraunhofer', 'k'), ('positive', '#9ec3e6'), ('complex', '#1f6fb4')):
        fn = f'{D}/{name}_{p}_line.npy'
        if os.path.exists(fn):
            g = np.load(fn); G = np.abs(np.fft.fftshift(np.fft.fft(g, pad))); ax[2].plot(xi, G / G.max(), color=c, lw=1, label=LAB[p])
            F = np.abs(np.fft.fftshift(np.fft.fft(np.abs(g) ** 2, pad))); ax[3].plot(xi, F / F.max(), color=c, lw=1, label=LAB[p])
    r = fr['a_over_d']
    for c0 in (-0.5, 0.5): ax[2].axvspan(c0 - r / 2, c0 + r / 2, color='orange', alpha=0.25)
    for c0 in (-1, 0, 1): ax[3].axvline(c0, color='orange', lw=0.8, alpha=0.6)
    for a_ in ax[2:]: a_.set_xlim(-2, 2); a_.set_xlabel(r'aperture coordinate $\xi/d$')
    ax[2].set_title('(c) |FT g| vs the slits (shaded)'); ax[3].set_title('(d) |FT f| vs the autocorrelation'); ax[2].legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(f'papers/paper3/figs/dslit_{name}.png', dpi=160); fig.savefig(f'papers/paper3/figs/dslit_{name}.pdf')
