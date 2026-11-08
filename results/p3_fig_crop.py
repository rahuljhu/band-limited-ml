"""Paper 3, Fig. 2: Fig. 1's truth on a record-like window of seven fringes (p3_cbench.py + p3_cbench_phase.py with
CB_TRUTH=tapered CB_M=3 CB_N=0: results/p3_cbench_tap). (a) the far field on the line and the window; (b) what the
window leaves of the aperture: |A| and the window's ground truth A * sinc_L (its Fourier coefficients a_k, k / L);
then against n, median over six data sets: (c) KL on the window (grey: slope 1/n); (d) slit separation; (e) fringe contrast; (f) relative
slit phase (about the fitted envelope centre; invariant to the aperture's position, which |g|^2 does not fix); (g) amplitude ratio given a coherent source (unequal slits);
(h) aperture error against A * sinc_L. Grey: the n -> inf fit (results/p3_cbench_tap_floor, the band of least KL), the
floor of the periodic class on this window. Methods: periodic two-slit kernel and KDE on all three truths; the three-mode model (run on the partially
coherent truth only) and the line kernel, which belongs on uncropped patterns, only in the table. Table tables/crop.tex. Run from the repo root."""
import json, glob, collections, numpy as np, matplotlib, matplotlib.ticker
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
PP, D = 'papers/submit/pra_complex', 'results/p3_cbench_tap'
R = collections.defaultdict(list)
for f in glob.glob(f'{D}/*.json'):
    d = json.load(open(f))
    if 'phase' in d:
        for e, s in d['phase'].items():
            if e != 'truth':
                d[e].update(s)
        R[(d['case'], d['n'])].append(d)
CASES = ['sym', 'asym', 'partial']; NS = sorted({n for _, n in R})
M = [('periodic', 'periodic two-slit kernel', 'C3', 's', '-'),                  # the line kernel: tables/crop.tex only
     ('kde', 'KDE', '0.4', 'x', ':')]          # the three-mode model (partially coherent truth only): tables/crop.tex
med = lambda c, n, e, m: np.median([d[e][m] for d in R[(c, n)] if e in d and m in d[e]]) if any(e in d and m in d[e] for d in R[(c, n)]) else np.nan
gm = lambda v: (lambda v: np.exp(np.mean(np.log(v))) if v else np.nan)([x for x in v if np.isfinite(x) and x > 0])
fig, ax = plt.subplots(2, 4, figsize=(7.0, 3.9)); ax = ax.ravel()
# (a), (b): the truth (equal slits), the window, A and A * sinc_L
D_, A_, NF, W = 0.017, 0.005, 1 << 16, 2048; nu = np.fft.fftfreq(NF)
sl = lambda c: np.where(np.abs(nu - c) < A_ / 2, np.cos(np.pi * np.abs(nu - c) / A_) ** 2, 0.0)
g = np.fft.fftshift(np.fft.ifft(sl(-D_ / 2) + sl(D_ / 2)))[NF // 2 - W // 2:NF // 2 + W // 2] * NF; I = np.abs(g) ** 2
Pf = 1 / D_; a0, a1 = int(round(1024 - 3.5 * Pf)), int(round(1024 + 3.5 * Pf)); L = a1 - a0; xs = np.arange(W) - 1024
a = ax[0]; a.plot(xs, I / I.max(), color='0.3', lw=0.7); a.axvspan(a0 - 1024, a1 - 1024, color='C3', alpha=0.12)
a.set_xlim(-700, 700); a.set_title('(a) far field and window', fontsize=8, loc='left'); a.set_xlabel('detector position (px)'); a.set_yticks([])
a = ax[1]; A = np.fft.fftshift(sl(-D_ / 2) + sl(D_ / 2)); nus = np.fft.fftshift(nu)
ak = np.fft.fft(g[a0:a1]); k = np.rint(np.fft.fftfreq(L) * L); o = np.argsort(k)
a.plot(nus / D_, A / A.max(), color='0.3', lw=1, label='aperture $A$')
a.plot(k[o] / L / D_, np.abs(ak[o]) / np.abs(ak).max(), 'o', color='C3', ms=2.5, label=r'$A*\mathrm{sinc}_L$ at $k/L$')
a.set_xlim(-1.2, 1.2); a.set_xlabel(r'spatial frequency / $d$'); a.set_title(r'(b) what the window keeps', fontsize=8, loc='left')
a.legend(fontsize=5.5, frameon=False, loc='upper center'); a.set_yticks([])
PAN = [('kl', CASES, 'KL on the window', '(c) intensity', ()), ('d_err', CASES, 'relative error', '(d) slit separation', ('kde',)),
       ('C_err', CASES, 'absolute error', '(e) fringe contrast', ()), ('phi_rel_err', ['asym'], 'error (rad)', '(f) relative slit phase', ('kde', 'periodic_three')),
       ('rho_err', ['asym'], 'absolute error', r'(g) $\rho$ given $V=1$', ('kde', 'periodic_three')),
       ('ap_err', ['sym', 'asym'], r'$\|\hat a-a\|$', r'(h) aperture error', ('kde', 'periodic_three'))]
for a, (m, cases, yl, title, skip) in zip(ax[2:], PAN):
    for e, nm, col, mk, ls in M:
        if e in skip:
            continue
        y = [gm([med(c, n, e, m) for c in cases]) for n in NS]
        if np.isfinite(y).any():
            a.loglog(NS, y, color=col, marker=mk, ls=ls, lw=1.0, ms=3, label=nm)
    fl = [d_ for d_ in (json.load(open(f)) for f in glob.glob(f'{D}_floor/*.json')) if d_['case'] in cases]
    if fl and m in ('ap_err', 'phi_rel_err', 'kl', 'C_err'):
        v = gm([min(f['fits'], key=lambda r: r['kl'])[m] for f in fl])
        if np.isfinite(v):
            a.axhline(v, color='0.6', lw=0.8, ls='--', label=r'periodic, $n\to\infty$')
    a.set_title(title, fontsize=8, loc='left'); a.set_xlabel('particles $n$'); a.set_ylabel(yl, fontsize=7)
ax[2].loglog(NS, 0.8 * gm([med(c, NS[-1], 'periodic', 'kl') for c in CASES]) * NS[-1] / np.array(NS, float), color='0.55', lw=0.8, ls=':')   # slope 1/n
h, l = ax[7].get_legend_handles_labels(); hk, lk = ax[2].get_legend_handles_labels(); h += [x for x, y in zip(hk, lk) if y not in l]; l += [y for y in lk if y not in l]
fig.legend(h, l, loc='upper center', ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.94))
for a_ in ax:                                         # minor labels only where the axis spans under a decade
    if a_.get_yscale() == 'log' and a_.get_ylim()[1] / a_.get_ylim()[0] > 10:
        a_.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.savefig(f'{PP}/figs/crop.pdf'); fig.savefig(f'{PP}/figs/crop.png', dpi=200)
cols = [('periodic', 'kl', 'KL per.', '{:.1e}'), ('line', 'kl', 'KL line', '{:.1e}'), ('kde', 'kl', 'KL KDE', '{:.1e}'),
        ('periodic', 'd_err', '$d$ per.', '{:.3f}'), ('line', 'd_err', '$d$ line', '{:.3f}'), ('periodic', 'C_err', '$C$ per.', '{:.3f}'),
        ('kde', 'C_err', '$C$ KDE', '{:.3f}'), ('periodic', 'phi_rel_err', r'$\phi$ per.', '{:.2f}'), ('periodic', 'rho_err', r'$\rho$ per.', '{:.2f}'),
        ('periodic', 'ap_err', 'ap. per.', '{:.2f}'), ('line', 'ap_err', 'ap. line', '{:.2f}'), ('periodic_three', 'V_hat', r'$\hat V$', '{:.2f}')]
T = [r'\begin{tabular}{ll' + 'r' * len(cols) + '}', r'\toprule', 'case & $n$ & ' + ' & '.join(c[2] for c in cols) + r' \\', r'\midrule']
for c, lab in (('sym', 'equal slits'), ('asym', 'unequal, phase'), ('partial', 'partially coherent')):
    for i, n in enumerate(NS):
        v = [med(c, n, e, m) for e, m, _, _ in cols]
        T.append(f"{lab if i == 0 else ''} & $10^{int(np.log10(n))}$ & " + ' & '.join('--' if not np.isfinite(x) else f.format(x) for x, (_, _, _, f) in zip(v, cols)) + r' \\')
    T.append(r'\addlinespace')
T = T[:-1] + [r'\bottomrule', r'\end{tabular}']
open(f'{PP}/tables/crop.tex', 'w').write('\n'.join(T) + '\n'); print(sum(len(v) for v in R.values()), 'data sets')
