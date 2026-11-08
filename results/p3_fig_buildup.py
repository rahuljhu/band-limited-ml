"""Paper 3, build-up figure: single photons of the heralded double slit arriving (top: the detected positions on the
camera, first n), the complex BLML estimate along the fringes from those n photons (middle: test counts shaded, the
estimate scaled to them), and the aperture |FT g| it implies (bottom: the two slits at +-d/2 emerge); right: held-out
log-likelihood per photon against n for complex BLML, the KDE and the histogram, on every record, relative to BLML at
full size. Inputs: results/p3_buildup/<set>.json, <set>_lines.npz (p3_buildup.py), the raw photons.
-> papers/submit/pra_complex/figs/buildup.pdf"""
import json, os, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
R = sys.argv[1] if len(sys.argv) > 1 else 'results/p3_buildup'
D = os.path.expanduser('~/data_repo/dslit_gpu/')
pts = np.load(D + 'dslit_photon_points.npy'); u = np.random.default_rng(0).random(len(pts))
o = np.argsort(pts[:, 0], kind='stable'); trv = o[(u[o] < 0.80)]
B = json.load(open(f'{R}/photon.json')); Z = np.load(f'{R}/photon_lines.npz'); TE = Z['TE']
P_fr = json.load(open('results/dslit_1d/photon/flat/photon_fraunhofer.json'))['params']['P']
show = [100, 300, 1000, int(B['rows'][-1]['n'])]
fig = plt.figure(figsize=(6.0, 4.4))
gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 0.8, 0.8], hspace=0.45, wspace=0.12)
x = np.arange(len(TE)); pad = 16 * len(TE); xi = np.fft.fftshift(np.fft.fftfreq(pad)) * P_fr
for j, n in enumerate(show):
    a = fig.add_subplot(gs[0, j]); s = trv[:n]
    a.scatter(pts[s, 1], pts[s, 2], s=0.6 if n > 1000 else 2.5, color='k', lw=0)   # fringe axis horizontal, as below
    a.set_xlim(0, 512); a.set_ylim(256, 0); a.set_xticks([]); a.set_yticks([]); a.set_aspect('auto')
    a.set_title(f'{n:,} photons' if n < show[-1] else f'all {n:,}', fontsize=8)
    main = n == show[-1]                          # the full record: the main fit of the double-slit section
    Pn = np.load('results/dslit_1d/photon/bg1/photon_complex_bg_P.npy').astype(float) if main else Z[f'P{n}']
    b = fig.add_subplot(gs[1, j])
    b.bar(x, TE, width=1, color='0.8', lw=0); b.plot(x, Pn / Pn.sum() * TE.sum(), color='C3', lw=1.0)
    b.set_xticks([]); b.set_yticks([]); b.set_xlim(0, len(TE))
    c = fig.add_subplot(gs[2, j]); g = np.load('results/dslit_1d/photon/bg1/photon_complex_bg_line.npy') if main else Z[f'n{n}']
    A = np.abs(np.fft.fftshift(np.fft.fft(g, pad))); A /= A.max(); Af = np.abs(np.fft.fftshift(np.fft.fft(np.abs(g) ** 2, pad))); Af /= Af.max()
    for v in (-0.5, 0.5):
        c.axvspan(v - 0.1, v + 0.1, color='C2', alpha=0.15, lw=0)
    c.plot(xi, Af, color='0.5', ls='--', lw=0.7); c.plot(xi, A, color='C3', lw=1.0)
    c.set_xlim(-1.4, 1.4); c.set_yticks([]); c.set_xticks([-1, 0, 1]); c.set_xlabel(r'$\xi/d$')
    if j == 0:
        a.set_ylabel('(a) detected', fontsize=8); b.set_ylabel('(b) estimate', fontsize=8); c.set_ylabel(r'(c) $|\hat g|$', fontsize=8)
fig.savefig('papers/submit/pra_complex/figs/buildup.pdf', bbox_inches='tight'); fig.savefig('papers/submit/pra_complex/figs/buildup.png', dpi=200, bbox_inches='tight')
print('done')
