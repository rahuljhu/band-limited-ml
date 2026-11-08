"""Figure for one bead: data vs BLML vs Gaussian, radial profiles, and Fourier views of g-hat and f-hat.
Usage: python plot_bead.py <k>"""
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
D = 'results/psf/'; k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
z = np.load(D + f'bead{k}.npz'); r = json.load(open(D + f'bead{k}.json'))
gr, cnt, c = z['gr'], z['cnt'].astype(float), z['center']; H = (cnt.shape[0] - 1) // 2
NA_L = r['fc_phys'] / 2                                          # pupil radius NA/lambda, cycles/px
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(2, 3, figsize=(11, 7.2))
ext = [gr[0], gr[-1], gr[0], gr[-1]]
vmax = cnt.max() / cnt.sum(); norm = LogNorm(vmax * 1e-4, vmax)
ax[0, 0].imshow(np.maximum(cnt / cnt.sum(), 1e-12), origin='lower', extent=[-H - .5, H + .5] * 2, norm=norm, cmap='magma')
ax[0, 0].set_title('(a) observed photons (per-pixel fraction)')
ax[0, 1].imshow(np.maximum(z['d_BLML'], 1e-12), origin='lower', extent=ext, norm=norm, cmap='magma')
ax[0, 1].set_title(f"(b) BLML density  (test LL {r['BLML']['ll']:.4f})")
im = ax[0, 2].imshow(np.maximum(z['d_Gaussian'], 1e-12), origin='lower', extent=ext, norm=norm, cmap='magma')
ax[0, 2].set_title(f"(c) 2-D Gaussian  (test LL {r['Gaussian']['ll']:.4f})")
for a in ax[0]:
    a.set_xlim(-12, 12); a.set_ylim(-12, 12); a.set_xlabel('x (pixels, 65 nm)')
ax[0, 0].set_ylabel('y (pixels)')
fig.colorbar(im, ax=ax[0, 2], fraction=0.046, pad=0.04, label='density (per px$^2$)')
# (d) radial profiles: empirical from all photons vs radial means of model densities
P = z['P']; rr = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]); edges = np.linspace(0, 14, 57)
h, _ = np.histogram(rr, edges); area = np.pi * np.diff(edges ** 2); mid = 0.5 * (edges[1:] + edges[:-1])
emp = h / len(P) / area; se = np.sqrt(np.maximum(h, 1)) / len(P) / area
GX, GY = np.meshgrid(gr, gr); RG = np.hypot(GX - c[0], GY - c[1]); idx = np.digitize(RG.ravel(), edges) - 1
def radial(d):
    d = d.ravel(); ok = (idx >= 0) & (idx < len(mid))
    return np.bincount(idx[ok], d[ok], len(mid)) / np.maximum(np.bincount(idx[ok], None, len(mid)), 1)
a = ax[1, 0]
a.errorbar(mid, emp, se, fmt='o', ms=3, color='0.2', label='photons (all)', zorder=5)
sty = {'BLML': ('#1f6fb4', '-', 'BLML (auto f$_c$)'), 'BLML_phys': ('#1f6fb4', ':', 'BLML at 2NA/$\\lambda$'),
       'GMM': ('#e08a00', '--', f"GMM (K={r['GMM']['K']})"), 'Gaussian': ('#b8322a', '-', 'Gaussian')}
for m, (col, ls, lab) in sty.items():
    a.plot(mid, np.maximum(radial(z['d_' + m]), 1e-9), ls, color=col, lw=1.6, label=lab)
a.set_yscale('log'); a.set_ylim(emp[emp > 0].min() * 0.3, emp.max() * 2); a.set_xlabel('radius (pixels)')
a.set_ylabel('azimuthal-mean density'); a.set_title('(d) radial profile: Airy rings'); a.legend(frameon=False, fontsize=8)
# airy minima for lambda/(2NA)-scaled radius: r = 0.61, 1.12, 1.62 * lambda/NA
for q in [0.610, 1.117, 1.619]:
    a.axvline(q / NA_L, color='0.75', lw=0.8, zorder=0)
# (e) |FT g-hat| with pupil circle; (f) |FT f-hat| with 2NA/lambda circle
nu = z['nu']; ftg = np.abs(z['ftg']); ftg /= ftg.max()
t = np.linspace(0, 2 * np.pi, 300)
a = ax[1, 1]
a.imshow(np.maximum(ftg, 1e-3), origin='lower', extent=[nu[0], nu[-1]] * 2, cmap='viridis', norm=LogNorm(1e-3, 1))
a.plot(NA_L * np.cos(t), NA_L * np.sin(t), 'w--', lw=1.2, label='pupil NA/$\\lambda$')
fc = r['BLML']['fc']; a.add_patch(plt.Rectangle((-fc[0] / 2, -fc[1] / 2), fc[0], fc[1], fill=False, ec='w', lw=0.8, ls=':'))
a.set_title('(e) |FT ĝ|  → energy inside pupil (log)'); a.set_xlabel('$\\nu_x$ (cycles/px)'); a.set_ylabel('$\\nu_y$')
a.legend(loc='upper right', fontsize=7, frameon=False, labelcolor='w')
fq = z['fq']; ftf = z['ftf']; ftf = ftf / ftf.max(); s = (np.abs(fq) <= 0.45)
a = ax[1, 2]
a.imshow(np.maximum(ftf[np.ix_(s, s)], 1e-4), origin='lower', extent=[fq[s][0], fq[s][-1]] * 2, cmap='viridis', norm=LogNorm(1e-4, 1))
a.plot(2 * NA_L * np.cos(t), 2 * NA_L * np.sin(t), 'w--', lw=1.2, label='OTF cutoff 2NA/$\\lambda$')
a.add_patch(plt.Rectangle((-fc[0], -fc[1]), 2 * fc[0], 2 * fc[1], fill=False, ec='w', lw=0.8, ls=':'))
a.set_title('(f) |FT f̂|  → OTF (log)'); a.set_xlabel('$\\nu_x$ (cycles/px)')
a.legend(loc='upper right', fontsize=7, frameon=False, labelcolor='w')
fig.suptitle(f"Bead at {tuple(r['bead'])}: {r['used']:,} photons, NA 1.3, 65 nm px, λ≈0.515 µm (assumed). "
             f"Test LL — BLML {r['BLML']['ll']:.4f}, BLML@phys {r['BLML_phys']['ll']:.4f}, "
             f"GMM {r['GMM']['ll']:.4f}, KDE {r['KDE']['ll']:.4f}, Gaussian {r['Gaussian']['ll']:.4f}", fontsize=8.5)
fig.tight_layout(); fig.savefig(D + f'bead{k}.png', dpi=170); fig.savefig(D + f'bead{k}.pdf')
print('saved', D + f'bead{k}.png')
