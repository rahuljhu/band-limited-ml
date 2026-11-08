"""Figure for the raw-count bead fit (fit_bead_raw.py): data, BLML bead component, test residuals,
radial profile with fitted background removed, |FT g-hat| with the pupil circle, radial OTF."""
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
D = 'results/psf/'; k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
z = np.load(D + f'bead{k}_raw.npz'); r = json.load(open(D + f'bead{k}_raw.json'))
cnt, tv, te, gr, c = z['cnt'].astype(float), z['tv'].astype(float), z['te'].astype(float), z['gr'], z['center']
H = (cnt.shape[0] - 1) // 2; B = cnt.size; ax = np.arange(-H, H + 1.0); X, Y = np.meshgrid(ax, ax)
pi, q = float(z['pi']), z['q']; FCP = r['fc_phys']; RHO = FCP / 2; fc = r['BLML+bg']['fc']
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(2, 3, figsize=(13, 8))
ext = [-H - .5, H + .5] * 2
# (a) raw counts
a = axs[0, 0]; im = a.imshow(cnt, origin='lower', extent=ext, norm=LogNorm(max(cnt.min(), 1), cnt.max()), cmap='magma')
a.set_title(f"(a) raw photon counts (bead + background)\n{int(cnt.sum()):,} photons, gain-corrected"); fig.colorbar(im, ax=a, fraction=0.046)
# (b) BLML bead component, fine grid, per-pixel probability units
d = z['d_blml']; d = d / d.sum() * (len(gr) / (2 * H + 1)) ** 2
a = axs[0, 1]; im = a.imshow(np.maximum(d, d.max() * 1e-5), origin='lower', extent=[gr[0], gr[-1]] * 2, norm=LogNorm(d.max() * 1e-5, d.max()), cmap='magma')
a.set_title(f"(b) BLML bead component ĝ² (background π = {pi:.3f} removed)\nfc = [{fc[0]:.3f}, {fc[1]:.3f}] cycles/px"); fig.colorbar(im, ax=a, fraction=0.046)
# (c) Pearson residuals of held-out counts under BLML+bg
P = pi / B + (1 - pi) * q; mu = te.sum() * P.reshape(cnt.shape); res = (te - mu) / np.sqrt(mu)
a = axs[0, 2]; im = a.imshow(res, origin='lower', extent=ext, cmap='RdBu_r', vmin=-4, vmax=4)
a.set_title(f"(c) test-count Pearson residuals, BLML+bg\nmean square {np.mean(res ** 2):.2f} (1 = Poisson)"); fig.colorbar(im, ax=a, fraction=0.046)
for a in axs[0]: a.set_xlabel('x (px, 65 nm)')
axs[0, 0].set_ylabel('y (px)')
# (d) radial profile: all counts minus BLML's fitted background, vs bead components
R = np.hypot(X - c[0], Y - c[1]); edges = np.arange(0, H + 1, 0.5); mid = 0.5 * (edges[1:] + edges[:-1]); ix = np.digitize(R.ravel(), edges) - 1
nbin = np.bincount(ix[(ix >= 0) & (ix < len(mid))], None, len(mid))
def radial(v):                                       # azimuthal mean; empty bins -> nan (not plotted)
    v = v.ravel(); ok = (ix >= 0) & (ix < len(mid))
    return np.where(nbin > 0, np.bincount(ix[ok], v[ok], len(mid)) / np.maximum(nbin, 1), np.nan)
bead_counts = cnt - pi * cnt.sum() / B; Nb = (1 - pi) * cnt.sum()
emp = radial(bead_counts) / Nb; se = np.sqrt(radial(cnt) / np.maximum(nbin, 1)) / Nb
a = axs[1, 0]; okp = np.nan_to_num(emp) > 0
a.errorbar(mid[okp], emp[okp], se[okp], fmt='o', ms=3, color='0.2', label='counts − fitted bg', zorder=5)
for key, col, ls, lab in [('q', '#1f6fb4', '-', f"BLML+bg ({r['BLML+bg']['ll']:.4f})"), ('q_phys', '#1f6fb4', ':', f"BLML+bg @2NA/λ ({r['BLML+bg@phys']['ll']:.4f})"),
                          ('q_gmm', '#e08a00', '--', f"GMM+bg K={r['GMM+bg']['K']} ({r['GMM+bg']['ll']:.4f})"), ('q_gauss', '#b8322a', '-', f"Gaussian+bg ({r['Gaussian+bg']['ll']:.4f})")]:
    a.plot(mid, np.maximum(radial(z[key]), 1e-9), ls, color=col, lw=1.6, label=lab)
for qq in [0.610, 1.117, 1.619]: a.axvline(qq / RHO, color='0.8', lw=0.8, zorder=0)    # r = q * lambda/NA, lambda/NA = 1/RHO px
a.set_yscale('log'); a.set_ylim(max(emp[okp].min() * 0.3, 1e-7), np.nanmax(emp) * 2); a.set_xlabel('radius (px)'); a.set_ylabel('bead probability per pixel')
a.set_title('(d) radial profile (grey lines: Airy minima)\nlegend: test LL'); a.legend(frameon=False, fontsize=7)
# (e) |FT g-hat| log, pupil circle
nu = z['nu']; F = np.abs(z['ftg']); F /= F.max(); t = np.linspace(0, 2 * np.pi, 200)
a = axs[1, 1]; a.imshow(np.maximum(F, 1e-3), origin='lower', extent=[nu[0], nu[-1]] * 2, cmap='viridis', norm=LogNorm(1e-3, 1))
a.plot(RHO * np.cos(t), RHO * np.sin(t), 'w--', lw=1, label='NA/λ')
a.add_patch(plt.Rectangle((-fc[0] / 2, -fc[1] / 2), fc[0], fc[1], fill=False, ec='w', ls=':', lw=0.8))
a.set_title('(e) |FT ĝ| (log): energy within the pupil\ndotted box = selected band'); a.set_xlabel('$\\nu_x$ (cycles/px)'); a.set_ylabel('$\\nu_y$')
a.legend(loc='upper right', fontsize=7, frameon=False, labelcolor='w')
# (f) radial OTF: background-removed counts vs BLML component, both / pixel sinc; ideal
nr = np.linspace(0, 0.5, 101); th = np.linspace(0, np.pi, 36, endpoint=False); NX = nr[:, None] * np.cos(th); NY = nr[:, None] * np.sin(th)
def otf(w):
    ph = np.exp(-2j * np.pi * ((X.ravel() - c[0])[:, None, None] * NX + (Y.ravel() - c[1])[:, None, None] * NY))
    return np.abs(np.tensordot(w.ravel(), ph, 1)).mean(1) / w.sum()
corr = (np.sinc(NX) * np.sinc(NY)).mean(1); s = np.clip(nr / FCP, 0, 1); ideal = 2 / np.pi * (np.arccos(s) - s * np.sqrt(1 - s ** 2))
a = axs[1, 2]
a.plot(nr, otf(bead_counts) / corr, 'o', ms=2.5, color='0.2', label='counts − bg, ÷ pixel sinc')
a.plot(nr, otf(z['q']) / corr, '-', color='#1f6fb4', lw=1.8, label='BLML bead component, ÷ pixel sinc')
a.plot(nr, ideal, '--', color='#b8322a', lw=1.3, label='ideal in-focus disk pupil')
a.axhline(1 / np.sqrt(cnt.sum()), color='0.6', ls=':', lw=1); a.text(0.38, 1.15 / np.sqrt(cnt.sum()), 'shot-noise floor 1/√N', fontsize=7, color='0.4')
a.axvline(FCP, color='0.75', lw=0.8); a.set_yscale('log'); a.set_ylim(3e-4, 1.2); a.set_xlim(0, 0.5)
a.set_xlabel('|ν| (cycles/px)'); a.set_ylabel('|OTF|'); a.set_title('(f) radial OTF; grey line = 2NA/λ'); a.legend(frameon=False, fontsize=7)
fig.suptitle(f"Bead {k} at {tuple(r['bead'])}, raw counts (no subtraction/clipping), EM background; NA 1.3, λ = 0.515 µm assumed. "
             f"Test LL: BLML+bg {r['BLML+bg']['ll']:.4f} | histogram {r['raw histogram']['ll']:.4f} | KDE {r['KDE']['ll']:.4f} | "
             f"GMM+bg {r['GMM+bg']['ll']:.4f} | Gaussian+bg {r['Gaussian+bg']['ll']:.4f}", fontsize=8.5)
fig.tight_layout(); fig.savefig(D + f'bead{k}_raw.png', dpi=160); fig.savefig(D + f'bead{k}_raw.pdf'); print('saved')
