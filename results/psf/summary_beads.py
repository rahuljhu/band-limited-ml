"""Summary over all isolated beads (pixel-level fits from fit_bead_px.py)."""
import glob, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = 'results/psf/'
R = [json.load(open(f)) for f in sorted(glob.glob(D + 'bead*_px.json'), key=lambda s: int(s.split('bead')[1].split('_')[0]))]
n = np.array([r['photons'] for r in R]); ll = lambda m: np.array([r[m]['ll'] for r in R])
bl = ll('BLML'); comps = [('BLML_phys', 'BLML at 2NA/λ'), ('GMM', 'GMM'), ('KDE', 'KDE'), ('Gaussian', 'Gaussian'),
                          ('empirical_train_hist', 'raw histogram')]
fcs = np.array([r['BLML']['fc'] for r in R]); edge = np.array([r['BLML']['at_grid_edge'] for r in R]); fcp = R[0]['fc_phys']
print(f'{len(R)} beads, photons {n.min():,}..{n.max():,} (median {int(np.median(n)):,})')
print(f"{'vs':16s} {'median Δ':>9s} {'mean Δ':>9s} {'BLML wins':>10s}")
summ = {}
for m, lab in comps:
    d = bl - ll(m); summ[m] = dict(median=float(np.median(d)), mean=float(d.mean()), wins=int((d > 0).sum()), n=len(d))
    print(f'{lab:16s} {np.median(d):+9.4f} {d.mean():+9.4f} {(d > 0).sum():5d}/{len(d)}')
print('selected fc: median x %.3f y %.3f; at grid edge %d/%d; phys %.3f' % (*np.median(fcs, 0), edge.sum(), len(R), fcp))
json.dump(dict(summary=summ, n_beads=len(R)), open(D + 'beads_summary.json', 'w'), indent=1)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
cols = ['#7fa9d1', '#e08a00', '#6a9a3a', '#b8322a', '0.35']
a = axs[0]; rng = np.random.default_rng(0)
for i, ((m, lab), col) in enumerate(zip(comps, cols)):
    d = bl - ll(m); a.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d, s=12, color=col, alpha=0.8)
    a.hlines(np.median(d), i - 0.3, i + 0.3, color='k', lw=1.5)
    a.text(i, 1.0, f"{summ[m]['wins']}/{len(d)}", transform=a.get_xaxis_transform(), ha='center', va='bottom', fontsize=8)
a.axhline(0, color='0.6', lw=0.8); a.set_xticks(range(len(comps))); a.set_xticklabels([l for _, l in comps], rotation=15)
a.set_yscale('symlog', linthresh=1e-3); a.set_ylabel('BLML − model, test LL (nats/photon)')
a.set_title('(a) BLML (auto fc) minus each model; >0 = BLML better\n(number = beads where BLML wins; bar = median)', fontsize=9)
a = axs[1]
for m, lab, col in [('GMM', 'vs GMM', cols[1]), ('KDE', 'vs KDE', cols[2]), ('empirical_train_hist', 'vs raw histogram', cols[4])]:
    a.scatter(n, bl - ll(m), s=14, color=col, label=lab)
a.axhline(0, color='0.6', lw=0.8); a.set_xscale('log'); a.set_yscale('symlog', linthresh=1e-3)
a.set_xlabel('photons in bead ROI'); a.set_ylabel('BLML − model, test LL'); a.legend(frameon=False, fontsize=8)
a.set_title('(b) advantage vs bead brightness', fontsize=9)
a = axs[2]
a.scatter(n, fcs[:, 0], s=14, color='#1f6fb4', label='fc x'); a.scatter(n, fcs[:, 1], s=14, marker='s', facecolor='none',
                                                                     edgecolor='#b8322a', label='fc y')
a.axhline(fcp, color='0.3', ls='--', lw=1); a.text(n.min(), fcp, ' 2NA/λ (λ=0.515 µm assumed)', va='bottom', fontsize=8)
a.axhline(0.5, color='0.7', ls=':', lw=1); a.text(n.min(), 0.5, ' pixel Nyquist / grid edge', va='bottom', fontsize=8, color='0.4')
a.set_xscale('log'); a.set_xlabel('photons in bead ROI'); a.set_ylabel('selected fc (cycles/px)'); a.legend(frameon=False, fontsize=8)
a.set_title('(c) validation-selected cutoff vs brightness', fontsize=9)
fig.suptitle(f'{len(R)} isolated beads, pixel-level BLMLQuick on counts; held-out pixel-count LL (multinomial 64/16/20 split per pixel)', fontsize=9.5)
fig.tight_layout(); fig.savefig(D + 'beads_summary.png', dpi=170); fig.savefig(D + 'beads_summary.pdf')
