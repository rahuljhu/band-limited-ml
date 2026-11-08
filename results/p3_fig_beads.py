"""Paper 3, beads figure: the whole 2048 x 2048 frame of fluorescent beads (plane 5), its complex BLML fit at the band
2NA/lambda (results/binned/frame_complex.npz: density and coefficients e), three isolated beads (training photons
against the fit), and the amplitude spectrum |DFT(e)| -- the field's spectrum, which fills the pupil disk of radius
NA/lambda and nothing outside (dashed: NA/lambda; dotted: 2NA/lambda, the band of the intensity).
-> papers/submit/pra_complex/figs/beads_frame.pdf"""
import sys, numpy as np, matplotlib
sys.path.insert(0, 'results/psf'); sys.path.insert(0, '.')
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
import psfkit as pk
plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
N = np.round(np.maximum(np.asarray(np.load('results/psf/stack_photons.npy', mmap_mode='r')[5]) / pk.GAIN_CORR, 0)).astype(np.int64)
TR = np.random.default_rng(0).binomial(N, 0.7)
Z = np.load('results/binned/frame_complex.npz'); f = Z['density'].astype(float); e = Z['e']
f = f / f.sum() * TR.sum()
beads = np.load('results/psf/beads_iso.npy'); H = 15
fig = plt.figure(figsize=(7.0, 4.1))
gs = fig.add_gridspec(2, 6, height_ratios=[1.35, 1], hspace=0.28, wspace=0.12)
vmax = np.quantile(TR, 0.9999); y0z, x0z, Lz = 900, 900, 320
dens = TR[tuple(beads.T)]; sel = [int(np.argsort(dens)[q]) for q in (len(beads) // 5, len(beads) // 2, 4 * len(beads) // 5)]
a0 = fig.add_subplot(gs[0, 0:2]); a0.imshow(np.maximum(TR, 0.5), cmap='magma', norm=LogNorm(0.5, vmax), interpolation='antialiased')
a0.add_patch(plt.Rectangle((x0z, y0z), Lz, Lz, fill=False, ec='c', lw=0.8)); a0.set_title('(a) photons, whole frame', loc='left', fontsize=8)
Wz = (slice(y0z, y0z + Lz), slice(x0z, x0z + Lz))
a1 = fig.add_subplot(gs[0, 2]); a1.imshow(np.maximum(TR[Wz], 0.5), cmap='magma', norm=LogNorm(0.5, vmax), interpolation='nearest')
a2 = fig.add_subplot(gs[0, 3]); a2.imshow(np.maximum(f[Wz], 0.5), cmap='magma', norm=LogNorm(0.5, vmax), interpolation='nearest')
a1.set_title('(b) photons', loc='left', fontsize=8); a2.set_title('fit', loc='left', fontsize=8)
for a in (a0, a1, a2):
    a.set_xticks([]); a.set_yticks([])
for j, k in enumerate(sel):
    y0, x0 = beads[k]; W = (slice(y0 - H, y0 + H + 1), slice(x0 - H, x0 + H + 1))
    for c, img in enumerate((TR[W], f[W])):
        a = fig.add_subplot(gs[1, 2 * j + c]); a.imshow(img, cmap='magma', norm=PowerNorm(0.5, 0, img.max()), interpolation='nearest')
        a.set_xticks([]); a.set_yticks([])
        a.set_title((f'(d) bead {j + 1}' if c == 0 else 'fit'), loc='left', fontsize=7)
    a0.add_patch(plt.Rectangle((x0 - H, y0 - H), 2 * H, 2 * H, fill=False, ec='w', lw=0.5))
a4 = fig.add_subplot(gs[0, 4:6])
S = np.abs(np.fft.fftshift(np.fft.fft2(e.astype(np.complex128)))); M = S.shape[0]; nu = np.fft.fftshift(np.fft.fftfreq(M))
c = M // 2; k = int(0.36 * M); S = S[c - k:c + k, c - k:c + k]; ext = [nu[c - k], nu[c + k - 1]] * 2
a4.imshow(np.log10(S / S.max() + 1e-6), cmap='viridis', vmin=-4, vmax=0, extent=[ext[0], ext[1], ext[1], ext[0]], interpolation='nearest')
for rr, ls in ((pk.FC_PHYS / 2, '--'), (pk.FC_PHYS, ':')):
    t_ = np.linspace(0, 2 * np.pi, 400); a4.plot(rr * np.cos(t_), rr * np.sin(t_), color='w', ls=ls, lw=0.8)
a4.set_xlabel('cycles / pixel'); a4.set_title(r'(c) field spectrum $|\hat g|$ (log)', loc='left', fontsize=8)
a4.set_yticks([-0.3, 0, 0.3]); a4.set_xticks([-0.3, 0, 0.3])
fig.savefig('papers/submit/pra_complex/figs/beads_frame.pdf', bbox_inches='tight'); fig.savefig('papers/submit/pra_complex/figs/beads_frame.png', dpi=200, bbox_inches='tight')
print('done')
