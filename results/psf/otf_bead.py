"""Radial OTF of bead k: empirical (DFT of pixel counts) vs BLMLQuick on counts (analytic FT of f = g^2,
fine frequency grid), each with and without dividing by the pixel sinc; vs the ideal disk-pupil OTF."""
import sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import blml_trivial as bt
from blml_trivial import make_modes
bt.DIRICHLET_LEGACY = False
D = 'results/psf/'; NA, LAM, PX = 1.3, 0.515, 0.065; FCP = 2 * NA / LAM * PX
k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
S = np.load(D + 'stack_photons.npy', mmap_mode='r'); im = np.asarray(S[5])
y0, x0 = np.load(D + 'beads_iso.npy')[k]; H = 15
roi = im[y0 - H:y0 + H + 1, x0 - H:x0 + H + 1].astype(float)
bg = np.median(np.concatenate([roi[0], roi[-1], roi[:, 0], roi[:, -1]]))
cnt = np.round(np.maximum(roi - bg, 0))
ax = np.arange(-H, H + 1.0); X, Y = np.meshgrid(ax, ax); Z = np.column_stack([X.ravel(), Y.ravel()])
c = np.array([(cnt * X).sum(), (cnt * Y).sum()]) / cnt.sum()            # centroid (removes linear phase)
psinc = lambda nx, ny: np.sinc(nx) * np.sinc(ny)
# empirical OTF: DFT of counts at arbitrary frequencies (no zero padding artefacts), |.| normalised
nr = np.linspace(0, 0.5, 101); th = np.linspace(0, np.pi, 36, endpoint=False)
NX = nr[:, None] * np.cos(th); NY = nr[:, None] * np.sin(th)
def dft(wts, pts):
    ph = np.exp(-2j * np.pi * ((pts[:, 0] - c[0])[:, None, None] * NX + (pts[:, 1] - c[1])[:, None, None] * NY))
    return np.abs(np.tensordot(wts, ph, 1)) / wts.sum()
emp = dft(cnt.ravel(), Z)
# BLML: FT g on a fine grid inside its box band, then FT f = autocorrelation of FT g (via FFT)
def blml_otf(fc):
    m = bt.BLMLQuick(dtype=np.float64).fit(Z, bt.with_lattice(make_modes([fc, fc], None, None, d=Z.shape[1]), 1.0), sample_weight=cnt.ravel())
    L = 256; du = fc / L; u = (np.arange(L) - L / 2 + 0.5) * du        # FT g lives in [-fc/2, fc/2]
    UX, UY = np.meshgrid(u, u); G = np.zeros((L, L), complex)
    for i in range(0, len(m.Z_), 200):
        z = m.Z_[i:i + 200] - c
        G += np.tensordot(m.e_hat_[i:i + 200], np.exp(-2j * np.pi * (z[:, 0, None, None] * UX + z[:, 1, None, None] * UY)), 1)
    G /= m.n_samples_
    P = np.zeros((2 * L, 2 * L), complex); P[:L, :L] = G
    A = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(P)) ** 2 * 0 + np.fft.fft2(P) * np.conj(np.fft.fft2(P)))) * du * du * (2 * L) ** 2
    nu = (np.arange(2 * L) - L) * du                                    # autocorrelation lag grid
    from scipy.interpolate import RegularGridInterpolator
    I = RegularGridInterpolator((nu, nu), np.abs(A), bounds_error=False, fill_value=0)
    v = I(np.stack([NY, NX], -1)); return v / v[0].mean()
bl_auto, bl_phys = blml_otf(0.5), blml_otf(FCP)
s = np.clip(nr / FCP, 0, 1); ideal = 2 / np.pi * (np.arccos(s) - s * np.sqrt(1 - s ** 2))
rad = lambda A: A.mean(1)
corr = rad(psinc(NX, NY))
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for a, logy in zip(axs, [False, True]):
    a.plot(nr, rad(emp), 'o', ms=2.5, color='0.55', label='pixel counts (raw)')
    a.plot(nr, rad(emp) / corr, 'o', ms=2.5, color='0.15', label='pixel counts ÷ pixel sinc')
    a.plot(nr, rad(bl_auto), '-', color='#9ec3e6', lw=1.5, label='BLML f̂ (fc 0.5), raw')
    a.plot(nr, rad(bl_auto) / corr, '-', color='#1f6fb4', lw=1.8, label='BLML f̂ (fc 0.5) ÷ pixel sinc')
    a.plot(nr, rad(bl_phys) / corr, ':', color='#1f6fb4', lw=1.8, label='BLML f̂ (fc 2NA/λ) ÷ pixel sinc')
    a.plot(nr, ideal, '--', color='#b8322a', lw=1.4, label='ideal in-focus disk pupil')
    a.axvline(FCP, color='0.7', lw=0.8); a.text(FCP, 0.02 if not logy else 2e-3, ' 2NA/λ', fontsize=8, color='0.4')
    a.set_xlabel('|ν| (cycles/px)'); a.set_xlim(0, 0.5)
    if logy: a.set_yscale('log'); a.set_ylim(1e-3, 1.2)
axs[0].set_ylabel('|OTF| (azimuthal mean)'); axs[0].legend(frameon=False, fontsize=7.5)
axs[0].set_title('(a) radial OTF, linear'); axs[1].set_title('(b) log scale: edge and beyond')
fig.suptitle(f'Bead {k}: pixel sinc correction at the band edge (pixel sinc at 0.328 ≈ {corr[np.argmin(abs(nr-FCP))]:.2f}); λ=0.515 µm assumed', fontsize=9)
fig.tight_layout(); fig.savefig(D + f'otf_bead{k}.png', dpi=170); fig.savefig(D + f'otf_bead{k}.pdf')
i = np.argmin(abs(nr - FCP)); j = np.argmin(abs(nr - 0.42))
print('at 0.328: emp raw %.4f corr %.4f | blml raw %.4f corr %.4f | ideal %.4f' % (rad(emp)[i], rad(emp)[i] / corr[i], rad(bl_auto)[i], rad(bl_auto)[i] / corr[i], ideal[i]))
print('at 0.42 : emp corr %.4f blml corr %.4f' % (rad(emp)[j] / corr[j], rad(bl_auto)[j] / corr[j]))
