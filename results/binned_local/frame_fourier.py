"""Fourier view of the full-frame complex BLML fit (plane 5), from its coefficients (no transform of the data):
ghat = FFT(e) on the band disc (the pupil times the emitter-map spectrum), fhat = FFT(|g|^2) (the frame's spectrum,
supported on the doubled disc: the OTF times the emitter spectrum), the radial MTF against the aberration-free
NA 1.3 / 515 nm incoherent OTF, and one isolated bead's pupil (modulus and phase, its position ramp removed).
Output papers/paper3/figs/frame_fourier.png and results/binned_local/frame_fourier.json"""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
S = '/private/tmp/claude-501/-Users-ra/e6cd45ce-b6f4-42cc-be9a-45c946facadf/scratchpad'
e = np.load('results/binned/frame_complex.npz')['e'].astype(np.complex128)
U = np.load(f'{S}/U_frame.npy').astype(np.complex128); F = np.abs(U) ** 2
m = e.shape[0]; nu = np.fft.fftshift(np.fft.fftfreq(m)); R = np.hypot(*np.meshgrid(nu, nu, indexing='ij'))
fc_g = 1.3 / 0.515 * 0.065                                           # NA / lambda per pixel: the amplitude band
G = np.fft.fftshift(np.fft.fft2(e)); G[R > fc_g] = 0
Fh = np.fft.fftshift(np.fft.fft2(F - F.mean()))
# radial MTF of the frame (|fhat| ring average, normalised at small nu) vs the ideal incoherent OTF
rb = np.linspace(0, 0.5, 201); rc = 0.5 * (rb[1:] + rb[:-1]); idx = np.digitize(R.ravel(), rb) - 1
ring = np.bincount(idx, weights=np.abs(Fh).ravel(), minlength=200)[:200] / np.maximum(np.bincount(idx, minlength=200)[:200], 1)
s = np.clip(rc / (2 * fc_g), 0, 1); otf = 2 / np.pi * (np.arccos(s) - s * np.sqrt(1 - s ** 2))
k0 = np.argmin(np.abs(rc - 0.02)); mtf = ring / ring[k0] * otf[k0]
# one bright isolated bead's pupil: its coefficients in a 64 px window, padded to 512
y0, x0 = 661, 1068; w = e[y0 - 32:y0 + 32, x0 - 32:x0 + 32]; P = np.fft.fftshift(np.fft.fft2(w, s=(512, 512)))
nup = np.fft.fftshift(np.fft.fftfreq(512)); Rp = np.hypot(*np.meshgrid(nup, nup, indexing='ij')); disk = Rp <= fc_g
Y, X = np.meshgrid(nup, nup, indexing='ij')
gy, gx = np.unravel_index(np.argmax(np.abs(np.fft.fft2(np.abs(w) ** 0 * w, s=(512, 512)))), (512, 512))
# remove the position ramp: weighted least squares of the unwrapped phase on a plane (phase gradients inside the disc)
ph = np.angle(P); A = np.abs(P); wt = (A * disk).ravel()
from numpy.linalg import lstsq
Z = np.exp(1j * ph)
dyp = np.angle(Z[1:, :] * np.conj(Z[:-1, :])); dxp = np.angle(Z[:, 1:] * np.conj(Z[:, :-1]))
my = disk[1:, :] & disk[:-1, :]; mx = disk[:, 1:] & disk[:, :-1]
ty = np.average(dyp[my], weights=(A[1:, :] * A[:-1, :])[my]); tx = np.average(dxp[mx], weights=(A[:, 1:] * A[:, :-1])[mx])
iy, ix = np.meshgrid(np.arange(512), np.arange(512), indexing='ij')
Pc = P * np.exp(-1j * (ty * iy + tx * ix)); Pc = Pc * np.exp(-1j * np.angle(np.sum(Pc[disk] * np.abs(Pc[disk]))))
phc = np.where(disk, np.angle(Pc), np.nan)
out = dict(mtf_at=[dict(nu=float(v), mtf=float(np.interp(v, rc, mtf)), ideal=float(np.interp(v, rc, otf))) for v in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)],
           pupil_phase_rms_rad=float(np.sqrt(np.average(phc[disk] ** 2, weights=np.abs(Pc[disk]) ** 2))),
           pupil_modulus_edge_over_centre=float(np.abs(Pc[disk & (Rp > 0.9 * fc_g)]).mean() / np.abs(Pc[Rp < 0.2 * fc_g]).mean()))
print(json.dumps(out, indent=1)); json.dump(out, open('results/binned_local/frame_fourier.json', 'w'), indent=1)
fig, ax = plt.subplots(2, 3, figsize=(15, 9.5)); ext = [nu[0], nu[-1], nu[0], nu[-1]]; z = 0.36
def crop(a): c = m // 2; k = int(z * m); return a[c - k:c + k, c - k:c + k]
ax[0, 0].imshow(np.log10(crop(np.abs(G)) + 1e-3 * np.abs(G).max()), extent=[-z, z, -z, z], cmap='magma'); ax[0, 0].set_title('log |ĝ|: frame amplitude spectrum (pupil × emitter speckle)')
ax[0, 1].imshow(np.log10(crop(np.abs(Fh)) + 1e-6 * np.abs(Fh).max()), extent=[-z, z, -z, z], cmap='magma'); ax[0, 1].set_title('log |f̂|: frame intensity spectrum (support 2NA/λ)')
for a_ in ax[0, :2]:
    for rr, c in ((fc_g, 'c'), (2 * fc_g, 'w')):
        a_.add_patch(plt.Circle((0, 0), rr, fill=False, ls='--', color=c, lw=0.8))
    a_.set_xlabel('cycles / px')
ax[0, 2].plot(rc, mtf, label='BLML frame fit (ring-averaged |f̂|)'); ax[0, 2].plot(rc, otf, '--', label='ideal incoherent OTF, NA 1.3, 515 nm')
ax[0, 2].axvline(0.5, color='grey', lw=0.5); ax[0, 2].set_xlim(0, 0.5); ax[0, 2].set_yscale('log'); ax[0, 2].set_ylim(1e-4, 1.5); ax[0, 2].legend(); ax[0, 2].set_title('radial MTF'); ax[0, 2].set_xlabel('cycles / px')
zp = 0.22; cp = 256; kp = int(zp * 512)
ax[1, 0].imshow(np.abs(F[y0 - 20:y0 + 21, x0 - 20:x0 + 21]), cmap='magma'); ax[1, 0].set_title('bead (fitted f), 41 px')
im = ax[1, 1].imshow(np.abs(Pc)[cp - kp:cp + kp, cp - kp:cp + kp], extent=[-zp, zp, -zp, zp], cmap='viridis'); ax[1, 1].set_title('bead pupil |P(ν)|'); plt.colorbar(im, ax=ax[1, 1])
im = ax[1, 2].imshow(phc[cp - kp:cp + kp, cp - kp:cp + kp], extent=[-zp, zp, -zp, zp], cmap='twilight', vmin=-np.pi, vmax=np.pi); ax[1, 2].set_title('bead pupil phase (position ramp removed), rad'); plt.colorbar(im, ax=ax[1, 2])
plt.tight_layout(); plt.savefig('papers/paper3/figs/frame_fourier.png', dpi=110)
