"""Paper 3, Fig. 2: why the positive root needs twice the band. A slit's in-focus far field g (real, band rho, changing
sign at the dark fringes), its modulus |g| (the positive root of f = g^2) and their spectra: g fills [-rho, rho],
|g| has kinks at the zeros and a spectrum that leaks far beyond rho, and f fills [-2 rho, 2 rho]. With a defocus the
field is complex and has no real root at all. -> papers/submit/pra_complex/figs/root.pdf"""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'serif'})
rho = 1.0; L = 2 ** 17; dx = 0.01; x = (np.arange(L) - L // 2) * dx
g = np.sinc(2 * rho * x)                                  # FT: a box on [-rho, rho]
nu = np.fft.fftshift(np.fft.fftfreq(L, dx))
spec = lambda a: np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(a)))) * dx
Sa, Sf = spec(np.abs(g)), spec(g ** 2); Sg = np.where(np.abs(nu) <= rho, 1.0, 0.0)   # the exact box
# defocus: a quadratic pupil phase makes the field complex
k = nu; pup = np.where(np.abs(k) <= rho, np.exp(1j * 2.0 * (2 * (k / rho) ** 2 - 1)), 0)
gd = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(pup))) / dx; gd = gd / np.abs(gd).max()
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.1), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1]))
s = np.abs(x) < 3.2
ax[0].axhline(0, color='0.8', lw=0.6)
ax[0].plot(x[s], g[s], color='C0', lw=1.2, label=r'field $g$ (band $\rho$)')
ax[0].plot(x[s], np.abs(g[s]), color='C3', lw=1.0, ls='--', label=r'positive root $|g|$')
ax[0].set_xlabel(r'position $x$ (units of $1/2\rho$)'); ax[0].set_title('(a) a slit in focus', loc='left', fontsize=8)
ax[0].legend(frameon=False, fontsize=7, loc='upper right'); ax[0].set_yticks([-0.2, 0, 0.5, 1])
m = np.abs(nu) < 3.2 * rho
for S, c, ls, lab in ((Sg, 'C0', '-', r'$\hat g$'), (Sa, 'C3', '--', r'$\widehat{|g|}$'), (Sf, 'k', ':', r'$\hat f=\widehat{g^2}$')):
    ax[1].semilogy(nu[m] / rho, np.where(S[m] / S.max() > 1e-6, S[m] / S.max(), np.nan), color=c, ls=ls, lw=1.1, label=lab)
for v in (-1, 1):
    ax[1].axvline(v, color='C0', lw=0.5, alpha=0.5); ax[1].axvline(2 * v, color='0.5', lw=0.5, alpha=0.5)
ax[1].set_ylim(3e-6, 3); ax[1].set_xlabel(r'frequency $\nu/\rho$'); ax[1].set_ylabel('spectrum (rel.)')
ax[1].set_title(r'(b) $|g|$ is not band-limited', loc='left', fontsize=8); ax[1].legend(frameon=False, fontsize=7, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0))
ax[2].axhline(0, color='0.8', lw=0.6)
ax[2].plot(x[s], gd.real[s], color='C0', lw=1.0, label=r'Re $g$'); ax[2].plot(x[s], gd.imag[s], color='C2', lw=1.0, label=r'Im $g$')
ax[2].plot(x[s], np.abs(gd[s]), color='k', lw=0.8, ls=':', label=r'$|g|$')
ax[2].set_xlabel(r'position $x$'); ax[2].set_title('(c) with defocus: no real root', loc='left', fontsize=8)
ax[2].legend(frameon=False, fontsize=7, loc='upper right')
fig.tight_layout(pad=0.4); fig.savefig('papers/submit/pra_complex/figs/root.pdf'); fig.savefig('papers/submit/pra_complex/figs/root.png', dpi=200)
print('fraction of |g| spectrum beyond rho:', float((Sa[np.abs(nu) > rho] ** 2).sum() / (Sa ** 2).sum()),
      'beyond 2 rho:', float((Sa[np.abs(nu) > 2 * rho] ** 2).sum() / (Sa ** 2).sum()))
