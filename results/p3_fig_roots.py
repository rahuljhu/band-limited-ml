"""Paper 3, the root ambiguity: two apertures with one intensity. The unequal slits of Fig. 1(b) (tapered, amplitude
ratio 0.6, phase step 1 rad) on the uncropped line of 2048 pixels; complex BLML with the band at the true cut-off
fitted to the exact density (10^9 times P, the limit n -> inf) from its default start, and from the true field. Both
are maxima of the same likelihood (to 1e-8 nats per particle). (a) the two apertures |A| and their phase across the
slits, aligned over the global phase, the twin and a shift; (b) their intensities on the detector; (c) the difference
of the two intensities, relative to the peak. -> papers/submit/pra_complex/figs/roots.pdf. Run from the repo root."""
import os, sys, numpy as np, matplotlib
os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_COLLAPSE', '1'); os.environ.setdefault('BLML_SOLVER', 'lbfgs')
sys.path.insert(0, '.'); sys.argv = sys.argv[:1]
from types import SimpleNamespace
from scipy.optimize import minimize_scalar
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
import dslit_job as dj
W, NF = 2048, 1 << 16
D_, A_, RHO, PHI = 0.017, 0.005, 0.6, 1.0                        # as p3_ladder.py
FC0 = D_ + A_; nu = np.fft.fftfreq(NF)
slit = lambda c: (lambda u: np.where(u < 1, np.cos(np.pi * u / 2) ** 2, 0.0))(np.abs(nu - c) / (A_ / 2))
field = lambda A: np.fft.fftshift(np.fft.ifft(A))[NF // 2 - W // 2:NF // 2 + W // 2] * NF
g_true = field(slit(-D_ / 2) + RHO * np.exp(1j * PHI) * slit(D_ / 2))
P = np.abs(g_true) ** 2; P /= P.sum(); Pinf = P * 1e9
dj.BFIX = np.array([np.nan]); line = lambda m: np.asarray(dj._np(m.Se_))[0] / m.n_
E0 = np.zeros((2, W), complex); E0[0] = Pinf * g_true / np.maximum(np.abs(g_true) ** 2, 1e-300); E0[1] = Pinf * 1e-4 / W
fits = {}
for lab, init in (('band, default start', None), ('band, started at the true field', SimpleNamespace(e_hat_=E0, shape_=(W,)))):
    Q, m, _ = dj.bg_fit(Pinf, [FC0], (0,), True, 1, max_sweeps=20000, tol=1e-9, init=init)
    fits[lab] = (Q / Q.sum(), line(m)); print(lab, 'loglik/n %.9f' % float((P * np.log(np.maximum(Q / Q.sum(), 1e-300))).sum()))
L8 = 8 * W; nuL = np.fft.fftshift(np.fft.fftfreq(L8)); tW = np.arange(W) - W // 2
ap = lambda g: np.fft.fftshift(np.fft.fft(np.r_[g, np.zeros(L8 - W)]) * np.exp(2j * np.pi * np.fft.fftfreq(L8) * (W // 2)))
R = ap(g_true); R /= np.linalg.norm(R)
def align(g):                                                     # global phase, twin, shift of the aperture
    best = None
    for gt in (g, np.conj(g)):
        Ad = lambda s: (lambda X: X / np.linalg.norm(X))(ap(gt * np.exp(2j * np.pi * s * tW)))
        r = minimize_scalar(lambda s: -abs(np.vdot(Ad(s), R)), bounds=(-3.0 / W, 3.0 / W), method='bounded')
        B = Ad(r.x); ip = np.vdot(B, R)
        if best is None or abs(ip) > best[0]:
            best = (abs(ip), B * np.exp(1j * np.angle(ip)))
    return best[1]
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.1), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1]))
sel = np.abs(nuL) < 0.8 * D_ * 1.5
a = ax[0]; ph = a.twinx()
for (lab, (Q, g)), col, ls in zip(fits.items(), ('C3', '0.2'), ('-', '--')):
    B = align(g); a.plot(nuL[sel] / D_, np.abs(B[sel]) / np.abs(R).max(), color=col, ls=ls, lw=1.1,
                         label='second root' if 'default' in lab else 'physical root (the slits)')
    on = sel & (np.abs(B) > 0.15 * np.abs(B).max())
    ph.plot(nuL[sel] / D_, np.where(on[sel], np.angle(B[sel]), np.nan), color=col, ls=':', lw=0.9)   # phase where |A| is not small
a.set_xlabel(r'spatial frequency / $d$'); a.set_ylabel(r'aperture $|A|$'); a.set_title('(a) two apertures', fontsize=8, loc='left')
ph.set_ylim(-np.pi, np.pi); ph.set_yticks([-3, 0, 3]); ph.set_ylabel('phase (rad), dotted', fontsize=6.5); ph.spines['top'].set_visible(False)
x = np.arange(W) - W // 2; b = ax[1]
for (lab, (Q, g)), col, ls in zip(fits.items(), ('C3', '0.2'), ('-', '--')):
    b.plot(x, Q / P.max(), color=col, ls=ls, lw=1.0)
b.legend(*a.get_legend_handles_labels(), fontsize=6, frameon=False, loc='upper right'); b.set_ylim(0, 1.25); b.set_xlim(-400, 400); b.set_yticks([]); b.set_xlabel('detector position (px)'); b.set_title('(b) their intensities', fontsize=8, loc='left')
c = ax[2]; (Q1, _), (Q2, _) = fits.values()
c.plot(x, (Q1 - Q2) / P.max(), color='k', lw=0.8); c.set_xlim(-400, 400); c.set_xlabel('detector position (px)')
c.set_title('(c) difference / peak', fontsize=8, loc='left'); c.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
fig.tight_layout(pad=0.3, w_pad=1.5)
fig.savefig('papers/submit/pra_complex/figs/roots.pdf'); fig.savefig('papers/submit/pra_complex/figs/roots.png', dpi=200)
print('max |difference| / peak %.1e' % (np.abs(Q1 - Q2).max() / P.max()))
