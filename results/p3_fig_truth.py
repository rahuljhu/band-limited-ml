"""Paper 3, Fig. 1 (the truths of p3_ladder.py and p3_cbench.py CB_TRUTH=tapered), from p3_fig_synth.py's truth figure:
2048 bins, three cases, the record-like window of seven fringes shaded. Formerly: the synthetic two-slit figures (truths as p3_dbench.py; fits from its results in results/dbench/p3_dbench).
  synth_truth.pdf    per case: the aperture A(nu) (|A| filled, phase as a line, d, a and the fill marked) and the far
                     field |g|^2 on the line with a histogram of n = 10^4 detected photons (bg 0, rep 0)
  synth_recover.pdf  per case, n = 10^4 (bg 0, rep 0), each method refitted at its selected hyperparameters: the
                     intensity near the centre against the truth, the recovered |A| and the aperture phase across the
                     slits (aligned to the truth over the global phase and the twin)
Run from the repo root (BLML_DATA_DIR not needed)."""
import os, sys, json, numpy as np, matplotlib
os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_SOLVER', 'lbfgs'); os.environ.setdefault('BLML_COLLAPSE', '1')
sys.path.insert(0, '.'); sys.argv = sys.argv[:1]
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
import dslit_job as dj
PP = 'papers/submit/pra_complex/figs'
W, NF = 2048, 1 << 16; D_, A_ = 0.017, 0.005; RHO, PHI, BETA, V = 0.6, 1.0, 2.0, 0.6; FILL = 2 * A_ / (D_ + A_)
nu = np.fft.fftfreq(NF)
def slit(c):
    u = np.abs(nu - c) / (A_ / 2); return np.where(u < 1, np.cos(np.pi * u / 2) ** 2, 0.0)
def field(A):
    return np.fft.fftshift(np.fft.ifft(A))[NF // 2 - W // 2:NF // 2 + W // 2] * NF
A1, A2 = slit(-D_ / 2), slit(D_ / 2)
CASES = [('sym', 'equal slits'), ('asym', 'unequal, phase 1 rad'), ('partial', 'partially coherent')]
def truth(case):
    if case == 'partial':
        g1, g2 = field(A1), field(A2); I = V * np.abs(g1 + g2) ** 2 + (1 - V) * (np.abs(g1) ** 2 + np.abs(g2) ** 2)
        return None, None, I / I.sum()
    A = {'sym': A1 + A2, 'asym': A1 + RHO * np.exp(1j * PHI) * A2,
         'defocus': (A1 + A2) * np.exp(1j * BETA * (nu / (D_ / 2 + A_ / 2)) ** 2)}[case]
    g = field(A); I = np.abs(g) ** 2; return A, g, I / I.sum()
def data(case, n=10000, rep=0, bg=0.0):
    P = truth(case)[2]; P = (1 - bg) * P + bg / W; P = P / P.sum()
    rng = np.random.default_rng(1000 * rep + n % 997); x = rng.choice(W, size=n, p=P); part = rng.choice(3, size=n, p=[0.64, 0.16, 0.20])
    TV = np.bincount(x[part < 2], minlength=W).astype(float); TE = np.bincount(x[part == 2], minlength=W).astype(float)
    return P, TV, TE
ns = np.fft.fftshift(nu); m_ap = np.abs(ns) < 0.02; u_ap = ns / D_            # aperture in units of d
xs = np.arange(W) - W // 2
# ---------------- figure 1: the truths and the data
fig, ax = plt.subplots(2, 3, figsize=(7.0, 3.4), gridspec_kw=dict(height_ratios=[1, 1.15]))
for j, (c, lab) in enumerate(CASES):
    A, g, P = truth(c); a = ax[0, j]
    if A is None:                                   # two mutually incoherent slit fields: their moduli
        for Ak, col in ((A1, 'C0'), (A2, 'C9')):
            a.fill_between(u_ap[m_ap], np.fft.fftshift(Ak)[m_ap], color=col, alpha=0.35, lw=0)
        a.text(0, 0.45, 'mutually\nincoherent\n$V=0.6$', ha='center', fontsize=6.5)
    else:
        As = np.fft.fftshift(A); a.fill_between(u_ap[m_ap], np.abs(As)[m_ap] / np.abs(As).max(), color='C0', alpha=0.35, lw=0)
        b = a.twinx(); ph = np.where(np.abs(As) > 1e-3, np.angle(As), np.nan)
        b.plot(u_ap[m_ap], ph[m_ap], color='C3', lw=1.1); b.set_ylim(-np.pi, np.pi); b.set_yticks([-3, 0, 3])
        b.spines['top'].set_visible(False); b.tick_params(labelsize=6, colors='C3')
        if j == 1:
            b.set_ylabel('phase (rad)', color='C3', fontsize=7)
    a.set_ylim(0, 1.32); a.set_xlim(-1.1, 1.1); a.set_xticks([-0.5, 0, 0.5]); a.set_xticklabels(['$-d/2$', '0', '$d/2$']); a.set_yticks([0, 1])
    a.set_title(f'({"abcd"[j]}) {lab}', fontsize=7.5, loc='left')
    if j == 0:
        a.set_ylabel(r'aperture $|A|$')
        a.annotate('', xy=(-0.5, 1.2), xytext=(0.5, 1.2), arrowprops=dict(arrowstyle='<->', lw=0.7)); a.text(0, 1.24, '$d$', ha='center', fontsize=7)
        a.annotate('', xy=(0.5 - A_ / D_ / 2, 1.05), xytext=(0.5 + A_ / D_ / 2, 1.05), arrowprops=dict(arrowstyle='<->', lw=0.7))
        a.text(0.5 + A_ / D_ / 2 + 0.05, 1.02, '$a$', fontsize=7)
        a.text(-1.08, 0.3, f'fill\n$w=\\frac{{2a}}{{d+a}}$\n$={FILL:.2f}$', fontsize=6)
    a = ax[1, j]; Pn, TV, TE = data(c); C = TV + TE
    C4 = C.reshape(-1, 4).sum(1) / 4; a.bar(xs[::4] + 1.5, C4, width=4, color='0.65', lw=0, label='$10^4$ photons (4-px bins)')
    a.plot(xs, P * C.sum(), color='k', lw=0.8, label=r'far field $|g|^2$')
    w0 = 3.5 / D_; a.axvspan(-w0, w0, color='C3', alpha=0.10, lw=0, label='record-like window' if j == 0 else None)
    a.set_xlim(-800, 800); a.set_xlabel('detector position (px)', fontsize=7); a.set_yticks([])
    if j == 0:
        a.set_ylabel('counts')
h_, l_ = ax[1, 0].get_legend_handles_labels(); ax[1, 2].legend(h_, ['far field', 'record-like\nwindow', '$10^4$ photons'], fontsize=5.5, frameon=False, loc='upper right', bbox_to_anchor=(1.1, 1.02))
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(f'{PP}/synth_truth.pdf'); fig.savefig(f'{PP}/synth_truth.png', dpi=200)
print('figure 1 done', flush=True)

