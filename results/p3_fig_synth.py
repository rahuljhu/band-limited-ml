"""Paper 3, the synthetic two-slit figures (truths as p3_dbench.py; fits from its results in results/dbench/p3_dbench).
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
W, NF = 768, 1 << 16; D_, A_ = 0.017, 0.005; RHO, PHI, BETA, V = 0.6, 1.0, 2.0, 0.6; FILL = 2 * A_ / (D_ + A_)
nu = np.fft.fftfreq(NF)
def slit(c):
    u = np.abs(nu - c) / (A_ / 2); return np.where(u < 1, np.cos(np.pi * u / 2) ** 2, 0.0)
def field(A):
    return np.fft.fftshift(np.fft.ifft(A))[NF // 2 - W // 2:NF // 2 + W // 2] * NF
A1, A2 = slit(-D_ / 2), slit(D_ / 2)
CASES = [('sym', 'equal slits'), ('asym', 'unequal, phase 1 rad'), ('defocus', 'defocus, 2 rad'), ('partial', 'partially coherent')]
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
fig, ax = plt.subplots(2, 4, figsize=(7.0, 3.4), gridspec_kw=dict(height_ratios=[1, 1.15]))
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
        if j == 2:
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
    a.bar(xs, C, width=1, color='0.75', lw=0, label='$10^4$ photons')
    a.plot(xs, P * C.sum(), color='k', lw=0.9, label=r'far field $|g|^2$')
    a.set_xlim(-250, 250); a.set_xlabel('detector position (px)', fontsize=7); a.set_yticks([])
    if j == 0:
        a.set_ylabel('counts'); a.legend(fontsize=6, frameon=False, loc='upper right')
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(f'{PP}/synth_truth768.pdf'); fig.savefig(f'{PP}/synth_truth768.png', dpi=200)
print('figure 1 done', flush=True)
# ---------------- figure 2: what each method recovers at n = 10^4 (bg 0, rep 0)
from blml_binned import _np
L8 = 8 * W; nuL = np.fft.fftfreq(L8); uL = np.fft.fftshift(nuL) / D_
def aperture(g):
    return np.fft.fftshift(np.fft.fft(np.r_[g, np.zeros(L8 - W)]) * np.exp(2j * np.pi * nuL * (W // 2)))
def align(Ah, Aref):
    """Ah aligned to Aref over the global phase and the twin conj(A(-nu)); both unit norm."""
    Ah = Ah / np.linalg.norm(Ah); R = Aref / np.linalg.norm(Aref); best = None
    for tw in (False, True):
        B = np.conj(Ah[::-1]) if tw else Ah
        B = np.roll(B, 1) if tw else B                     # fftshifted grid: -nu maps index k to L8 - k
        ip = np.vdot(B, R); e = np.sqrt(max(2 - 2 * abs(ip), 0))
        if best is None or e < best[0]:
            best = (e, B * np.exp(1j * np.angle(ip)))
    return best[1]
NREC = int(os.environ.get('P3_FIG_N', '100000'))   # the data-set size of figure 2
METH = [('blml_slits', 'complex BLML, two-slit kernel', 'C3', '-'), ('blml', 'complex BLML, band', 'C0', '-'),
        ('fresnel', 'Fresnel model', 'C2', '-.'), ('hio_true_support', 'HIO, true support', 'C4', (0, (4, 1.5))), ('kde', 'KDE', '0.45', ':')]
fig, ax = plt.subplots(3, 4, figsize=(7.0, 5.6), gridspec_kw=dict(height_ratios=[1, 1, 0.9]))
for j, (c, lab) in enumerate(CASES):
    r = json.load(open(f'results/dbench/p3_dbench/{c}_bg0_n{NREC}_0.json'))
    A, g, P = truth(c); Pn, TV, TE = data(c, n=NREC); dj.BFIX = np.array([np.nan]); bflat = np.ones((1, W)) / W
    fits = {}
    s0 = json.load(open(f'results/dbench/p3_dbench/{c}_bg0_n{NREC}_0_s0.json'))   # flat background (N = 0), as Sec. III
    sl, inc = (s0['blml_three'], 2) if c == 'partial' else (s0['blml_slits0'], 0)  # partially coherent: the three-mode model
    m = dj.bg_fit(TV, [s0['blml0']['fc']], (0,), True, 1, max_sweeps=5000, tol=1e-6)[1]
    fits['blml'] = (np.asarray(_np(m.density_)), np.asarray(_np(m.Se_))[0] / m.n_)
    m = dj.bg_fit(TV, [sl['fc']], (0,), True, 1, max_sweeps=5000, tol=1e-6, fill=sl['fill'], incoh=inc)[1]
    fits['blml_slits'] = (np.asarray(_np(m.density_)), np.asarray(_np(m.Se_))[0] / m.n_)
    fits['kde'] = (dj.em(TV, bflat, dj.kde_q((r['kde']['h'],)))[0], None)
    th = dj.ref_fit(TV, bflat, 0, False); th = dj.ref_fit(TV, bflat, 0, True, th0=th)
    fits['fresnel'] = (dj.ref_P(th, bflat, 0, True)[0], dj.ref_parts(th, (W,), 0, True)[1])
    nuW = np.fft.fftfreq(W); Asup = (A1 + A2) if A is None else A
    supp = np.interp(nuW, np.fft.fftshift(nu), np.fft.fftshift(np.abs(Asup))) > 1e-3 * np.abs(Asup).max()
    amp = np.sqrt(TV / TV.sum()); best = None
    for sd in range(20):
        rr = np.random.default_rng(sd); gg = np.fft.ifftshift(amp * np.exp(2j * np.pi * rr.random(W))); Aw = np.where(supp, np.fft.fft(gg), 0)
        for it in range(500):
            gg = np.fft.ifftshift(amp) * np.exp(1j * np.angle(np.fft.ifft(Aw))); An = np.fft.fft(gg)
            Aw = np.where(supp, An, Aw - 0.9 * An) if it < 400 else np.where(supp, An, 0)
        gh = np.fft.fftshift(np.fft.ifft(Aw)); Lh = dj.ll(np.abs(gh) ** 2 + 1e-12, TV)
        if best is None or Lh > best[0]:
            best = (Lh, gh)
    fits['hio_true_support'] = (np.abs(best[1]) ** 2 + 1e-12, best[1])
    # row 1: intensity near the centre
    a = ax[0, j]; z = np.abs(xs) <= 150
    a.fill_between(xs[z], 0, P[z] / P[z].max() * 0.07, color='k', alpha=0.08, lw=0, label='truth (scaled)')
    for k, nm, col, ls in METH:
        Q = fits[k][0] / fits[k][0].sum(); a.plot(xs[z], (Q[z] - P[z]) / P[z].max(), color=col, ls=ls, lw=1.2 if k == 'hio_true_support' else 0.9, label=nm)
    a.axhline(0, color='0.6', lw=0.4); a.set_ylim(-0.06, 0.08)
    a.set_title(f'({"abcd"[j]}) {lab}', fontsize=7.5, loc='left'); a.set_xlabel('detector position (px)', fontsize=7)
    if j == 0:
        a.set_ylabel('fit $-$ truth')
    # rows 2-3: aperture modulus and phase, aligned to the truth
    Aref = aperture(g) if g is not None else None          # aperture() is already centred (fftshifted)
    a2, a3 = ax[1, j], ax[2, j]; mm = np.abs(uL) < 1.1
    if Aref is not None:
        Rn = Aref / np.linalg.norm(Aref)
        a2.fill_between(uL[mm], np.abs(Rn[mm]) / np.abs(Rn).max(), color='k', alpha=0.15, lw=0)
        on = np.abs(Rn) > 0.15 * np.abs(Rn).max()
        a3.plot(uL[mm & on], np.angle(Rn)[mm & on], 'o', color='k', ms=1.2, alpha=0.3)
    else:
        for Ak in (A1, A2):
            Ak_ = aperture(field(Ak)); a2.fill_between(uL[mm], np.abs(Ak_[mm]) / np.abs(Ak_).max(), color='k', alpha=0.15, lw=0)
    for k, nm, col, ls in METH:
        if fits[k][1] is None:
            continue
        Ah = aperture(fits[k][1])
        if Aref is not None:
            Ah = align(Ah, Aref)
        a2.plot(uL[mm], np.abs(Ah[mm]) / np.abs(Ah).max(), color=col, ls=ls, lw=1.2 if k == 'hio_true_support' else 0.9)
        if Aref is not None:
            sel = mm & on; a3.plot(uL[sel], np.angle(Ah[sel]), '.', color=col, ms=1.5)
    for a_ in (a2, a3):
        a_.set_xlim(-1.1, 1.1); a_.set_xticks([-0.5, 0, 0.5]); a_.set_xticklabels(['$-d/2$', '0', '$d/2$'])
    a2.set_yticks([0, 1]); a3.set_ylim(-np.pi, np.pi); a3.set_yticks([-3, 0, 3])
    if Aref is None:
        a3.text(0, 0, 'no single field', ha='center', fontsize=7); a3.set_yticks([])
    if j == 0:
        a2.set_ylabel(r'aperture $|\hat A|$'); a3.set_ylabel('aperture phase')
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=3.2)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.94))
fig.savefig(f'{PP}/synth_recover.pdf'); fig.savefig(f'{PP}/synth_recover.png', dpi=200)
print('figure 2 done')
