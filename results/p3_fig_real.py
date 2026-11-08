"""Paper 3, the real double-slit figures.
  real_data.pdf     per record: the detections on the detector (photons: the 512 x 256 crop, the fringe axis horizontal as below;
                    electrons: the detector around the 18-row line, the line marked) and the counts along the fringes
  real_fits.pdf     per record: the test counts along the fringes (4-pixel bins, centre) with complex BLML (band, N = 1),
                    the two-slit kernel (N = 3, CV), the KDE and the Fresnel model; below, each method's test
                    log-likelihood minus complex BLML's on the paper's split (+- paired SE), the KDE's mean and range
                    over six splits, and the replicas' expectation for the KDE
  real_aperture.pdf photons and electrons: |A_hat| and the aperture phase across the slits from complex BLML (band),
                    the two-slit kernel, HIO given the published slits and the Fresnel fit, against the published slits
Run from the repo root: BLML_DATA_DIR=~/data_repo/dslit_gpu .venv/bin/python results/p3_fig_real.py"""
import os, sys, json, glob, numpy as np, matplotlib
os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_SOLVER', 'lbfgs')
sys.argv = sys.argv[:1]; sys.path.insert(0, '.')
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
import dslit_job as dj
from blml_binned import _np
from p3_buildup import setup
PP = 'papers/submit/pra_complex/figs'; D = os.path.expanduser('~/data_repo/dslit_gpu/'); R = 'results/p3synth_lbfgs'
SETS = [('photon', 'photons, single mode'), ('ghost', 'ghost diffraction'), ('electron', 'electrons'),
        ('photon_mm', 'photons, multimode')]
PUB = dict(photon=100 / 500, ghost=100 / 500, electron=62 / 272, photon_mm=100 / 500)   # published a / d
def split(name):
    x, W, bflat, trv, te = setup(name); u = np.random.default_rng(0).random(len(x)); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))
    return x, W, np.bincount(x[part < 2], minlength=W).astype(float), np.bincount(x[part == 2], minlength=W).astype(float)
# ---------------- figure: the records
fig, ax = plt.subplots(2, 4, figsize=(7.0, 3.6), gridspec_kw=dict(height_ratios=[1.3, 1]))
for j, (s, lab) in enumerate(SETS):
    a = ax[0, j]; p = np.load(D + f'dslit_{s}_points.npy')
    if s == 'electron':
        ev = np.load(os.path.expanduser('~/data_repo/unl_dslit/electrons.npy')); sel = (ev[:, 2] > 140) & (ev[:, 2] < 230)
        a.scatter(ev[sel, 1], ev[sel, 2], s=0.3, color='k', lw=0, alpha=0.6)
        a.axhspan(175.5, 193.5, color='C3', alpha=0.15, lw=0); a.set_ylim(230, 140); a.set_xlim(0, 640)
        a.text(5, 172, 'line', color='C3', fontsize=6.5)
    else:
        k = p if len(p) < 30000 else p[np.random.default_rng(0).choice(len(p), 30000, replace=False)]
        a.scatter(k[:, 1], k[:, 2], s=0.3 if len(p) < 30000 else 0.08, color='k', lw=0, alpha=0.6)   # fringe axis horizontal
        a.set_xlim(0, 512); a.set_ylim(256, 0)
    a.set_xticks([]); a.set_yticks([]); a.set_title(f'({"abcd"[j]}) {lab}', fontsize=7.5, loc='left')
    a.text(0.98, 0.02, f'{len(p):,} particles', transform=a.transAxes, ha='right', fontsize=6.5, color='C0')
    a = ax[1, j]; x, W, TV, TE = split(s); C = TV + TE
    a.bar(np.arange(W), C, width=1, color='0.45', lw=0); a.set_xlim(0, W); a.set_yticks([])
    a.set_xlabel('position along the fringes (px)', fontsize=7)
    if j == 0:
        a.set_ylabel('counts')
fig.tight_layout(pad=0.3, w_pad=0.6)
fig.savefig(f'{PP}/real_data.pdf'); fig.savefig(f'{PP}/real_data.png', dpi=200)
print('records done', flush=True)
# ---------------- figure: fits and test results
T2 = json.load(open('results/p3_dslit_table2.json')); SC = json.load(open('results/p3_split_check.json'))
fits_cache = {}
fig, ax = plt.subplots(2, 4, figsize=(7.0, 4.2), gridspec_kw=dict(height_ratios=[1, 1.15]))
SHOW = [('complex BLML, $N=1$', 'complex BLML, band', 'C0', '-'), ('two-slit', 'complex BLML, two-slit kernel (CV)', 'C3', '-'),
        ('KDE', 'KDE', '0.4', '--'), ('Fresnel', 'Fresnel model', 'C2', '-.')]
for j, (s, lab) in enumerate(SETS):
    x, W, TV, TE = split(s); dj.BFIX = np.array([np.nan])
    Pb = np.load(f'{R}/bg1/{s}/{s}_complex_bg_P.npy').astype(float)
    C = {json.load(open(f))['config']: json.load(open(f)) for f in glob.glob(f'results/p3_slits_diag/{s}_K1-N[0-9].json')}
    dg = C[max(C, key=lambda k: C[k]['cv_best'])]                    # cut-off, fill and background harmonics by CV
    m = dj.bg_fit(TV, [dg['fc']], (dg['N'],), True, 1, max_sweeps=5000, tol=1e-6, fill=dg['fill'])[1]; Ps = np.asarray(_np(m.density_)); fits_cache[s] = (dg, m)
    Pk = np.load(f'{R}/flat/{s}/{s}_kde_P.npy').astype(float); Pf = np.load(f'results/dslit_1d/{s}/flat/{s}_fresnel_P.npy').astype(float)
    curves = {'complex BLML, $N=1$': Pb, 'two-slit': Ps, 'KDE': Pk, 'Fresnel': Pf}
    a = ax[0, j]; c0 = int(np.argmax(np.convolve(TV + TE, np.ones(15), 'same'))); half = 110
    lo, hi = max(0, c0 - half), min(W, c0 + half); b4 = 4; edges = np.arange(lo, hi + 1, b4)
    tc = np.add.reduceat(TE[lo:hi], np.arange(0, hi - lo, b4))[:len(edges) - 1]
    a.bar(edges[:-1] + b4 / 2, tc, width=b4, color='0.82', lw=0, label='test counts')
    for k, nm, col, ls in SHOW:
        P = curves[k]
        if P is None:
            continue
        P = P / P.sum() * TE.sum(); a.plot(np.arange(lo, hi), P[lo:hi] * b4, color=col, ls=ls, lw=0.9, label=nm)
    a.set_xlim(lo, hi); a.set_yticks([]); a.set_title(f'({"abcd"[j]}) {lab}', fontsize=7.5, loc='left'); a.set_xlabel('px', fontsize=7)
    # bottom: differences to complex BLML (N = 1)
    a = ax[1, j]; rows = {r['model']: r for r in T2[s]['rows']}
    order = ['complex BLML, flat background', f"two-slit kernel, CV ($N={dg['N']}$)", 'streaming, $N=1$', 'adaptive KDE', 'KDE', 'Fresnel', 'Fraunhofer', 'histogram']
    labs = ['BLML, EM bg', 'two-slit kernel', 'streaming', 'adaptive KDE', 'KDE', 'Fresnel', 'Fraunhofer', 'histogram']
    ys = np.arange(len(order))[::-1]
    for y, k, lb in zip(ys, order, labs):
        if k in rows:
            r = rows[k]; a.errorbar(r['diff'] * 1e3, y, xerr=r['se'] * 1e3, fmt='o', ms=2.5, color='k', lw=0.7, capsize=0)
    yk = ys[order.index('KDE')]; d6 = np.array([v['diff'] for v in SC[s]]) * -1e3       # KDE - BLML over six splits
    a.plot([d6.min(), d6.max()], [yk - 0.35] * 2, color='C3', lw=2, alpha=0.5); a.plot(d6.mean(), yk - 0.35, '|', color='C3', ms=6)
    rep = [json.load(open(f)) for f in glob.glob(f'{R}/p3_synth/{s}_base_*.json')]
    g = -np.array([d_['blml_bg1']['test'] - d_['kde']['test'] for d_ in rep]) * 1e3
    a.axvspan(g.mean() - g.std(ddof=1), g.mean() + g.std(ddof=1), ymin=0, ymax=1, color='C0', alpha=0.08, lw=0)
    a.axvline(0, color='0.5', lw=0.5); a.set_yticks(ys); a.set_yticklabels(labs if j == 0 else [], fontsize=6.5)
    a.set_xlabel(r'$\Delta$ test LL vs BLML ($10^{-3}$ nats)', fontsize=6.5)
    a.set_xlim(max(-30, a.get_xlim()[0]), max(2, a.get_xlim()[1]))
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=5, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.95))
fig.savefig(f'{PP}/real_fits.pdf'); fig.savefig(f'{PP}/real_fits.png', dpi=200)
print('fits done', flush=True)
# ---------------- figure: aperture and phase from the particles
fig, ax = plt.subplots(2, 2, figsize=(7.0, 3.9), gridspec_kw=dict(height_ratios=[1, 0.8]))
for j, s in enumerate(('photon', 'electron')):
    x, W, TV, TE = split(s); th = np.array(json.load(open(f'results/dslit_1d/{s}/flat/{s}_fresnel.json'))['theta'])
    Pfr = dj.ref_desc(th, True)['P']; dnu = 1.0 / Pfr; r = PUB[s]
    L = 16 * W; nuL = np.fft.fftfreq(L); uL = np.fft.fftshift(nuL) / dnu
    ap = lambda g: np.fft.fftshift(np.fft.fft(np.r_[g, np.zeros(L - W)]) * np.exp(2j * np.pi * nuL * (W // 2)))
    band = json.load(open(f'{R}/bg1/{s}/{s}_complex_bg.json'))['fc'][0]
    gb = np.asarray(_np(dj.bg_fit(TV, [band], (1,), True, 1, max_sweeps=5000, tol=1e-6)[1].Se_))[0]
    dg, m = fits_cache[s]; gs = np.asarray(_np(m.Se_))[0]
    gf = dj.ref_parts(th, (W,), 0, True)[1]
    nuW = np.fft.fftfreq(W); supp = (np.abs(np.abs(nuW) - dnu / 2) <= r * dnu / 2)          # the published slits
    amp = np.sqrt(TV / TV.sum()); best = None
    for sd in range(20):
        rr = np.random.default_rng(sd); gg = np.fft.ifftshift(amp * np.exp(2j * np.pi * rr.random(W))); Aw = np.where(supp, np.fft.fft(gg), 0)
        for it in range(500):
            gg = np.fft.ifftshift(amp) * np.exp(1j * np.angle(np.fft.ifft(Aw))); An = np.fft.fft(gg)
            Aw = np.where(supp, An, Aw - 0.9 * An) if it < 400 else np.where(supp, An, 0)
        gh = np.fft.fftshift(np.fft.ifft(Aw)); Lh = dj.ll(np.abs(gh) ** 2 + 1e-12, TV)
        if best is None or Lh > best[0]:
            best = (Lh, gh)
    curves = [(gb, 'complex BLML, band', 'C0', '-'), (gs, 'two-slit kernel, CV', 'C3', '-'),
              (best[1], 'HIO, published slits', 'C4', ':'), (gf, 'Fresnel fit', 'C2', '-.')]
    if s == 'electron':                              # the two-slit kernel at the published geometry, N = 3 (ties the band)
        gp = np.asarray(_np(dj.bg_fit(TV, [(1 + r) * dnu], (3,), True, 1, max_sweeps=5000, tol=1e-6, fill=2 * r / (1 + r))[1].Se_))[0]
        curves.append((gp, 'two-slit kernel, published slits', 'C3', '--'))
    a, b = ax[0, j], ax[1, j]; mm = np.abs(uL) < 1.2
    for v in (-0.5, 0.5):
        a.axvspan(v - r / 2, v + r / 2, color='0.88', lw=0); b.axvspan(v - r / 2, v + r / 2, color='0.88', lw=0)
    for g, nm, col, ls in curves:
        A = ap(g); a.plot(uL[mm], np.abs(A[mm]) / np.abs(A).max(), color=col, ls=ls, lw=1, label=nm)
        on = mm & (np.abs(A) > 0.2 * np.abs(A).max()); ref = np.angle(A[np.argmin(np.abs(uL + 0.5))])
        b.plot(uL[on], np.angle(A[on] * np.exp(-1j * ref)), '.', color=col, ms=1.3)
    a.set_xlim(-1.2, 1.2); b.set_xlim(-1.2, 1.2); b.set_ylim(-np.pi, np.pi); b.set_yticks([-3, 0, 3])
    for a_ in (a, b):
        a_.set_xticks([-0.5, 0, 0.5]); a_.set_xticklabels(['$-d/2$', '0', '$d/2$'])
    a.set_title(f'({"ab"[j]}) {"photons, single mode" if s == "photon" else "electrons"}', fontsize=7.5, loc='left'); a.set_yticks([0, 1])
    a.text(0.99, 0.97, f"CV: $N={dg['N']}$, $w={dg['fill']:g}$", transform=a.transAxes, ha='right', va='top', fontsize=6.5, color='C3')
    if j == 0:
        a.set_ylabel(r'$|\hat A|$'); b.set_ylabel('phase (rad)\nrel. to left slit', fontsize=7)
h, l = ax[0, 1].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
fig.savefig(f'{PP}/real_aperture.pdf'); fig.savefig(f'{PP}/real_aperture.png', dpi=200)
print('aperture done')
