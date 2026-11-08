"""Paper 3, the double-slit tables and figures with the two-slit kernel as the estimator (results of p3_slit_main.py in
results/p3_slit_main): per record on the paper's split (seed 0) every method's test log-likelihood and its paired
difference to the two-slit kernel with the estimated geometry (tables/dslit_<set>.tex); over six splits and five
replicas the two-slit kernel (estimated and published geometry) minus the KDE (tables/dslit_splits.tex); figures
real_fits.pdf and real_aperture.pdf. Run from the repo root with BLML_DATA_DIR=~/data_repo/dslit_gpu."""
import os, sys, json, glob, numpy as np, matplotlib
os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_SOLVER', 'lbfgs')
sys.argv = sys.argv[:1]; sys.path.insert(0, '.')
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
import dslit_job as dj
from blml_binned import _np
from p3_buildup import setup
PP = 'papers/submit/pra_complex'; R = 'results/p3synth_lbfgs'; R0 = 'results/dslit_1d'; M = 'results/p3_slit_main'
SETS = [('photon', 'photons, single mode'), ('ghost', 'ghost diffraction'), ('electron', 'electrons'), ('photon_mm', 'photons, multimode')]
PUB = dict(photon=100 / 500, ghost=100 / 500, photon_mm=100 / 500, electron=62 / 272)
def split(name):
    x, W, bflat, trv, te = setup(name); u = np.random.default_rng(0).random(len(x)); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))
    return x, W, bflat, np.bincount(x[part < 2], minlength=W).astype(float), np.bincount(x[part == 2], minlength=W).astype(float)
fitslit = lambda TV, c: dj.bg_fit(TV, [c['fc']], (c['N'],), True, 1, max_sweeps=5000, tol=1e-6, fill=c['fill'])
OUT = {}
for s, lab in SETS:
    r0 = json.load(open(f'{M}/{s}_s0.json')); x, W, bflat, TV, TE = split(s); dj.BFIX = np.array([np.nan])
    Qe, me, be = fitslit(TV, r0['est']); Qp, mp, bp = fitslit(TV, r0['pub'])
    P = {'two-slit kernel, estimated': (Qe, be, r0['est']), 'two-slit kernel, published slits': (Qp, bp, r0['pub'])}
    for lab_, d, p in (('complex BLML, band (no aperture)', f'{R}/bg1/{s}', 'complex_bg'), ('KDE', f'{R}/flat/{s}', 'kde'),
                       ('adaptive KDE', f'{R}/flat/{s}', 'akde'), ('Fresnel', f'{R0}/{s}/flat', 'fresnel'),
                       ('Fraunhofer', f'{R0}/{s}/flat', 'fraunhofer'), ('histogram', f'{R0}/{s}/flat', 'hist')):
        j = json.load(open(f'{d}/{s}_{p}.json')); P[lab_] = (np.load(f'{d}/{s}_{p}_P.npy').astype(float), float(np.sum(j.get('beta', [0]))),
                                                            dict(fc=(j['fc'][0] if isinstance(j.get('fc'), list) else j.get('fc')), fill=None))
    fr = json.load(open(f'{R0}/{s}/flat/{s}_fraunhofer.json'))['params']; phys = (1 + fr['a_over_d']) / fr['P']
    ref = np.log(np.maximum(Qe / Qe.sum(), 1e-300)); rows = []
    for k, (Q, beta, c) in P.items():
        L = np.log(np.maximum(Q / Q.sum(), 1e-300)); d = np.repeat(L - ref, TE.astype(int))
        rows.append(dict(model=k, test=float((TE * L).sum() / TE.sum()), diff=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))),
                         beta=beta, ratio=None if c.get('fc') is None else c['fc'] / phys, fill=c.get('fill'), N=c.get('N')))
    rows.sort(key=lambda r: -r['test']); OUT[s] = dict(rows=rows, est=r0['est'], pub=r0['pub'])
    T = [r'\begin{tabular}{lrrrr}', r'\toprule', r'model & test LL & $-$ two-slit, est.\ ($\pm$ SE) & $\beta$ & $f_c$/physical \\', r'\midrule']
    for r in rows:
        dcol = '--' if r['model'] == 'two-slit kernel, estimated' else f"{r['diff']:+.4f} ({r['se']:.4f})"
        fcc = '--' if r['ratio'] is None else f"{r['ratio']:.2f}" + (f" ($w={r['fill']:.2f}$, $N={r['N']}$)" if r['fill'] is not None else '')
        T.append(f"{r['model']} & {r['test']:.4f} & {dcol} & {'--' if r['model'] == 'histogram' else format(r['beta'], '.2f')} & {fcc} \\\\")
    T += [r'\bottomrule', r'\end{tabular}']
    open(f'{PP}/tables/dslit_{s}.tex', 'w').write('\n'.join(T) + '\n')
json.dump(OUT, open('results/p3_slit_tables.json', 'w'), indent=1)
# splits and replicas
L = [r'\begin{tabular}{lccccc}', r'\toprule',
     r' & \multicolumn{3}{c}{record, six splits} & \multicolumn{2}{c}{five replicas} \\', r'\cmidrule(lr){2-4}\cmidrule(lr){5-6}',
     r'record & estimated $-$ KDE & published $-$ KDE & $f_c$/phys., $w$ & estimated $-$ KDE & KL: est./KDE \\', r'\midrule']
SPL = {}
for s, lab in SETS:
    S = [json.load(open(f)) for f in sorted(glob.glob(f'{M}/{s}_s[0-9].json'))]
    Rp = [json.load(open(f)) for f in sorted(glob.glob(f'{M}/{s}_rep*_s0.json'))]
    de = np.array([v['est_minus_kde']['diff'] for v in S]); dp = np.array([v['pub_minus_kde']['diff'] for v in S])
    ge = np.array([v['est_minus_kde']['diff'] for v in Rp]) if Rp else np.array([np.nan])
    kr = np.mean([v['est']['kl'] for v in Rp]) / np.mean([v['kde']['kl'] for v in Rp]) if Rp else np.nan
    fcr = np.array([v['est']['ratio'] for v in S]); ws = np.array([v['est']['fill'] for v in S])
    SPL[s] = dict(est=de.tolist(), pub=dp.tolist(), rep=ge.tolist(), kde_minus_est=(-de).tolist())
    rng = lambda a: f'{a.min():.2f}' if a.min() == a.max() else f'{a.min():.2f}--{a.max():.2f}'
    L.append(f"{lab} & ${de.mean():+.4f}$ (${de.min():+.4f}$, ${de.max():+.4f}$) & ${dp.mean():+.4f}$ & {rng(fcr)}, {rng(ws)} & "
             f"${np.nanmean(ge):+.4f}\\pm{np.nanstd(ge, ddof=1) if len(ge) > 1 else 0:.4f}$ & {kr:.2f} \\\\")
L += [r'\bottomrule', r'\end{tabular}']
open(f'{PP}/tables/dslit_splits.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
json.dump(SPL, open('results/p3_slit_splits.json', 'w'), indent=1)
# ---------------- figure: fits and scores
fig, ax = plt.subplots(2, 4, figsize=(7.0, 4.2), gridspec_kw=dict(height_ratios=[1, 1.0]))
for j, (s, lab) in enumerate(SETS):
    x, W, bflat, TV, TE = split(s); O = OUT[s]
    Qe = fitslit(TV, O['est'])[0]; Qp = fitslit(TV, O['pub'])[0]
    Pk = np.load(f'{R}/flat/{s}/{s}_kde_P.npy').astype(float); Pf = np.load(f'{R0}/{s}/flat/{s}_fresnel_P.npy').astype(float)
    a = ax[0, j]; c0 = int(np.argmax(np.convolve(TV + TE, np.ones(15), 'same'))); half = 110
    lo, hi = max(0, c0 - half), min(W, c0 + half); b4 = 4; edges = np.arange(lo, hi + 1, b4)
    tc = np.add.reduceat(TE[lo:hi], np.arange(0, hi - lo, b4))[:len(edges) - 1]
    a.bar(edges[:-1] + b4 / 2, tc, width=b4, color='0.84', lw=0, label='test counts')
    for Q, nm, col, ls in ((Qe, 'two-slit kernel, estimated', 'C3', '-'), (Qp, 'two-slit kernel, published slits', 'C1', '--'),
                           (Pk, 'KDE', '0.35', ':'), (Pf, 'Fresnel model', 'C2', '-.')):
        Q = Q / Q.sum() * TE.sum(); a.plot(np.arange(lo, hi), Q[lo:hi] * b4, color=col, ls=ls, lw=0.9, label=nm)
    a.set_xlim(lo, hi); a.set_yticks([]); a.set_title(f'({"abcd"[j]}) {lab}', fontsize=7.5, loc='left'); a.set_xlabel('px', fontsize=7)
    a = ax[1, j]; rows = {r['model']: r for r in O['rows']}
    order = ['two-slit kernel, published slits', 'complex BLML, band (no aperture)', 'adaptive KDE', 'KDE', 'Fresnel', 'Fraunhofer', 'histogram']
    labs = ['two-slit, published', 'band, no aperture', 'adaptive KDE', 'KDE', 'Fresnel', 'Fraunhofer', 'histogram']
    ys = np.arange(len(order))[::-1]
    for y, k in zip(ys, order):
        r = rows[k]; a.errorbar(r['diff'] * 1e3, y, xerr=r['se'] * 1e3, fmt='o', ms=2.5, color='k', lw=0.7, capsize=0)
    yk = ys[order.index('KDE')]; d6 = np.array(SPL[s]['kde_minus_est']) * 1e3
    a.plot([d6.min(), d6.max()], [yk - 0.35] * 2, color='C3', lw=2, alpha=0.5); a.plot(d6.mean(), yk - 0.35, '|', color='C3', ms=6)
    g = -np.array(SPL[s]['rep']) * 1e3
    if np.all(np.isfinite(g)):
        a.axvspan(g.mean() - g.std(ddof=1), g.mean() + g.std(ddof=1), color='C0', alpha=0.1, lw=0)
    a.axvline(0, color='0.5', lw=0.5); a.set_yticks(ys); a.set_yticklabels(labs if j == 0 else [], fontsize=6.5)
    a.set_xlabel(r'$\Delta$ test LL ($10^{-3}$ nats)', fontsize=6.5); a.set_xlim(max(-30, a.get_xlim()[0]), max(2, a.get_xlim()[1]))
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=5, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.95))
fig.savefig(f'{PP}/figs/real_fits.pdf'); fig.savefig(f'{PP}/figs/real_fits.png', dpi=200)
# ---------------- figure: aperture and phase
fig, ax = plt.subplots(2, 2, figsize=(7.0, 3.9), gridspec_kw=dict(height_ratios=[1, 0.8]))
for j, s in enumerate(('photon', 'electron')):
    x, W, bflat, TV, TE = split(s); O = OUT[s]
    th = np.array(json.load(open(f'{R0}/{s}/flat/{s}_fresnel.json'))['theta']); dnu = 1.0 / dj.ref_desc(th, True)['P']; r = PUB[s]
    L8 = 16 * W; nuL = np.fft.fftfreq(L8); uL = np.fft.fftshift(nuL) / dnu
    ap = lambda g: np.fft.fftshift(np.fft.fft(np.r_[g, np.zeros(L8 - W)]) * np.exp(2j * np.pi * nuL * (W // 2)))
    ge = np.asarray(_np(fitslit(TV, O['est'])[1].Se_))[0]; gp = np.asarray(_np(fitslit(TV, O['pub'])[1].Se_))[0]
    gf = dj.ref_parts(th, (W,), 0, True)[1]
    nuW = np.fft.fftfreq(W); supp = np.abs(np.abs(nuW) - dnu / 2) <= r * dnu / 2; amp = np.sqrt(TV / TV.sum()); best = None
    for sd in range(20):
        rr = np.random.default_rng(sd); gg = np.fft.ifftshift(amp * np.exp(2j * np.pi * rr.random(W))); Aw = np.where(supp, np.fft.fft(gg), 0)
        for it in range(500):
            gg = np.fft.ifftshift(amp) * np.exp(1j * np.angle(np.fft.ifft(Aw))); An = np.fft.fft(gg)
            Aw = np.where(supp, An, Aw - 0.9 * An) if it < 400 else np.where(supp, An, 0)
        gh = np.fft.fftshift(np.fft.ifft(Aw)); Lh = dj.ll(np.abs(gh) ** 2 + 1e-12, TV)
        if best is None or Lh > best[0]:
            best = (Lh, gh)
    a, b = ax[0, j], ax[1, j]; mm = np.abs(uL) < 1.2
    for v in (-0.5, 0.5):
        a.axvspan(v - r / 2, v + r / 2, color='0.88', lw=0); b.axvspan(v - r / 2, v + r / 2, color='0.88', lw=0)
    for g, nm, col, ls in ((ge, 'two-slit kernel, estimated', 'C3', '-'), (gp, 'two-slit kernel, published slits', 'C1', '--'),
                           (best[1], 'HIO, published slits', 'C4', ':'), (gf, 'Fresnel fit', 'C2', '-.')):
        A = ap(g); a.plot(uL[mm], np.abs(A[mm]) / np.abs(A).max(), color=col, ls=ls, lw=1, label=nm)
        on = mm & (np.abs(A) > 0.2 * np.abs(A).max()); ref = np.angle(A[np.argmin(np.abs(uL + 0.5))])
        b.plot(uL[on], np.angle(A[on] * np.exp(-1j * ref)), '.', color=col, ms=1.3)
    a.set_xlim(-1.2, 1.2); b.set_xlim(-1.2, 1.2); b.set_ylim(-np.pi, np.pi); b.set_yticks([-3, 0, 3]); a.set_yticks([0, 1])
    for a_ in (a, b):
        a_.set_xticks([-0.5, 0, 0.5]); a_.set_xticklabels(['$-d/2$', '0', '$d/2$'])
    e = O['est']; a.set_title(f'({"ab"[j]}) {"photons, single mode" if s == "photon" else "electrons"}', fontsize=7.5, loc='left')
    a.text(0.99, 0.97, f"estimated: $N={e['N']}$, $w={e['fill']:.2f}$; published $w={2 * r / (1 + r):.2f}$", transform=a.transAxes,
           ha='right', va='top', fontsize=6.3, color='C3')
    if j == 0:
        a.set_ylabel(r'$|\hat A|$'); b.set_ylabel('phase (rad)\nrel. to left slit', fontsize=7)
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))
fig.savefig(f'{PP}/figs/real_aperture.pdf'); fig.savefig(f'{PP}/figs/real_aperture.png', dpi=200)
print('done')
