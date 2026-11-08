"""Paper 3, double slit, 1-d protocol: tables, figure and numbers from results/dslit_1d (results/dslit_1d_runs.sh,
the stream_bg runs, results/dslit_1d_table.py, results/dslit_1d_valcurve.py).
Writes papers/paper3/tables/dslit_<set>.tex (photon, electron, photon_mm), papers/paper3/figs/dslit.pdf/png and
papers/paper3/numbers_dslit.tex. Figure rows: single-mode photons, electrons. Columns: (a) test counts along the
fringes with complex BLML (background mode N = 1) and the Fraunhofer model; (b) validation log-likelihood against
fc / physical band for N = 0, 1, 2 (total nats over the validation particles, relative to the maximum); (c) |FT g|
of complex BLML along the fringes against the aperture (abscissa xi / d = nu P: slits of width a / d at +-1/2) and
|FT |g|^2|, the aperture autocorrelation; (d) the streaming fc as particles arrive, against the batch fc and the
physical band. Run from the repo root."""
import os, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R0, PP = 'results/dslit_1d', 'papers/paper3'; DATA = os.path.expanduser('~/data_repo/dslit_gpu/')
G = os.path.expanduser('~/data_repo/glasgow_dslit/')
T = json.load(open(f'{R0}/table.json')); TAG = dict(photon='Ph', electron='El', photon_mm='Mm', ghost='Gh')
LAB = {'complex, bg N=1': 'BLML complex, $N=1$', 'complex, bg N=0': 'BLML complex, $N=0$', 'complex, bg N=2': 'BLML complex, $N=2$',
       'KDE, flat bg': 'KDE', 'adaptive KDE, flat bg': 'adaptive KDE', 'Fresnel, flat bg': 'Fresnel', 'Fraunhofer, flat bg': 'Fraunhofer', 'histogram': 'histogram'}
num = {}
fmt = lambda v, d=4: f'{v:.{d}f}'
big = lambda v: f'{int(v):,}'.replace(',', '{,}')


def data(name):
    pts = np.load(DATA + f'dslit_{name}_points.npy'); SH = np.load(DATA + f'dslit_{name}_b.npy').shape[-2:]
    axis = int(np.load(DATA + f'dslit_{name}_axis.npy'))
    x = np.clip(np.floor(pts[:, 1 + axis] + 0.5).astype(int), 0, SH[axis] - 1)
    te = np.random.default_rng(0).random(len(pts)) >= 0.80
    return pts, x, te, SH[axis]


def ft_heights(line, P, pad):
    A = np.abs(np.fft.fft(line, pad)); A = A / A.max(); xi = np.fft.fftfreq(pad) * P
    return {c: float(A[np.abs(xi - c) < 0.12].max()) for c in (-1.0, -0.5, 0.0, 0.5, 1.0)}


# ---------------------------------------------------------------- tables and numbers
for name, t in TAG.items():
    tb = T[name]; rows = {r['model']: r for r in tb['rows']}; ref = rows['complex, bg N=1']
    fr = json.load(open(f'{R0}/{name}/flat/{name}_fraunhofer.json'))['params']; phys = tb['phys']
    pts, x, te, W = data(name)
    with open(f'{PP}/tables/dslit_{name}.tex', 'w') as f:
        f.write('\\begin{tabular}{lrrrr}\n\\toprule\nmodel & test LL & $-$ BLML $N{=}1$ ($\\pm$ SE) & $\\beta$ & $f_c$ / physical \\\\\n\\midrule\n')
        for r in [r for r in tb['rows'] if r['model'] in LAB]:          # blurred references: text only
            d = '--' if r['model'] == 'complex, bg N=1' else f"{r['diff']:+.4f} ({r['se']:.4f})"
            be = '--' if not r['beta'] else f"{r['beta']:.2f}"
            fq = '--' if r['fc_ratio'] is None else f"{r['fc_ratio']:.2f}"
            f.write(f"{LAB[r['model']]} & {r['test_ll']:.4f} & {d} & {be} & {fq} \\\\\n")
        f.write('\\bottomrule\n\\end{tabular}\n')
    num[f'{t}N'] = big(len(pts)); num[f'{t}NTest'] = big(te.sum()); num[f'{t}NVal'] = big(json.load(open(f'{R0}/{name}/bg1/{name}_complex_bg.json'))['n_val'])
    num[f'{t}Bins'] = str(W); num[f'{t}Vis'] = fmt(fr['V'], 2); num[f'{t}Period'] = fmt(fr['P'], 1); num[f'{t}AD'] = fmt(fr['a_over_d'], 3); num[f'{t}Phys'] = fmt(phys, 4)
    num[f'{t}Fc'] = fmt(ref['fc'], 4); num[f'{t}Ratio'] = fmt(ref['fc_ratio'], 2)
    rr = [rows[f'complex, bg N={N}']['fc_ratio'] for N in (0, 1, 2)]; bb = [rows[f'complex, bg N={N}']['beta'] for N in (0, 1, 2)]
    num[f'{t}RatioRange'] = f'{min(rr):.2f}\\mbox{{--}}{max(rr):.2f}' if f'{min(rr):.2f}' != f'{max(rr):.2f}' else f'{min(rr):.2f}'
    num[f'{t}Beta'] = fmt(ref['beta'], 2); num[f'{t}BetaRange'] = f'{min(bb):.2f}\\mbox{{--}}{max(bb):.2f}'
    num[f'{t}LLCx'] = fmt(ref['test_ll'], 4)
    for m, lab in (('KDE, flat bg', 'KDE'), ('adaptive KDE, flat bg', 'AKDE'), ('Fresnel, flat bg', 'Fresnel'), ('Fraunhofer, flat bg', 'Fr'), ('histogram', 'Hist')):
        num[f'{t}D{lab}'] = fmt(-rows[m]['diff'], 4); num[f'{t}SE{lab}'] = fmt(rows[m]['se'], 4); num[f'{t}Beta{lab}'] = fmt(rows[m]['beta'], 2)
    dN = [abs(rows[f'complex, bg N={N}']['diff']) / max(rows[f'complex, bg N={N}']['se'], 1e-12) for N in (0, 2)]
    num[f'{t}DNmax'] = fmt(max(abs(rows[f'complex, bg N={N}']['diff']) for N in (0, 2)), 4)
    line = np.load(f'{R0}/{name}/bg1/{name}_complex_bg_line.npy'); h = ft_heights(line, fr['P'], 16 * W)
    hf = ft_heights(np.abs(line) ** 2, fr['P'], 16 * W)
    num[f'{t}Slit'] = f'{min(h[-0.5], h[0.5]):.2f}\\mbox{{--}}{max(h[-0.5], h[0.5]):.2f}' if f'{min(h[-0.5], h[0.5]):.2f}' != f'{max(h[-0.5], h[0.5]):.2f}' else f'{h[0.5]:.2f}'
    num[f'{t}Centre'] = fmt(h[0.0], 2); num[f'{t}AutoSide'] = fmt(max(hf[-1.0], hf[1.0]), 2)
    vp = f'{R0}/{name}/val_curve.json'
    if os.path.exists(vp):                                  # the validation plateau (N = 1): below / above the band
        V = json.load(open(vp)); fq = np.array(V['fc']) / phys; vv = np.array(V['val']['1']) * V['n_val']; vv = vv - vv.max()
        num[f'{t}Below'] = f'{-vv[fq < 0.85].max():.0f}'; num[f'{t}Flat'] = fmt(fq[vv > -2].max(), 1)
    sp = f'{R0}/{name}/stream/{name}_stream_bg.json'
    if os.path.exists(sp):
        s = json.load(open(sp)); Ps = np.load(sp.replace('.json', '_P.npy')).astype(float)
        Pb = np.load(f'{R0}/{name}/bg1/{name}_complex_bg_P.npy').astype(float)
        d = np.log(Ps)[x[te]] - np.log(Pb)[x[te]]
        num[f'{t}StreamFc'] = fmt(s['fc'][0], 4); num[f'{t}StreamRatio'] = fmt(s['fc'][0] / phys, 2)
        num[f'{t}StreamBurn'] = fmt(s['burn_in_fc'][0], 4); num[f'{t}StreamBurnRatio'] = fmt(s['burn_in_fc'][0] / phys, 2)
        num[f'{t}StreamEpochs'] = str(s['epochs']); num[f'{t}StreamBatches'] = str(s['n_batches'])
        num[f'{t}StreamD'] = fmt(d.mean(), 4); num[f'{t}StreamSE'] = fmt(d.std(ddof=1) / np.sqrt(len(d)), 4)
        num[f'{t}StreamBeta'] = fmt(s['beta'][0], 2); num[f'{t}StreamS'] = f"{s['s']:.0f}"
num['MmFold'] = str(int(round(len(data('photon_mm')[0]) / len(data('photon')[0]))))
num['ElBetaMeas'] =fmt(float(np.load(DATA + 'dslit_electron_beta.npy')[0]), 2)
lg = json.load(open(G + 'Heralded_Diffraction_SM_clean.json'))
num.update(PhFrameN=big(lg['photons_clean']), PhPixels=big(lg['pixels_raw']), PhClusters=big(lg['clusters']),
           PhStreaks=str(lg['streak_clusters']), PhHot=str(lg['hot_pixels']), PhReadouts=big(lg['readouts']))
open(f'{PP}/numbers_dslit.tex', 'w').write(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in num.items()))

# ---------------------------------------------------------------- figure
plt.rcParams.update({'font.size': 6, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False, 'legend.fontsize': 6})
fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.6))
for r, (name, title) in enumerate((('photon', 'Photons (single mode)'), ('electron', 'Electrons'))):
    pts, x, te, W = data(name); phys = T[name]['phys']
    fr = json.load(open(f'{R0}/{name}/flat/{name}_fraunhofer.json'))['params']; nT = te.sum()
    Ct = np.bincount(x[te], minlength=W)
    Pb = np.load(f'{R0}/{name}/bg1/{name}_complex_bg_P.npy').astype(float)
    Pf = np.load(f'{R0}/{name}/flat/{name}_fraunhofer_P.npy').astype(float)
    a = ax[r, 0]; a.bar(np.arange(W), Ct, width=1.0, color='0.8', label='test counts')
    a.plot(nT * Pb, color='C3', lw=1.2, label='BLML complex, $N=1$'); a.plot(nT * Pf, color='C0', lw=0.9, ls='--', label='Fraunhofer')
    a.set_xlabel('pixel along the fringes'); a.set_ylabel('counts'); a.set_title(f'(a) {title.split(" ")[0].lower()}: test counts', fontsize=7); a.legend(frameon=False, fontsize=5.5, loc='upper right')
    V = json.load(open(f'{R0}/{name}/val_curve.json')); fc = np.array(V['fc']); a = ax[r, 1]
    for N, c in zip(('0', '1', '2'), ('C1', 'C3', 'C2')):
        v = np.array(V['val'][N]) * V['n_val']; a.plot(fc / phys, v - v.max(), marker='.', ms=3, color=c, lw=0.9, label=f'$N={N}$')
    a.axvline(1.0, color='k', lw=0.6, ls=':'); a.set_xscale('log'); a.set_ylim(-60, 5)
    a.set_xlabel('$f_c$ / physical band'); a.set_ylabel('validation LL $-$ max (nats)'); a.set_title('(b) band selection', fontsize=7); a.legend(frameon=False, fontsize=5.5)
    line = np.load(f'{R0}/{name}/bg1/{name}_complex_bg_line.npy'); pad = 16 * W; xi = np.fft.fftshift(np.fft.fftfreq(pad)) * fr['P']
    Ag = np.abs(np.fft.fftshift(np.fft.fft(line, pad))); Af = np.abs(np.fft.fftshift(np.fft.fft(np.abs(line) ** 2, pad)))
    a = ax[r, 2]; ad = fr['a_over_d']
    for c in (-0.5, 0.5):
        a.axvspan(c - ad / 2, c + ad / 2, color='C2', alpha=0.15, lw=0)
    a.plot(xi, Ag / Ag.max(), color='C3', lw=1.1, label='$|\\hat g|$ (BLML complex)'); a.plot(xi, Af / Af.max(), color='0.4', lw=0.9, ls='--', label='$|\\hat f|$')
    a.set_xlim(-1.6, 1.6); a.set_xlabel('$\\xi / d$'); a.set_title('(c) aperture (shaded: slits)', fontsize=7); a.legend(frameon=False, fontsize=5.5, loc='upper right')
    a = ax[r, 3]; s = json.load(open(f'{R0}/{name}/stream/{name}_stream_bg.json'))
    hs = [h for h in s['history'] if 'n' in h]; seen = np.cumsum([1.0] * len(hs))
    fcs = np.array([h['fc'][0] for h in hs]) / phys
    a.plot(np.arange(1, len(hs) + 1) / s['n_batches'], fcs, color='C3', lw=1.0, label='streaming')
    a.axhline(T[name]['rows'][[q['model'] for q in T[name]['rows']].index('complex, bg N=1')]['fc_ratio'], color='C3', ls='--', lw=0.8, label='batch')
    a.axhline(1.0, color='k', lw=0.6, ls=':'); a.set_xlabel('epochs'); a.set_ylabel('$f_c$ / physical band')
    a.set_title('(d) streaming', fontsize=7); a.legend(frameon=False, fontsize=5.5)
fig.tight_layout(); fig.savefig(f'{PP}/figs/dslit.pdf'); fig.savefig(f'{PP}/figs/dslit.png', dpi=200)
print(num)
