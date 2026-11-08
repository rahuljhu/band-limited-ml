"""Figures for the full-lattice BLML report (BLMLBinned, BLMLEmpirical): full frame, bead zoom, held-out scores,
bead windows, convergence, streaming, speed. Reads results/binned/, results/binned_local/, results/binned_parity_cpu.json.
Usage: python results/binned_figs.py OUTDIR"""
import sys, os, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
R = 'results/binned/'; L = 'results/binned_local/'
INK, MUTED, GRID = '#22313a', '#6b7a83', '#d9e0e4'
C = dict(complex='#0f6e74', real='#5b7f9b', empirical='#c9822b', blur='#8d969b', histogram='#c5ccd0', stream='#2a9d8f')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
                     'axes.titlecolor': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150,
                     'savefig.bbox': 'tight', 'savefig.facecolor': 'white', 'grid.color': GRID, 'grid.linewidth': 0.6})
FRAME = json.load(open(R + 'frame.json')) if os.path.exists(R + 'frame.json') else None
N = np.load('/private/tmp/claude-501/-Users-ra/e6cd45ce-b6f4-42cc-be9a-45c946facadf/scratchpad/gpu_data/psf_plane5.npy').astype(np.int64)
rng = np.random.default_rng(0); TR = rng.binomial(N, 0.7)                      # the job's split (seed 0)
Y0, X0 = 661, 1068

def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig)

# 1. full frame: counts and BLML density, and the bead-0 zoom
if FRAME is not None and os.path.exists(R + 'frame_complex.npz'):
    f = np.load(R + 'frame_complex.npz')['density'].astype(float); f = f / f.sum() * TR.sum()
    emp = np.load(R + 'frame_empirical.npz')['density'].astype(float) if os.path.exists(R + 'frame_empirical.npz') else None
    if emp is not None:
        emp = emp / emp.sum() * TR.sum()
    vmin, vmax = 5, max(TR.max(), f.max())
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
    for a, img, t in [(ax[0], np.maximum(TR, 0.5), 'Photon counts (70 % training split)'), (ax[1], f, 'Complex BLML density, whole frame')]:
        im = a.imshow(img, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='cividis', interpolation='nearest')
        a.set_title(t, loc='left'); a.set_xticks([]); a.set_yticks([])
        a.add_patch(plt.Rectangle((X0 - 48, Y0 - 48), 96, 96, fill=False, ec='#f2a541', lw=1.2))
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01); cb.set_label('photons per pixel (log)', color=INK)
    save(fig, 'frame.png')
    # zoom
    h = 48; sl = (slice(Y0 - h, Y0 + h), slice(X0 - h, X0 + h))
    panels = [(np.maximum(TR[sl], 0.5), 'Counts'), (f[sl], 'BLML, whole frame')]
    if emp is not None:
        panels.append((emp[sl], 'Paper 7, whole frame'))
    if os.path.exists(L + 'win31.npy'):
        w31 = np.load(L + 'win31.npy').astype(float); W = np.full((2 * h, 2 * h), np.nan)
        W[h - 15:h + 16, h - 15:h + 16] = w31 / w31.sum() * TR[Y0 - 15:Y0 + 16, X0 - 15:X0 + 16].sum(); panels.append((W, 'BLML, 31 px crop'))
    fig, ax = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4))
    for a, (img, t) in zip(ax, panels):
        a.set_facecolor('#e6eaec')
        a.imshow(img, norm=LogNorm(vmin=10, vmax=TR[sl].max()), cmap='cividis', interpolation='nearest'); a.set_title(t, loc='left')
        a.set_xticks([]); a.set_yticks([])
    save(fig, 'zoom.png')
    # radial profile around bead 0
    yy, xx = np.mgrid[-h:h, -h:h]; rr = np.hypot(yy, xx); edges = np.arange(0, 46, 1.5)
    prof = lambda img: np.array([np.nanmedian(img[(rr >= a) & (rr < b)]) for a, b in zip(edges[:-1], edges[1:])])
    fig, a = plt.subplots(figsize=(6.4, 3.8)); mid = 0.5 * (edges[1:] + edges[:-1])
    a.semilogy(mid, prof(TR[sl].astype(float)), 'o', ms=3.5, color=INK, label='counts (median per ring)')
    a.semilogy(mid, prof(f[sl]), '-', color=C['complex'], lw=2, label='BLML, whole frame')
    if emp is not None:
        a.semilogy(mid, prof(emp[sl]), '--', color=C['empirical'], lw=1.6, label='paper 7, whole frame')
    a.axvline(15.5, color=MUTED, lw=0.8, ls=':'); a.text(15.9, a.get_ylim()[1] * 0.5, '31 px crop edge', color=MUTED, fontsize=8)
    a.axvline(25.5, color=MUTED, lw=0.8, ls=':'); a.text(25.9, a.get_ylim()[1] * 0.2, '51 px', color=MUTED, fontsize=8)
    a.set_xlabel('distance from bead 0 (pixels)'); a.set_ylabel('photons per pixel'); a.grid(True, which='major'); a.legend(frameon=False, fontsize=8)
    a.set_title('Bead 0: the PSF wings reach background only at r ≈ 25 px', loc='left')
    save(fig, 'profile.png')

# 2. held-out score on the full frame
if FRAME is not None:
    B = FRAME['baselines']; rows = [('histogram', B['histogram']['test']), ('Gaussian blur', B['gauss_blur']['test'])]
    for k, lab in (('real', 'BLML real'), ('complex', 'BLML complex'), ('empirical', 'paper 7 (ridge)')):
        if k in FRAME:
            rows.append((lab, FRAME[k]['test']))
    if os.path.exists(R + 'stream.json'):
        rows.append(('BLML streaming, 10 batches', json.load(open(R + 'stream.json'))[-1]['test']))
    base = B['histogram']['test']
    fig, a = plt.subplots(figsize=(7, 3.2)); labs = [r[0] for r in rows]; vals = [1e3 * (r[1] - base) for r in rows]
    cols = [C['histogram'], C['blur'], C['real'], C['complex'], C['empirical'], C['stream']][:len(rows)]
    cmap = {'histogram': C['histogram'], 'Gaussian blur': C['blur'], 'BLML real': C['real'], 'BLML complex': C['complex'],
            'paper 7 (ridge)': C['empirical'], 'BLML streaming, 10 batches': C['stream']}
    a.barh(labs, vals, color=[cmap[l] for l in labs], height=0.6)
    for y, v in enumerate(vals):
        a.text(v + 0.2, y, f'+{v:.2f}' if v > 0 else f'{v:.2f}', va='center', fontsize=8, color=INK)
    a.invert_yaxis(); a.set_xlabel('held-out log-likelihood per photon minus histogram (×10⁻³)'); a.grid(True, axis='x')
    a.set_title(f"Whole frame, {FRAME['photons'] / 1e6:.0f}M photons: held-out 20 %", loc='left')
    save(fig, 'heldout.png')

# 3. bead windows: per-bead difference to complex BLML
if os.path.exists(L + 'bead_windows.json'):
    W = json.load(open(L + 'bead_windows.json'))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    meths = [('emp_ridge', 'paper 7 (ridge)', C['empirical']), ('gauss_blur', 'Gaussian blur', C['blur']),
             ('histogram', 'histogram', C['histogram']), ('blml_real', 'BLML real', C['real'])]
    for a, win in zip(ax, (31, 51)):
        X = [r for r in W if r['window'] == win]
        for j, (k, lab, col) in enumerate(meths):
            d = 1e3 * np.array([r[k] - r['blml_complex'] for r in X])
            a.scatter(d, np.full(len(d), j) + np.random.default_rng(j).uniform(-0.18, 0.18, len(d)), s=10, color=col, alpha=0.85)
            a.plot([d.mean()] * 2, [j - 0.3, j + 0.3], color=INK, lw=1.6)
        a.axvline(0, color=C['complex'], lw=1.2); a.set_yticks(range(len(meths))); a.set_yticklabels([m[1] for m in meths])
        a.set_title(f'{win} × {win} px windows ({len(X)} beads)', loc='left'); a.grid(True, axis='x')
        a.set_xlabel('held-out log-lik. per photon minus complex BLML (×10⁻³)')
    ax[0].invert_yaxis()
    save(fig, 'beads.png')

# 4. convergence on the full frame: T4 run (old Anderson, never accepted) vs CPU run (host normal-equation Anderson)
if FRAME is not None and 'complex' in FRAME and 'merit_per_photon' in FRAME['complex']:
    runs = [('T4 float32, Jacobi only (Anderson step failing)', FRAME['complex'], C['real'])]
    if os.path.exists(L + 'frame_cpu.json'):
        runs.append(('CPU float64, Jacobi + Anderson', json.load(open(L + 'frame_cpu.json')), C['complex']))
    best = max(max(np.array(r['merit_per_photon']) - r['merit_per_photon'][0]) for _, r, _ in runs)   # gains: precision offsets cancel
    fig, a = plt.subplots(1, 2, figsize=(10, 3.4))
    for lab, r, col in runs:
        rho = np.array(r['rho_history']); mp = np.array(r['merit_per_photon']); it = np.arange(len(mp)) * 10
        a[0].semilogy(np.arange(len(rho)) * 10, rho, color=col, lw=1, label=lab)
        gap = best - (mp - mp[0]); m = gap > 0
        a[1].loglog(it[m] + 10, gap[m], color=col, lw=1.6, label=lab)
    a[0].set_xlabel('sweep'); a[0].set_ylabel('max over bins of |ρ_b|'); a[0].grid(True); a[0].set_title('Stationarity residual (worst bin)', loc='left')
    a[1].set_xlabel('sweep'); a[1].set_ylabel('log-lik. gain per photon short of best'); a[1].grid(True, which='major')
    a[1].set_title('Likelihood convergence, 2048² frame', loc='left'); a[1].legend(frameon=False, fontsize=8)
    save(fig, 'convergence.png')

# 5. streaming
if os.path.exists(R + 'stream.json') and FRAME is not None:
    S = json.load(open(R + 'stream.json')); ph = np.array([s['photons'] for s in S]) / 1e6; te = np.array([s['test'] for s in S])
    fig, a = plt.subplots(figsize=(6.4, 3.6))
    a.plot(ph, te, 'o-', color=C['stream'], label='streaming BLML (300 sweeps per batch)')
    a.axhline(FRAME['baselines']['gauss_blur']['test'], color=C['blur'], ls='--', lw=1, label='Gaussian blur, all photons')
    a.axhline(FRAME['baselines']['histogram']['test'], color=C['histogram'], ls='--', lw=1, label='histogram, all photons')
    if 'complex' in FRAME:
        a.axhline(FRAME['complex']['test'], color=C['complex'], lw=1.2, label='one-shot BLML, all photons')
    a.set_xlabel('training photons streamed (millions)'); a.set_ylabel('held-out log-lik. per photon'); a.grid(True)
    a.legend(frameon=False, fontsize=8); a.set_title(f"Streaming the frame: {np.mean([s['time_s'] for s in S]):.0f} s per batch on a T4", loc='left')
    save(fig, 'stream.png')

# 6. speed
P = json.load(open(R + 'parity.json')) if os.path.exists(R + 'parity.json') else []
big = [r for r in P if '256' in r['cell']]
fig, a = plt.subplots(figsize=(7, 3.0))
labs = ['BLMLQuick, 51² bead window (5 starts)', 'BLMLBinned CPU, 51² window']; vals = [170.0, 0.5]; cols = [C['histogram'], C['complex']]
if big:
    labs += ['BLMLBinned Colab CPU, 256²', 'BLMLBinned T4 float64, 256²', 'BLMLBinned T4 float32, 256²']
    vals += [big[0]['cpu64_t'], big[0]['gpu64_t'], big[0]['gpu32_t']]; cols += [C['real'], C['complex'], C['stream']]
a.barh(labs, vals, color=cols, height=0.6); a.set_xscale('log'); a.invert_yaxis(); a.grid(True, axis='x')
for y, v in enumerate(vals):
    a.text(v * 1.1, y, f'{v:.1f} s', va='center', fontsize=8, color=INK)
a.set_xlabel('time to converge (s, log scale)'); a.set_title('Complex rank-1 fit time', loc='left')
save(fig, 'speed.png')
print('figures in', OUT, sorted(os.listdir(OUT)))
