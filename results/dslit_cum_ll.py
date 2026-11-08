"""Paper 3, double slit, 1-d protocol: cumulative test log-likelihood by frame (photons: readout; electrons: the
Master_list frame) under the batch complex BLML fit with the background mode N = 1 (results/dslit_1d/<set>/bg1).
Panels per set: the cumulative sum of log P at each test particle; the same minus the average rate (a drift or jump
there is a stretch of frames the fit predicts worse or better than average); test particles per frame bin. Jumps are
flagged as frame bins whose mean log P is beyond 4 standard errors of the overall mean.
Usage: python results/dslit_cum_ll.py [out.png]"""
import os, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = sys.argv[1] if len(sys.argv) > 1 else 'results/dslit_1d/cum_ll.png'; DATA = os.path.expanduser('~/data_repo/dslit_gpu/')
fig, ax = plt.subplots(3, 2, figsize=(11, 7), sharex='col')
for c, name in enumerate(('photon', 'electron')):
    pts = np.load(DATA + f'dslit_{name}_points.npy'); SH = np.load(DATA + f'dslit_{name}_b.npy').shape[-2:]
    axis = int(np.load(DATA + f'dslit_{name}_axis.npy'))
    x = np.clip(np.floor(pts[:, 1 + axis] + 0.5).astype(int), 0, SH[axis] - 1)
    te = np.random.default_rng(0).random(len(pts)) >= 0.80
    P = np.load(f'results/dslit_1d/{name}/bg1/{name}_complex_bg_P.npy').astype(float)
    fr = pts[te, 0]; lp = np.log(np.maximum(P, 1e-300))[x[te]]
    cum = np.cumsum(lp); dev = cum - lp.mean() * np.arange(1, len(lp) + 1)
    ax[0, c].plot(fr, cum, lw=1); ax[0, c].set_ylabel('cumulative test LL (nats)'); ax[0, c].set_title(f'{name}: {te.sum()} test particles')
    ax[1, c].plot(fr, dev, lw=1); ax[1, c].axhline(0, color='k', lw=0.5)
    sd = lp.std(); n = np.arange(1, len(lp) + 1)
    band = 2 * sd * np.sqrt(n * (1 - n / len(lp)))                  # +-2 sd of a random-order bridge
    ax[1, c].fill_between(fr, -band, band, color='0.85', lw=0); ax[1, c].set_ylabel('minus average rate (nats)')
    edges = np.linspace(fr.min(), fr.max() + 1, 41); k = np.digitize(fr, edges) - 1
    cnt = np.bincount(k, minlength=40)[:40]; mu = np.array([lp[k == i].mean() if cnt[i] else np.nan for i in range(40)])
    z = (mu - lp.mean()) / (sd / np.sqrt(np.maximum(cnt, 1)))
    ax[2, c].bar(edges[:-1], cnt, width=np.diff(edges), align='edge', color='0.6'); ax[2, c].set_ylabel('test particles per bin')
    ax[2, c].set_xlabel('frame' if name == 'electron' else 'readout')
    flag = np.nonzero(np.abs(z) > 4)[0]
    for i in flag:
        ax[1, c].axvspan(edges[i], edges[i + 1], color='r', alpha=0.2)
    print(f'{name}: mean log P {lp.mean():.4f}, sd {sd:.3f}; max |bridge| {np.abs(dev).max():.1f} nats vs 2 sd envelope max {band.max():.1f}; '
          f'frame bins beyond 4 SE: {[(int(edges[i]), int(edges[i + 1]), round(float(z[i]), 1)) for i in flag]}; '
          f'max |z| {np.nanmax(np.abs(z)):.1f}')
fig.tight_layout(); fig.savefig(OUT, dpi=110); print('wrote', OUT)
