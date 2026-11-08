"""Figure: SID 10011, 0.04 s, the largest half-Nyquist gain over the histogram (+0.072 / photon on the 10 s photons):
the short frame (histogram, bilinear demosaic for display), BLML binned real at half-Nyquist (fc = 0.25 per channel
sub-lattice, no selection), and the 10 s exposure. Same white balance and display scale (from the 10 s frame) for all.
Output papers/paper7/figs/sid_10011_0.04s_halfnyq.png"""
import sys, json, numpy as np, rawpy
sys.path.insert(0, '.')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import laplace, gaussian_filter, map_coordinates
import blml_trivial as bt
from blml_binned import BLMLBinned, _np
D = __import__('os').path.expanduser('~/data_repo/sid'); PT = json.load(open('results/sid_photon_transfer.json'))
CH = {'R': (0, 0), 'G1': (0, 1), 'G2': (1, 0), 'B': (1, 1)}; W = 512
src = open('results/sid_benchmark.py').read()
exec(src[src.index('def raw(f):'):src.index('def loglik(pred, cnt, floor):')])     # raw, crop_origin, sub, full_grid, interp_grid
scene, expo = '10011', '0.04s'
Lfull = raw(f'{scene}_00_10s'); y0, x0 = crop_origin(Lfull); Lc = Lfull[y0:y0 + W, x0:x0 + W]
g = {ch['ch']: ch['e_per_adu'] for ch in PT[f'{scene}_{expo}']['channels']}
A = raw(f'{scene}_00_{expo}')[y0:y0 + W, x0:x0 + W]
hist, blml, gt = {}, {}, {}
for c, (cy, cx) in CH.items():
    k = f'{cy}{cx}'; C = np.clip(sub(A, c) * g[k], 0, None)
    m = BLMLBinned(tol=1e-5, max_sweeps=500, ll_tol=1e-9).fit(C, bt.make_modes([0.25, 0.25], ['sinc', 'sinc']), h=2.0, origin=(cy, cx))
    blml[c] = full_grid(m, c) * 4.0 * C.sum(); hist[c] = interp_grid(C, c, 1)
    gt[c] = interp_grid(np.clip(sub(Lc, c) * g[k], 0, None), c, 1)
def rgb(P, ref):
    img = np.stack([P['R'], (P['G1'] + P['G2']) / 2, P['B']], -1)
    wb = np.array([ref['R'].mean(), (ref['G1'].mean() + ref['G2'].mean()) / 2, ref['B'].mean()])
    img = img / img.reshape(-1, 3).mean(0) * wb / wb                     # each method at its own exposure scale
    refimg = np.stack([ref['R'], (ref['G1'] + ref['G2']) / 2, ref['B']], -1); refimg = refimg / refimg.reshape(-1, 3).mean(0)
    return np.clip(img / np.percentile(refimg, 99.5), 0, 1) ** (1 / 2.2)
fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
for a_, (t, P) in zip(ax, (('short 0.04 s frame (histogram, bilinear demosaic)', hist),
                           ('BLML binned real, half Nyquist (no tuning)', blml), ('10 s exposure (ground truth)', gt))):
    a_.imshow(rgb(P, gt)); a_.set_title(t, fontsize=11); a_.set_xticks([]); a_.set_yticks([])
fig.suptitle('SID 10011 at 0.04 s (~3 e-/px, read noise ~5 e-): held-out log-likelihood on the 10 s photons, '
             'histogram -10.6970, BLML -10.6253 (+0.072 per photon)', fontsize=10)
plt.tight_layout(); plt.savefig('papers/paper7/figs/sid_10011_0.04s_halfnyq.png', dpi=110)
