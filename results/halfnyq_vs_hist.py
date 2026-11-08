"""Does BLML with g band-limited at half the lattice Nyquist (f at the full Nyquist band: fc = 0.5 cycles / px in our
convention, f's band = fc) beat the histogram on held-out log-likelihood? No selection of fc; lam = 1e-8 (numerical).
The histogram gets a pseudo-count a chosen on validation counts (a = 0 would give empty pixels probability 0).
  SID (real data only):   the three scenes at 0.1 s and 0.04 s, per Bayer channel on its own sub-lattice (Nyquist 0.25 cycles / px:
         fc = 0.25), real data only (electrons clipped at 0); frame _00 fits, frame _01 chooses a, the 10 s frame's
         photons are the test
Output results/halfnyq_vs_hist.json"""
import os, sys, json, time, numpy as np
sys.path.insert(0, '.')
import matplotlib; from PIL import Image
import blml_trivial as bt
from blml_binned import BLMLBinned, BLMLEmpirical, _np
from scipy.ndimage import laplace, gaussian_filter, map_coordinates

A_GRID = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
def ll(p, m):
    p = np.maximum(p, 1e-300); p = p / p.sum(); return float((m * np.log(p)).sum() / m.sum())

def pixel_integrated(m, pixel=1.0, q=4):
    """p_b = the integral of the fitted f over each site's photosite (pixel x pixel, centred on the site), by Simpson's
    rule on f at spacing pixel / q (q even; positive weights, so p_b >= 0 as f = |g|^2 >= 0; exact for the smooth,
    band-limited f up to O(spacing^4)). f on that grid by predict_grid (the lattice refined h q / pixel times). Sites on
    the lattice's outer rows / columns: the part of the photosite inside the evaluated span, rescaled to the full area."""
    h = float(m.h_[0]); r = int(round(h * q / pixel)); G = m.predict_grid(r); d = G.ndim
    w1 = np.ones(q + 1); w1[1:-1:2] = 4; w1[2:-1:2] = 2; w1 = w1 / w1.sum()            # Simpson on [-pixel/2, pixel/2]
    k = q // 2; step = r; Gp = np.pad(G, [(k, k)] * d, mode='constant', constant_values=np.nan)
    out = 0.0; wsum = 0.0
    import itertools
    for offs in itertools.product(range(-k, k + 1), repeat=d):
        sl = tuple(slice(k + o, k + o + G.shape[j], step) for j, o in enumerate(offs))
        v = Gp[sl]; w = np.prod([w1[o + k] for o in offs]); ok = ~np.isnan(v)
        out = out + np.where(ok, v, 0.0) * w; wsum = wsum + ok * w
    return out / wsum * pixel ** d

def fits(C, fc, h=1.0, origin=0.0):
    modes = bt.make_modes([fc, fc], ['sinc', 'sinc']); out = {}
    for lab, mk in (('BLML binned real', lambda: BLMLBinned(tol=1e-5, max_sweeps=500, ll_tol=1e-9).fit(C, modes, h=h, origin=origin)),
                    ('BLML binned complex', lambda: BLMLBinned(tol=1e-5, max_sweeps=500, ll_tol=1e-9).fit(C, modes, h=h, origin=origin, complex_g=True))):
        t = time.time(); m = mk(); out[lab + ' (bin centre)'] = (_np(m.density_), time.time() - t)
        out[lab + ' (pixel integral)'] = (np.maximum(pixel_integrated(m), 0.0), time.time() - t)
    return out

res = {}
import rawpy
D = __import__('os').path.expanduser('~/data_repo/sid'); PT = json.load(open('results/sid_photon_transfer.json'))
CH = {'R': (0, 0), 'G1': (0, 1), 'G2': (1, 0), 'B': (1, 1)}; W = 512
src = open('results/sid_benchmark.py').read()
exec(src[src.index('def raw(f):'):src.index('def full_grid(m, c):')])   # raw, crop_origin, sub
for scene in ('10003', '10011', '10022'):
    Lfull = raw(f'{scene}_00_10s'); y0, x0 = crop_origin(Lfull); Lc = Lfull[y0:y0 + W, x0:x0 + W]
    for expo in ('0.1s', '0.04s'):
        g = {ch['ch']: ch['e_per_adu'] for ch in PT[f'{scene}_{expo}']['channels']}
        A = raw(f'{scene}_00_{expo}')[y0:y0 + W, x0:x0 + W]; B = raw(f'{scene}_01_{expo}')[y0:y0 + W, x0:x0 + W]
        tot = {}
        for c, (cy, cx) in CH.items():
            k = f'{cy}{cx}'; C = np.clip(sub(A, c) * g[k], 0, None); V = np.clip(sub(B, c) * g[k], 0, None)
            T = np.clip(sub(Lc, c) * g[k], 0, None); T = np.where(sub(Lc, c) < 15000, T, 0.0)
            a = max(A_GRID, key=lambda a_: ll(C + a_, V))
            parts = {'histogram (+a)': C + a}
            for fc_, tag in ((0.25, 'half Nyquist'), (0.5, 'full Nyquist')):
                for lab, (p, _) in fits(C, fc_, h=2.0, origin=(cy, cx)).items():
                    parts[f'{lab}, {tag}'] = p
            for lab, p in parts.items():
                pp = p / p.sum(); s_ = tot.setdefault(lab, [0.0, 0.0])
                with np.errstate(divide='ignore'):
                    s_[0] += float((T * np.log(pp)).sum()) if np.all(pp[T > 0] > 0) else -np.inf   # zero mass on a test photon: -inf
                s_[1] += float(T.sum())
        row = {k_: dict(test_10s=v[0] / v[1]) for k_, v in tot.items()}
        res[f'sid_{scene}_{expo}'] = row
        print(f'SID {scene} {expo}: ' + '  '.join(f"{k_} {v['test_10s']:.5f}" for k_, v in row.items()), flush=True)
json.dump(res, open('results/halfnyq_vs_hist.json', 'w'), indent=1)
