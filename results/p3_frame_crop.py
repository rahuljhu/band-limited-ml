"""Paper 3, beads from the full-frame fits (binned_gpu_job.py 'frame': the whole 2048 x 2048 plane-5 frame, train 70% /
val 10% / test 20% by binomial thinning, rng 0; complex, real and empirical BLML at the physical band 2NA/lambda, ball
kernel, no crop and no floor): each fitted density cropped to every isolated bead's 31 x 31 window and renormalised
there, scored on the window's test photons, paired against (a) the frame-level Gaussian blur (sigma on the frame's
validation, as the job's baseline, cropped the same way) and (b) a window KDE with the EM uniform floor (bandwidth on
the window's validation counts, fitted on its training counts), as paper 3 / 7.
Also the CPU frame fits (results/binned_local: frame_cpu, and frame_bg with the background mode) when present.
-> results/p3_frame_crop.json"""
import sys, json, numpy as np
sys.path.insert(0, 'results/psf'); sys.path.insert(0, '.')
import psfkit as pk
from scipy.ndimage import gaussian_filter
H = 15
N = np.round(np.maximum(np.asarray(np.load('results/psf/stack_photons.npy', mmap_mode='r')[5]) / pk.GAIN_CORR, 0)).astype(np.int64)
r = np.random.default_rng(0); TR = r.binomial(N, 0.7); rest = N - TR; VA = r.binomial(rest, 1 / 3); TE = rest - VA
F = {l: np.load(f'results/binned/frame_{l}.npz')['density'].astype(np.float64) for l in ('complex', 'real', 'empirical')}
sg = json.load(open('results/binned/frame.json'))['baselines']['gauss_blur']['sigma']
import os
for l, f in (('complex_cpu', 'results/binned_local/frame_cpu.npz'), ('complex_bg', 'results/frame_bg/oneshot_bg.npz'), ('complex_stream_bg', 'results/frame_bg/stream_bg.npz')):
    if os.path.exists(f):                                   # cpu: CPU float64 4000 sweeps; bg: T4 float32 (frame_bg_gpu_job.py)
        F[l] = np.load(f)['density'].astype(np.float64)
F['frame_blur'] = gaussian_filter(TR.astype(float), sg, mode='constant') + 1e-3


def em(Nw, q, iters=80, tol=1e-7):
    Nw = np.asarray(Nw, float).ravel(); B = Nw.size; pi, r_, prev = 0.1, np.full(B, 0.1), -np.inf
    for _ in range(iters):
        qd = q(Nw * (1 - r_)); P = pi / B + (1 - pi) * qd; r_ = (pi / B) / P; pi = float((Nw * r_).sum() / Nw.sum())
        P = pi / B + (1 - pi) * qd; L = pk.score(P, Nw)
        if L - prev < tol:
            break
        prev = L
    return P / P.sum()


rows = []
for k in range(len(np.load('results/psf/beads_iso.npy'))):
    y0, x0 = np.load('results/psf/beads_iso.npy')[k]; W = (slice(y0 - H, y0 + H + 1), slice(x0 - H, x0 + H + 1))
    tr, va, te = TR[W], VA[W], TE[W]; w = te.ravel().astype(float); sh = tr.shape
    h = max(pk.KDE_H, key=lambda h_: pk.score(em(tr, lambda x: pk.kde(x, h_, sh)), va.ravel()))
    P = {l: (lambda p: p / p.sum())(np.maximum(F[l][W].ravel(), 1e-300)) for l in F}
    P['kde_window'] = em(tr, lambda x: pk.kde(x, h, sh))
    row = dict(bead=k, kde_h=h, **{l: pk.score(P[l], w) for l in P})
    for l in [l for l in F if l != 'frame_blur']:
        for ref in ('kde_window', 'frame_blur') + (('complex', 'complex_cpu') if l in ('complex_bg', 'complex_stream_bg') else ()):
            d = np.log(P[l]) - np.log(P[ref]); mu = float((w * d).sum() / w.sum())
            row[f'{l}-{ref}'] = (mu, float(np.sqrt((w * (d - mu) ** 2).sum() / w.sum()) / np.sqrt(w.sum())))
    rows.append(row)
json.dump(rows, open('results/p3_frame_crop.json', 'w'), indent=1)
for l in [l for l in F if l != 'frame_blur']:
    for ref in ('kde_window', 'frame_blur') + (('complex', 'complex_cpu') if l in ('complex_bg', 'complex_stream_bg') else ()):
        d = np.array([x[f'{l}-{ref}'][0] for x in rows]); z = d / np.array([x[f'{l}-{ref}'][1] for x in rows])
        print(f'{l:9s} vs {ref:10s}: median {np.median(d):+.5f}  wins {(z > 2).sum()} ties {(abs(z) <= 2).sum()} losses {(z < -2).sum()}')
