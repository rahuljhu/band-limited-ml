"""Simulated bead images with a known pixel distribution; exact KL(truth || fit) vs photon count.
Truth: |FT(pupil * exp(i W))|^2 on a 4x oversampled 128-px field, pupil radius NA/lambda = FC_PHYS/2
cycles/px, sub-pixel offset, box-integrated over pixels (so the pixel image is exactly band-limited),
plus a flat background (7% of window photons), cut to the 31x31 window.
Cases: airy (in focus), defocus (Zernike defocus, 1 rad), defocus+fp (same, x a fixed 3% per-pixel gain
pattern = not band-limited, the camera fixed pattern of real sCMOS data).
Per (case, n, seed): counts ~ Multinomial(n, p); train/val 80/20 by thinning for selection; refit on all.
Usage: python simulate.py <case> <n> <seed>  -> results/psf/v3/sim/<case>_<n>_<seed>.json"""
import sys, json, time, numpy as np
sys.path.insert(0, 'results/psf'); import psfkit as pk
case, n, seed = sys.argv[1], int(float(sys.argv[2])), int(sys.argv[3])
H = 15; Z = pk.lattice(H); RHO = pk.FC_PHYS / 2

def truth(case):
    F, OS = 128, 4; M = F * OS; f = np.fft.fftfreq(M, d=1.0 / OS)          # cycles/px
    FX, FY = np.meshgrid(f, f); R = np.hypot(FX, FY) / RHO; pupil = (R <= 1).astype(complex)
    W = 1.0 * (2 * R ** 2 - 1) if case.startswith('defocus') else 0 * R
    off = (0.3, -0.2)                                                       # sub-pixel bead position (px)
    h = np.fft.ifft2(pupil * np.exp(1j * W) * np.exp(-2j * np.pi * (FX * off[0] + FY * off[1])))
    I = np.fft.fftshift(np.abs(h) ** 2)
    # pixel integration: sum OSxOS blocks; with fftshift the field centre (M/2) is a pixel centre after offsetting by OS/2
    I = np.roll(I, (OS // 2, OS // 2), (0, 1)).reshape(F, OS, F, OS).sum((1, 3))
    c = F // 2; win = I[c - H:c + H + 1, c - H:c + H + 1]; win = win / win.sum()
    p = 0.93 * win + 0.07 / win.size
    if case.endswith('+fp'):
        p = p * (1 + 0.03 * np.random.default_rng(123).standard_normal(p.shape)); p = np.maximum(p, 1e-12)
    return p / p.sum()

p = truth(case); rng = np.random.default_rng(seed)
N = rng.multinomial(n, p.ravel()).reshape(p.shape); tr, va = pk.split(N, seed, f=(0.8, 0.2))[:2]
t0 = time.time(); out = pk.run_all(Z, tr, va, None, shape=p.shape, truth=p)
res = dict(case=case, n=n, seed=seed, s=time.time() - t0, **{m: {kk: v for kk, v in d.items() if kk != '_P'} for m, d in out.items()})
import os; os.makedirs('results/psf/v3/sim', exist_ok=True)
json.dump(res, open(f'results/psf/v3/sim/{case}_{n}_{seed}.json', 'w'), indent=1)
print(case, n, seed, {m: round(d['kl'], 5) for m, d in out.items()}, f'{res["s"]:.0f}s', flush=True)
