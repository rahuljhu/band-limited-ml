"""2-D: does rank K help when f is an incoherent sum? f = |h|^2 integrated over 1-px pixels (8x8 sub-samples).
Fits: histogram; real K=1..3; complex K=1..3 (ball kernel at fc = 2 rho)."""
import sys, time, json, numpy as np
sys.path.insert(0, 'results/psf'); import blml_trivial as bt
exec(open('results/psf/complex2d.py').read().split("if __name__")[0])
def pixel_psf(case):
    k = np.fft.fftfreq(NF, d=1.0 / OS); KX, KY = np.meshgrid(k, k, indexing='ij'); r = np.hypot(KX, KY) / RHO
    ph = {'airy': 0 * r, 'defocus': 2.0 * (2 * r ** 2 - 1)}[case]
    I = np.abs(np.fft.fftshift(np.fft.ifft2(np.where(r <= 1, np.exp(1j * ph), 0)))) ** 2
    c = NF // 2; I = np.roll(np.roll(I, OS // 2, 0), OS // 2, 1)               # pixel centre at c
    M = NF // OS; return I[:M * OS, :M * OS].reshape(M, OS, M, OS).mean((1, 3))   # pixel integrals
case, H, n = sys.argv[1], int(sys.argv[2]), int(float(sys.argv[3]))
P = pixel_psf(case); c0 = P.shape[0] // 2; pb = P[c0 - H:c0 + H + 1, c0 - H:c0 + H + 1]; p = (pb / pb.sum()).ravel()
ii, jj = np.meshgrid(np.arange(-H, H + 1.0), np.arange(-H, H + 1.0), indexing='ij'); Z = np.c_[ii.ravel(), jj.ravel()]
N = np.random.default_rng(0).multinomial(n, p); kl = lambda q: float((p * np.log(p / (np.maximum(q, 1e-300) / np.maximum(q, 1e-300).sum()))).sum())
out = dict(case=case, window=2 * H + 1, n=n, hist=kl(N + .5))
for cg in [False, True]:
    for K in [1, 2, 3]:
        t0 = time.time(); m = bt.BLMLQuick(dtype=np.float64, tol=1e-6).fit(Z, bt.with_lattice(bt.make_modes(FC, ['ball', 'ball'], None, rank=K), 1.0), sample_weight=N, complex_g=cg)
        out[f"{'complex' if cg else 'real'}_K{K}"] = [kl(m.predict(Z)), round(time.time() - t0, 1), bool(m.converged_)]
print(json.dumps(out), flush=True)
