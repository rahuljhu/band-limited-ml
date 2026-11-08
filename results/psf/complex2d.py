"""2-D far field: disk pupil radius rho = NA/lambda (cycles/px), h = IFT(pupil e^{i phase}), f = |h|^2 sampled at
pixel centres (no pixel blur). Fits on the window: histogram, real BLML (ball kernel, fc = 2 rho), complex BLML."""
import sys, time, json, numpy as np
sys.path.insert(0, 'results/psf'); import psfkit as pk; import blml_trivial as bt
FC = pk.FC_PHYS; RHO = FC / 2; OS = 8; NF = 2048                        # field grid: 1/OS px, NF samples
def field(case):
    k = np.fft.fftfreq(NF, d=1.0 / OS); KX, KY = np.meshgrid(k, k, indexing='ij'); r = np.hypot(KX, KY) / RHO
    ph = {'airy': 0 * r, 'defocus': 2.0 * (2 * r ** 2 - 1), 'astig': 1.5 * r ** 2 * np.cos(2 * np.arctan2(KY, KX))}[case]
    h = np.fft.fftshift(np.fft.ifft2(np.where(r <= 1, np.exp(1j * ph), 0)))
    return h[::OS, ::OS]                                                   # pixel-centre samples, centre at index NF//OS//2
def run(case, H, n, seed):
    h = field(case); c0 = h.shape[0] // 2; hw = h[c0 - H:c0 + H + 1, c0 - H:c0 + H + 1]
    pb = np.abs(hw) ** 2; p = (pb / pb.sum()).ravel()
    ii, jj = np.meshgrid(np.arange(-H, H + 1.0), np.arange(-H, H + 1.0), indexing='ij'); Z = np.c_[ii.ravel(), jj.ravel()]
    N = np.random.default_rng(seed).multinomial(n, p); kl = lambda q: float((p * np.log(p / (np.maximum(q, 1e-300) / np.maximum(q, 1e-300).sum()))).sum())
    out = {'case': case, 'window': 2 * H + 1, 'n': n, 'seed': seed, 'hist': kl(N + 0.5)}
    for lab, kw in [('real', {}), ('complex', {'complex_g': True})]:
        t0 = time.time(); m = bt.BLMLQuick(dtype=np.float64, tol=1e-6, seed=seed).fit(Z, bt.with_lattice(bt.make_modes(FC, ['ball', 'ball']), 1.0), sample_weight=N, **kw)
        out[lab] = kl(m.predict(Z)); out[lab + '_s'] = time.time() - t0; out[lab + '_conv'] = bool(m.converged_)
        if kw:
            g = m.sqrt_density(Z); U = np.vstack([g.real, g.imag]); sv = np.linalg.svd(U, compute_uv=False); out['complex_realness'] = float(sv[1] / sv[0])
    return out
if __name__ == '__main__':
    case, H, n, seed = sys.argv[1], int(sys.argv[2]), int(float(sys.argv[3])), int(sys.argv[4])
    r = run(case, H, n, seed); print(json.dumps(r), flush=True)
