"""Paper 3, Sec. 4.2: pupil phase from a single image. Fit complex BLML (ball kernel, fc = 2NA/lambda) to
counts sampled from |h|^2 (pixel centres, 61x61 window), FFT of g on a fine grid -> pupil estimate, fit Zernike
coefficients (amplitude-weighted, r <= 0.9 rho), compare with the true coefficients. Piston and tilt are
nuisance (removed); twin: the even modes may flip sign together. Usage: sec42_phase.py <case> <n> <seed>"""
import sys, json, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'results/psf')
import blml_trivial as bt, psfkit as pk
from blml_trivial import make_modes
FC = pk.FC_PHYS; RHO = FC / 2; OS = 8; NF = 2048; H = 30
# Zernike (unit disk, r in [0,1]): name, function, parity under (x,y) -> (-x,-y) (even: twin flips sign)
ZN = [('piston', lambda r, t: 1 + 0 * r, 'nuis'), ('tilt_x', lambda r, t: r * np.cos(t), 'nuis'),
      ('tilt_y', lambda r, t: r * np.sin(t), 'nuis'),
      ('defocus', lambda r, t: 2 * r ** 2 - 1, 'even'), ('astig_0', lambda r, t: r ** 2 * np.cos(2 * t), 'even'),
      ('astig_45', lambda r, t: r ** 2 * np.sin(2 * t), 'even'),
      ('coma_x', lambda r, t: (3 * r ** 3 - 2 * r) * np.cos(t), 'odd'), ('coma_y', lambda r, t: (3 * r ** 3 - 2 * r) * np.sin(t), 'odd'),
      ('trefoil_x', lambda r, t: r ** 3 * np.cos(3 * t), 'odd'), ('trefoil_y', lambda r, t: r ** 3 * np.sin(3 * t), 'odd'),
      ('spherical', lambda r, t: 6 * r ** 4 - 6 * r ** 2 + 1, 'even')]
CASES = {'defocus': {'defocus': 1.0}, 'astig': {'astig_0': 1.0}, 'coma': {'coma_x': 0.5},
         'mixed': {'defocus': 0.4, 'astig_45': 0.5, 'coma_y': 0.3, 'spherical': 0.2}}
def phase_of(coef, r, t):
    return sum(c * dict((n, f) for n, f, _ in ZN)[name](r, t) for name, c in coef.items())
case, n, seed = sys.argv[1], int(float(sys.argv[2])), int(sys.argv[3])
coef = CASES[case]
k = np.fft.fftfreq(NF, d=1.0 / OS); KX, KY = np.meshgrid(k, k, indexing='ij'); r = np.hypot(KX, KY) / RHO; t = np.arctan2(KY, KX)
h = np.fft.fftshift(np.fft.ifft2(np.where(r <= 1, np.exp(1j * phase_of(coef, r, t)), 0)))[::OS, ::OS]; c = h.shape[0] // 2
hw = h[c - H:c + H + 1, c - H:c + H + 1].ravel(); p = np.abs(hw) ** 2; p /= p.sum()
g1 = np.arange(-H, H + 1.0); I, J = np.meshgrid(g1, g1, indexing='ij'); Z = np.c_[I.ravel(), J.ravel()]
N = np.random.default_rng(seed).multinomial(n, p)
m = bt.BLMLQuick(dtype=np.float64, tol=1e-5).fit(Z, bt.with_lattice(make_modes([FC, FC], ['ball', 'ball'], None), 1.0), sample_weight=N, complex_g=True)
L = 128; dx = 0.25; xs = np.arange(-L, L) * dx; X, Y = np.meshgrid(xs, xs, indexing='ij')
g = m.sqrt_density(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
Ph = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g))); kk = np.fft.fftshift(np.fft.fftfreq(2 * L, d=dx))
QX, QY = np.meshgrid(kk, kk, indexing='ij'); qr = np.hypot(QX, QY) / RHO; qt = np.arctan2(QY, QX); inside = qr <= 0.9
amp = np.abs(Ph[inside]); wts = amp / amp.sum()
# phase relative to the amplitude-weighted mean (piston), unwrap-free: fit Zernikes to angle(Ph * conj(ref)) iteratively
A = np.stack([f(qr[inside], qt[inside]) for _, f, _ in ZN], 1); est = np.zeros(len(ZN))
for _ in range(20):                                   # Gauss-Newton on the wrapped residual
    res = np.angle(Ph[inside] * np.exp(-1j * (A @ est)))
    est += np.linalg.lstsq(A * np.sqrt(wts)[:, None], res * np.sqrt(wts), rcond=None)[0]
true = np.array([coef.get(nm, 0.0) for nm, _, _ in ZN]); par = np.array([pp for _, _, pp in ZN])
keep = par != 'nuis'
err_direct = est[keep] - true[keep]
twin = est.copy(); twin[par == 'even'] *= -1                    # twin flips the even modes
err_twin = twin[keep] - true[keep]
use_twin = np.linalg.norm(err_twin) < np.linalg.norm(err_direct)
err = err_twin if use_twin else err_direct
out = dict(case=case, n=n, seed=seed, twin_chosen=bool(use_twin), rms_err=float(np.sqrt(np.mean(err ** 2))),
           true_rms=float(np.sqrt(np.mean(true[keep] ** 2))), modes=[nm for (nm, _, pp) in ZN if pp != 'nuis'],
           est=(twin if use_twin else est)[keep].tolist(), true=true[keep].tolist(), conv=bool(m.converged_))
print(json.dumps(out), flush=True)
import os; os.makedirs('results/complex/sec42', exist_ok=True)
json.dump(out, open(f'results/complex/sec42/{case}_{n}_{seed}.json', 'w'))
