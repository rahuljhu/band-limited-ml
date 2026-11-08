"""Complex BLML (g in C, f = |g|^2) on binned data: does the ML g come out real (up to a global phase)?
c = a + i b; f(z) = (Sa)^2 + (Sb)^2; norm a'Sa + b'Sb. Random complex starts, L-BFGS.
'realness' = s2/s1 of the 2 x B matrix [Sa; Sb] (0 = real up to a global phase)."""
import sys, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, 'results/psf'); import psfkit as pk; import blml_trivial as bt
FC = pk.FC_PHYS; RHO = FC / 2
def ff_truth(case, z):
    nu = np.linspace(-RHO, RHO, 2001); dnu = nu[1] - nu[0]
    ph = 2.0 * (2 * (nu / RHO) ** 2 - 1) if case == 'defocus' else 0 * nu
    h = (np.exp(1j * ph)[None] * np.exp(2j * np.pi * z[:, None] * nu[None])).sum(1) * dnu
    return np.abs(h) ** 2, h
def complex_blml(z, N, seed):
    n = N.sum(); B = len(z); S = FC * np.sinc(FC * (z[:, None] - z[None]))
    def L(x):
        a, b = x[:B], x[B:]; ua, ub = S @ a, S @ b; f = ua ** 2 + ub ** 2 + 1e-300; E = a @ ua + b @ ub
        val = -(N @ np.log(f) - n * np.log(E)) / n
        ga = -(2 * S @ (N * ua / f) - 2 * n * ua / E) / n; gb = -(2 * S @ (N * ub / f) - 2 * n * ub / E) / n
        return val, np.r_[ga, gb]
    rng = np.random.default_rng(seed); x0 = rng.normal(size=2 * B)
    r = minimize(L, x0, jac=True, method='L-BFGS-B', options={'maxiter': 200000, 'maxfun': 400000, 'gtol': 1e-12, 'ftol': 1e-16})
    a, b = r.x[:B], r.x[B:]; U = np.vstack([S @ a, S @ b]); sv = np.linalg.svd(U, compute_uv=False)
    f = (U ** 2).sum(0); return f / f.sum(), sv[1] / sv[0], r.fun
n = 10 ** 5
for case in ['focus', 'defocus']:
    for H in [15, 150]:
        z = np.arange(-H, H + 1.0); pb, h = ff_truth(case, z); p = pb / pb.sum()
        N = np.random.default_rng(0).multinomial(n, p); kl = lambda q: float((p * np.log(p / (q / q.sum()))).sum())
        line = f'{case:8s} {2*H+1:4d}px | hist {kl(N + .5):.5f}  unsigned {kl(np.maximum(bt.BLMLQuick(dtype=np.float64, tol=1e-6).fit(z[:, None], bt.with_lattice(bt.make_modes(FC, None, None, d=z[:, None].shape[1]), 1.), sample_weight=N).predict(z), 1e-300)):.5f}'
        if case == 'focus':
            line += f'  oracle-signed {kl(np.maximum(bt.BLMLQuick(dtype=np.float64, tol=1e-6).fit(z[:, None], bt.with_lattice(bt.make_modes(FC, None, None, d=z[:, None].shape[1]), 1.), sample_weight=N, signs=np.sign(h.real + 1e-12)).predict(z), 1e-300)):.5f}'
        line += f'  | true h: Im/Re ratio {np.linalg.svd(np.vstack([h.real, h.imag]), compute_uv=False)[::-1][0] / np.linalg.svd(np.vstack([h.real, h.imag]), compute_uv=False)[0]:.3f}'
        print(line, flush=True)
        for s in range(4):
            q, real, nll = complex_blml(z, N, s)
            print(f'      complex start {s}: KL {kl(q):.5f}  realness s2/s1 {real:.2e}  -LL/n {nll:.6f}', flush=True)
