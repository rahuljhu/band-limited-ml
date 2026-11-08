"""Far-field test where sqrt(f) IS band-limited: in focus, h(x) = 2*rho*sinc(2*rho*x) (real, BL at rho = NA/lambda),
f = h^2 (no pixel integration: p_b proportional to f(z_b)). Rank-1 BLML at fc = 2*rho is then the exact model.
Window sweep x n; KL(p || q) over the window, q_b = ghat^2(z_b)."""
import sys, numpy as np
sys.path.insert(0, 'results/psf'); import psfkit as pk; import blml_trivial as bt
FC = pk.FC_PHYS; f = lambda x: np.sinc(FC * x) ** 2                    # h = sinc(2 rho x), 2 rho = FC
for n in [10 ** 4, 10 ** 5, 10 ** 6]:
    for H in [15, 50, 150]:
        Z = np.arange(-H, H + 1.0)[:, None]; z = Z[:, 0]; pb = f(z) * FC; out = 1 - pb.sum(); p = pb / pb.sum()
        r = {k: [] for k in ['hist', 'BLML@phys', 'BLML@phys signed-init', 'BLML@0.5']}
        for s in range(3):
            N = np.random.default_rng(s).multinomial(n, p)
            kl = lambda q: float((p * np.log(p / (np.maximum(q, 1e-300) / np.maximum(q, 1e-300).sum()))).sum())
            r['hist'].append(kl(N + 0.5))
            r['BLML@phys'].append(kl(bt.BLMLQuick(dtype=np.float64).fit(Z, bt.with_lattice(bt.make_modes(FC, None, None, d=Z.shape[1]), 1.0), sample_weight=N).predict(Z)))
            r['BLML@0.5'].append(kl(bt.BLMLQuick(dtype=np.float64).fit(Z, bt.with_lattice(bt.make_modes(0.5, None, None, d=Z.shape[1]), 1.0), sample_weight=N).predict(Z)))
            # oracle sign pattern: fit with the exact model by L-BFGS from c = S^-1 sign(h) sqrt(N/n) (tests local optima)
            S = FC * np.sinc(FC * (z[:, None] - z[None])); from scipy.optimize import minimize
            u0 = np.sign(np.sinc(FC * z) + 1e-12) * np.sqrt((N + .5) / (N + .5).sum()); a0 = np.linalg.lstsq(S, u0, rcond=1e-6)[0]
            def L(a):
                u = S @ a; E = a @ u; return -(N @ np.log(u ** 2 + 1e-300) - n * np.log(E)) / n, -(2 * S @ (N / u) - 2 * n * u / E) / n
            a = minimize(L, a0, jac=True, method='L-BFGS-B', options={'maxiter': 20000, 'gtol': 1e-12, 'ftol': 1e-15}).x
            r['BLML@phys signed-init'].append(kl((S @ a) ** 2))
        print(f'n={n:>8d} window {2*H+1:4d}px outside {out:.4f} | ' + '  '.join(f'{k} {np.mean(v):.6f}' for k, v in r.items()), flush=True)
