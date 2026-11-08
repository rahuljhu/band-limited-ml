"""Checks of blml_binned.BLMLEmpirical (paper 7) against dense linear algebra on small lattices (exact Gram from the
kernel classes): ridge and l2 solutions, l1 KKT conditions, the window / plane normalisations, and held-out
selection of lam on a binomial split. Usage: python results/empirical_checks.py [cpu|gpu]"""
import sys, json, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_binned import BLMLEmpirical, _np
engine = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
rng = np.random.default_rng(3)
def hist(shape, n, f):
    g = np.meshgrid(*[np.arange(m, dtype=float) for m in shape], indexing='ij'); p = f(*g); p = p / p.sum()
    return rng.multinomial(n, p.ravel()).reshape(shape)
c1 = lambda x: np.exp(-0.5 * ((x - 14) / 3) ** 2) + 0.5 * np.exp(-0.5 * ((x - 27) / 2) ** 2) + 0.02
c2 = lambda x, y: np.exp(-0.5 * (((x - 11) / 2.5) ** 2 + ((y - 13) / 3) ** 2)) + 0.4 * np.exp(-0.5 * (((x - 17) / 2) ** 2 + ((y - 6) / 2) ** 2)) + 0.01
rows = []
for name, N, modes, h in [('d1 sinc', hist((40,), 3000, c1), bt.make_modes([0.35], ['sinc']), 1.0),
                          ('d2 ball', hist((24, 24), 8000, c2), bt.make_modes([0.4, 0.4], ['ball', 'ball']), 1.0),
                          ('d2 sinc h=0.5', hist((24, 24), 8000, c2), bt.make_modes([0.8, 0.6], ['sinc', 'sinc']), 0.5)]:
    m0 = BLMLEmpirical(engine=engine).fit(N, modes, h=h); Z = m0.centres(); D = Z[:, None, :] - Z[None, :, :]; sp = m0.specs_[0]; S = 1.0
    for kind, dims in bt.kernel_units(sp['kinds'], N.ndim):
        S = S * bt.kernel_class(kind).eval(bt.unit_theta(kind, sp['fc'], sp['periods'], dims), [D[..., j] for j in dims])
    n = N.sum(); Dl = h ** N.ndim; t = np.sqrt(N.ravel() / (n * Dl)); B = len(t)
    for reg, lam in [('ridge', 1e-2), ('ridge', 1e-4), ('l2', 1e-3), ('l1', 1e-4), ('l1', 1e-3)]:
        m = BLMLEmpirical(reg=reg, lam=lam, tol=1e-10, engine=engine).fit(N, modes, h=h); c = _np(m.c_raw_).ravel()
        if reg == 'ridge':
            cd = n * np.linalg.solve(S + lam * np.eye(B), t); err = float(np.max(np.abs(c - cd)) / np.max(np.abs(cd)))
        elif reg == 'l2':
            cd = n * np.linalg.solve(S @ S + lam * np.eye(B), S @ t); err = float(np.max(np.abs(c - cd)) / np.max(np.abs(cd)))
        else:                                                    # KKT: |grad| <= lam, = -lam sign(c) on the support
            a = c / n; gr = 2 * S @ (S @ a - t); on = np.abs(a) > 0
            err = float(max(np.max(np.abs(gr[on] + lam * np.sign(a[on]))) if on.any() else 0, np.max(np.abs(gr[~on])) - lam if (~on).any() else 0) / lam)
        wn = float(_np(m.density_).sum() * Dl)
        r = dict(case=name, reg=reg, lam=lam, err=err, iters=int(m.n_iter_), window_sum=wn, plane_norm=m.plane_norm_, window_norm=m.window_norm_,
                 sparsity=m.sparsity_)
        rows.append(r); print(f"{name:14s} {reg:5s} lam {lam:.0e}: {'rel err vs dense' if reg != 'l1' else 'KKT viol/lam'} {err:.1e}  iters {m.n_iter_}  "
                              f"sum f Delta {wn:.6f}  int g^2/window {m.plane_norm_ / m.window_norm_:.4f}  nonzero {m.sparsity_:.2f}", flush=True)
    tr = rng.binomial(N, 0.8); va = N - tr
    best = BLMLEmpirical.select(tr, va, modes, [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1], reg='ridge', engine=engine)
    print(f"{name:14s} select ridge: lam {best.lam:.0e}  path {[(l, round(s, 4)) for l, s in best.path_]}", flush=True)
json.dump(rows, open(f'results/empirical_checks_{engine}.json', 'w'), indent=1)
