"""Parity of blml_binned.BLMLBinned (full lattice, FFT) with BLMLQuick on counts on a fixed lattice (with_lattice),
cell by cell of the capability matrix (kind x d x g x rank). Synthetic histograms with empty bins. For each cell:
mean log-likelihood per count of both, max relative difference of f at the bins, and of f at off-lattice points
(predict), convergence of both. Real rank 1 is convex: parity must be to the solver tolerance; complex / rank K
are non-convex: both use one start (real root + phases); a higher likelihood of either is reported, not hidden.
Usage: python results/binned_parity.py [cpu|gpu] [out.json]"""
import sys, json, time, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_binned import BLMLBinned

engine = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
# The library tabulates radial kernels on an r grid with RADIAL_Q = 4 points per lattice step (linear interpolation,
# compiled into its numba kernels; ~1% error at fc h ~ 0.4, the fixed-lattice regime). BLMLBinned evaluates kernels
# exactly. Every cell therefore also gets an exact reference: the dense Gram from the kernel classes on the occupied
# bins, and (real rank 1, convex) a sequential coordinate-descent solve with it; both fits' stationarity residuals
# under the exact Gram are reported.
out = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else f'results/binned_parity_{engine}.json'
rng = np.random.default_rng(1)

def hist(shape, n, f):
    g = np.meshgrid(*[np.arange(m, dtype=float) for m in shape], indexing='ij')
    p = f(*g); p = p / p.sum()
    return rng.multinomial(n, p.ravel()).reshape(shape)

def cells():
    c1 = lambda x: np.exp(-0.5 * ((x - 14) / 3) ** 2) + 0.5 * np.exp(-0.5 * ((x - 27) / 2) ** 2) + 0.02
    yield 'd1 sinc real', hist((40,), 3000, c1), bt.make_modes([0.35], ['sinc']), {}
    yield 'd1 sinc complex', hist((40,), 3000, c1), bt.make_modes([0.35], ['sinc']), dict(complex_g=True)
    yield 'd1 sinc rank2 shared complex', hist((40,), 3000, c1), bt.make_modes([0.35], ['sinc'], rank=2), dict(complex_g=True)
    yield 'd1 sinc rank2 shared real', hist((40,), 3000, c1), bt.make_modes([0.35], ['sinc'], rank=2), {}
    yield 'd1 sinc rank2 per-mode complex', hist((40,), 3000, c1), bt.make_modes([[0.35], [0.12]], ['sinc']), dict(complex_g=True)
    cp = lambda x: np.exp(np.cos(2 * np.pi * x / 32)) + 0.3 * np.exp(-0.5 * ((x - 5) / 1.5) ** 2)
    yield 'd1 dirichlet exact period real', hist((32,), 3000, cp), bt.make_modes([0.3], ['dirichlet'], periods=[32.0]), {}
    yield 'd1 dirichlet exact period complex', hist((32,), 3000, cp), bt.make_modes([0.3], ['dirichlet'], periods=[32.0]), dict(complex_g=True)
    yield 'd1 dirichlet longer period real', hist((32,), 3000, cp), bt.make_modes([0.3], ['dirichlet'], periods=[50.0]), {}
    c2 = lambda x, y: np.exp(-0.5 * (((x - 11) / 2.5) ** 2 + ((y - 13) / 3) ** 2)) + 0.4 * np.exp(-0.5 * (((x - 17) / 2) ** 2 + ((y - 6) / 2) ** 2)) + 0.01
    yield 'd2 ball real', hist((24, 24), 8000, c2), bt.make_modes([0.4, 0.4], ['ball', 'ball']), {}
    yield 'd2 ball complex', hist((24, 24), 8000, c2), bt.make_modes([0.4, 0.4], ['ball', 'ball']), dict(complex_g=True)
    yield 'd2 sinc x sinc real', hist((24, 24), 8000, c2), bt.make_modes([0.4, 0.3], ['sinc', 'sinc']), {}
    yield 'd2 sinc x dirichlet complex', hist((24, 24), 8000, c2), bt.make_modes([0.4, 0.3], ['sinc', 'dirichlet'], periods=[None, 24.0]), dict(complex_g=True)
    yield 'd2 ball rank2 shared complex', hist((24, 24), 8000, c2), bt.make_modes([0.4, 0.4], ['ball', 'ball'], rank=2), dict(complex_g=True)
    yield 'd2 ball rank2 per-mode complex', hist((24, 24), 8000, c2), bt.make_modes([[0.4, 0.4], [0.15, 0.15]], ['ball', 'ball']), dict(complex_g=True)
    c3 = lambda x, y, z: np.exp(-0.5 * (((x - 5) / 1.5) ** 2 + ((y - 4) / 2) ** 2 + ((z - 6) / 1.5) ** 2)) + 0.02
    yield 'd3 sinc real', hist((10, 10, 10), 6000, c3), bt.make_modes([0.5, 0.5, 0.5], ['sinc'] * 3), {}
    yield 'd3 ball complex', hist((10, 10, 10), 6000, c3), bt.make_modes([0.5] * 3, ['ball'] * 3), dict(complex_g=True)
    yield 'd2 ball real h=0.5 origin', hist((24, 24), 8000, c2), bt.make_modes([0.8, 0.8], ['ball', 'ball']), dict(h=0.5, origin=(-3.0, 1.0))

rows = []
for name, N, modes, kw in cells():
    h = kw.pop('h', 1.0); origin = np.broadcast_to(np.asarray(kw.pop('origin', 0.0), float), (N.ndim,))
    cplx = kw.get('complex_g', False)
    t = time.time(); mb = BLMLBinned(engine=engine, tol=1e-7).fit(N, modes, h=h, origin=origin, complex_g=cplx); tb = time.time() - t
    Zc = mb.centres(); occ = N.ravel() > 0
    lib = bt.BLMLQuick(dtype=np.float64, tol=1e-7); lib.n_starts = (0, 0)
    t = time.time(); lib.fit(Zc[occ], bt.with_lattice(modes, h), sample_weight=N.ravel()[occ].astype(float), complex_g=cplx); tl = time.time() - t
    fl = lib.predict(Zc[occ]); fb = np.asarray(mb.density_.get() if hasattr(mb.density_, 'get') else mb.density_).ravel()[occ]
    w = N.ravel()[occ]; n = w.sum()
    ll_l = float((w * np.log(fl)).sum() / n); ll_b = mb.loglik_
    Xq = Zc[occ][:: max(1, occ.sum() // 50)] + 0.37 * h                          # off-lattice points
    fq_l, fq_b = lib.predict(Xq), mb.predict(Xq)
    r = dict(cell=name, engine=engine, ll_binned=ll_b, ll_quick=ll_l, dll=ll_b - ll_l,
             f_bins_maxrel=float(np.max(np.abs(fb - fl) / fl)), f_off_maxrel=float(np.max(np.abs(fq_b - fq_l) / np.maximum(fq_l, 1e-12 * fq_l.max()))),
             conv_binned=bool(mb.converged_), sweeps_binned=int(mb.n_sweeps_), conv_quick=bool(lib.converged_), t_binned=tb, t_quick=tl)
    Zo = Zc[occ]; Dz = Zo[:, None, :] - Zo[None, :, :]
    Sx = []
    for sp in mb.specs_:
        Kv = 1.0
        for kind, dims in bt.kernel_units(sp['kinds'], N.ndim):
            Kv = Kv * bt.kernel_class(kind).eval(bt.unit_theta(kind, sp['fc'], sp['periods'], dims), [Dz[..., j] for j in dims])
        Sx.append(Kv)
    def stat_resid(E):                                   # max_b rho_b under the exact Gram, E (B_occ, K)
        U = np.stack([Sx[k] @ E[:, k] for k in range(E.shape[1])], 1); nr = (np.abs(E) ** 2).sum(1)
        return float(np.max(np.sqrt((np.abs(U * (nr / (n * w))[:, None] - E) ** 2).sum(1)) / np.sqrt(nr)))
    Eb = np.asarray(mb.e_hat_.get() if hasattr(mb.e_hat_, 'get') else mb.e_hat_).reshape(mb.K_, -1)[:, occ].T
    El = lib.e_hat_ if lib.e_hat_.ndim == 2 else lib.e_hat_[:, None]
    r['stat_exact_binned'], r['stat_exact_quick'] = stat_resid(Eb), stat_resid(El)
    if mb.K_ == 1 and not cplx:                          # convex: exact dense reference solve
        S0 = Sx[0]; s0 = S0[0, 0]; e = np.sqrt(n * w / s0); u = S0 @ e
        for _ in range(20000):
            for b in range(len(e)):
                t = bt.cd_root(u[b] - s0 * e[b], n * w[b], s0); u += S0[:, b] * (t - e[b]); e[b] = t
            if np.max(np.abs(e * u / (n * w) - 1)) < 1e-10:
                break
        fr = u ** 2 / n ** 2; r['ll_exact_ref'] = float((w * np.log(fr)).sum() / n)
        r['dll_vs_exact'] = ll_b - r['ll_exact_ref']; r['f_bins_maxrel_vs_exact'] = float(np.max(np.abs(fb - fr) / fr))
    if mb.shared_:
        r['cert_binned'] = mb.certificate()
    rows.append(r)
    print(f"{name:34s} dll {r['dll']:+.2e}  f_bins {r['f_bins_maxrel']:.1e}  f_off {r['f_off_maxrel']:.1e}  "
          f"conv {r['conv_binned']}/{r['conv_quick']} sweeps {r['sweeps_binned']}  t {tb:.2f}s/{tl:.2f}s"
          + f"  stat(exact) {r['stat_exact_binned']:.1e}/{r['stat_exact_quick']:.1e}"
          + (f"  vs exact dll {r['dll_vs_exact']:+.1e} f {r['f_bins_maxrel_vs_exact']:.1e}" if 'dll_vs_exact' in r else '')
          + (f"  cert {r['cert_binned']:.4f}" if 'cert_binned' in r else ''), flush=True)
json.dump(rows, open(out, 'w'), indent=1)
