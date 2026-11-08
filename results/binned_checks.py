"""Checks of blml_binned beyond parity with BLMLQuick (see binned_parity.py):
  grid      predict_grid(r) equals predict at the refined lattice points
  grad      fc_grad (envelope theorem) equals the central difference of the refitted loglik_ (warm-started)
  cert      certificate() (Lanczos, FFT mat-vec) equals lambda_max of the dense D^1/2 S D^1/2 / n
  stream    BLMLBinnedStreaming over binomially thinned batches (decay 1) equals the one-shot fit on their sum;
            decay < 1 equals the one-shot fit on the decayed sum
Usage: python results/binned_checks.py [cpu|gpu] [out.json]"""
import sys, json, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_binned import BLMLBinned, BLMLBinnedStreaming, _np

engine = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
out = sys.argv[2] if len(sys.argv) > 2 else f'results/binned_checks_{engine}.json'
rng = np.random.default_rng(2)

def hist(shape, n, f):
    g = np.meshgrid(*[np.arange(m, dtype=float) for m in shape], indexing='ij')
    p = f(*g); p = p / p.sum(); return rng.multinomial(n, p.ravel()).reshape(shape)

c1 = lambda x: np.exp(-0.5 * ((x - 14) / 3) ** 2) + 0.5 * np.exp(-0.5 * ((x - 27) / 2) ** 2) + 0.02
c2 = lambda x, y: np.exp(-0.5 * (((x - 11) / 2.5) ** 2 + ((y - 13) / 3) ** 2)) + 0.4 * np.exp(-0.5 * (((x - 17) / 2) ** 2 + ((y - 6) / 2) ** 2)) + 0.01
CASES = [('d1 sinc real', hist((40,), 3000, c1), lambda fc: bt.make_modes([fc], ['sinc']), 0.35, {}),
         ('d1 sinc complex', hist((40,), 3000, c1), lambda fc: bt.make_modes([fc], ['sinc']), 0.35, dict(complex_g=True)),
         ('d2 ball real', hist((24, 24), 8000, c2), lambda fc: bt.make_modes([fc, fc], ['ball', 'ball']), 0.4, {}),
         ('d2 ball complex', hist((24, 24), 8000, c2), lambda fc: bt.make_modes([fc, fc], ['ball', 'ball']), 0.4, dict(complex_g=True)),
         ('d2 ball rank2 complex', hist((24, 24), 8000, c2), lambda fc: bt.make_modes([fc, fc], ['ball', 'ball'], rank=2), 0.4, dict(complex_g=True))]
rows = []
for name, N, mk, fc, kw in CASES:
    m = BLMLBinned(engine=engine, tol=1e-9).fit(N, mk(fc), **kw); r = dict(case=name, engine=engine)
    # grid
    G = m.predict_grid(3); fine = np.meshgrid(*[np.arange(s) / 3.0 for s in G.shape], indexing='ij')
    Xf = np.stack([x.ravel() for x in fine], 1); sel = np.arange(0, len(Xf), max(1, len(Xf) // 300))
    r['grid_maxrel'] = float(np.max(np.abs(G.ravel()[sel] - m.predict(Xf[sel])) / np.maximum(m.predict(Xf[sel]), 1e-300)))
    # grad (one parameter: the fc of the unit, all columns together for ball)
    g = np.concatenate(m.fc_grad()); d = 1e-4; ll = []
    for sgn in (1, -1):
        mm = BLMLBinned(engine=engine, tol=1e-10).fit(N, mk(fc * np.exp(sgn * d)), init=m, **kw); ll.append(mm.loglik_)
    fd = (ll[0] - ll[1]) / (2 * d)
    r['grad'] = float(np.nansum(g)); r['grad_fd'] = float(fd); r['grad_relerr'] = float(abs(np.nansum(g) - fd) / max(abs(fd), 1e-12))
    # cert
    occ = N.ravel() > 0; Zo = m.centres()[occ]; Dz = Zo[:, None, :] - Zo[None, :, :]; sp = m.specs_[0]; Kv = 1.0
    for kind, dims in bt.kernel_units(sp['kinds'], N.ndim):
        Kv = Kv * bt.kernel_class(kind).eval(bt.unit_theta(kind, sp['fc'], sp['periods'], dims), [Dz[..., j] for j in dims])
    E = _np(m.e_hat_).reshape(m.K_, -1)[:, occ]; Dd = (np.abs(E) ** 2).sum(0) / N.ravel()[occ]
    lam = np.linalg.eigvalsh(np.sqrt(Dd)[:, None] * Kv * np.sqrt(Dd)[None])[-1] / m.n_
    r['cert'] = m.certificate(); r['cert_dense'] = float(lam); r['cert_err'] = abs(r['cert'] - lam)
    # stream: 4 thinned batches
    parts, rest = [], N.copy()
    for k in range(3):
        b = rng.binomial(rest, 1.0 / (4 - k)); parts.append(b); rest = rest - b
    parts.append(rest)
    s = BLMLBinnedStreaming(engine=engine, tol=1e-9, fc_update=None)
    for b in parts:
        s.partial_fit(b, mk(fc), **kw)
    r['stream_dll'] = s.loglik_ - m.loglik_; r['stream_fmaxrel'] = float(np.max(np.abs(_np(s.density_) - _np(m.density_))[N > 0] / _np(m.density_)[N > 0]))
    dec = 0.7; s2 = BLMLBinnedStreaming(engine=engine, tol=1e-9, decay=dec, fc_update=None)
    for b in parts:
        s2.partial_fit(b, mk(fc), **kw)
    Nd = sum(b * dec ** (len(parts) - 1 - i) for i, b in enumerate(parts))
    m2 = BLMLBinned(engine=engine, tol=1e-9).fit(Nd, mk(fc), **kw)
    r['decay_dll'] = s2.loglik_ - m2.loglik_
    r['stream_sweeps'] = int(s.n_sweeps_)
    rows.append(r)
    print(f"{name:24s} grid {r['grid_maxrel']:.1e}  grad {r['grad']:+.5f} fd {r['grad_fd']:+.5f} (rel {r['grad_relerr']:.1e})  "
          f"cert {r['cert']:.6f} dense {r['cert_dense']:.6f}  stream dll {r['stream_dll']:+.1e} f {r['stream_fmaxrel']:.1e}  "
          f"decay dll {r['decay_dll']:+.1e}", flush=True)
json.dump(rows, open(out, 'w'), indent=1)
