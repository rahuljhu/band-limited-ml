"""BLMLBinned on the GPU: CPU/GPU parity, GPU checks, then the full 2048 x 2048 bead frame (no crop).
Run: BLML_DATA_SRC=<dir with psf_plane5.npy> ./colab_job.sh start binned binned_gpu_job.py T4
Local smoke test: BLML_ENGINE=cpu BLML_FRAME_CROP=257 BLML_PARTS=frame python binned_gpu_job.py
Each part writes one JSON (and npz) into BLML_OUT_DIR and is skipped if it exists (resumable)."""
import os, sys, json, time, subprocess
OUT = os.environ.get('BLML_OUT_DIR', 'outputs_binned'); CODE = os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('BLML_DATA_DIR', '.'); ENGINE = os.environ.get('BLML_ENGINE', 'gpu')
PARTS = os.environ.get('BLML_PARTS', 'parity,checks,frame,stream').split(',')
CROP = int(os.environ.get('BLML_FRAME_CROP', '0'))
os.makedirs(OUT, exist_ok=True)
if ENGINE == 'gpu' and subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda13x', 'numba'], check=True)
sys.path.insert(0, CODE)
import numpy as np
import blml_trivial as bt
from blml_binned import BLMLBinned, BLMLBinnedStreaming, _np

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def done(name):
    return os.path.exists(os.path.join(OUT, name))

def sync():
    if ENGINE == 'gpu':
        import cupy as cp; cp.cuda.Device().synchronize()

rng = np.random.default_rng(1)
def hist(shape, n, f):
    g = np.meshgrid(*[np.arange(m, dtype=float) for m in shape], indexing='ij')
    p = f(*g); p = p / p.sum(); return rng.multinomial(n, p.ravel()).reshape(shape)

def cells():
    c1 = lambda x: np.exp(-0.5 * ((x - 14) / 3) ** 2) + 0.5 * np.exp(-0.5 * ((x - 27) / 2) ** 2) + 0.02
    cp_ = lambda x: np.exp(np.cos(2 * np.pi * x / 32)) + 0.3 * np.exp(-0.5 * ((x - 5) / 1.5) ** 2)
    c2 = lambda x, y: np.exp(-0.5 * (((x - 11) / 2.5) ** 2 + ((y - 13) / 3) ** 2)) + 0.4 * np.exp(-0.5 * (((x - 17) / 2) ** 2 + ((y - 6) / 2) ** 2)) + 0.01
    c3 = lambda x, y, z: np.exp(-0.5 * (((x - 5) / 1.5) ** 2 + ((y - 4) / 2) ** 2 + ((z - 6) / 1.5) ** 2)) + 0.02
    yield 'd1 sinc real', hist((40,), 3000, c1), bt.make_modes([0.35], ['sinc']), {}
    yield 'd1 sinc complex', hist((40,), 3000, c1), bt.make_modes([0.35], ['sinc']), dict(complex_g=True)
    yield 'd1 sinc rank2 shared complex', hist((40,), 3000, c1), bt.make_modes([0.35], ['sinc'], rank=2), dict(complex_g=True)
    yield 'd1 sinc rank2 per-mode complex', hist((40,), 3000, c1), bt.make_modes([[0.35], [0.12]], ['sinc']), dict(complex_g=True)
    yield 'd1 dirichlet exact period real', hist((32,), 3000, cp_), bt.make_modes([0.3], ['dirichlet'], periods=[32.0]), {}
    yield 'd1 dirichlet exact period complex', hist((32,), 3000, cp_), bt.make_modes([0.3], ['dirichlet'], periods=[32.0]), dict(complex_g=True)
    yield 'd2 ball real', hist((24, 24), 8000, c2), bt.make_modes([0.4, 0.4], ['ball', 'ball']), {}
    yield 'd2 ball complex', hist((24, 24), 8000, c2), bt.make_modes([0.4, 0.4], ['ball', 'ball']), dict(complex_g=True)
    yield 'd2 sinc x dirichlet complex', hist((24, 24), 8000, c2), bt.make_modes([0.4, 0.3], ['sinc', 'dirichlet'], periods=[None, 24.0]), dict(complex_g=True)
    yield 'd2 ball rank2 per-mode complex', hist((24, 24), 8000, c2), bt.make_modes([[0.4, 0.4], [0.15, 0.15]], ['ball', 'ball']), dict(complex_g=True)
    yield 'd3 sinc real', hist((10, 10, 10), 6000, c3), bt.make_modes([0.5] * 3, ['sinc'] * 3), {}
    yield 'd3 ball complex', hist((10, 10, 10), 6000, c3), bt.make_modes([0.5] * 3, ['ball'] * 3), dict(complex_g=True)
    c512 = lambda x, y: sum(np.exp(-0.5 * (((x - a) / 1.6) ** 2 + ((y - b) / 1.6) ** 2)) for a, b in [(60, 70), (180, 40), (120, 200), (220, 220)]) + 0.05
    yield 'd2 ball complex 256^2', hist((256, 256), 400000, c512), bt.make_modes([0.33, 0.33], ['ball', 'ball']), dict(complex_g=True)

# ---------------------------------------------------------------- A: CPU / GPU parity
if 'parity' in PARTS and not done('parity.json'):
    rows = []
    for name, N, modes, kw in cells():
        res = {}
        for tag, eng, dt in [('cpu64', 'cpu', np.float64), ('gpu64', 'gpu', np.float64), ('gpu32', 'gpu', np.float32)]:
            if eng == 'gpu' and ENGINE != 'gpu':
                continue
            tol = 1e-7 if dt == np.float64 else 1e-4
            t = time.time(); m = BLMLBinned(engine=eng, dtype=dt, tol=tol, max_sweeps=50000).fit(N, modes, **kw); sync(); dt_s = time.time() - t
            res[tag] = dict(ll=m.loglik_, f=_np(m.density_).ravel(), sweeps=int(m.n_sweeps_), conv=bool(m.converged_), t=dt_s,
                            cert=(m.certificate() if m.shared_ else None))
        occ = N.ravel() > 0; ref = res['cpu64']
        r = dict(cell=name, **{f'{k}_{q}': v[q] for k, v in res.items() for q in ('ll', 'sweeps', 'conv', 't', 'cert')})
        for k in res:
            if k != 'cpu64':
                r[f'{k}_dll'] = res[k]['ll'] - ref['ll']
                r[f'{k}_fmaxrel'] = float(np.max(np.abs(res[k]['f'][occ] - ref['f'][occ]) / ref['f'][occ]))
        rows.append(r); log(json.dumps({k: (round(v, 8) if isinstance(v, float) else v) for k, v in r.items()}))
        json.dump(rows, open(os.path.join(OUT, 'parity.partial.json'), 'w'), indent=1)
    json.dump(rows, open(os.path.join(OUT, 'parity.json'), 'w'), indent=1)

# ---------------------------------------------------------------- B: GPU checks (grid, grad, cert, stream)
if 'checks' in PARTS and not done('checks.json') and ENGINE == 'gpu':
    c2 = lambda x, y: np.exp(-0.5 * (((x - 11) / 2.5) ** 2 + ((y - 13) / 3) ** 2)) + 0.4 * np.exp(-0.5 * (((x - 17) / 2) ** 2 + ((y - 6) / 2) ** 2)) + 0.01
    rows = []
    for name, cplx in [('d2 ball real', False), ('d2 ball complex', True)]:
        N = hist((24, 24), 8000, c2); mk = lambda fc: bt.make_modes([fc, fc], ['ball', 'ball']); fc = 0.4
        m = BLMLBinned(engine='gpu', tol=1e-8).fit(N, mk(fc), complex_g=cplx)
        G = m.predict_grid(3); fine = np.meshgrid(*[np.arange(s) / 3.0 for s in G.shape], indexing='ij')
        Xf = np.stack([x.ravel() for x in fine], 1); sel = np.arange(0, len(Xf), max(1, len(Xf) // 300))
        r = dict(case=name, grid_maxrel=float(np.max(np.abs(G.ravel()[sel] - m.predict(Xf[sel])) / m.predict(Xf[sel]))))
        g = float(np.nansum(m.fc_grad()[0])); d = 1e-4
        ll = [BLMLBinned(engine='gpu', tol=1e-10).fit(N, mk(fc * np.exp(sg * d)), init=m, complex_g=cplx).loglik_ for sg in (1, -1)]
        r.update(grad=g, grad_fd=(ll[0] - ll[1]) / (2 * d), cert=m.certificate())
        mc = BLMLBinned(engine='cpu', tol=1e-8).fit(N, mk(fc), complex_g=cplx); r['cert_cpu'] = mc.certificate()
        parts, rest = [], N.copy()
        for k in range(3):
            b = rng.binomial(rest, 1.0 / (4 - k)); parts.append(b); rest = rest - b
        parts.append(rest); s = BLMLBinnedStreaming(engine='gpu', tol=1e-8, fc_update=None)
        for b in parts:
            s.partial_fit(b, mk(fc), complex_g=cplx)
        r['stream_dll'] = s.loglik_ - m.loglik_
        rows.append(r); log(json.dumps(r))
    json.dump(rows, open(os.path.join(OUT, 'checks.json'), 'w'), indent=1)

# ---------------------------------------------------------------- C: the full frame
FC = 2 * 1.3 / 0.515 * 0.065                                     # 2 NA / lambda in cycles per pixel
def frame_counts():
    N = np.load(os.path.join(DATA, 'psf_plane5.npy')).astype(np.int64)
    if CROP:
        y0, x0 = 661, 1068; H = CROP // 2; N = N[y0 - H:y0 + H + 1, x0 - H:x0 + H + 1]
    return N

def split(N, seed=0):
    r = np.random.default_rng(seed); tr = r.binomial(N, 0.7); rest = N - tr
    va = r.binomial(rest, 1 / 3); return tr, va, rest - va

def score(P, N):
    return float((N * np.log(np.maximum(P, 1e-300))).sum() / N.sum())

def baselines(tr, va, te):
    """Histogram + pseudo-count a, and a Gaussian-blurred histogram (sigma), each chosen on va; scored on te."""
    from scipy.ndimage import gaussian_filter
    out = {}
    best = max(((score((tr + a) / (tr + a).sum(), va), a) for a in [0.01, 0.1, 0.3, 1, 3]))
    a = best[1]; out['histogram'] = dict(a=a, val=best[0], test=score((tr + a) / (tr + a).sum(), te))
    cand = []
    for sg in [0.5, 0.7, 1.0, 1.4, 2.0]:
        P = gaussian_filter(tr.astype(float), sg, mode='constant') + 1e-3; P = P / P.sum(); cand.append((score(P, va), sg, P))
    v, sg, P = max(cand, key=lambda t: t[0]); out['gauss_blur'] = dict(sigma=sg, val=v, test=score(P, te))
    return out

if 'frame' in PARTS and not done('frame.json'):
    N = frame_counts(); tr, va, te = split(N); tag = f'{N.shape[0]}x{N.shape[1]}'
    log(f'frame {tag}: {int(N.sum())} photons; train {int(tr.sum())}')
    res = dict(shape=list(N.shape), photons=int(N.sum()), baselines=baselines(tr, va, te))
    log(f"baselines {res['baselines']}")
    modes = bt.make_modes([FC, FC], ['ball', 'ball'])
    dt = np.float32 if ENGINE == 'gpu' else np.float64
    for label, cplx in [('complex', True), ('real', False)]:
        t = time.time()
        m = BLMLBinned(engine=ENGINE, dtype=dt, tol=1e-4, max_sweeps=int(os.environ.get('BLML_MAX_SWEEPS') or '20000'), verbose=True,
                       ll_tol=float(os.environ.get('BLML_LL_TOL', '1e-9')))
        m.fit(tr, modes, complex_g=cplx); sync(); tfit = time.time() - t
        f = _np(m.density_).astype(np.float64); P = f / f.sum()
        res[label] = dict(time_s=tfit, sweeps=int(m.n_sweeps_), converged=bool(m.converged_), residual=float(m.residual_),
                          anderson=int(m.n_anderson_), damped=int(m.n_damped_), gradient=int(m.n_gradient_), stop=m.stop_reason_,
                          train=score(P, tr), val=score(P, va), test=score(P, te),
                          mass_in_frame=float(f.sum()), rho_history=[float(x) for x in m.rho_history_[::10]],
                          merit_per_photon=[float(x) / m.n_ for x in m.merit_history_[::10]])
        log(f"{label}: {json.dumps({k: v for k, v in res[label].items() if k != 'rho_history'})}")
        np.savez_compressed(os.path.join(OUT, f'frame_{label}.npz'), density=f.astype(np.float32), e=_np(m.e_hat_[0]).astype(np.complex64 if cplx else np.float32))
        if label == 'complex':                                   # 2x refined density around bead 0 for figures
            y0, x0 = (661, 1068) if not CROP else (N.shape[0] // 2, N.shape[1] // 2)
            res['complex']['bead0_window'] = [y0, x0]
            json.dump(res, open(os.path.join(OUT, 'frame.partial.json'), 'w'), indent=1)
        json.dump(res, open(os.path.join(OUT, 'frame.partial.json'), 'w'), indent=1)
    from blml_binned import BLMLEmpirical                    # paper 7: normalised on the frame, ridge, lam on val
    t = time.time(); e = BLMLEmpirical.select(tr, va, modes, [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], reg='ridge', engine=ENGINE,
                                              dtype=dt, tol=1e-6, max_iter=3000); sync()
    f = _np(e.density_).astype(np.float64); P = f / f.sum()
    res['empirical'] = dict(time_s=time.time() - t, lam=e.lam, path=e.path_, iters=int(e.n_iter_), test=score(P, te), val=score(P, va))
    log(f"empirical: {json.dumps(res['empirical'])}")
    np.savez_compressed(os.path.join(OUT, 'frame_empirical.npz'), density=f.astype(np.float32))
    json.dump(res, open(os.path.join(OUT, 'frame.json'), 'w'), indent=1)

# ---------------------------------------------------------------- D: streaming the frame in 10 photon batches
if 'stream' in PARTS and not done('stream.json'):
    N = frame_counts(); tr, va, te = split(N)
    r = np.random.default_rng(5); rest = tr.copy(); batches = []
    for k in range(10):
        b = r.binomial(rest, 1.0 / (10 - k)); batches.append(b); rest = rest - b
    dt = np.float32 if ENGINE == 'gpu' else np.float64
    s = BLMLBinnedStreaming(engine=ENGINE, dtype=dt, tol=1e-4, fc_update=None, batch_sweeps=int(os.environ.get('BLML_BATCH_SWEEPS', '300')))
    modes = bt.make_modes([FC, FC], ['ball', 'ball']); rows = []; acc = np.zeros_like(tr)
    for i, b in enumerate(batches):
        t = time.time(); s.partial_fit(b, modes, complex_g=True); sync(); acc = acc + b
        f = _np(s.density_).astype(np.float64); P = f / f.sum()
        rows.append(dict(batch=i + 1, photons=int(acc.sum()), time_s=time.time() - t, sweeps=int(s.n_sweeps_),
                         residual=float(s.residual_), test=score(P, te)))
        log(json.dumps(rows[-1]))
        json.dump(rows, open(os.path.join(OUT, 'stream.partial.json'), 'w'), indent=1)
    json.dump(rows, open(os.path.join(OUT, 'stream.json'), 'w'), indent=1)
# ---------------------------------------------------------------- E: rank of the intensity, real rank K vs complex
# complex rank 1 = real rank 2 (|a + ib|^2 = a^2 + b^2); a gain at real K = 3 that stops there fits the vectorial PSF
if 'rank' in PARTS and not done('rank.json'):
    N = frame_counts(); tr, va, te = split(N)
    dt = np.float32 if ENGINE == 'gpu' else np.float64
    pj = os.path.join(OUT, 'rank.partial.json'); res = json.load(open(pj)) if os.path.exists(pj) else {}
    fits = [('real1', 1, False), ('complex1', 1, True), ('real2', 2, False), ('real3', 3, False), ('real4', 4, False),
            ('complex2', 2, True), ('real6', 6, False)]                  # complex2 = real4's model class
    keep = os.environ.get('BLML_RANK_FITS') or 'real1,complex1,real2,real3,real4'
    for label, K, cplx in [f for f in fits if f[0] in keep.split(',')]:
        if label in res:
            continue
        t = time.time()
        m = BLMLBinned(engine=ENGINE, dtype=dt, tol=1e-4, max_sweeps=int(os.environ.get('BLML_MAX_SWEEPS') or '3000'), verbose=True,
                       ll_tol=float(os.environ.get('BLML_LL_TOL', '1e-9')))
        m.fit(tr, bt.make_modes([FC, FC], ['ball', 'ball'], rank=K), complex_g=cplx); sync(); tfit = time.time() - t
        f = _np(m.density_).astype(np.float64); P = f / f.sum()
        U = _np(m.Se_).reshape(K, -1); w = f.ravel() > 10 * np.median(f)          # bead bins
        R = np.concatenate([U.real, U.imag]) if cplx else U.real                   # real amplitudes: gauge-invariant Gram
        spec = lambda X: sorted((np.linalg.eigvalsh(X @ X.T) / (X ** 2).sum()).tolist(), reverse=True)
        res[label] = dict(K=K, complex=cplx, time_s=tfit, sweeps=int(m.n_sweeps_), stop=m.stop_reason_, residual=float(m.residual_),
                          train=score(P, tr), val=score(P, va), test=score(P, te),
                          mode_spectrum_all=spec(R), mode_spectrum_beads=spec(R[:, w]))
        log(f"{label}: {json.dumps(res[label])}")
        np.savez_compressed(os.path.join(OUT, f'rank_{label}.npz'), Se=_np(m.Se_).astype(np.complex64 if cplx else np.float32))
        json.dump(res, open(pj, 'w'), indent=1)
    json.dump(res, open(os.path.join(OUT, 'rank.json'), 'w'), indent=1)
log('all parts done')
