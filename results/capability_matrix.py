"""The core's capability matrix (REFACTOR_KERNELS.md, part 2): one small case per cell -- kernel kind (sinc, dirichlet,
ball) x unit (column / radial group / mapped unit) x fit (one-shot, dep, counts, counts + dep, complex, rank 2;
streaming with trainable parameters, dep, counts, complex, rank 2) -- plus the kernel derivative check (every declared
parameter and every coordinate shift of every kind, lattice rows and point sums, against finite differences of the
kernel classes' formulas). A cell passes when it runs and gives finite numbers, or when it raises its error by design
(BY_DESIGN); anything else fails. The numbers are saved per engine; `compare` checks CPU = GPU.
    capability_matrix.py cpu|gpu [out.json]      run the cells on one engine
    capability_matrix.py compare cpu.json gpu.json
CELLS="substring,..." runs a subset."""
import os, sys, json, time, traceback, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np
warnings.filterwarnings('ignore')
import blml_trivial as bt
from blml_trivial import kernel_units, kernel_class, unit_theta
from blml_rolling import BLMLQuickStreaming

# errors by design: (cell substring, error substring)
BY_DESIGN = [('mapped', 'counts', 'a mapped unit needs a lattice regular in its outputs'),
             ('streaming tol, split roles', 'tol needs bcd_frac = 1')]
TOL = dict(ll=2e-2, grad=5e-2, fc=0.2, period=0.5, ess=0.1, mean=0.15, epochs=0, residual=0.3)  # sampling: different random streams

rng = np.random.default_rng(0)
N = 2000
hr = rng.integers(0, 24, N).astype(float)
DATA = {'sinc': np.c_[rng.normal(size=N), rng.normal(size=N) * 0.7],
        'dirichlet': np.c_[np.sin(2 * np.pi * hr / 24) + 0.5 * rng.normal(size=N), hr],
        'ball': np.c_[rng.normal(size=N), rng.normal(size=N)]}
Z4 = rng.normal(size=(N, 4)); Q4 = np.linalg.qr(np.random.default_rng(5).normal(size=(4, 4)))[0]
DATA['mapped'] = Z4 @ Q4
YLAB = (DATA['sinc'][:, 0] * DATA['sinc'][:, 1] > 0).astype(float)
YCON = YLAB + 0.4 * rng.normal(size=N)             # a continuous dep column (counts: Nyquist on the lattice)
W0 = np.linalg.qr(np.random.default_rng(1).normal(size=(4, 2)))[0]


def base_mode(kind, fc=None):
    """One mode over the 2 columns of DATA[kind] (or the 4 inputs of a mapped unit); fc None: trainable from 0.8."""
    f = lambda v: dict(value=v, trainable=True) if fc is None else v
    if kind == 'sinc':
        return {0: dict(kind='sinc', fc=f(0.8)), 1: dict(kind='sinc', fc=f(1.0))}
    if kind == 'dirichlet':
        return {0: dict(kind='sinc', fc=f(0.8)), 1: dict(kind='dirichlet', fc=dict(value=0.3, trainable=False), period=24.0)}
    if kind == 'ball':
        return {(0, 1): dict(kind='ball', fc=f(0.8))}
    mk = kind.split('-')[1]
    unit = dict(kind=mk, W=W0, fc=f(0.6) if mk == 'ball' else f([0.6, 0.6]))
    if mk == 'dirichlet':
        unit['period'] = 12.0
        unit['fc'] = dict(value=[0.6, 0.6], trainable=False)
    return {(0, 1, 2, 3): unit}


def d_of(kind):
    return 4 if kind.startswith('mapped') else 2


def X_of(kind):
    return DATA['mapped'] if kind.startswith('mapped') else DATA[kind]


def with_dep(mode, d, fc=None):
    """The mode with a dep column d: the class label (fc 10000, fixed) or, with fc, a continuous column."""
    m = dict(mode); m[d] = dict(kind='sinc', fc=dict(value=10000.0 if fc is None else fc, trainable=False), dep=True)
    return m


def counts_of(X, h):
    u, inv = np.unique(np.round(X / h).astype(np.int64), axis=0, return_inverse=True)
    return u * h, np.bincount(np.ravel(inv)).astype(float)


def finite(v):
    return bool(np.all(np.isfinite(np.asarray(v, float))))


def cells(eng):
    Est = lambda **k: bt.BLMLQuick(engine=eng, dtype=np.float64, **k)
    out = []
    for kind in ['sinc', 'dirichlet', 'ball', 'mapped-sinc', 'mapped-dirichlet', 'mapped-ball']:
        out += kind_cells(kind, Est, eng)
    out.append(('kernel derivatives (rows, rows_multi, points) = FD', lambda: kernel_fd(eng)))
    return out


def kind_cells(kind, Est, eng):
    """The cells of one kernel kind (a function per kind: the cells' closures bind its own data)."""
    out = []
    if True:
        X = X_of(kind); d = d_of(kind); Xq = X[:300]
        fixed = [base_mode(kind, fc=True)]                   # given fc (the one-shot fits)
        Xy = np.c_[X, YLAB]; dep = [with_dep(fixed[0], d)]

        def one_shot(modes=fixed, X=X, **kw):
            m = Est().fit(X, modes, **kw)
            return dict(ll=float(np.mean(np.log(np.maximum(m.predict(Xq if X.shape[1] == d else np.c_[Xq, YLAB[:300]]), 1e-300)))))

        def one_dep():
            m = Est().fit(Xy, dep)
            return dict(ll=float(np.mean(np.log(np.maximum(m.predict_ratio(Xy[:300]), 1e-300)))))

        Xc = np.c_[X, YCON]; depc = [with_dep(fixed[0], d, fc=1.0)]

        def one_counts(with_y=False):
            h = 0.08
            Z, w = counts_of(Xc if with_y else X, h)
            md = bt.with_lattice(depc if with_y else fixed, h)
            m = Est().fit(Z, md, sample_weight=w)
            q = Xc[:300] if with_y else Xq
            f = m.predict_ratio(q) if with_y else m.predict(q)
            return dict(ll=float(np.mean(np.log(np.maximum(f, 1e-300)))))

        def one_rank2():
            return one_shot(modes=fixed * 2)

        def one_grad():
            m = Est().fit(X, fixed)
            units = [dims if kind_ != 'dirichlet' else (dims, 'period') for kind_, dims in kernel_units(m.kinds_, len(m.kinds_))]
            g = m.param_grad(Xq, units, eps_rel=1e-3).mean(0)
            return dict(grad=[float(v) for v in g])

        def stream(modes=None, X=X, epochs=2, tol=None, **kw):
            md = modes if modes is not None else [base_mode(kind)]
            s = BLMLQuickStreaming(engine=eng, burn_in=800, lr=0.05, param_every=1, dtype=np.float64,
                                   **kw).fit(X, md, epochs=epochs, batch=400, tol=tol)
            ff = s.ff                                   # real rank 1: e_b > 0 at the optimum, so never below 0
            if ff is not None and not ff.cx and ff.K == 1 and np.min(ff.e) < 0:
                raise RuntimeError(f"{int((ff.e < 0).sum())} negative coefficients (real rank 1)")
            q = Xq if X.shape[1] == d else np.c_[Xq, YLAB[:300]]
            f = s.predict_ratio(q) if X.shape[1] != d else s.predict(q)
            r = dict(ll=float(np.mean(np.log(np.maximum(f, 1e-300)))), fc=np.ravel(bt.modes_fc(s.modes_)).tolist())
            return r if tol is None else dict(r, epochs=s.epochs_, residual=s.residual_)

        STOP = dict(epochs=4, tol=1e-2)                  # stopping on the solver's residual (at most 4 passes)

        def stream_dep():
            return stream(modes=[with_dep(base_mode(kind), d)], X=Xy)

        def stream_tol_fixed():
            """Every parameter frozen: the coefficients alone, run until the solver's residual meets tol."""
            frz = lambda c: dict(c, fc=dict(value=bt.param_value(c['fc'])[0], trainable=False),
                                 **(dict(trainable=False) if 'W' in c else {}))     # a plain fc is trainable by default
            m = {k: frz(c) for k, c in base_mode(kind, fc=0.8).items()}
            r = stream(modes=[m], epochs=40, tol=1e-3)
            if r['residual'] >= 1e-3:
                raise RuntimeError(f"not converged in {r['epochs']} epochs: residual {r['residual']:.2e}")
            return r

        def stream_counts(with_y=False, select_=False, tol=None):
            h = 0.08
            Z, w = counts_of(Xc if with_y else X, h)
            base = base_mode(kind) if not select_ else {k: {a: v for a, v in c.items() if a != 'fc' or bt.param_value(v)[1] is False}
                                                         for k, c in base_mode(kind).items()}
            md = bt.with_lattice([with_dep(base, d, fc=1.0)] if with_y else [base], h)
            s = BLMLQuickStreaming(engine=eng, burn_in=800, lr=0.05, param_every=1, dtype=np.float64).fit(
                Z, md, batch=400, sample_weight=w, **(dict(epochs=2) if tol is None else STOP))
            q = Xc[:300] if with_y else Xq
            f = s.predict_ratio(q) if with_y else s.predict(q)
            r = dict(ll=float(np.mean(np.log(np.maximum(f, 1e-300)))))
            return r if tol is None else dict(r, epochs=s.epochs_, residual=s.residual_)

        def one_select(counts=False):
            from selector import select, select_counts
            tmpl = [with_dep({k: {a: v for a, v in c.items() if a != 'fc' or bt.param_value(v)[1] is False}
                              for k, c in fixed[0].items()}, d, fc=1.0)]
            if counts:
                h = 0.08; Z, w = counts_of(Xc, h); wtr = np.random.default_rng(1).binomial(w.astype(np.int64), 0.8)
                md, best = select_counts(Z, wtr, w - wtr, h, tmpl, engine=eng)
            else:
                md, best = select(Xc, tmpl, search='shared', engine=eng)
            return dict(ll=float(best), fc=np.ravel(bt.modes_fc(md)).tolist())

        def other_mode():
            """A second mode with another kernel on the same columns (rank 2, per-mode kinds)."""
            if kind == 'sinc':
                return {(0, 1): dict(kind='ball', fc=dict(value=0.6, trainable=True))}
            if kind == 'ball':
                return {0: dict(kind='sinc', fc=dict(value=0.6, trainable=True)), 1: dict(kind='sinc', fc=dict(value=0.9, trainable=True))}
            if kind == 'dirichlet':
                return {0: dict(kind='sinc', fc=dict(value=0.6, trainable=True)), 1: dict(kind='sinc', fc=dict(value=0.4, trainable=True))}
            mk = kind.split('-')[1]                       # the same map, another kernel on its outputs
            unit = dict(base_mode(kind)[(0, 1, 2, 3)])
            unit.pop('period', None)
            if mk == 'ball':
                unit.update(kind='sinc', fc=dict(value=[0.5, 0.7], trainable=True))
            else:
                unit.update(kind='ball', fc=dict(value=0.5, trainable=True))
            return {(0, 1, 2, 3): unit}

        def fixed_of(mode):
            return {k: {a: (bt.param_value(v)[0] if a == 'fc' and isinstance(v, dict) and v.get('trainable', True) else v)
                        for a, v in c.items()} for k, c in mode.items()}

        def one_mixed():
            return one_shot(modes=[fixed[0], fixed_of(other_mode())])

        def one_grad_modes():
            m = Est().fit(X, [fixed[0], fixed_of(other_mode())])
            dl = len(m.modes_[0]['kinds'])
            pn = lambda kd: 'period' if kd == 'dirichlet' else 'fc'      # dirichlet fc: whole harmonics, no gradient
            u0 = kernel_units(m.modes_[0]['kinds'], dl)[0]
            units = [(1, dims, pn(kd)) for kd, dims in kernel_units(m.modes_[1]['kinds'], dl)] + [(0, u0[1], pn(u0[0]))]
            return dict(grad=[float(v) for v in m.param_grad(Xq, units, eps_rel=1e-3).mean(0)])

        def one_grad_dep():
            m = Est().fit(Xy, dep)
            kk = m._joint.kinds_ if hasattr(m, '_joint') else m.kinds_
            units = [(dims, 'period' if kind_ == 'dirichlet' else 'fc') for kind_, dims in kernel_units(kk, len(kk))
                     if dims != [len(kk) - 1]]
            return dict(grad=[float(v) for v in m.param_grad(Xy[:300], units, eps_rel=1e-3).mean(0)])

        def one_sample():
            m = Est().fit(Xc, depc)
            s, ess = m.sample_conditional(Xq[:20], n_samples=50, n_prop=200)
            return dict(ess=float(ess), mean=float(np.mean(s)))

        def stream_mixed():
            return stream(modes=[base_mode(kind), other_mode()])

        def stream_hill_rank2():
            md = [base_mode(kind), base_mode(kind)]
            return stream(modes=md, fc_update='hillclimb')

        def stream_period_k(**kw):
            if kind != 'dirichlet':
                raise SkipCell('no period')
            md = base_mode(kind); md[1] = dict(md[1], period=dict(value=23.0, trainable=True))
            md2 = {0: dict(kind='sinc', fc=dict(value=10000.0, trainable=False), dep=True), 1: md[1]}
            K = kw.pop('K', 1)
            s = BLMLQuickStreaming(engine=eng, burn_in=800, lr=0.05, param_every=1, dtype=np.float64, **kw).fit(
                np.c_[YLAB, X[:, 1]], [md2] * K, epochs=2, batch=400)
            return dict(period=[float(p) for p in bt.modes_period(s.modes_)[:, 1]])

        def stream_period():
            if kind == 'mapped-dirichlet':                 # a trainable period on a map's outputs (conditioning)
                unit = dict(base_mode(kind)[(0, 1, 2, 3)], period=dict(value=11.0, trainable=True))
                s = BLMLQuickStreaming(engine=eng, burn_in=800, lr=0.05, param_every=1, dtype=np.float64).fit(
                    np.c_[X, YLAB], [with_dep({(0, 1, 2, 3): unit}, d)], epochs=2, batch=400)
                return dict(period=[float(p) for p in np.ravel(bt.modes_period(s.modes_)) if np.isfinite(p)])
            if kind != 'dirichlet':
                raise SkipCell('no period')
            md = base_mode(kind); md[1] = dict(md[1], period=dict(value=23.0, trainable=True))
            Xr = np.c_[YLAB, X[:, 1]]                       # the period of a conditioning column
            md2 = {0: dict(kind='sinc', fc=dict(value=10000.0, trainable=False), dep=True), 1: md[1]}
            s = BLMLQuickStreaming(engine=eng, burn_in=800, lr=0.05, param_every=1, dtype=np.float64).fit(
                Xr, [md2], epochs=2, batch=400)
            return dict(period=float(bt.modes_period(s.modes_)[0][1]))

        for fit, fn in (('one-shot', one_shot), ('one-shot dep', one_dep), ('one-shot counts', one_counts),
                        ('one-shot counts dep', lambda: one_counts(True)),
                        ('one-shot complex', lambda: one_shot(complex_g=True)), ('one-shot rank 2', one_rank2),
                        ('param_grad', one_grad), ('select dep', one_select),
                        ('select_counts dep', lambda: one_select(True)),
                        ('streaming', stream), ('streaming dep', stream_dep), ('streaming counts', stream_counts),
                        ('streaming counts dep', lambda: stream_counts(True)),
                        ('streaming counts dep, burn-in selection', lambda: stream_counts(True, True)),
                        ('streaming complex', lambda: stream(complex_g=True)),
                        ('streaming rank 2', lambda: stream(modes=[base_mode(kind)] * 2)),
                        ('streaming trainable period', stream_period),
                        ('one-shot rank 2, a kernel per mode', one_mixed), ('param_grad per-mode', one_grad_modes),
                        ('param_grad conditional', one_grad_dep), ('sample_conditional', one_sample),
                        ('streaming rank 2, a kernel per mode', stream_mixed),
                        ('streaming rank 2 hill-climb', stream_hill_rank2),
                        ('streaming rank 2 trainable period', lambda: stream_period_k(K=2)),
                        ('streaming complex trainable period', lambda: stream_period_k(complex_g=True)),
                        ('streaming tol', lambda: stream(**STOP)),
                        ('streaming tol dep', lambda: stream(modes=[with_dep(base_mode(kind), d)], X=Xy, **STOP)),
                        ('streaming tol counts', lambda: stream_counts(tol=True)),
                        ('streaming tol complex', lambda: stream(complex_g=True, **STOP)),
                        ('streaming tol rank 2', lambda: stream(modes=[base_mode(kind)] * 2, **STOP)),
                        ('streaming tol, split roles', lambda: stream(bcd_frac=0.5, **STOP)),
                        ('streaming tol, fixed parameters', stream_tol_fixed)):
            out.append((f'{kind} | {fit}', fn))
    return out


class SkipCell(Exception):
    pass


def kernel_fd(eng='cpu'):
    """Every declared parameter and coordinate shift of every kind: lattice rows / fused rows / point sums vs a
    forward difference of the kernel classes' exact formulas; returns the worst relative error. eng 'gpu': the CUDA
    kernels too (float64 and float32), against the CPU kernels entry by entry."""
    from blml_kernels import ProductKernel
    from blml_local import rows, rows_multi, point_rows, point_rows_multi
    r_ = np.random.default_rng(0)

    def exact(A, B, w, kinds, fc, per):
        G = np.ones((len(A), len(B)))
        for kind, dims in kernel_units(kinds, A.shape[1]):
            G *= kernel_class(kind).eval(unit_theta(kind, fc, per, dims), [A[:, k, None] - B[None, :, k] for k in dims])
        return G @ w
    worst = 0.0; gpu_worst = {False: 0.0, True: 0.0}
    for kinds, per, fc in ((['sinc', 'sinc'], [None, None], np.array([1.0, 0.7])),
                           (['sinc', 'dirichlet'], [None, 3.0], np.array([1.0, 2.0])),
                           (['ball', 'ball', 'sinc'], [None, None, None], np.array([0.8, 0.8, 1.1]))):
        d = len(kinds); h = 0.04 / fc if kinds[0] == 'ball' else np.array([0.05, 3.0 / 60])
        Zi = r_.integers(-15, 15, size=(200, d)); w = r_.normal(size=200); units = kernel_units(kinds, d)
        want = []
        for ui, (kind, dims) in enumerate(units):
            want += [(ui, 'period' if kind == 'dirichlet' else 'fc')] + [(ui, ('shift', i)) for i in range(len(dims))]
        tab = ProductKernel(kinds, per).tables(fc, per, h, Zi, want)
        ZA = Zi[:6]; A, B = ZA * h, Zi * h; um = np.ones((2, len(units)), np.int64)
        got = [rows(ZA, Zi, w, tab, True)[1], rows_multi(ZA, Zi, np.c_[w, w], tab, um, True)[1][:, 1],
               point_rows(A, B, w, kinds, fc, per, want, True)[1],
               point_rows_multi(A, B, np.c_[w, w], kinds, fc, per, um, want, True)[1][:, 0]]
        if eng == 'gpu':
            import blml_local_gpu as G
            for f32 in (False, True):
                gg = [G.rows(ZA, Zi, w, tab, True, f32=f32)[1], G.rows_multi(ZA, Zi, np.c_[w, w], tab, um, True, f32=f32)[1][:, 1],
                      G.point_rows(A, B, w, kinds, fc, per, want, True, f32=f32)[1]]
                for a, b in zip(gg, got[:3]):
                    rel = float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))
                    gpu_worst[f32] = max(gpu_worst[f32], rel)
        for q, (ui, p) in enumerate(want):
            dims = units[ui][1]; e = 1e-6
            if isinstance(p, str):
                f2, p2 = fc.copy(), list(per)
                if p == 'fc':
                    f2[dims] *= np.exp(e)
                else:
                    p2[dims[0]] *= np.exp(e)
                fd = (exact(A, B, w, kinds, f2, p2) - exact(A, B, w, kinds, fc, per)) / e
            else:
                A2 = A.copy(); A2[:, dims[p[1]]] += e
                fd = (exact(A2, B, w, kinds, fc, per) - exact(A, B, w, kinds, fc, per)) / e
            sc = np.abs(fd).max() + 1e-12
            worst = max(worst, max(float(np.abs(gq[:, q] - fd).max()) / sc for gq in got))
    assert worst < 1e-3, f'kernel derivative off by {worst:.1e}'
    if eng == 'gpu':
        assert gpu_worst[False] < 1e-9 and gpu_worst[True] < 1e-3, f'GPU derivative kernels vs CPU: {gpu_worst}'
        return dict(fd_worst=worst, gpu64_vs_cpu=gpu_worst[False], gpu32_vs_cpu=gpu_worst[True])
    return dict(fd_worst=worst)


def run(eng, path):
    """RESUME=1: keep the cells already in path (a reclaimed VM loses no finished cell)."""
    only = [s for s in os.environ.get('CELLS', '').split(',') if s]
    res = json.load(open(path)) if (os.environ.get('RESUME') and os.path.exists(path)) else {}
    for name, fn in cells(eng):
        if only and not any(s in name for s in only):
            continue
        if name in res and res[name]['state'] != 'FAIL':
            continue
        t = time.time()
        try:
            v = fn()
            bad = [k for k, x in v.items() if not finite(x)]
            res[name] = dict(state='ok' if not bad else 'FAIL', s=round(time.time() - t, 1), **v,
                             **({'error': f'non-finite {bad}'} if bad else {}))
        except SkipCell as e:
            res[name] = dict(state='n/a', why=str(e))
        except Exception as e:
            msg = f'{type(e).__name__}: {e}'
            by = any(all(p in name for p in pats[:-1]) and pats[-1] in msg for pats in BY_DESIGN)
            res[name] = dict(state='by design' if by else 'FAIL', error=msg, s=round(time.time() - t, 1))
            if not by:
                traceback.print_exc(limit=3)
        r = res[name]
        print(f"{r['state']:9s} {name:45s} {r.get('error', '') or {k: v for k, v in r.items() if k not in ('state', 's')}}",
              flush=True)
        json.dump(res, open(path, 'w'), indent=1)
    n = {s: sum(r['state'] == s for r in res.values()) for s in ('ok', 'by design', 'n/a', 'FAIL')}
    print('SUMMARY', n, flush=True)


def compare(pa, pb):
    a, b = json.load(open(pa)), json.load(open(pb)); bad = 0
    for name in a:
        if name not in b or a[name]['state'] != 'ok' or b[name]['state'] != 'ok':
            if name in b and a[name]['state'] != b[name]['state']:
                print(f"FAIL {name}: cpu {a[name]['state']} gpu {b[name]['state']}"); bad += 1
            continue
        diffs = []
        for k, va in a[name].items():
            if k in ('state', 's', 'fd_worst', 'gpu64_vs_cpu', 'gpu32_vs_cpu'):
                continue
            vb = b[name][k]; tol = TOL.get(k, TOL['ll'])
            if k == 'fc' and 'rank 2' in name:            # rank K: the modes' cut-offs are not identified (the split
                continue                                  # of f between modes is flat); the log-likelihood is compared
            if k == 'fc':
                ok = np.allclose(va, vb, rtol=tol)
            else:
                ok = np.allclose(va, vb, atol=tol, rtol=tol)
            if not ok:
                diffs.append(f'{k}: {np.round(va, 4)} vs {np.round(vb, 4)}')
        bad += bool(diffs)
        print(('FAIL ' if diffs else 'PASS ') + name + ('' if not diffs else ': ' + '; '.join(diffs)))
    print('SUMMARY', bad, 'mismatches')


if __name__ == '__main__':
    if sys.argv[1] == 'compare':
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else f'capability_{sys.argv[1]}.json')
