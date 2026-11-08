"""GPU row kernels vs the CPU ones (refactor stages 2-4): rows, rows_multi (unit masks), cols (with the joint +
marginal pass), point_rows, and ProductKernel tables built on the device, on small lattices covering every column
kind (offset sinc rows, a rank column, folded dirichlet, a radial group) with derivative tables (fc, period,
'shift'), float64 and float32. Prints PASS / FAIL per check; exits non-zero on a failure. Seconds on a T4."""
import os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np
import cupy as cp
import blml_local as L
import blml_local_gpu as G
from blml_kernels import ProductKernel, rank_coords
from blml_trivial import kernel_units

rng = np.random.default_rng(0)
fails = []


def check(name, a, b, tol):
    a, b = np.asarray(a), np.asarray(b)
    err = float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(a))), 1e-300))
    ok = a.shape == b.shape and err < tol
    print(f"{'PASS' if ok else 'FAIL'} {name}: shape {a.shape} vs {b.shape}, rel err {err:.1e}", flush=True)
    if not ok:
        fails.append(name)


CASES = {
    'sinc + rank label': (['sinc', 'sinc', 'sinc'], [None, None, None],
                          lambda n: np.c_[rng.integers(-40, 40, n), rng.integers(-30, 30, n), rng.integers(0, 3, n) * 700000],
                          np.array([0.7, 0.9, 10000.0]), np.array([0.02, 0.03, 1e-6])),
    'sinc + dirichlet': (['sinc', 'dirichlet'], [None, 24.0],
                         lambda n: np.c_[rng.integers(-50, 50, n), rng.integers(0, 48, n)],
                         np.array([0.8, 0.3]), np.array([0.02, 0.5])),
    'radial + sinc': (['ball:0', 'ball:0', 'sinc'], [None, None, None],
                      lambda n: np.c_[rng.integers(-30, 30, n), rng.integers(-30, 30, n), rng.integers(-20, 20, n)],
                      np.array([0.6, 0.6, 0.9]), np.array([0.03, 0.03, 0.03])),
}

for cname, (kinds, periods, draw, fc, h) in CASES.items():
    try:
        d = len(kinds); Z = np.unique(draw(400).astype(np.int64), axis=0); B = len(Z)
        units = kernel_units(kinds, d)
        want = [(ui, 'fc') for ui, (kind, _) in enumerate(units) if kind == 'sinc' or kind.startswith('ball')]
        want += [(ui, 'period') for ui, (kind, _) in enumerate(units) if kind == 'dirichlet']
        want += [(ui, 'shift') for ui, (kind, dims) in enumerate(units) if kind == 'sinc' and dims == [0]]
        w = rng.normal(size=B); Wt = rng.normal(size=(B, 2))
        umask = np.array([[1] * len(units), [0 if dims == [d - 1] else 1 for _, dims in units]], np.int64)
        tc = ProductKernel(kinds, periods).tables(fc, periods, h, Z, want)
        for f32, tol in ((False, 1e-9), (True, 2e-4)):
            tg = ProductKernel(kinds, periods, engine='gpu', f32=f32).tables(fc, periods, h, Z, want)
            tag = f'{cname} f32={f32}'
            check(f'{tag}: device table T', tc['T'], cp.asnumpy(tg['T']), 1e-6 if f32 else 1e-12)
            s0, d0 = L.rows(Z, Z, w, tc, grad=True); s1, d1 = G.rows(Z, Z, w, tg, grad=True, f32=f32)
            check(f'{tag}: rows', s0, s1, tol); check(f'{tag}: rows grad', d0, d1, tol)
            s0, d0 = L.rows_multi(Z, Z[:50], Wt[:50], tc, umask, grad=True); s1, d1 = G.rows_multi(Z, Z[:50], Wt[:50], tg, umask, grad=True, f32=f32)
            check(f'{tag}: rows_multi', s0, s1, tol); check(f'{tag}: rows_multi grad', d0, d1, tol)
            c0 = L.cols(Z, Z[:40], tc)
            out = cp.empty((B, 40), cp.float32 if f32 else cp.float64); out_m = cp.empty_like(out)
            ZA, ZB = cp.asarray(rank_coords(Z, tg)), cp.asarray(rank_coords(Z[:40], tg))
            G.cols(ZA, ZB, tg, out, out_m, depmask=1 << (d - 1), f32=f32)
            check(f'{tag}: cols (joint)', c0, cp.asnumpy(out), tol)
            if not kinds[d - 1].startswith('ball'):     # the marginal of the other columns
                tcm = ProductKernel(kinds[:-1], periods[:-1]).tables(fc[:-1], periods[:-1], h[:-1], Z[:, :-1])
                check(f'{tag}: cols (marginal, depmask)', L.cols(Z[:, :-1], Z[:40, :-1], tcm), cp.asnumpy(out_m), tol)
        A = rng.normal(size=(60, d)) * h * 20; Bp = Z[:80] * h
        p0 = L.point_rows(A, Bp, w[:80], kinds, fc, periods, [], False)[0]
        p1 = G.point_rows(A, Bp, w[:80], kinds, fc, periods, [], False)[0]
        check(f'{cname}: point_rows', p0, p1, 1e-9)
    except Exception:
        print(f'FAIL {cname}: exception', flush=True); traceback.print_exc(); fails.append(cname)

print('SUMMARY', 'all passed' if not fails else f'{len(fails)} failed: {fails}', flush=True)
sys.exit(1 if fails else 0)
