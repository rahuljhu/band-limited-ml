"""One GPU run, accuracy first: (1) the touched-bin kernels (blml_local_gpu) against the numba ones on random
inputs -- sinc, folded and unfolded Dirichlet, radial ball, derivatives, complex weights; (2) results/parity.py
gpu (the 17 library checks), written to the json path given. Prints PASS / FAIL lines. Small: a few minutes.
Usage: gpu_check.py out.json"""
import os, sys, time, json, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import blml_trivial as bt
bt.DIRICHLET_LEGACY = False
import blml_local as L
import blml_local_gpu as G

rng = np.random.default_rng(0)
ok_all = True


def check(name, a, b, tol=1e-9):
    global ok_all
    a, b = np.asarray(a), np.asarray(b)
    err = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))
    ok = err <= tol
    ok_all &= ok
    print(f"{'PASS' if ok else 'FAIL'} kernel {name}: rel err {err:.1e}", flush=True)


cases = [(['sinc', 'sinc'], [None, None], [0.4, 0.7], [(0, 'fc'), (1, 'fc')], [True, True]),
         (['sinc', 'dirichlet', 'sinc'], [None, 24.0, None], [0.5, 0.3, 0.6], [(0, 'fc'), (1, 'period')], [True, True, True]),
         (['sinc', 'dirichlet'], [None, 24.0], [0.5, 0.3], [(0, 'fc'), (1, 'period')], [True, False]),
         (['ball', 'ball', 'sinc'], [None, None, None], [0.5, 0.5, 0.7], [(0, 'fc'), (1, 'fc')], [True] * 3)]
for kinds, per, fc, want, fold in cases:
    d = len(kinds); fc = np.array(fc)
    h = np.where(np.array([k == 'dirichlet' for k in kinds]), 24.0 / 96, 0.1 / fc)
    ZB = rng.integers(0, 60, (3000, d)); ZA = ZB[rng.choice(3000, 200, replace=False)]
    for k in range(d):
        if kinds[k] == 'dirichlet' and fold[k]:
            ZB[:, k] %= 96; ZA[:, k] %= 96
    w = rng.normal(size=3000)
    tab = L.lattice_tables(kinds, fc, per, h, ZB.max(0) - ZB.min(0), want, fold)
    tag = '+'.join(kinds) + ('' if all(fold) else ' unfolded')
    A = rng.uniform(0, 6, (200, d)); Bp = rng.uniform(0, 6, (3000, d))
    for f32, tol in ((False, 1e-9), (True, 1e-4)):          # float32 arithmetic, float64 sums
        pr = ' f32' if f32 else ''
        for grad in (False, True):
            a, b = G.rows(ZA, ZB, w, tab, grad, f32=f32), L.rows(ZA, ZB, w, tab, grad)
            check(f'rows {tag} grad={grad}{pr}', a[0], b[0], tol); check(f'rows {tag} grad={grad} d{pr}', a[1], b[1], tol) if grad else None
        a = G.point_rows(A, Bp, w, kinds, fc, per, want, True, f32=f32); b = L.point_rows(A, Bp, w, kinds, fc, per, want, True)
        check(f'points {tag}{pr}', a[0], b[0], tol); check(f'points {tag} d{pr}', a[1], b[1], tol)
wc = rng.normal(size=3000) + 1j * rng.normal(size=3000)          # complex weights through LocalFit's wrapper
fg, fc_ = L._complex_ok(G.point_rows), L._complex_ok(L.point_rows)
A = rng.uniform(0, 6, (200, 2)); Bp = rng.uniform(0, 6, (3000, 2))
check('points complex', fg(A, Bp, wc, ['sinc', 'sinc'], np.array([0.4, 0.7]), [None, None])[0],
      fc_(A, Bp, wc, ['sinc', 'sinc'], np.array([0.4, 0.7]), [None, None])[0])
# end to end in float32 on the GPU: touched-bin streaming at a fixed fc against the exact float64 fit
X2 = np.c_[rng.normal(size=4000), rng.normal(size=4000) * 0.7]; Q2 = np.c_[rng.normal(size=1000), rng.normal(size=1000) * 0.7]
from blml_rolling import BLMLQuickStreaming
md = [{j: dict(kind='sinc', fc=dict(value=1.5, trainable=False)) for j in range(2)}]
ll = lambda m: float(np.mean(np.log(np.maximum(m.predict(Q2), 1e-300))))
s32 = BLMLQuickStreaming(engine='gpu', burn_in=1000, dtype=np.float32).fit(X2, md, epochs=3, batch=500)
ex = bt.BLMLQuick(dtype=np.float64).fit(X2, bt.make_modes([1.5, 1.5], ['sinc', 'sinc']))
d32 = abs(ll(s32) - ll(ex)); ok_all &= d32 < 2e-3
print(f"{'PASS' if d32 < 2e-3 else 'FAIL'} f32 streaming vs exact: {ll(s32):.4f} vs {ll(ex):.4f}", flush=True)
print('KERNELS', 'ALL PASS' if ok_all else 'SOME FAIL', flush=True)

import parity                                                    # the library checks, engine='gpu'
sys.argv = ['parity.py', 'gpu', sys.argv[1] if len(sys.argv) > 1 else 'parity_gpu.json']
parity.run('gpu', sys.argv[2])
print('GPU CHECK DONE', flush=True)
