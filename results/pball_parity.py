"""Parity of the periodic ball (kind 'pball': harmonics k / P with ||k / P|| <= fc / 2 over a group of periodic columns,
a tuple-key unit like 'ball'): kernel against the explicit harmonic sum, one column against dirichlet, d/dt against
finite differences, the row / column / point kernels (numba on the CPU, CUDA with 'gpu') against the kernel, and one
fit through every estimator on a 31 x 31 window: the real fits (binned and quick, CD and L-BFGS) reach one optimum;
the complex fits are checked by polishing (each solver started at another's optimum stays there; streaming's end point
is a stationary point of the batch fit), complex fits on the plane having many local maxima for any 2-d kernel.
  python results/pball_parity.py [cpu|gpu]   -> results/pball_parity_<engine>.json"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import blml_trivial as bt
from blml_binned import BLMLBinned
from blml_kernels import ProductKernel, BinnedGram
ENG = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
out, ok_all = {}, True


def check(name, err, tol):
    global ok_all
    ok = bool(err <= tol); ok_all &= ok; out[name] = dict(err=float(err), tol=tol, ok=ok)
    print(f"{'ok ' if ok else 'BAD'} {name}: {err:.2e} (tol {tol:g})", flush=True)


P, fc = 31.0, 0.33
K = bt.kernel_class('pball'); th = dict(fc=[fc, fc], period=np.array([P, P]))
Kv, sc = bt.pball_harmonics(fc, [P, P]); out['n_harmonics'] = int(len(Kv))
rng = np.random.default_rng(0)
t = rng.uniform(-40, 40, (2, 3000))
if ENG == 'cpu':
    ref = sc * np.cos(2 * np.pi * (Kv @ t) / P).sum(0)
    check('kernel = harmonic sum', np.abs(K.eval(th, [t[0], t[1]]) - ref).max(), 1e-12)
    x = rng.uniform(-500, 500, 2000)
    check('one column = dirichlet', np.abs(K.eval(dict(fc=[0.0214], period=np.array([392.0])), [x])
                                           - bt.kernel_class('dirichlet').eval(dict(fc=[0.0214], period=392.0), [x])).max(), 1e-12)
    eps = 1e-5
    for i in range(2):
        tp = [t[0] + (eps if i == 0 else 0), t[1] + (eps if i == 1 else 0)]
        tm = [t[0] - (eps if i == 0 else 0), t[1] - (eps if i == 1 else 0)]
        check(f'd/dt{i} = finite difference', np.abs(K.eval_dlog(th, ('shift', i), [t[0], t[1]])
                                                      - (K.eval(th, tp) - K.eval(th, tm)) / (2 * eps)).max(), 1e-8)
# row, column and point kernels against the kernel (with a sinc column alongside the group)
kinds, per = ['pball:0-1', 'pball:0-1', 'sinc'], [P, P, None]
fcv, h = np.array([fc, fc, 0.2]), np.array([1.0, 1.0, 0.7])
pk = ProductKernel(kinds, per, fold=[True] * 3, engine=ENG)
ZA = np.c_[rng.integers(0, 31, 60), rng.integers(0, 31, 60), rng.integers(-20, 20, 60)]
ZB = np.c_[rng.integers(0, 31, 80), rng.integers(0, 31, 80), rng.integers(-20, 20, 80)]
want = [(1, ('shift', 0)), (1, ('shift', 1)), (0, 'fc')]
tab = pk.tables(fcv, per, h, np.r_[ZA, ZB], want=want)
G = BinnedGram(ProductKernel(kinds, per, fold=[True] * 3), fcv, h).gram_int(ZA, ZB)
w = rng.normal(size=80)
thg, ths = bt.unit_theta('pball:0-1', fcv, per, [0, 1]), bt.unit_theta('sinc', fcv, per, [2])
D = [(ZA[:, k, None] - ZB[None, :, k]) * h[k] for k in range(3)]
Ks = bt.kernel_class('sinc'); Ksv = Ks.eval(ths, [D[2]])
drefs = [K.eval_dlog(thg, ('shift', 0), D[:2]) * Ksv, K.eval_dlog(thg, ('shift', 1), D[:2]) * Ksv,
         K.eval(thg, D[:2]) * Ks.eval_dlog(ths, 'fc', [D[2]])]
A = rng.uniform(0, 31, (50, 3)); B = rng.uniform(0, 31, (70, 3)); wv = rng.normal(size=70)
DD = [A[:, k, None] - B[None, :, k] for k in range(3)]
pref = [(K.eval(thg, DD[:2]) * Ks.eval(ths, [DD[2]])) @ wv,
        (K.eval_dlog(thg, ('shift', 0), DD[:2]) * Ks.eval(ths, [DD[2]])) @ wv,
        (K.eval_dlog(thg, ('shift', 1), DD[:2]) * Ks.eval(ths, [DD[2]])) @ wv,
        (K.eval(thg, DD[:2]) * Ks.eval_dlog(ths, 'fc', [DD[2]])) @ wv]
get = lambda v: np.asarray(v.get() if hasattr(v, 'get') else v)
if ENG == 'cpu':
    from blml_local import rows, cols, point_rows
else:
    from blml_local_gpu import rows, cols, point_rows
s, ds = (get(v) for v in rows(ZA, ZB, w, tab, grad=True))
check(f'{ENG} rows = Gram', np.abs(s - G @ w).max(), 1e-12)
for q, nm in enumerate(['d/dt0', 'd/dt1', 'sinc d/dlog fc']):
    check(f'{ENG} rows {nm} = kernel', np.abs(ds[:, q] - drefs[q] @ w).max(), 1e-12)
if ENG == 'gpu':                                   # the device kernel takes rank coordinates on the device
    import cupy as cp
    from blml_kernels import rank_coords
    o = cp.zeros((60, 80)); cols(cp.asarray(rank_coords(ZA, tab)), cp.asarray(rank_coords(ZB, tab)), tab, o)
else:
    o = np.empty((60, 80)); cols(ZA, ZB, tab, o)
check(f'{ENG} cols = Gram', np.abs(get(o) - G).max(), 1e-12)
sp, dsp = (get(v) for v in point_rows(A, B, wv, kinds, fcv, per, want=want, grad=True))
check(f'{ENG} point rows = kernel', np.abs(sp - pref[0]).max(), 1e-10)
for q, nm in enumerate(['d/dt0', 'd/dt1', 'sinc d/dlog fc']):
    check(f'{ENG} point {nm} = kernel', np.abs(dsp[:, q] - pref[q + 1]).max(), 1e-10)
# one fit through every estimator: a field with the disk's harmonics on the 31 x 31 window
W = int(P); yy, xx = np.meshgrid(np.arange(W), np.arange(W), indexing='ij')
coef = rng.normal(size=len(Kv)) + 1j * rng.normal(size=len(Kv))
g = np.exp(2j * np.pi * (np.outer(yy.ravel(), Kv[:, 0]) + np.outer(xx.ravel(), Kv[:, 1])) / P) @ coef
pr = np.abs(g) ** 2 + 0.05 * np.abs(g).max() ** 2; pr /= pr.sum()
idx = rng.choice(W * W, size=20000, p=pr); Tc = np.bincount(idx, minlength=W * W).reshape(W, W).astype(float)
gr = np.real(g); prr = gr ** 2 / (gr ** 2).sum(); idr = rng.choice(W * W, size=20000, p=prr)
Tr = np.bincount(idr, minlength=W * W).reshape(W, W).astype(float)
modes = [{(0, 1): dict(kind='pball', fc=fc, period=[P, P])}]
modes_bg = modes + [{(0, 1): dict(kind='pball', fc=1.5 * 2 / P, period=[P, P])}]
lls = {}
# real fits: one start, every estimator the same optimum. Complex fits on the plane have many local maxima (for any
# 2-d kernel, ball included, and a single complex mode is never certified there: a sum of squares fits finite data
# better), so their parity is the polish test: each solver started at another's optimum stays there.
T, ix = Tr, idr
for solver in ('cd', 'lbfgs'):
    lls[f'binned {solver} real'] = float(BLMLBinned(tol=1e-7, max_sweeps=20000, ll_tol=1e-12, engine=ENG, dtype=np.float64,
                                                     solver=solver).fit(T, modes).loglik_)
Xr = np.c_[idr // W, idr % W].astype(float)
Quick = (lambda solver: bt.BLMLQuick(engine=ENG, solver=solver, dtype=np.float64, tol=1e-7)) if ENG == 'cpu' else \
    (lambda solver: __import__('blml_gpu')._BLMLQuickCUDA(solver=solver, tol=1e-7))
qll = lambda q, X: float(np.mean(np.log(np.maximum(get(q.predict(X)), 1e-300))))
for solver in ('cd', 'lbfgs'):
    lls[f'quick {solver} real'] = qll(Quick(solver).fit(Xr, modes), Xr)
b = list(lls.values())
check('real fits agree across estimators', max(b) - min(b), 1e-4)
Xc = np.c_[idx // W, idx % W].astype(float)
for lab, md in (('signal', modes), ('signal+bg', modes_bg)):
    a = BLMLBinned(tol=1e-8, max_sweeps=20000, ll_tol=1e-13, engine=ENG, dtype=np.float64, solver='lbfgs').fit(Tc, md, complex_g=True)
    c = BLMLBinned(tol=1e-8, max_sweeps=20000, ll_tol=1e-13, engine=ENG, dtype=np.float64, solver='cd').fit(Tc, md, complex_g=True, init=a)
    a2 = BLMLBinned(tol=1e-8, max_sweeps=20000, ll_tol=1e-13, engine=ENG, dtype=np.float64, solver='lbfgs').fit(Tc, md, complex_g=True, init=c)
    lls[f'binned complex {lab}: lbfgs / cd from it / lbfgs from that'] = [float(a.loglik_), float(c.loglik_), float(a2.loglik_)]
    check(f'binned complex {lab}: cd polish of the lbfgs optimum', abs(c.loglik_ - a.loglik_), 1e-5)
    check(f'binned complex {lab}: lbfgs polish of the cd optimum', abs(a2.loglik_ - c.loglik_), 1e-5)
try:
    q1 = Quick('lbfgs').fit(Xc, modes, complex_g=True); q2 = Quick('cd').fit(Xc, modes, complex_g=True, init=q1)
    lls['quick complex: lbfgs / cd from it'] = [qll(q1, Xc), qll(q2, Xc)]
    check('quick complex: cd polish of the lbfgs optimum', abs(qll(q2, Xc) - qll(q1, Xc)), 1e-5)
except Exception as ex:
    lls['quick complex'] = f'ERR {type(ex).__name__}: {ex}'[:200]; print('quick complex', lls['quick complex']); ok_all = False
out['ll'] = lls
for k, v in lls.items():
    print('  ', k, v)
try:                        # streaming: its end point is a stationary point of the batch fit (both batch solvers started
    from blml_binned import BLMLBinnedStreaming   # there stay); which local maximum it reaches depends on its path
    batches = [np.bincount(idx[i::10], minlength=W * W).reshape(W, W).astype(float) for i in range(10)]
    mdf = [{(0, 1): dict(kind='pball', fc=dict(value=fc, trainable=False), period=[P, P])}]
    ms = BLMLBinnedStreaming(engine=ENG, tol=1e-7, max_sweeps=20000, solver='lbfgs').fit(batches, mdf, epochs=3, complex_g=True)
    sl = float(np.sum(Tc * np.log(np.maximum(get(ms.density_), 1e-300))) / Tc.sum()); out['stream_ll'] = sl
    for solver in ('lbfgs', 'cd'):
        mb = BLMLBinned(tol=1e-8, max_sweeps=30000, ll_tol=1e-13, engine=ENG, solver=solver).fit(Tc, mdf, complex_g=True, init=ms)
        check(f'streaming end point is a batch stationary point ({solver} stays)', abs(float(mb.loglik_) - sl), 1e-5)
except Exception as ex:
    out['stream_ll'] = f'ERR {type(ex).__name__}: {ex}'[:300]; print('streaming', out['stream_ll']); ok_all = False
out['ok'] = ok_all
json.dump(out, open(f'results/pball_parity_{ENG}.json', 'w'), indent=1)
print('ALL OK' if ok_all else 'SOME CHECKS FAILED')
