"""Parity of the periodic two-slit band (kind 'dirichlet' with fill, token 'dirichlet:<fill>'): kernel against the
explicit harmonic sum, derivatives against finite differences, numba point rows against the kernel, fill = 1 against the
plain dirichlet band, and one fit through every CPU estimator (binned CD / L-BFGS, quick CD / L-BFGS, streaming; real
and complex). GPU cells: python results/pslits_parity.py gpu (on a GPU host) -> the same fits with engine='gpu'.
  python results/pslits_parity.py [cpu|gpu]   -> results/pslits_parity_<engine>.json"""
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import blml_trivial as bt
from blml_binned import BLMLBinned
ENG = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
out, ok_all = {}, True
def check(name, err, tol):
    global ok_all
    ok = bool(err <= tol); ok_all &= ok; out[name] = dict(err=float(err), tol=tol, ok=ok)
    print(f"{'ok ' if ok else 'BAD'} {name}: {err:.2e} (tol {tol:g})", flush=True)
P, fc, w = 392.0, 0.0214, 0.47
th = dict(fc=[fc], period=P)
K = bt.kernel_class(f'dirichlet:{w!r}'); K0 = bt.kernel_class('dirichlet')
t = np.random.default_rng(0).uniform(-600, 600, 4000)
N = bt.dirichlet_harmonics(fc, P); lo = int(np.ceil((1 - w) * fc * P / 2 - 1e-9))
ks = np.array([k for k in range(-N, N + 1) if abs(k) >= lo])
ref = np.real(np.exp(2j * np.pi * np.outer(t, ks) / P).sum(1)) / P
out['harmonics_kept'] = ks.tolist()
if ENG == 'cpu':
    check('kernel = harmonic sum', np.abs(K.eval(th, [t]) - ref).max(), 1e-10)
    check('fill 1 = plain dirichlet', np.abs(bt.kernel_class('dirichlet:1.0').eval(th, [t]) - K0.eval(th, [t])).max(), 1e-12)
    eps = 1e-5
    fd = (K.eval(th, [t + eps]) - K.eval(th, [t - eps])) / (2 * eps)
    check('d/dt = finite difference', np.abs(K.eval_dshift(th, [t]) - fd).max(), 1e-6)
    tp = np.random.default_rng(1).uniform(0, 300, 2000); dP = 1e-4          # d/dlog P at fixed harmonics, raw offsets
    Kp = lambda Pv: (np.real(np.exp(2j * np.pi * np.outer(tp, ks) / Pv).sum(1)) / Pv)
    fdp = (Kp(P * np.exp(dP)) - Kp(P * np.exp(-dP))) / (2 * dP)
    check('d/dlog P = finite difference', np.abs(K.eval_dlog(th, 'period', [tp]) - fdp).max(), 1e-6)
    row = K.lattice(th, [1.0], 0); j = np.arange(len(row)); jm = np.minimum(j, len(row) - j)
    check('lattice row = kernel', np.abs(row - K.eval(th, [jm.astype(float)])).max(), 1e-12)
    from blml_local import _point_args, _point_rows_numba, PD_DIR_PERIOD
    A = np.random.default_rng(2).uniform(0, P, (300, 1)); B = np.random.default_rng(3).uniform(0, P, (400, 1))
    wv = np.random.default_rng(4).normal(size=400); kinds = [f'dirichlet:{w!r}']
    want = [(0, 'period'), (0, 'shift')]
    s, ds = _point_rows_numba(A, B, wv, *_point_args(A, B, kinds, np.array([fc]), [P], want), True)
    D = A[:, 0][:, None] - B[:, 0][None, :]
    check('numba rows = kernel', np.abs(s - K.eval(th, [D.ravel()]).reshape(D.shape) @ wv).max(), 1e-9)
    check('numba d/dlog P = kernel', np.abs(ds[:, 0] - K.eval_dlog(th, 'period', [D.ravel()]).reshape(D.shape) @ wv).max(), 1e-8)
    check('numba d/dt = kernel', np.abs(ds[:, 1] - K.eval_dshift(th, [D.ravel()]).reshape(D.shape) @ wv).max(), 1e-8)
if ENG == 'gpu':                                     # the CUDA point rows (blml_local_gpu) against the kernel
    from blml_local_gpu import point_rows
    A = np.random.default_rng(2).uniform(0, P, (300, 1)); B = np.random.default_rng(3).uniform(0, P, (400, 1))
    wv = np.random.default_rng(4).normal(size=400); D = A[:, 0][:, None] - B[:, 0][None, :]
    res = point_rows(A, B, wv, [f'dirichlet:{w!r}'], np.array([fc]), [P], want=[(0, 'period'), (0, 'shift')], grad=True)
    s, ds = (np.asarray(v.get() if hasattr(v, 'get') else v) for v in res)
    check('cuda rows = kernel', np.abs(s - K.eval(th, [D.ravel()]).reshape(D.shape) @ wv).max(), 1e-9)
    check('cuda d/dlog P = kernel', np.abs(ds[:, 0] - K.eval_dlog(th, 'period', [D.ravel()]).reshape(D.shape) @ wv).max(), 1e-8)
    check('cuda d/dt = kernel', np.abs(ds[:, 1] - K.eval_dshift(th, [D.ravel()]).reshape(D.shape) @ wv).max(), 1e-8)
# one fit through every estimator: a two-slit periodic field on the circle, data on the lattice of 392 bins
W = int(P); x0 = np.arange(W)
g = np.exp(2j * np.pi * np.outer(x0, ks) / P) @ (np.random.default_rng(5).normal(size=len(ks)) + 1j * np.random.default_rng(6).normal(size=len(ks)))
pr = np.abs(g) ** 2 + 0.05 * np.abs(g).max() ** 2; pr /= pr.sum()
xs = np.random.default_rng(7).choice(W, size=20000, p=pr); T = np.bincount(xs, minlength=W).astype(float); Tc, xc = T, xs
modes = [{0: dict(kind='dirichlet', fc=fc, period=P, fill=w)}]
modes_bg = modes + [{0: dict(kind='dirichlet', fc=1.5 * 2 / P, period=P)}]
assert bt.make_modes([fc], [f'dirichlet:{w!r}'], periods=[P])[0] == bt.make_modes([fc], [f'dirichlet:{w!r}'], periods=[P])[0]
lls = {}
gr = np.cos(2 * np.pi * np.outer(x0, np.abs(ks[ks > 0])) / P) @ np.random.default_rng(8).normal(size=(ks > 0).sum())   # a real field
prr = gr ** 2 / (gr ** 2).sum(); xr = np.random.default_rng(9).choice(W, size=20000, p=prr); Tr = np.bincount(xr, minlength=W).astype(float)
for cg in (False, True):
    T, xs = (Tc, xc) if cg else (Tr, xr)                     # real fits: data from a real field (g^2, zeros included)
    for solver in ('cd', 'lbfgs'):
        for lab, md in (('signal', modes), ('signal+bg', modes_bg)):
            if not cg and lab == 'signal+bg':
                continue
            m = BLMLBinned(tol=1e-7, max_sweeps=20000, ll_tol=1e-12, engine=ENG, dtype=np.float64, solver=solver).fit(T, md, complex_g=cg)
            lls[f'binned {solver} complex={cg} {lab}'] = float(m.loglik_)
    Xp = (xs + 0.0)[:, None]
    for solver in ('cd', 'lbfgs'):
        try:
            q = bt.BLMLQuick(engine=ENG, solver=solver, dtype=np.float64, tol=1e-7).fit(Xp, modes, complex_g=cg) if ENG == 'cpu' else \
                __import__('blml_gpu')._BLMLQuickCUDA(solver=solver, tol=1e-7).fit(Xp, modes, complex_g=cg)
            lls[f'quick {solver} complex={cg} signal'] = float(np.mean(np.log(np.maximum(q.predict(Xp), 1e-300))))
        except Exception as ex:
            lls[f'quick {solver} complex={cg} signal'] = f'ERR {type(ex).__name__}: {ex}'[:200]
out['ll'] = lls
for cg in (False, True):
    b = [v for k, v in lls.items() if f'complex={cg} signal' in k and 'signal+bg' not in k and isinstance(v, float)]
    if b:
        check(f'fit LL agrees across estimators (complex={cg})', max(b) - min(b), 2e-3)
    bb = [v for k, v in lls.items() if f'complex={cg} signal+bg' in k and isinstance(v, float)]
    if len(bb) > 1:
        check(f'fit LL agrees cd/lbfgs with bg (complex={cg})', max(bb) - min(bb), 1e-4)
for k, v in lls.items():
    print('  ', k, v)
try:
    from blml_binned import BLMLBinnedStreaming
    batches = [np.bincount(xs[i::10], minlength=W).astype(float) for i in range(10)]
    ms = BLMLBinnedStreaming(engine=ENG, tol=1e-6, max_sweeps=5000, solver='lbfgs').fit(batches, [{0: dict(kind='dirichlet', fc=dict(value=fc, trainable=False), period=P, fill=w)}],
                                                                                epochs=3, complex_g=True)
    out['stream_ll'] = float(np.sum(T * np.log(np.maximum(bt_np := np.asarray(ms.density_ if not hasattr(ms.density_, 'get') else ms.density_.get()), 1e-300))) / T.sum())
    check('streaming = batch (complex)', abs(out['stream_ll'] - lls['binned lbfgs complex=True signal']), 5e-3)
except Exception as ex:
    out['stream_ll'] = f'ERR {type(ex).__name__}: {ex}'[:300]; print('streaming', out['stream_ll']); ok_all = False
out['ok'] = ok_all
json.dump(out, open(f'results/pslits_parity_{ENG}.json', 'w'), indent=1)
print('ALL OK' if ok_all else 'SOME CHECKS FAILED')
