"""One GPU run for the streaming additions (fused fit, score switch, projection), accuracy first:
(1) the CUDA cross sums against numba (float64, float32);
(2) results/parity.py gpu (the library checks, incl. fused conditional streaming), json to the path given;
(3) score='dependence' on 2-d rings (no projection), GPU and CPU side by side;
(4) the mapped-sinc node (10-d hidden rings, n = 4000, 3 epochs), GPU and CPU side by side.
Usage: gpu_check2.py parity_out.json   Prints PASS / FAIL and RESULT lines. Small: a few minutes."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, warnings
warnings.filterwarnings('ignore')
import blml_trivial as bt
bt.DIRICHLET_LEGACY = False
import blml_local as L, blml_local_gpu as G
from blml_rolling import BLMLQuickStreaming

rng = np.random.default_rng(0)
A = rng.normal(size=(300, 3)); B = rng.normal(size=(3000, 3)); W = rng.normal(size=(3000, 2)); fc = np.array([0.7, 0.9, 1.3])
c = L.cross_sums(A, B, W, fc)
for f32, tol in ((False, 1e-7), (True, 1e-5)):          # float64: summation order only (atomics vs sequential)
    g = G.cross_sums(A, B, W, fc, f32=f32)
    e = max(np.abs(g[i] - c[i]).max() / np.abs(c[i]).max() for i in (0, 1))
    print(f"{'PASS' if e < tol else 'FAIL'} cross_sums f32={f32}: rel err {e:.1e}", flush=True)

import parity
parity.run('gpu', sys.argv[1] if len(sys.argv) > 1 else 'parity_gpu.json')

modes2 = [{0: dict(kind='sinc', fc=0.4), 1: dict(kind='sinc', fc=0.4), 2: dict(kind='sinc', fc=dict(value=10000.0, trainable=False), dep=True)}]
r2 = np.random.default_rng(1)
def rings(n, P=2, Q=None):
    Z = r2.normal(size=(n, P)); y = (np.hypot(Z[:, 0], Z[:, 1]) > 1.18).astype(float)
    return np.c_[(Z @ Q) if Q is not None else Z, y]
X, Xt = rings(8000), rings(3000)
for eng in ('cpu', 'gpu'):
    t = time.time()
    s = BLMLQuickStreaming(burn_in=2000, score='dependence', engine=eng, dtype=np.float64).fit(X, modes2, epochs=2, batch=1000)
    p1 = s.predict_ratio(np.c_[Xt[:, :2], np.ones(len(Xt))]); p0 = s.predict_ratio(np.c_[Xt[:, :2], np.zeros(len(Xt))])
    print('RESULT ' + json.dumps(dict(test='dependence rings 2-d', engine=eng, fc=np.round(bt.modes_fc(s.modes_)[0][:2], 3).tolist(),
                                      acc=round(float(np.mean((p1 > p0) == (Xt[:, 2] == 1))), 3), d2=round(s.preq_[-1], 4),
                                      s=round(time.time() - t, 1))), flush=True)

P = 10; Qr = np.linalg.qr(np.random.default_rng(99).normal(size=(P, P)))[0]
X, Xt = rings(4000, P, Qr), rings(3000, P, Qr); Bt = Qr[:2].T
modes = [{tuple(range(P)): dict(kind='sinc', k=2, fc=0.5),
          P: dict(kind='sinc', fc=dict(value=10000.0, trainable=False), dep=True)}]
for eng in ('cpu', 'gpu'):
    t = time.time()
    s = BLMLQuickStreaming(burn_in=1500, engine=eng, lr=0.1, param_every=1, seed=3, dtype=np.float64).fit(X, modes, epochs=3, batch=500)
    p1 = s.predict_ratio(np.c_[Xt[:, :P], np.ones(len(Xt))]); p0 = s.predict_ratio(np.c_[Xt[:, :P], np.zeros(len(Xt))])
    ang = np.degrees(np.arccos(np.clip(np.linalg.svd(s.W_.T @ Bt, compute_uv=False), -1, 1)))
    print('RESULT ' + json.dumps(dict(test='projection node 10-d rings', engine=eng, angles=np.round(ang, 1).tolist(),
                                      fc=np.round(bt.modes_fc(s.modes_)[0][:2], 3).tolist(),
                                      acc=round(float(np.mean((p1 > p0) == (Xt[:, P] == 1))), 3), s=round(time.time() - t, 1))), flush=True)
print('GPU CHECK2 DONE', flush=True)
