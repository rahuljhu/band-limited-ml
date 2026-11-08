"""The held-out conditional score vs fc at a fixed W, for the hidden-plane data of fc_epochs_t1proj.py (same seed, D,
n): W = [P0, cos(a) P1 + sin(a) q] with q a noise direction, a = 0 / 11 / 30 / 90 deg (11: where the 20-d run's second
direction stalls with fc ~3). One-shot BLMLQuick fits of (u1, u2, y) at fixed fc = (FC1, fc2, 10000) over a grid of fc2;
score = mean log p(y | u) on fresh test data (the joint normalised over y), and the accuracy. Tests whether (W at 11 deg,
fc2 ~3) is a local optimum: best fc2 at 11 deg high, while at 0 deg fc2 ~0.4 beats it.
Env: T1P_D (20), T1P_N (20000), T1P_SHIFT (3), LS_FC1 (0.35), LS_ANGLES (0,11,30,90). Writes
results/t1proj_fc_landscape_D<D>[_a<angles>].json."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np, warnings
warnings.filterwarnings('ignore')
from p1_truth import sample_f1
import blml_trivial as bt

D, N = int(os.environ.get('T1P_D', 20)), int(os.environ.get('T1P_N', 20000))
SHIFT, FC1 = float(os.environ.get('T1P_SHIFT', 3.0)), float(os.environ.get('LS_FC1', 0.35))
rng = np.random.default_rng(0)                                       # as fc_epochs_t1proj.py
Rot = np.linalg.qr(rng.normal(size=(D, D)))[0]
P = Rot[:, :2]


def draw(m):
    y = np.r_[np.zeros(m // 2), np.ones(m - m // 2)]
    Z = np.stack([sample_f1(rng, m) for _ in range(D)], 1); Z[:, :2] += SHIFT * y[:, None]
    p = rng.permutation(m)
    return (Z @ Rot.T)[p], y[p]


X, y = draw(N); Xt, yt = draw(20000)
modes = lambda fc2: bt.make_modes([FC1, fc2, 10000.0], ['sinc'] * 3)
rows = []
ANG = os.environ.get('LS_ANGLES', '0,11,30,90')
for a in [float(x) for x in ANG.split(',')]:
    r = np.radians(a)
    W = np.c_[P[:, 0], np.cos(r) * P[:, 1] + np.sin(r) * Rot[:, 2]]
    U, Ut = X @ W, Xt @ W
    for fc2 in (0.2, 0.3, 0.4, 0.6, 0.8, 1.2, 1.6, 2.4, 3.2):
        t = time.time()
        m = bt.BLMLQuick(dtype=np.float64).fit(np.c_[U, y], modes(fc2))
        f0 = np.maximum(m.predict(np.c_[Ut, np.zeros(len(Ut))]).ravel(), 1e-300)
        f1 = np.maximum(m.predict(np.c_[Ut, np.ones(len(Ut))]).ravel(), 1e-300)
        ll = float(np.mean(np.where(yt == 1, np.log(f1), np.log(f0)) - np.log(f0 + f1)))
        acc = float(np.mean((f1 > f0) == (yt == 1)))
        rows.append(dict(angle=a, fc2=fc2, cll=ll, acc=acc))
        print(f"angle {a:4.0f}  fc2 {fc2:4.1f}  cond ll {ll:.4f}  acc {acc:.4f}  ({time.time() - t:.0f}s)", flush=True)
json.dump(dict(D=D, n=N, fc1=FC1, rows=rows), open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                           f't1proj_fc_landscape_D{D}' + ('' if 'LS_ANGLES' not in os.environ else '_a' + ANG.replace(',', '-')) + '.json'), 'w'), indent=1)
