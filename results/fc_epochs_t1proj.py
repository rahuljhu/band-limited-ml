"""fc and W over epochs when the classes live in a hidden plane: 20-d data, two coordinates carry two classes (paper 1's
T1 product, class 1 shifted by SHIFT in both), the other 18 are T1 noise shared by the classes; then a random rotation
hides the plane. The node: a LinearMap unit 20 -> 2 from a random start, the class a dep column (fc 10000, fixed),
conditional score, map_norm off, param_every 1. Plain vs split (bcd_frac 0.5, freeze_frac 0). Prints, per epoch, fc
and the principal angles between W and the hidden plane (trace_trainable), and the final test accuracy vs Bayes.
Env: T1P_D (20), T1P_N (20000), T1P_EPOCHS (20), T1P_SHIFT (3), T1P_CFG, T1P_WSEED (1), T1P_WFIX=1 (W frozen at the true
plane: fc only), T1P_IGRAD=1 (implicit_grad: the coefficients' one-sweep response in the gradient). Writes
results/fc_epochs_t1proj_<cfg>_D<D>[_wfix][_igrad].json."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np, warnings
warnings.filterwarnings('ignore')
from p1_truth import sample_f1, f1, FC_TRUE
from blml_rolling import BLMLQuickStreaming

D, N, EP = int(os.environ.get('T1P_D', 20)), int(os.environ.get('T1P_N', 20000)), int(os.environ.get('T1P_EPOCHS', 20))
SHIFT, WSEED = float(os.environ.get('T1P_SHIFT', 3.0)), int(os.environ.get('T1P_WSEED', 1))
rng = np.random.default_rng(0)
Rot = np.linalg.qr(rng.normal(size=(D, D)))[0]                      # the hidden plane: Rot[:, :2]
P = Rot[:, :2]


def draw(m):
    y = np.r_[np.zeros(m // 2), np.ones(m - m // 2)]
    Z = np.stack([sample_f1(rng, m) for _ in range(D)], 1); Z[:, :2] += SHIFT * y[:, None]
    p = rng.permutation(m)
    return (Z @ Rot.T)[p], y[p], Z[p]


X, y, _ = draw(N); Xt, yt, Zt = draw(20000)
L0 = np.sum(np.log(f1(Zt[:, :2])), 1); L1 = np.sum(np.log(f1(Zt[:, :2] - SHIFT)), 1)
bayes_acc = float(np.mean((L1 > L0) == (yt == 1)))
angles = lambda W: np.degrees(np.arccos(np.clip(np.linalg.svd(P.T @ np.linalg.qr(W)[0], compute_uv=False), -1, 1)))
WFIX = os.environ.get('T1P_WFIX') == '1'                             # W frozen at the true plane: fc only
IGRAD = os.environ.get('T1P_IGRAD') == '1'
W0 = P.copy() if WFIX else np.linalg.qr(np.random.default_rng(WSEED).normal(size=(D, 2)))[0]
unit = dict(kind='sinc', W=dict(value=W0, trainable=not WFIX))
modes = [{tuple(range(D)): unit, D: dict(kind='sinc', fc=dict(value=10000.0, trainable=False), dep=True)}]
cfgs = {'plain': dict(), 'split': dict(bcd_frac=0.5, freeze_frac=0.0)}
which = os.environ.get('T1P_CFG', 'both')
print(f'Bayes acc {bayes_acc:.4f}; start angles {np.round(angles(W0), 1).tolist()}', flush=True)
key = tuple(range(D))
for name, kw in cfgs.items():
    if which not in ('both', name):
        continue
    t = time.time()
    s = BLMLQuickStreaming(burn_in=1500, lr=0.05, param_every=1, trace_trainable=True, dtype=np.float64,
                           implicit_grad=IGRAD, **kw).fit(np.c_[X, y], modes, epochs=EP, batch=500)
    hs = [h for h in s.history_ if 'trainable' in h]
    steps_ep = len(hs) / EP
    rows = []
    for e in range(EP):
        h = hs[min(len(hs) - 1, int(round((e + 1) * steps_ep)) - 1)]['trainable'][key]
        rows.append(dict(fc=np.round(np.ravel(h['fc']), 3).tolist(), angles=np.round(angles(h.get('W', W0)), 1).tolist()))
    q1 = s.predict_ratio(np.c_[Xt, np.ones(len(Xt))]); q0 = s.predict_ratio(np.c_[Xt, np.zeros(len(Xt))])
    acc = float(np.mean((q1 > q0) == (yt == 1)))
    out = dict(cfg=name, n=N, epochs=EP, shift=SHIFT, wseed=WSEED, acc=acc, bayes_acc=bayes_acc, s=round(time.time() - t, 1),
               per_epoch=rows)
    tag = f"{name}_D{D}" + ('_wfix' if WFIX else '') + ('_igrad' if IGRAD else '')
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'fc_epochs_t1proj_{tag}.json'), 'w'), indent=1)
    print(f"{name}: acc {acc:.4f} (Bayes {bayes_acc:.4f})  ({out['s']:.0f}s)", flush=True)
    for e, r in enumerate(rows):
        print(f"  epoch {e + 1:2d}: fc {r['fc']}  angles {r['angles']}", flush=True)
