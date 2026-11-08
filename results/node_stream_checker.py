"""Paper 6: the streaming dependence node (BLMLQuickStreaming with a mapped sinc unit u = x W, binary label as a dep column at a
fixed integer cut-off, score='cond_loglik') on the planted checkerboard and quadrant XOR in 50-d, n = 5000 (+2000
held out), CPU. Starts: random, SAVE, and the true plane (a diagnostic: does streaming keep it, at what scale).
Writes results/node_stream_checker.json."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, warnings
warnings.filterwarnings('ignore')
from blml_rolling import BLMLQuickStreaming
from blml_node import save
from node_stage1 import make, angles, P

N, NT = 5000, 2000
out = {}
for kind in ('checker', 'xor'):
    rng = np.random.default_rng(11)
    X, y, B = make(kind, N, rng); Xt, yt, _ = make(kind, NT, rng)
    starts = {'random': None, 'SAVE': np.linalg.qr(save(X - X.mean(0), y, 2))[0], 'true plane': B}
    dep = dict(kind='sinc', fc=dict(value=10000.0, trainable=False), dep=True)
    for name, Q0 in starts.items():
        unit = dict(kind='sinc', fc=0.5, **(dict(k=2) if Q0 is None else dict(W=Q0)))
        t = time.time()
        s = BLMLQuickStreaming(burn_in=1500, lr=0.1, param_every=1, seed=5, dtype=np.float64).fit(
            np.c_[X, y], [{tuple(range(P)): unit, P: dep}], epochs=8, batch=500)
        p1 = s.predict_ratio(np.c_[Xt, np.ones(NT)]); p0 = s.predict_ratio(np.c_[Xt, np.zeros(NT)])
        acc = float(np.mean((p1 > p0) == (yt == 1)))
        a = angles(s.W_, B)
        r = dict(acc=round(acc, 3), angles=[round(float(v), 1) for v in a], fc=np.round(s._fc[:2], 3).tolist(), preq_first=round(s.preq_[0], 3), preq_last=round(float(np.mean(s.preq_[-5:])), 3),
                 s=round(time.time() - t, 1))
        out[f'{kind}|{name}'] = r
        print(f"{kind:7s} start={name:10s} acc {r['acc']:.3f} angles {r['angles']} fc {r['fc']} "
              f"preq {r['preq_first']} -> {r['preq_last']} ({r['s']:.0f}s)", flush=True)
        json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'node_stream_checker.json'), 'w'), indent=1)
print('DONE')
