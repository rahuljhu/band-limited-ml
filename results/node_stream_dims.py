"""Paper 6: the streaming dependence node against the ambient dimension D. Planted checkerboard and quadrant XOR in a
2-d plane, rotated at random into D = 5, 10, 20, 50 coordinates (the rest Gaussian noise); n = 5000 + 2000 held out;
BLMLQuickStreaming with a mapped sinc unit (u = x W, k = 2 outputs, fc trainable, started at 4), label as a dep column at
a fixed integer cut-off, 8 epochs of batches of 500, score 'cond_loglik' (and 'dependence' at D = 50). Starts: the
true plane, SAVE, random. Held-out accuracy of the node's own conditional and principal angles to the true plane.
Writes results/node_stream_dims.json."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, warnings
warnings.filterwarnings('ignore')
from blml_rolling import BLMLQuickStreaming
import blml_trivial as bt
from blml_node import save
from node_stage1 import angles

N, NT = 5000, 2000


def make(kind, n, D, rng):
    Z = rng.normal(size=(n, D))
    if kind == 'xor':
        y = (Z[:, 0] * Z[:, 1] > 0).astype(float)
    elif kind == 'rings':
        y = (np.hypot(Z[:, 0], Z[:, 1]) > 1.18).astype(float)
    elif kind == 'blobs':                               # shifted means along the first planted direction
        y = rng.integers(0, 2, n).astype(float); Z[:, 0] += 1.5 * (2 * y - 1)
    else:
        y = ((np.floor(Z[:, 0] / 0.75) + np.floor(Z[:, 1] / 0.75)) % 2).astype(float)
    Q = np.linalg.qr(np.random.default_rng(99 + D).normal(size=(D, D)))[0]
    return Z @ Q, y, Q[:2].T


if __name__ == '__main__':
    out = {}
    OUTF = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get('NODE_OUT', 'node_stream_dims.json'))
    runs = [(D, 'cond_loglik') for D in (5, 10, 20, 50)] + [(50, 'dependence')]
    if os.environ.get('NODE_RUNS'):                      # e.g. "5:cond_loglik,50:dependence"
        runs = [(int(r.split(':')[0]), r.split(':')[1]) for r in os.environ['NODE_RUNS'].split(',')]
    for D, score in runs:
        for kind in ('checker', 'xor'):
            rng = np.random.default_rng(11); X, y, B = make(kind, N, D, rng); Xt, yt, _ = make(kind, NT, D, rng)
            Sv = np.linalg.qr(save(X - X.mean(0), y, 2))[0]; rnd = np.linalg.qr(np.random.default_rng(5).normal(size=(D, 2)))[0]
            for start, W0 in (('true plane', B), ('SAVE', Sv), ('random', rnd)):
                modes = [{tuple(range(D)): dict(kind='sinc', W=W0, fc=4.0),
                          D: dict(kind='sinc', fc=dict(value=10000.0, trainable=False), dep=True)}]
                t = time.time()
                s = BLMLQuickStreaming(burn_in=1500, score=score, lr=0.05, param_every=1, dtype=np.float64).fit(
                    np.c_[X, y], modes, epochs=8, batch=500)
                p1 = s.predict_ratio(np.c_[Xt, np.ones(NT)]); p0 = s.predict_ratio(np.c_[Xt, np.zeros(NT)])
                r = dict(D=D, score=score, kind=kind, start=start, acc=round(float(np.mean((p1 > p0) == (yt == 1))), 3),
                         angles=[round(float(a), 1) for a in angles(s.W_, B)],
                         fc=np.round(bt.modes_fc(s.modes_)[0][:2], 2).tolist(), s=round(time.time() - t, 1))
                out[f'{D}|{score}|{kind}|{start}'] = r
                print(f"D={D:2d} {score:11s} {kind:7s} start={start:10s} acc {r['acc']:.3f} angles {r['angles']} fc {r['fc']} ({r['s']:.0f}s)", flush=True)
                json.dump(out, open(OUTF, 'w'), indent=1)
    print('DONE', flush=True)
