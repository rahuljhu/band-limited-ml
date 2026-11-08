"""Paper 6: the streaming node projecting to k = 5 dims (more than the planted 2) from D = 10, 20 (and k = 2 for
comparison, from node_stream_dims.json). Planted checkerboard and quadrant XOR as node_stream_dims.py; starts:
the true plane (padded with random directions), SAVE (top k), random. Angles: between the planted plane and the
node's k-dim subspace (0: the plane lies in it). NODE_IGRAD=1: implicit_grad. NODE_KIND: the unit's kernel (sinc, ball). Writes results/node_stream_k5.json."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, warnings
warnings.filterwarnings('ignore')
from blml_rolling import BLMLQuickStreaming
import blml_trivial as bt
from blml_node import save
from node_stream_dims import make, N, NT

K = int(os.environ.get('NODE_K', 5))
def angles_in(W, B):                                   # planted plane vs the node's k-dim span
    Qw, Qb = np.linalg.qr(W)[0], np.linalg.qr(B)[0]
    return np.degrees(np.arccos(np.clip(np.linalg.svd(Qb.T @ Qw, compute_uv=False), -1, 1)))

out = {}
OUTF = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get('NODE_OUT', 'node_stream_k5.json'))
for D in [int(v) for v in os.environ.get('NODE_D', '10,20').split(',')]:
    for kind in os.environ.get('NODE_KINDS', 'checker,xor').split(','):
        Nn = int(os.environ.get('NODE_N', N))
        rng = np.random.default_rng(11); X, y, B = make(kind, Nn, D, rng); Xt, yt, _ = make(kind, NT, D, rng)
        Sv = np.linalg.qr(save(X - X.mean(0), y, K))[0]
        pad = np.linalg.qr(np.c_[B, np.random.default_rng(3).normal(size=(D, K - 2))])[0] if K > 2 else B
        rnd = np.linalg.qr(np.random.default_rng(int(os.environ.get('NODE_WSEED', 5))).normal(size=(D, K)))[0]
        sel = os.environ.get('NODE_STARTS', 'true plane,SAVE,random').split(',')
        for start, W0 in [(a, b) for a, b in (('true plane', pad), ('SAVE', Sv), ('random', rnd)) if a in sel]:
            sc_env = os.environ.get('NODE_FC', ','.join(['4'] * K))   # 'none': the burn-in's shared search picks fc
            ukind = os.environ.get('NODE_KIND', 'sinc')               # 'ball': one round band over the k outputs (one fc)
            fc0 = [float(v) for v in sc_env.split(',')] if sc_env != 'none' else None
            unit = dict(kind=ukind, W=W0, **({} if fc0 is None else dict(fc=fc0[0] if ukind == 'ball' else fc0)))
            modes = [{tuple(range(D)): unit, D: dict(kind='sinc', fc=dict(value=10000.0, trainable=False), dep=True)}]
            eng = os.environ.get('NODE_ENGINE', 'cpu'); ep = int(os.environ.get('NODE_EPOCHS', 8)); t = time.time()
            s = BLMLQuickStreaming(burn_in=1500, engine=eng, lr=0.05, trace_trainable=True,
                                   param_every=int(os.environ.get('NODE_PARAM_EVERY', 1)),
                                   map_norm=os.environ.get('NODE_MAP_NORM', 'off'),
                                   bcd_frac=float(os.environ.get('NODE_BCD', 1.0)), freeze_frac=float(os.environ.get('NODE_FREEZE', 0.0)),
                                   implicit_grad=os.environ.get('NODE_IGRAD') == '1',
                                   dtype=np.float32 if eng == 'gpu' else np.float64).fit(np.c_[X, y], modes, epochs=ep, batch=500)
            p1 = s.predict_ratio(np.c_[Xt, np.ones(NT)]); p0 = s.predict_ratio(np.c_[Xt, np.zeros(NT)])
            r = dict(D=D, k=K, n=Nn, epochs=ep, engine=eng, fc0=sc_env, kind=kind, start=start, acc=round(float(np.mean((p1 > p0) == (yt == 1))), 3),
                     angles=[round(float(a), 1) for a in angles_in(s.W_, B)],
                     unit_kind=ukind, fc=np.round(bt.modes_fc(s.modes_)[0][:K], 2).tolist(), s=round(time.time() - t, 1),
                     trace=[dict(angles=[round(float(v), 2) for v in angles_in(h['trainable'][tuple(range(D))]['W'], B)],
                                 fc=np.round(h['trainable'][tuple(range(D))]['fc'], 3).tolist(),
                                 preq=round(float(h['preq']), 4)) for h in s.history_ if 'trainable' in h])
            out[f'{D}|{kind}|{start}'] = r
            print(f"n={Nn} epochs={ep} {eng} fc0={sc_env} D={D:2d} k={K} {kind:7s} start={start:10s} acc {r['acc']:.3f} angles {r['angles']} fc {r['fc']} ({r['s']:.0f}s)", flush=True)
            json.dump(out, open(OUTF, 'w'), indent=1)
print('DONE', flush=True)
