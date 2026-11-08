"""Diagnose: held-out LL and relative ISE of BLMLQuick at fixed cut-offs, 2-D T1/T2, n = 1e4 (synthetic-study
samples, rep 0). Isotropic sweep and a 2-D grid. Writes results/synstream/fc_surface.json."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')
import numpy as np
import blml_trivial as bt
from blml_trivial import make_modes, BLMLQuick
from p1_truth import Truth, DefensiveProposal
bt.DIRICHLET_LEGACY = False
N, D = 10000, 2
G = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
out = {}
for kind in ('T1', 'T2'):
    tr = Truth(kind, D); rng = np.random.default_rng(20_000 + D)
    Xf = tr.sample(rng, 100000); ff = tr.pdf(Xf)
    q = DefensiveProposal(tr, scale=6.0); Xq = q.sample(rng, 200000); fq, qq = tr.pdf(Xq), q.pdf(Xq)
    X = tr.sample(np.random.default_rng([3, D, N, 0]), N)
    res = []
    for a in G:
        for b in G:
            m = BLMLQuick(dtype=np.float64).fit(X, make_modes([a, b], ['sinc'] * D))
            pf, pq = m.predict(Xf), m.predict(Xq)
            ll = float(np.mean(np.log(np.maximum(pf, 1e-300))))
            ise = float(np.mean((pq - fq) ** 2 / qq)) / float(ff.mean())
            res.append(dict(fc=[a, b], ll=ll, rel_ise=ise))
    out[kind] = res
    bl = max(res, key=lambda r: r['ll']); bi = min(res, key=lambda r: r['rel_ise'])
    print(kind, 'argmax LL', bl['fc'], f"{bl['ll']:.4f}", '| argmin ISE', bi['fc'], f"{bi['rel_ise']:.2e}", flush=True)
    for r in res:
        if r['fc'][0] == r['fc'][1]:
            print(f"  iso {r['fc'][0]:.2f}: LL {r['ll']:.4f}  relISE {r['rel_ise']:.2e}")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'synstream', 'fc_surface.json'), 'w'))
