"""Trainable fc (Adam), T1 d=4 n=20k, 4 epochs: each streaming ingredient switched off (patched here; library untouched).
  base        the library
  no_removal  revisits not scored leave-batch-out
  no_qfix     Q not re-estimated from the sampled rows (q_rate 0: Q only as cd tracks it)
  no_dq       the normaliser term dQ dropped from the gradient
usage: stream_ablation.py variant [variant ...]"""
import sys, time, numpy as np
sys.path.insert(0, 'results/rolling')
import blml_trivial as bt, blml_local as L, blml_rolling as BR
from blml_trivial import make_modes, modes_fc
from p1_truth import Truth
import p1_estimators as E
from synth_stream import eval_sets
bt.DIRICHLET_LEGACY = False
BS, INIT, STEP = L.FusedFit.batch_stats, L.FusedFit.__init__, BR.BLMLQuickStreaming._step

def no_removal(self, X, idx, bc, I, wt, want, revisit, *a, **kw):
    return BS(self, X, idx, bc, I, wt, want, False, *a, **kw)

def init_q0(self, *a, **kw):
    INIT(self, *a, **kw); self.q_rate = 0.0

def no_dq(self, *a, **kw):
    lp, SeI, st = BS(self, *a, **kw)
    for s in st or []:
        s['dQ'] = np.zeros_like(s['dQ'])
    return lp, SeI, st

PATCH = dict(base={}, no_removal={(L.FusedFit, 'batch_stats'): no_removal}, no_qfix={(L.FusedFit, '__init__'): init_q0},
             no_dq={(L.FusedFit, 'batch_stats'): no_dq})
ORIG = {(L.FusedFit, 'batch_stats'): BS, (L.FusedFit, '__init__'): INIT, (BR.BLMLQuickStreaming, '_step'): STEP}
d, n = 4, 20000; truth = Truth('T1', d); ev = eval_sets(truth, d)
for v in sys.argv[1:]:
    for rep in range(2):
        for (c, a), f in ORIG.items():
            setattr(c, a, PATCH[v].get((c, a), f))
        X = truth.sample(np.random.default_rng([3, d, n, rep]), n); out = []
        def cb(ep, s):
            r = E.metrics(s.predict(ev['Xf']), ev['ff'], s.predict(ev['Xq']), ev['fq'], ev['qq'], ev['int_f2'])
            out.append(f"{r['rel_ise']:.2e} fc{np.round(modes_fc(s.modes_)[0], 2).tolist()}")
        t0 = time.time()
        BR.BLMLQuickStreaming(search='auto', seed=rep, dtype=np.float64, rebin_tol=1.0).fit(X, make_modes(np.nan, ['sinc'] * d), epochs=4,
                                                                              batch=n // 20, on_epoch=cb)
        print(f'{v:10s} rep {rep} ({time.time() - t0:.0f}s): ' + ' | '.join(out), flush=True)
