"""The derivative-free row passes: FusedFit._rm_block (solver block kernel + matrix product) against the fused rows
kernel, every kind x real / complex / rank 2 / conditional (joint + marginal), at the fit's parameters and at others
(fc x 1.3: the tables a parameter step or a hill-climb candidate uses). Usage: block_rows_check.py [cpu|gpu]"""
import sys, numpy as np
import os
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)) + '/..', os.path.dirname(os.path.abspath(__file__))]
import blml_trivial as bt, blml_local as L

from blml_rolling import BLMLQuickStreaming
from capability_matrix import DATA, YLAB, base_mode, with_dep
bt.DIRICHLET_LEGACY = False
eng = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
rng = np.random.default_rng(0); worst = 0.0
for kind in ('sinc', 'dirichlet', 'ball', 'mapped-sinc'):
    X = DATA['mapped'] if kind.startswith('mapped') else DATA[kind]; d = X.shape[1]
    cases = {'real': ([base_mode(kind)], X, {}), 'complex': ([base_mode(kind)], X, dict(complex_g=True)),
             'rank 2': ([base_mode(kind)] * 2, X, {})}
    if not kind.startswith('mapped'):
        cases['conditional'] = ([with_dep(base_mode(kind), d)], np.c_[X, YLAB + 0.4 * rng.normal(size=len(X))], {})
    for name, (md, Xi, kw) in cases.items():
        s = BLMLQuickStreaming(engine=eng, burn_in=800, dtype=np.float64, **kw).fit(Xi, md, epochs=1, batch=400); ff = s.ff
        I = np.sort(rng.choice(len(ff.cnt), 300, replace=False))
        for k in range(ff.K):
            tab = ff.tables_k(k); E = np.ascontiguousarray(ff.e[:, :, k])
            a = ff._rm(ff.Zi[I], ff.Zi, E, tab, k, grad=True)[0]       # the fused rows kernel (this engine)
            b = ff._rm_block(ff.Zi[I], ff.Zi, E, tab, k)[0]
            rel = float(np.abs(a - b).max() / np.abs(a).max()); worst = max(worst, rel)
            fcS = ff.fcK * 1.3                                        # tables at other parameters (Adam step, hill-climb)
            tabS = ff.tables_k(k, (), fcS[k], ff.periodsK[k])
            aS = ff._rm(ff.Zi[I], ff.Zi, E, tabS, k, grad=True)[0]
            bS = ff._rm(ff.Zi[I], ff.Zi, E, tabS, k, fcK=fcS, periodsK=ff.periodsK)[0]
            relS = float(np.abs(aS - bS).max() / np.abs(aS).max()); worst = max(worst, relS)
            print(f'{kind:12s} {name:11s} mode {k} M={ff.M}: max rel diff {rel:.1e}, at fc x 1.3 {relS:.1e}')
print('WORST', worst)
