"""Paper 7, hidden-phase section: held-out score per count of the empirical estimators (+sqrt h, signed, complex, real
rank 2) against complex BLML (BLMLBinned) and the truth, on the sign-changing synthetic truths of binned_learn_checks;
and the alternation's convergence. Output results/empirical_phase_table.json"""
import sys, json, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_trivial import kernel_class
from blml_binned import BLMLBinned, BLMLEmpirical, _np
exec(open('results/binned_learn_checks.py').read().split('def fd_check')[0].split('rows = []')[1])
sc = lambda p, te: float((te * np.log(np.maximum(p / p.sum(), 1e-300))).sum() / te.sum())
res = {}
for name, p, fc, kinds in [('1d sinc', truth((64,), 'sinc', 0.25), [0.25], ['sinc']),
                           ('2d ball', truth((32, 32), 'ball', 0.3), [0.3, 0.3], ['ball', 'ball'])]:
    tr, te = draws(p, 20000, 4), draws(p, 20000, 5); r = dict(truth=sc(p, te))
    for lab, ekw, fkw, K in [('+sqrt h', {}, {}, 1), ('signed', dict(retrieve=True), {}, 1),
                             ('complex', {}, dict(complex_g=True), 1), ('real rank 2', {}, {}, 2)]:
        m = BLMLEmpirical(lam=1e-4, tol=1e-10, outer_tol=1e-12, max_outer=20000, **ekw).fit(tr, bt.make_modes(fc, kinds, rank=K), **fkw)
        J = np.array(m.J_history_)
        r[lab] = dict(test=m.score(te), outer=m.n_outer_, J_rise=float(np.max(np.diff(J))) if len(J) > 1 else 0.0)
    for lab, cg in [('BLML real', False), ('BLML complex', True)]:
        m = BLMLBinned(tol=1e-9).fit(tr, bt.make_modes(fc, kinds), complex_g=cg); r[lab] = dict(test=sc(_np(m.density_), te))
    res[name] = r; print(name, json.dumps(r))
json.dump(res, open('results/empirical_phase_table.json', 'w'), indent=1)
