"""Paper 7 Table phase_conv: convergence of the alternation (complex rank 1, ridge lam 1e-3, 1-d sinc truth): J, last
decrease, |grad J| of the reduced objective, min |g_b| / sqrt(h_b). Output results/empirical_phase_conv.json"""
import sys, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_trivial import kernel_class
from blml_binned import BLMLEmpirical, _np
exec(open('results/binned_learn_checks.py').read().split('def fd_check')[0].split('rows = []')[1])
p1 = truth((64,), 'sinc', 0.25); tr = draws(p1, 20000, 1)
rows = []
for mo in (100, 1000, 10000, 50000):
    m = BLMLEmpirical(lam=1e-3, tol=1e-12, outer_tol=0, max_outer=mo).fit(tr, bt.make_modes([0.3], ['sinc']), complex_g=True)
    st = m._ift_state(); C, U, t, a, As = st['C'], st['U'], st['t'], st['a'], st['As']
    q = ((a / m.n_ - t) / As)[None] * U; g = (2 / m.n_) * m._Sall(q) + 2 * m.lam * U / m.n_ ** 2
    J = np.array(m.J_history_)
    rows.append(dict(outer=mo, J=float(J[-1]), dJ=float(J[-2] - J[-1]), grad=float(np.linalg.norm(g)), min_ratio=float(np.min((a / m.n_)[m.occ_] / t[m.occ_])))); print(rows[-1], flush=True)
import json; json.dump(rows, open('results/empirical_phase_conv.json', 'w'), indent=1)
