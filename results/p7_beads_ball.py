"""Paper 7 against paper 3's bead estimator: real and complex BLML with the ball kernel (circular pupil) and the EM
background, refitted on train + validation at the cut-off paper 3 selected (results/psf/v5/bead<k>.json; selector
search on validation, fc up to 1/px), exactly as results/psf/run_bead_v5.py, and paired over test photons with the
empirical fit of results/p7_beads (refitted here at its selected (fc, lam)).
Usage: python results/p7_beads_ball.py <bead>  -> results/p7_beads/ball_b<bead>.json"""
import sys, json, time, numpy as np
sys.path.insert(0, 'results/psf'); sys.path.insert(0, '.')
import psfkit as pk, blml_trivial as bt
from blml_trivial import with_lattice, make_modes
from blml_binned import BLMLEmpirical, _np
k = int(sys.argv[1]); H = 15; Z = pk.lattice(H)
cnt, _ = pk.bead_counts(k, H); tr, va, te = pk.split(cnt); B = cnt.size; N = (tr + va).astype(float)
v5, p7 = json.load(open(f'results/psf/v5/bead{k}.json')), json.load(open(f'results/p7_beads/b{k}_full_0.json'))


def em(N, q, iters=60, tol=1e-7):
    N = np.asarray(N, float).ravel(); pi, r, prev = 0.1, np.full(N.shape, 0.1), -np.inf
    for _ in range(iters):
        qd = q(N * (1 - r)); P = pi / B + (1 - pi) * qd; r = (pi / B) / P; pi = float((N * r).sum() / N.sum())
        P = pi / B + (1 - pi) * qd; L = pk.score(P, N)
        if L - prev < tol:
            break
        prev = L
    return P / P.sum()


def q_ball(fc, cg):
    st = {}
    def f(w):
        m = bt.BLMLQuick(dtype=np.float64, tol=1e-5).fit(Z, with_lattice(make_modes(list(fc), ['ball', 'ball'], None), 1.0),
                                                        sample_weight=np.where(w > 1e-6, w, 0), complex_g=cg, init=st.get('m'))
        st['m'] = m; q = np.maximum(m.predict(Z), 1e-300); return q / q.sum()
    return f


def q_emp(w):
    m = BLMLEmpirical(reg='ridge', lam=p7['empirical']['lam']).fit(w.reshape(cnt.shape), bt.make_modes(list(p7['empirical']['fc']), ['sinc', 'sinc']))
    d = np.maximum(_np(m.density_), 1e-300).ravel(); return d / d.sum()


lw = lambda P: np.log(P)
w = te.ravel().astype(float); Pe = em(N, q_emp); res = dict(bead=k, empirical=pk.score(Pe, w))
for lab, cg in (('real_ball', False), ('complex_ball', True)):
    t = time.time(); P = em(N, q_ball(v5[lab]['fc'], cg)); d = lw(Pe) - lw(P)
    mu = float((w * d).sum() / w.sum()); sd = np.sqrt(float((w * (d - mu) ** 2).sum() / w.sum()))
    res[lab] = dict(test=pk.score(P, w), v5_test=v5[lab]['test_ll'], fc=v5[lab]['fc'], emp_minus=mu, se=sd / np.sqrt(w.sum()), s=time.time() - t)
json.dump(res, open(f'results/p7_beads/ball_b{k}.json', 'w'), indent=1)
print(k, {l: (round(res[l]['emp_minus'], 5), round(res[l]['se'], 5), round(res[l]['test'] - res[l]['v5_test'], 6), round(res[l]['s'])) for l in ('real_ball', 'complex_ball')}, flush=True)
