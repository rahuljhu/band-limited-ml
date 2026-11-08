"""Paper 3, bead diagnostic: is complex BLML's deficit against the KDE a window-truncation effect? For bead k, fit on
the (2H+1)^2 window for H = 15 (paper 3) and H = 25, and score both methods on the test photons of the same central
31 x 31 pixels (probabilities renormalised there). Complex and real BLML: ball kernel at paper 3's selected cut-off
(results/psf/v5), EM uniform floor over the fitting window; KDE: Gaussian with the same floor, bandwidth on validation
over the fitting window. One binomial split of the H = 25 window (seed 0); the H = 15 fit uses its central part.
Usage: python results/p3_window.py <bead>  -> results/p3_window/b<bead>.json"""
import os, sys, json, numpy as np
sys.path.insert(0, 'results/psf'); sys.path.insert(0, '.')
import psfkit as pk, blml_trivial as bt
from blml_trivial import with_lattice, make_modes
k = int(sys.argv[1]); OUT = 'results/p3_window'; os.makedirs(OUT, exist_ok=True)
v5 = json.load(open(f'results/psf/v5/bead{k}.json'))
big, _ = pk.bead_counts(k, 25)
if big.shape != (51, 51):
    sys.exit(f'{k}: too close to the frame edge')
TR, VA, TE = pk.split(big); C = slice(10, 41)                      # central 31 x 31
te_c = TE[C, C].ravel().astype(float)


def em(N, q, iters=60, tol=1e-7):
    N = np.asarray(N, float).ravel(); B = N.size; pi, r, prev = 0.1, np.full(N.shape, 0.1), -np.inf
    for _ in range(iters):
        qd = q(N * (1 - r)); P = pi / B + (1 - pi) * qd; r = (pi / B) / P; pi = float((N * r).sum() / N.sum())
        P = pi / B + (1 - pi) * qd; L = pk.score(P, N)
        if L - prev < tol:
            break
        prev = L
    return P / P.sum()


def q_ball(Z, fc, cg):
    st = {}
    def f(w):
        m = bt.BLMLQuick(dtype=np.float64, tol=1e-5).fit(Z, with_lattice(make_modes(list(fc), ['ball', 'ball'], None), 1.0),
                                                        sample_weight=np.where(w > 1e-6, w, 0), complex_g=cg, init=st.get('m'))
        st['m'] = m; q = np.maximum(m.predict(Z), 1e-300); return q / q.sum()
    return f


def centre(P, H):
    P = P.reshape(2 * H + 1, 2 * H + 1); P = P[C, C] if H == 25 else P; return (P / P.sum()).ravel()


res = dict(bead=k)
for H in (15, 25):
    cut = slice(None) if H == 25 else C
    tr, va = TR[cut, cut], VA[cut, cut]; Z = pk.lattice(H); sh = tr.shape
    edge = np.ones(sh, bool); edge[1:-1, 1:-1] = False
    row = dict(edge=float((tr + va)[edge].sum() / (tr + va).sum()))
    h = max(pk.KDE_H, key=lambda h_: pk.score(em(tr, lambda w: pk.kde(w, h_, sh)), va.ravel()))
    Pk = centre(em(tr + va, lambda w: pk.kde(w, h, sh)), H); row['kde'] = pk.score(Pk, te_c)
    for lab, cg in (('real_ball', False), ('complex_ball', True)):
        P = centre(em(tr + va, q_ball(Z, v5[lab]['fc'], cg)), H); d = np.log(P) - np.log(Pk)
        mu = float((te_c * d).sum() / te_c.sum()); sd = np.sqrt(float((te_c * (d - mu) ** 2).sum() / te_c.sum()))
        row[lab] = dict(test=pk.score(P, te_c), minus_kde=mu, se=sd / np.sqrt(te_c.sum()))
    res[f'H{H}'] = row
    print(k, H, f"edge {row['edge']:.3f} kde h {h}", {l: (round(row[l]['minus_kde'], 5), round(row[l]['se'], 5)) for l in ('real_ball', 'complex_ball')}, flush=True)
json.dump(res, open(f'{OUT}/b{k}.json', 'w'), indent=1)
