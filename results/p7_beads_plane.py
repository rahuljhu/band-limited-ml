"""Paper 7 control: the empirical estimator at its selected (fc, lam) (results/p7_beads/b<k>_full_0.json) refitted with
normalise='plane' (int g^2 = 1, as ML BLML) in place of 'window': splits the gain over ML into normalisation and the
interpolation itself. Same EM floor, train + validation, full test counts. -> adds 'emp_plane' to results/p7_beads/summary.json"""
import sys, json, glob, numpy as np
sys.path.insert(0, 'results/psf'); sys.path.insert(0, '.')
import psfkit as pk, blml_trivial as bt
from blml_binned import BLMLEmpirical, _np
S = json.load(open('results/p7_beads/summary.json'))
for row in S['rows']:
    k = row['bead']; r = json.load(open(f'results/p7_beads/b{k}_full_0.json'))
    cnt, _ = pk.bead_counts(k, 15); tr, va, te = pk.split(cnt); B = cnt.size; N = (tr + va).astype(float)
    fc, lam = r['empirical']['fc'], r['empirical']['lam']
    def q(w):
        m = BLMLEmpirical(reg='ridge', lam=lam, normalise='plane').fit(w, bt.make_modes(list(fc), ['sinc', 'sinc']))
        d = np.maximum(_np(m.density_), 1e-300); return d / d.sum()
    pi, r_, prev = 0.1, np.full(N.shape, 0.1), -np.inf
    for _ in range(60):
        qd = q(N * (1 - r_)); P = pi / B + (1 - pi) * qd; r_ = (pi / B) / P; pi = float((N * r_).sum() / N.sum())
        P = pi / B + (1 - pi) * qd; L = pk.score(P.ravel(), N.ravel())
        if L - prev < 1e-7:
            break
        prev = L
    row['emp_plane'] = pk.score((P / P.sum()).ravel(), te.ravel())
    print(k, round(row['empirical'] - row['emp_plane'], 5), round(row['emp_plane'] - row['ml'], 5), flush=True)
json.dump(S, open('results/p7_beads/summary.json', 'w'), indent=1)
