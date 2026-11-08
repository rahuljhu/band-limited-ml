"""Paper 3, beads with the camera's fixed pattern in the model. The sCMOS pixels carry a per-pixel offset pattern
o_b (photons) shared by every exposure, so binomial thinning gives it to train and test alike. o_b is calibrated on
the stack's planes not adjacent to the analysed one (median over planes of the image minus its Gaussian(4 px)
smooth; results/psf/pattern_o.npy). Background component: expected counts c + f o_b (c = pi n / B the uniform level,
f the split's share of the exposure), so P_b = pi bg_b + (1 - pi) q_b with bg ~ c + f o_b refitted with pi by EM.
Same background for complex and real BLML (ball kernel at paper 3's selected cut-off) and the Gaussian KDE (bandwidth on
validation). Scored on test photons, paired against the KDE.
Usage: python results/p3_pattern.py <bead> -> results/p3_pattern/b<bead>.json"""
import os, sys, json, numpy as np
sys.path.insert(0, 'results/psf'); sys.path.insert(0, '.')
import psfkit as pk, blml_trivial as bt
from blml_trivial import with_lattice, make_modes
from scipy.ndimage import gaussian_filter
OUT = 'results/p3_pattern'; os.makedirs(OUT, exist_ok=True); PF = 'results/psf/pattern_o.npy'
if not os.path.exists(PF):
    S = np.load('results/psf/stack_photons.npy', mmap_mode='r')
    R = [(lambda im: im - gaussian_filter(im, 4))(np.asarray(S[p], float) / pk.GAIN_CORR) for p in range(S.shape[0]) if abs(p - 5) > 1]
    np.save(PF, np.median(np.array(R), 0).astype(np.float32))
k = int(sys.argv[1]); H = 15; Z = pk.lattice(H); v5 = json.load(open(f'results/psf/v5/bead{k}.json'))
cnt, (y0, x0) = pk.bead_counts(k, H); tr, va, te = pk.split(cnt)
o = np.load(PF, mmap_mode='r')[y0 - H:y0 + H + 1, x0 - H:x0 + H + 1].astype(float).ravel()


def em(N, q, f, pattern=True, iters=80, tol=1e-7):
    N = np.asarray(N, float).ravel(); n, B = N.sum(), N.size; pi, prev = 0.1, -np.inf; r = np.full(B, 0.1)
    for _ in range(iters):
        qd = q(N * (1 - r)); c = pi * n / B
        bg = np.maximum(c + f * o, 1e-3 * c) if pattern else np.full(B, c); bg = bg / bg.sum()
        P = pi * bg + (1 - pi) * qd; r = pi * bg / P; pi = float((N * r).sum() / n); P = pi * bg + (1 - pi) * qd
        L = pk.score(P, N)
        if L - prev < tol:
            break
        prev = L
    return P / P.sum()


def q_ball(fc, cg):
    st = {}
    def g(w):
        m = bt.BLMLQuick(dtype=np.float64, tol=1e-5).fit(Z, with_lattice(make_modes(list(fc), ['ball', 'ball'], None), 1.0),
                                                        sample_weight=np.where(w > 1e-6, w, 0), complex_g=cg, init=st.get('m'))
        st['m'] = m; qq = np.maximum(m.predict(Z), 1e-300); return qq / qq.sum()
    return g


w = te.ravel().astype(float); res = dict(bead=k)
for pat in (False, True):
    h = max(pk.KDE_H, key=lambda h_: pk.score(em(tr, lambda x: pk.kde(x, h_, cnt.shape), 0.64, pat), va.ravel()))
    Pk = em(tr + va, lambda x: pk.kde(x, h, cnt.shape), 0.80, pat); row = dict(kde_h=h, kde=pk.score(Pk, w))
    for lab, cg in (('real_ball', False), ('complex_ball', True)):
        P = em(tr + va, q_ball(v5[lab]['fc'], cg), 0.80, pat); d = np.log(P) - np.log(Pk)
        mu = float((w * d).sum() / w.sum()); sd = np.sqrt(float((w * (d - mu) ** 2).sum() / w.sum()))
        row[lab] = dict(test=pk.score(P, w), minus_kde=mu, se=sd / np.sqrt(w.sum()))
    res['pattern' if pat else 'uniform'] = row
    print(k, 'pattern' if pat else 'uniform', f"kde h {h} kde {row['kde']:.5f}",
          {l: (round(row[l]['test'], 5), round(row[l]['minus_kde'], 5), round(row[l]['se'], 5)) for l in ('real_ball', 'complex_ball')}, flush=True)
json.dump(res, open(f'{OUT}/b{k}.json', 'w'), indent=1)
