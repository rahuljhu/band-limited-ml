"""Paper 3, bead windows with the periodic ball (kind 'pball': the pupil's disk of harmonics of the 31 x 31 window,
normalised on the window), the cut-off chosen on validation from 0.6--1.25 times paper 3's selected ball cut-off, plus
the window's base harmonic, in place of the ball kernel normalised on the plane (p3_window_bgmode.py). Otherwise as there.
Usage: python results/p3_window_pball.py  -> results/p3_window_pball.json
Formerly: bead windows with the background as a second band-limited mode in place of the EM floor: does it close the
windows' gap to the KDE? Per bead, the 31 x 31 window (same split as p3_window.py: central part of the 51 x 51 split),
complex BLML (ball at paper 3's selected cut-off, results/psf/v5) plus the window's base harmonic (dirichlet, period 31),
fitted by BLMLBinned on train + val; the KDE (Gaussian + EM floor, bandwidth on validation) as in p3_window.py; both
scored on the window's test photons, paired."""
import os, sys, json, time, numpy as np
sys.path.insert(0, 'results/psf'); sys.path.insert(0, '.')
import psfkit as pk, blml_trivial as bt
from blml_binned import BLMLBinned, _np
src = open('results/p3_window.py').read(); exec(src[src.index('def em('):src.index('def q_ball(')])
rows = []
for k in range(int(os.environ.get('NBEADS', '53'))):
    v5 = json.load(open(f'results/psf/v5/bead{k}.json')); big, _ = pk.bead_counts(k, 25)
    if big.shape != (51, 51):
        continue
    t = time.time(); TR, VA, TE = pk.split(big); C = slice(10, 41); tr, va, te = TR[C, C], VA[C, C], TE[C, C]
    sh = tr.shape; te_c = te.ravel().astype(float)
    h = max(pk.KDE_H, key=lambda h_: pk.score(em(tr, lambda w: pk.kde(w, h_, sh)), va.ravel()))
    Pk = em(tr + va, lambda w: pk.kde(w, h, sh))
    fc0 = np.asarray(v5['complex_ball']['fc'], float); W = sh[0]
    bgm = bt.make_modes([1.0 / W, 1.0 / W], ['dirichlet', 'dirichlet'], periods=[W, W])[0]
    per = lambda r: [{(0, 1): dict(kind='pball', fc=float(r * fc0[0]), period=[W, W])}, bgm]
    fit = lambda T, r: BLMLBinned(tol=1e-6, max_sweeps=20000, ll_tol=1e-12).fit(T.astype(float), per(r), complex_g=True)
    val = {}
    for r in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25):
        f_ = _np(fit(tr, r).density_).ravel(); val[r] = float((va.ravel() * np.log(np.maximum(f_ / f_.sum(), 1e-300))).sum())
    rs = max(val, key=val.get); m = fit(tr + va, rs)
    f = _np(m.density_).ravel(); P = f / f.sum(); d = np.log(np.maximum(P, 1e-300)) - np.log(Pk)
    mu = float((te_c * d).sum() / te_c.sum()); se = np.sqrt(float((te_c * (d - mu) ** 2).sum() / te_c.sum()) / te_c.sum())
    E, U = _np(m.e_hat_), _np(m.Se_); Q = np.array([float(np.real(np.sum(np.conj(E[j]) * U[j]))) for j in range(2)])
    rows.append(dict(bead=k, ratio=rs, minus_kde=mu, se=se, bg_share=float(Q[1] / Q.sum()), sweeps=m.n_sweeps_, s=time.time() - t))
    print(k, f'{mu:+.5f} +- {se:.5f} bg {Q[1] / Q.sum():.3f} {time.time() - t:.1f}s', flush=True)
z = np.array([r['minus_kde'] / r['se'] for r in rows])
out = dict(rows=rows, wins=int((z > 2).sum()), ties=int((abs(z) <= 2).sum()), losses=int((z < -2).sum()),
           median=float(np.median([r['minus_kde'] for r in rows])))
json.dump(out, open('results/p3_window_pball.json', 'w'), indent=1)
print({k: v for k, v in out.items() if k != 'rows'})
