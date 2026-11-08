"""Paper 4, synthetic: the rank test of coherence. Two point emitters, separation s pixels along the diagonal (so f is not separable: along one axis it would be a
1-d density, which always has one complex root), on a periodic L x L window;
each emits the band-limited field h(x - x_j) (a Dirichlet kernel of N harmonics per axis, periodic over the window, so
no mass leaves it). Mutual coherence gamma in [0, 1]: the detected intensity is f = sum_k lam_k |sum_j U_jk h_j|^2 with
(lam, U) the eigenpairs of [[1, gamma], [gamma, 1]]: rank 1 at gamma = 1 (one field), rank 2 below. Photons on the
pixel centres (multinomial; train, validation and test drawn independently). Complex BLML of rank K = 1 and 2 with the
true band (dirichlet modes, N harmonics), scored on test photons (paired); K selected on validation.
Usage: python results/p4_synth.py <n> <seed>  -> results/p4_synth/n<n>_<seed>.json"""
import sys, json, time, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_binned import BLMLBinned, _np
n, seed = int(sys.argv[1]), int(sys.argv[2]); L, NH = 32, 4
x = np.arange(L)
def h1(c):                                   # periodic Dirichlet field, N harmonics, centred at c
    m = np.arange(-NH, NH + 1); return np.exp(2j * np.pi * np.outer(x - c, m) / L).sum(1) / (2 * NH + 1)
def field(cx, cy):
    return np.outer(h1(cy), h1(cx))
mode = bt.make_modes([(NH + 0.5) * 2 / L] * 2, ['dirichlet', 'dirichlet'], periods=[float(L), float(L)])[0]
ll = lambda P, C: float((C * np.log(np.maximum(P / P.sum(), 1e-300))).sum() / C.sum())
res = dict(n=n, seed=seed, L=L, harmonics=NH, rows=[])
for s in (2.0, 4.0):
    d = s / 2 / np.sqrt(2); H = [field(L / 2 - d, L / 2 - d), field(L / 2 + d, L / 2 + d)]   # diagonal: not separable
    for gamma in (0.0, 0.5, 0.9, 1.0):
        lam, U = np.linalg.eigh(np.array([[1.0, gamma], [gamma, 1.0]]))
        f = sum(max(lam[k], 0) * np.abs(U[0, k] * H[0] + U[1, k] * H[1]) ** 2 for k in range(2)); p = (f / f.sum()).ravel()
        rng = np.random.default_rng([seed, int(10 * s), int(100 * gamma)])
        TR, VA, TE = [rng.multinomial(m, p).reshape(L, L).astype(float) for m in (n, n // 4, 10 ** 5)]
        row = dict(s=s, gamma=gamma, entropy=float(-(p[p > 0] * np.log(p[p > 0])).sum()))
        for K in (1, 2):
            t = time.time()
            m = BLMLBinned(tol=1e-6, max_sweeps=20000, ll_tol=1e-12).fit(TR, [mode] * K, complex_g=True)
            P = np.maximum(_np(m.density_), 1e-300).ravel(); E, Sx = _np(m.e_hat_), _np(m.Se_)
            G = np.array([[np.sum(np.conj(E[a]) * Sx[b]) for b in range(len(E))] for a in range(len(E))])   # mode Gram <g_a, g_b>
            Q = np.clip(np.linalg.eigvalsh((G + G.conj().T) / 2), 0, None)                             # rotation-invariant spectrum
            row[f'K{K}'] = dict(val=ll(P, VA.ravel()), test=ll(P, TE.ravel()), kl=float((p * (np.log(np.maximum(p, 1e-300)) - np.log(P / P.sum()))).sum()),
                               spectrum=sorted((Q / Q.sum()).tolist(), reverse=True), stop=m.stop_reason_, s=time.time() - t)
        row['kl_gain'] = row['K1']['kl'] - row['K2']['kl']; row['true_spectrum'] = sorted((np.clip(lam, 0, None) / 2).tolist(), reverse=True); row['val_pick'] = 2 if row['K2']['val'] > row['K1']['val'] else 1
        res['rows'].append(row)
        print(n, seed, f"s={s} gamma={gamma}: KL K1 {row['K1']['kl']:.5f} K2 {row['K2']['kl']:.5f} pick K{row['val_pick']} "
              f"shares K2 {np.round(row['K2']['spectrum'], 3).tolist()} true {np.round(sorted(lam / 2, reverse=True), 3).tolist()} "
              f"{row['K1']['stop']}/{row['K2']['stop']} {row['K1']['s']:.0f}+{row['K2']['s']:.0f}s", flush=True)
json.dump(res, open(f'results/p4_synth/n{n}_{seed}.json', 'w'), indent=1)
