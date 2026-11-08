"""Paper 3, Sec. 4.2 diagnostic: escape local maxima with the certificate eigenvector (rank staircase).
default fit -> cert; if lambda_max/n > 1: add the top generalized eigenvector beta (S D S beta = lam S beta) as a
second complex mode, fit rank 2, collapse to the dominant mode (eigvec of C^H S C), refit rank 1; repeat.
Usage: sec42_staircase.py <case> <n> <seed>"""
from blml_trivial import with_lattice
import sys, json, types, time, numpy as np
src = open('results/complex/sec42_truthstart.py').read(); exec(src.split("out = dict(")[0])
Zk = Z[N > 0]; Nk = N[N > 0]
def fit(init=None, rank=1):
    return bt.BLMLQuick(dtype=np.float64, tol=1e-5).fit(Z, with_lattice(bt.make_modes([FC, FC], ['ball', 'ball'], None, rank=rank), 1.0), sample_weight=N, complex_g=True, init=None if init is None else types.SimpleNamespace(Z_=Zk, e_hat_=init))
def cert(m):
    c, beta = m.certificate(return_vector=True); return c, beta, None
def fbins(m):
    C = m.e_hat_ if m.e_hat_.ndim == 2 else m.e_hat_[:, None]
    return (np.abs(m.kernel_matvec(C) / m.n_samples_) ** 2).sum(1)      # f at the fitted (count > 0) bins
def llm(m): return float(np.sum(Nk * np.log(np.maximum(fbins(m), 1e-300))))
def report(tag, m, extra={}):
    r = dict(ll_minus_truth=llm(m) - ll(np.abs(hw) ** 2), cert=cert(m)[0], conv=bool(m.converged_), **extra)
    if m.e_hat_.ndim == 1:
        r['rms_err'], r['twin'], r['est'] = zern(m)
    print(case, tag, r, flush=True); return r
out = dict(case=case, n=n, seed=seed); t0 = time.time()
mT = fit(np.where(N > 0, N / np.conj(np.where(hw == 0, 1, hw)), 0)[N > 0]); out['truth_start'] = report('truth_start', mT)
m = fit(); out['default'] = report('default', m); best = llm(m)
for it in range(3):
    c, beta, S = cert(m)
    if c <= 1 + 1e-6: break
    e = m.e_hat_; eps = 0.3 * np.linalg.norm(e) / np.linalg.norm(beta)
    m2 = fit(np.c_[e, eps * beta], rank=2); r2 = report(f'step{it}_rank2', m2)
    C = m2.e_hat_; G = C.conj().T @ m2.kernel_matvec(C); w, U = np.linalg.eigh(G); e1 = C @ U[:, -1]
    m1 = fit(e1); l1 = llm(m1)
    out[f'step{it}'] = report(f'step{it}_rank1', m1, dict(rank2=r2, mode_share=float(w[-1] / w.sum()), t=time.time() - t0))
    if l1 <= best + 1e-6: break
    m, best = m1, l1
json.dump(out, open(f'results/complex/sec42_init/stair_{case}_{n}_{seed}.json', 'w'), indent=1, default=str)
