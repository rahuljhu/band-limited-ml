"""Small-n check against the original paper's orthant search: for n samples (continuous, unbinned),
solve the real BLML in every orthant c0 in {+-1}^n (2^n convex solves, c0 ~ -c0 so 2^(n-1)), take the best,
and compare with complex BLML (BLMLTrivial-style dense Gram, Gauss-Seidel from the positive real start).
Log-likelihood per sample: mean log g(x_i)^2 with int g^2 = 1, g = (1/n) sum c_i K(x - x_i)."""
import sys, itertools, json, numpy as np
sys.path.insert(0, '.'); import blml_trivial as bt
FC = 0.328
def S_of(x): return FC * np.sinc(FC * (x[:, None] - x[None]))
def real_orthant(S, sg):                    # solve c_i (Sc)_i = n with c = sg * t, t > 0 (fixed point / Newton)
    M = S * sg[:, None] * sg[None]; n = len(sg)
    t, *_ = bt.solve_rho_positive_orthant(lambda v: M @ v, n, tol=1e-11)
    return sg * t
def ll(S, c):
    n = len(c); u = S @ c / n; return float(np.mean(np.log(np.abs(u) ** 2)) - np.log(np.real(np.vdot(c, S @ c)) / n ** 2))
def complex_gs(S, seed):
    n = len(S); a = real_orthant(S, np.ones(n)); rng = np.random.default_rng(seed)
    e = a * np.exp(0.1j * rng.standard_normal(n)); er, ei = e.real.copy(), e.imag.copy(); Se = S @ e; sr, si = Se.real.copy(), Se.imag.copy()
    w = np.ones(n)
    for it in range(200000):
        perm = rng.permutation(n).astype(np.int64)
        bt._complex_gs_block(perm, np.ascontiguousarray(S[:, perm]), er, ei, sr, si, w, float(n), S[0, 0])
        e = er + 1j * ei
        if np.max(np.abs(np.conj(e) * (sr + 1j * si) / n - 1)) < 1e-10: break
    return e
def sample(case, n, rng):
    xs = np.linspace(-80, 80, 320001)
    if case == 'focus': p = np.sinc(FC * xs) ** 2
    else:  # two-lobe real BL density with sign change: h = sinc(FC x - 1.5) - sinc(FC x + 1.5)
        p = (np.sinc(FC * xs - 1.5) - np.sinc(FC * xs + 1.5)) ** 2
    p /= p.sum(); return rng.choice(xs, size=n, p=p)
rows = []
for case in ['focus', 'dipole']:
    for n in [8, 12, 16]:
        for seed in range(5):
            x = sample(case, n, np.random.default_rng(100 * n + seed)); S = S_of(x)
            best, best_sg, lls = -np.inf, None, []
            for bits in itertools.product([1.0, -1.0], repeat=n - 1):
                sg = np.r_[1.0, bits]; c = real_orthant(S, sg); v = ll(S, c); lls.append(v)
                if v > best: best, best_sg = v, sg
            allpos = ll(S, real_orthant(S, np.ones(n)))
            bqp = max(itertools.product([1.0, -1.0], repeat=n - 1), key=lambda b: np.r_[1.0, b] @ S @ np.r_[1.0, b])
            bqp_ll = ll(S, real_orthant(S, np.r_[1.0, bqp]))
            e = complex_gs(S, seed); lc = ll(S, e); U = np.vstack([e.real, e.imag]); sv = np.linalg.svd(np.vstack([(S @ e).real, (S @ e).imag]), compute_uv=False)
            same = bool(np.all(np.sign((S @ e * np.exp(-1j * np.angle((S @ e)[0]))).real) == best_sg * np.sign(best_sg[0]))) if sv[1] / sv[0] < 1e-6 else None
            rows.append(dict(case=case, n=n, seed=seed, best_orthant=best, all_positive=allpos, bqp_orthant=bqp_ll, complex=lc,
                             complex_realness=float(sv[1] / sv[0]), complex_matches_best_signs=same, rank_of_allpos=int((np.array(lls) > allpos + 1e-12).sum())))
            r = rows[-1]; print(f"{case:6s} n={n:2d} s={seed}: best orthant {r['best_orthant']:.5f}  all+ {r['all_positive']:.5f}  BQP {r['bqp_orthant']:.5f}  complex {r['complex']:.5f} realness {r['complex_realness']:.1e} same-signs {same}", flush=True)
json.dump(rows, open('results/complex/bqp_exhaustive.json', 'w'), indent=1)
