"""Paper 3, Sec. 4.2: beam search over the two row roots of coordinate descent.
Beam of K complex coefficient vectors, started at the K library starts (3 HIO + 2 symmetry breaking). One step = one
bin b (random order per sweep, shared by the beam): every member gives two children e_b = t R_b/|R_b| with t a root
of S_bb t^2 + |R_b| t - n w_b = 0 (t+ > 0: aligned with the neighbours' field; t- < 0: anti-aligned), and the K best
of the 2K children by train LL = sum_b w_b log|Se|_b^2 - n log(e^H S e) survive. Greedy = always t+ (plain
sequential CD) from the same starts, no pruning. After the sweeps, the best member of each is polished by the
library solver (warm start) and scored. Dense S (B x B): experiment only, not solver code.
Usage: sec42_beam.py <case> <n> <seed> [K] [sweeps]"""
import sys, json, time, types, numpy as np
src = open('results/complex/sec42_staircase.py').read(); exec(src.split("\nout = dict(")[0])
Kb = int(sys.argv[4]) if len(sys.argv) > 4 else 5
SWEEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 200
w = Nk.astype(float); B = len(w); nn = w.sum()
kern = bt.lattice_kernel(np.array([FC, FC]), np.ones(2), ['ball:0', 'ball:0'], [None, None], 2)
S = kern(Zk.astype(float), Zk.astype(float)); sbb = float(S[0, 0])
starts = bt.complex_starts(np.round(Zk).astype(np.int64), w, np.ones(2), np.array([FC, FC]), ['ball:0', 'ball:0'],
                           3, 2, seed=0)[:Kb]
E0 = np.array([e for _, _, e in starts])


def LL(E, SE):
    return np.log(np.maximum(np.abs(SE) ** 2, 1e-300)) @ w - nn * np.log(np.real(np.sum(np.conj(E) * SE, 1)))


def run(beam):
    E = E0.copy(); SE = E @ S; K = len(E); rng = np.random.default_rng(1)
    hist, kept_minus = [], []
    for sweep in range(SWEEPS):
        km = 0
        for b in rng.permutation(B):
            R = SE[:, b] - sbb * E[:, b]; aR = np.abs(R); u = R / np.maximum(aR, 1e-300)
            disc = np.sqrt(aR ** 2 + 4 * sbb * nn * w[b])
            tp, tm = (-aR + disc) / (2 * sbb), (-aR - disc) / (2 * sbb)
            if not beam:
                d = tp * u - E[:, b]; SE += d[:, None] * S[b][None]; E[:, b] = tp * u
                continue
            par = np.r_[np.arange(K), np.arange(K)]; new = np.r_[tp * u, tm * u]
            Ec = E[par]; SEc = SE[par] + (new - Ec[:, b])[:, None] * S[b][None]; Ec[:, b] = new
            top = np.argsort(-LL(Ec, SEc))[:K]
            km += int(np.sum(top >= K)); E, SE = Ec[top], SEc[top]
        SE = E @ S                                   # drop accumulated rounding
        hist.append(LL(E, SE).tolist()); kept_minus.append(km)
    return E, hist, kept_minus


out = dict(case=case, n=n, seed=seed, K=Kb, sweeps=SWEEPS, B=B, starts=[(f, s) for f, s, _ in starts])
t0 = time.time(); mT = fit(np.where(N > 0, N / np.conj(np.where(hw == 0, 1, hw)), 0)[N > 0])
out['truth_start'] = report('truth_start', mT); ll0 = ll(np.abs(hw) ** 2)
for lab, beam in [('greedy', False), ('beam', True)]:
    t1 = time.time(); E, hist, km = run(beam); i = int(np.argmax(hist[-1]))
    m = fit(E[i])                                    # polish the best member to convergence
    r = report(lab, m, dict(ll_after_sweeps=[h - ll0 for h in hist[-1]], kept_minus_per_sweep=km[:10] + km[-3:],
                            total_kept_minus=int(np.sum(km)), s=time.time() - t1))
    out[lab] = r
out['s'] = time.time() - t0
json.dump(out, open(f'results/complex/sec42_init/beam_{case}_{n}_{seed}.json', 'w'), indent=1, default=str)
print('done', case, seed, '%.0fs' % out['s'], flush=True)
