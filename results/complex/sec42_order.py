"""Paper 3, Sec. 4.2: does the update order decide the basin? From each of the 5 library starts (3 HIO + 2 symmetry
breaking): (a) the library solver (block-Jacobi + Anderson) from the start; (b) SEQ sweeps of sequential single-bin
CD (aligned root, random order per sweep), then the library solver. Phase rms error and LL - truth per start.
Dense S (B x B) for the sequential sweeps: experiment only.
Usage: sec42_order.py <case> <n> <seed> [SEQ]"""
import sys, json, time, numpy as np
src = open('results/complex/sec42_staircase.py').read(); exec(src.split("\nout = dict(")[0])
SEQ = int(sys.argv[4]) if len(sys.argv) > 4 else 30
w = Nk.astype(float); B = len(w); nn = w.sum()
kern = bt.lattice_kernel(np.array([FC, FC]), np.ones(2), ['ball:0', 'ball:0'], [None, None], 2)
S = kern(Zk.astype(float), Zk.astype(float)); sbb = float(S[0, 0])
starts = bt.complex_starts(np.round(Zk).astype(np.int64), w, np.ones(2), np.array([FC, FC]), ['ball:0', 'ball:0'], 3, 2, seed=0)


def seq(e, sweeps):
    e = e.astype(complex).copy(); Se = S @ e; rng = np.random.default_rng(1)
    for _ in range(sweeps):
        for b in rng.permutation(B):
            R = Se[b] - sbb * e[b]; aR = abs(R)
            t = (-aR + np.sqrt(aR ** 2 + 4 * sbb * nn * w[b])) / (2 * sbb)
            new = t * R / max(aR, 1e-300); Se += (new - e[b]) * S[b]; e[b] = new
        Se = S @ e
    return e


out = dict(case=case, n=n, seed=seed, SEQ=SEQ, B=B); t0 = time.time()
out['truth_start'] = report('truth_start', fit(np.where(N > 0, N / np.conj(np.where(hw == 0, 1, hw)), 0)[N > 0]))
out['starts'] = []
for k, (fam, sc, e0) in enumerate(starts):
    ra = report(f'{fam}{k}_library', fit(e0))
    t1 = time.time(); e1 = seq(e0, SEQ); ts = time.time() - t1
    rb = report(f'{fam}{k}_seq{SEQ}', fit(e1), dict(seq_s=ts))
    out['starts'].append(dict(family=fam, start_score=sc, library=ra, seq=rb))
out['s'] = time.time() - t0
json.dump(out, open(f'results/complex/sec42_init/order_{case}_{n}_{seed}.json', 'w'), indent=1, default=str)
print('done', case, n, seed, '%.0fs' % out['s'], flush=True)
