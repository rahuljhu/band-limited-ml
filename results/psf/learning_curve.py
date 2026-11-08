"""Real-bead learning curve: thin train+val by q to reach n photons, keep the full test set.
Usage: python learning_curve.py <bead k> <n> <seed>  -> results/psf/v3/lc/b<k>_<n>_<seed>.json"""
import sys, json, os, time, numpy as np
sys.path.insert(0, 'results/psf'); import psfkit as pk
k, n, seed = int(sys.argv[1]), int(float(sys.argv[2])), int(sys.argv[3]); H = 15; Z = pk.lattice(H)
cnt, pos = pk.bead_counts(k, H); tr, va, te = pk.split(cnt)
q = min(1.0, n / (tr + va).sum()); tr_n, va_n = pk.thin(tr, q, 100 + seed), pk.thin(va, q, 200 + seed)
t0 = time.time(); out = pk.run_all(Z, tr_n, va_n, te, shape=cnt.shape)
res = dict(bead=k, n=int((tr_n + va_n).sum()), n_target=n, seed=seed, s=time.time() - t0,
           **{m: {kk: v for kk, v in d.items() if kk != '_P'} for m, d in out.items()})
os.makedirs('results/psf/v3/lc', exist_ok=True); json.dump(res, open(f'results/psf/v3/lc/b{k}_{n}_{seed}.json', 'w'), indent=1)
print(k, n, seed, {m: round(d['ll'], 5) for m, d in out.items()}, f'{res["s"]:.0f}s', flush=True)
