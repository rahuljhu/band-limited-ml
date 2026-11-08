"""Real vs complex BLML on one bead (raw counts, same thinning split as v3).  Usage: python run_bead_complex.py <k>"""
import sys, json, time, numpy as np
sys.path.insert(0, 'results/psf'); import psfkit as pk
k = int(sys.argv[1]); H = 15; Z = pk.lattice(H)
cnt, pos = pk.bead_counts(k, H); tr, va, te = pk.split(cnt)
M = ['cBLML', 'cBLML@phys', 'cBLML+bg', 'cBLML+bg@phys', 'BLML', 'BLML+bg', 'BLML+bg@phys', 'histogram', 'KDE', 'GMM+bg']
t0 = time.time(); out = pk.run_all(Z, tr, va, te, shape=cnt.shape, methods=M)
res = dict(bead=k, photons=int(cnt.sum()), s=time.time() - t0, **{m: {kk: v for kk, v in d.items() if kk != '_P'} for m, d in out.items()})
import os; os.makedirs('results/psf/v4', exist_ok=True)
json.dump(res, open(f'results/psf/v4/bead{k}.json', 'w'), indent=1)
np.savez(f'results/psf/v4/bead{k}_P.npz', cnt=cnt, tr=tr, va=va, te=te, **{m.replace('@', '_').replace('+', '_'): d['_P'] for m, d in out.items()})
print(k, {m: round(d['ll'], 5) for m, d in out.items()}, {m: d.get('fc') for m, d in out.items() if 'fc' in d}, f'{res["s"]:.0f}s', flush=True)
