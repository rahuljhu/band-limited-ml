"""All methods on one real bead (raw counts, per-pixel thinning).  Usage: python run_bead.py <k>"""
import sys, json, time, numpy as np
sys.path.insert(0, 'results/psf'); import psfkit as pk
k = int(sys.argv[1]); H = 15; Z = pk.lattice(H)
cnt, pos = pk.bead_counts(k, H); tr, va, te = pk.split(cnt)
t0 = time.time(); out = pk.run_all(Z, tr, va, te, shape=cnt.shape)
res = dict(bead=k, pos=pos, photons=int(cnt.sum()), n_trainval=int((tr + va).sum()), n_test=int(te.sum()), s=time.time() - t0,
           **{m: {kk: v for kk, v in d.items() if kk != '_P'} for m, d in out.items()})
json.dump(res, open(f'results/psf/v3/bead{k}.json', 'w'), indent=1)
np.savez(f'results/psf/v3/bead{k}_P.npz', cnt=cnt, tr=tr, va=va, te=te, **{m.replace('@', '_').replace('+', '_'): d['_P'] for m, d in out.items()})
print(k, {m: round(d['ll'], 5) for m, d in out.items()}, f'{res["s"]:.0f}s', flush=True)
