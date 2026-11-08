"""Bead k, no background model: cut-off selection on counts (selector.select_counts, no grid, fc <= 1/px).
Models: real r1, complex r1, complex r2 (shared fc), real 2 separate modes. Fit on train, validate on val."""
import sys, json, time, numpy as np
sys.path.insert(0, 'results/psf'); sys.path.insert(0, '.')
import psfkit as pk, selector as sel, blml_trivial as bt
k = int(sys.argv[1]) if len(sys.argv) > 1 else 0; H = 15; Z = pk.lattice(H)
cnt, pos = pk.bead_counts(k, H); tr, va, te = pk.split(cnt)
out = dict(bead=k, photons=int(cnt.sum()), fc_phys=pk.FC_PHYS)
for lab, kw in [('real_r1', {}), ('complex_r1', dict(complex_g=True)), ('complex_r2', dict(complex_g=True, rank=2)),
                ('real_2modes', dict(rank=2, separate_modes=True))]:
    tmpl = bt.make_modes(np.nan, d=2, rank=kw.pop('rank', 1))
    t0 = time.time(); sm, ll = sel.select_counts(Z, tr.ravel(), va.ravel(), 1.0, tmpl, tag=lab, **kw)
    F = bt.modes_fc(sm); fc = F if kw.get('separate_modes') else F[0]
    out[lab] = dict(fc=np.asarray(fc).tolist(), val_ll=ll, s=time.time() - t0)
    print(lab, out[lab], flush=True)
    json.dump(out, open(f'results/psf/v4/bead{k}_select.json', 'w'), indent=1)
