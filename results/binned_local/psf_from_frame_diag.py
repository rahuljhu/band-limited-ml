"""Diagnosis of psf_from_frame.py: (self) each bead scored under its own aligned fitted window through the same
shift / flat-background / window pipeline (tests the pipeline: should match the frame fit); (local) the PSF averaged
over the bead's 30 nearest other beads (tests field dependence); and the global-PSF loss against field position.
Output results/binned_local/psf_from_frame_diag.json"""
import json, numpy as np
src = open('results/binned_local/psf_from_frame.py').read().split('t0 = time.time()')[0]
exec(src)
J = json.load(open('results/binned_local/psf_from_frame.json'))
tree = cKDTree(np.c_[y, x]); rng = np.random.default_rng(1); sel = rng.choice(nb, 150, replace=False)
rows = []
for i in sel:
    ct, ce = win(tr, i, hw), win(te, i, hw)
    sA, n = score(win(F, i, hw), ce)
    own = average([i], F, rounds=2)                                     # its own fitted window, aligned
    _, a, b, q = fit_bead(own, ct); sS, _ = score(a * q + b, ce)
    _, nbr = tree.query([y[i], x[i]], k=31); nbr = [j for j in nbr if j != i][:30]
    loc = average(nbr, F); _, a, b, q = fit_bead(loc, ct); sL, _ = score(a * q + b, ce)
    rows.append(dict(i=int(i), y=int(y[i]), x=int(x[i]), n=n, self=(sS - sA) / n, local=(sL - sA) / n))
    print(f'bead {i:4d} ({y[i]:4d},{x[i]:4d})  self-A {(sS - sA) / n:+.4f}  local-A {(sL - sA) / n:+.4f}', flush=True)
w = np.array([r_['n'] for r_ in rows])
summ = {k: float(np.average([r_[k] for r_ in rows], weights=w)) for k in ('self', 'local')}
pb = J['per_bead']; dB = np.array([p['B'] - p['A'] for p in pb]); yy = np.array([p['y'] for p in pb]); xx = np.array([p['x'] for p in pb])
rad = np.hypot(yy - 1024, xx - 1024)
summ['global_loss_vs_radius_corr'] = float(np.corrcoef(rad, dB)[0, 1])
summ['global_loss_by_radius_quartile'] = [float(dB[(rad >= lo) & (rad < hi)].mean()) for lo, hi in
                                          zip(np.percentile(rad, [0, 25, 50, 75]), np.percentile(rad, [25, 50, 75, 100.01]))]
print(json.dumps(summ, indent=1))
json.dump(dict(summary=summ, rows=rows), open('results/binned_local/psf_from_frame_diag.json', 'w'), indent=1)
