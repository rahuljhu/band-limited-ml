"""Paper 7: the 53 isolated beads at full counts (results/p7_beads/b<k>_full_0.json): empirical - ML per photon against
the share of the window's counts on its edge ring (where the window cuts a neighbour). -> results/p7_beads/summary.json"""
import sys, json, glob, numpy as np
sys.path.insert(0, 'results/psf'); sys.path.insert(0, '.'); import psfkit as pk
rows = []
for f in sorted(glob.glob('results/p7_beads/b*_full_0.json')):
    r = json.load(open(f)); c, _ = pk.bead_counts(r['bead'], 15); m = np.ones(c.shape, bool); m[1:-1, 1:-1] = False
    rows.append(dict(bead=r['bead'], edge=float(c[m].sum() / c.sum()), d=r['emp_minus_ml']['mean'], se=r['emp_minus_ml']['se'],
                     per_px=r['per_pixel'], **{k: r[k]['test'] for k in ('ml', 'empirical', 'histogram', 'kde')},
                     lam=r['empirical']['lam'], lam_edge=r['empirical']['lam_at_edge']))
e, d, se = (np.array([x[k] for x in rows]) for k in ('edge', 'd', 'se'))
z = d / se; rho = float(np.corrcoef(np.argsort(np.argsort(e)), np.argsort(np.argsort(d)))[0, 1])
lo = e < np.median(e)
S = dict(n=len(rows), wins=int((z > 2).sum()), losses=int((z < -2).sum()), ties=int((abs(z) <= 2).sum()), median_d=float(np.median(d)),
         spearman=rho, median_edge=float(np.median(e)), median_d_low_edge=float(np.median(d[lo])), median_d_high_edge=float(np.median(d[~lo])),
         kde_beats_ml=int(sum(x['kde'] > x['ml'] for x in rows)), emp_beats_kde=int(sum(x['empirical'] > x['kde'] for x in rows)),
         lam_edge=int(sum(x['lam_edge'] for x in rows)), per_px=[float(np.min([x['per_px'] for x in rows])), float(np.max([x['per_px'] for x in rows]))], rows=rows)
json.dump(S, open('results/p7_beads/summary.json', 'w'), indent=1)
print({k: v for k, v in S.items() if k != 'rows'})
