"""Paper 3 / 7, electron double slit: BLMLEmpirical, rank-1 complex (hidden-phase retrieval, window normalisation) on the
raw counts, no background model (no EM): f = |g|^2 normalised over the crop is the density scored; fc on validation;
test LL per electron; spectra. The same split as dslit_job.py.
Step 1 (lam): at the ML complex fit's fc, on the training counts, the fit for lam = 1e-2 ... 1e-7; lam is the smallest value whose inner CG solves converge (residual < tol within the cap)
and whose result agrees with the next larger lam (relative change of f < 1e-3): the ridge is only a numerical
stabiliser, the band does the regularisation.
Step 2: fc along the fringes on validation (across-line fc fixed at the ML fit's), refit on train + validation, score.
Output results/dslit_final/electron_empirical.json (+ _P.npy, _line.npy for dslit_analyze.py)."""
import os, sys, json, time, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_binned import BLMLEmpirical, _np
D = os.path.expanduser('~/data_repo/dslit_gpu/'); OUT = 'results/dslit_electron_aux'
pts = np.load(D + 'dslit_electron_points.npy'); b = np.load(D + 'dslit_electron_b.npy')[0]; BETA = float(np.load(D + 'dslit_electron_beta.npy')[0])
SH = b.shape; axis = 1
ij = [np.clip(np.floor(pts[:, k] + 0.5).astype(int), 0, SH[k - 1] - 1) for k in (1, 2)]
u = np.random.default_rng(0).random(len(pts)); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))
def img(s):
    C = np.zeros(SH); np.add.at(C, (ij[0][s], ij[1][s]), 1); return C
TR, VA, TE, TV = img(part == 0), img(part == 1), img(part == 2), img(part < 2)
ml = json.load(open(f'{OUT}/electron_complex.json')); fr = json.load(open(f'{OUT}/electron_fraunhofer.json'))['params']
def ll(P, N):
    P = P / P.sum(); return float((N * np.log(np.maximum(P, 1e-300))).sum() / N.sum())
def efit(w, fc, lam, init=None, outer_tol=1e-5):
    m = BLMLEmpirical(reg='ridge', lam=lam, tol=1e-6, max_iter=2000, max_outer=400, outer_tol=outer_tol, retrieve=True).fit(
        w, bt.make_modes(list(fc), ['sinc', 'sinc']), complex_g=True, init=init)
    q = np.maximum(_np(m.density_), 1e-300); return q / q.sum(), m
res = dict(ml_fc=ml['fc'], lam_scan=[])
# ---- step 1: lam
w = TR
prev_q, lam_sel = None, None
for lam in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
    t = time.time(); q, m = efit(w, ml['fc'], lam)
    dq = None if prev_q is None else float(np.abs(q - prev_q).sum())
    row = dict(lam=lam, cg_converged=bool(m.converged_), cg_residual=float(m.residual_), outer=int(m.n_outer_), outer_crit=float(m.outer_crit_),
               change_vs_previous=dq, val_ll=ll(q, VA), s=time.time() - t)
    res['lam_scan'].append(row); print(json.dumps(row), flush=True)
    if row['cg_converged']: lam_sel = lam
    prev_q = q
res['lam'] = lam_sel; print('lam selected (smallest with converged CG):', lam_sel, flush=True)
# ---- step 2: fc along the fringes on validation, then refit and score
grid = {}
for f1 in (0.012, 0.016, 0.02, 0.024, 0.032, 0.04, 0.056):
    fc = [ml['fc'][0], f1]; P, m = efit(TR, fc, lam_sel); grid[f1] = ll(P, VA); print('fc', f1, round(grid[f1], 5), flush=True)
f1 = max(grid, key=grid.get); fc = [ml['fc'][0], f1]
t = time.time(); P, m = efit(TV, fc, lam_sel)
G = _np(m.Se_)[0]; line = G[int(round(fr['t0'])), :]
res.update(part='empirical', fc=fc, val_grid={str(k): v for k, v in grid.items()}, test_ll=ll(P, TE), beta=[0.0],
           window_norm=float(m.window_norm_), plane_norm=float(m.plane_norm_), outer=int(m.n_outer_), s=time.time() - t)
np.save(f'{OUT}/electron_empirical_P.npy', (P / P.sum()).astype(np.float32)); np.save(f'{OUT}/electron_empirical_line.npy', line)
json.dump(res, open(f'{OUT}/electron_empirical.json', 'w'), indent=1)
print({k: v for k, v in res.items() if k not in ('lam_scan', 'val_grid')}, flush=True)
