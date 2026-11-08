"""Paper 3 / 7, electron double slit: BLMLEmpiricalStreaming, rank-1 complex, one mode on the raw counts (no background
term), the kernel fixed at the binned ML complex fit's fc (results/dslit_final/electron_complex.json), and the ridge lam
learned online by Adam on log lam from the prequential score. FCUPD=1: fc learned in the same pass by the library's
streaming learner (implicit prequential gradient, Adam on log fc), from FC0 (default the ML fit's fc); ANSC=1: the
variance-stabilised target sqrt(N + 3/8).
Each batch (NB electrons, arrival order; train + validation electrons, the test electrons held out) is scored by the
current fit before it is absorbed; d score / d log lam by the implicit-function theorem at the fitted coefficients,
with the estimator's own pieces (_ift_state, _dP_de, _hvp, _free): lam enters the ridge objective only through
lam c'Sc / n^2, so d grad / d log lam = 2 lam U / n^2 and d score / d log lam = -<v, 2 lam U / n^2>, H v = d score / dc.
A finite-difference check of that derivative on the first learning batch (warm refits at lam e^{+-h}) is printed.
Adam: blml_maps.adam_step on log lam (lr, betas as the streaming fc learner), every PE batches.
Output results/dslit_final/electron_empirical_lam.json (history: n, preq, lam), test LL of the final fit."""
import os, sys, json, time, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_binned import BLMLEmpiricalStreaming, _np, _icg, _rdot
from blml_maps import adam_step
D = os.path.expanduser('~/data_repo/dslit_gpu/'); OUT = 'results/dslit_electron_aux'
NB, PE, LR, LAM0, OUTER = int(os.environ.get('NB', 200)), 1, float(os.environ.get('LR', 0.3)), float(os.environ.get('LAM0', 1e-2)), 60
pts = np.load(D + 'dslit_electron_points.npy'); SH = np.load(D + 'dslit_electron_b.npy').shape[-2:]
ij = [np.clip(np.floor(pts[:, k] + 0.5).astype(int), 0, SH[k - 1] - 1) for k in (1, 2)]
u = np.random.default_rng(0).random(len(pts)); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))
def img(s):
    C = np.zeros(SH); np.add.at(C, (ij[0][s], ij[1][s]), 1); return C
TE = img(part == 2); stream = np.nonzero(part < 2)[0]                     # arrival order (the points are in arrival order)
fc = json.load(open(f'{OUT}/electron_complex.json'))['fc']
FCUPD = os.environ.get('FCUPD', '0') == '1'                                 # learn fc too (the library's streaming fc learner)
if os.environ.get('FC0'): fc = [float(v) for v in os.environ['FC0'].split(',')]
modes = bt.make_modes(list(fc), ['sinc', 'sinc'])
def score(m, N):
    p = _np(m.density_).astype(float); p = p / p.sum(); return float((N * np.log(np.maximum(p, 1e-300))).sum() / N.sum())
def lam_grad(m, counts):
    xp = m.xp; st = m._ift_state(); F = st['F']; occ = counts > 0
    eps = 1e-3 * float(np.median(_np(F)[occ])) / (float(F.sum()) * m.Delta_)       # density units, as the fc learner
    b = m._dP_de(st, xp.asarray(counts), eps); mask = m._free(st)
    v, it, res = _icg(xp, lambda x: m._hvp(st, x) * mask, b * mask, 1e-6, 300)
    return -_rdot(xp, v, (2 * m.lam * st['U'] / m.n_ ** 2) * mask), eps
def preq(m, counts, eps=0.0):
    F = (np.abs(_np(m.Se_)) ** 2).sum(0); f = F / F.sum(); occ = counts > 0
    return float((counts[occ] * np.log(np.maximum(f[occ] + eps, 1e-300))).sum() / counts.sum())
m = BLMLEmpiricalStreaming(reg='ridge', lam=LAM0, anscombe=bool(int(os.environ.get('ANSC', '0'))), tol=1e-6, max_iter=2000, max_outer=400, outer_tol=1e-5, retrieve=True,
                           fc_update='adam' if FCUPD else None, lr=LR, param_every=PE, burn_in=1000, batch_outer=OUTER)
adam = [0.0, 0.0, 0]; hist = []; pool = []; t0 = time.time(); checked = False
for b0 in range(0, len(stream), NB):
    s = np.zeros(len(pts), bool); s[stream[b0:b0 + NB]] = True; C = img(s)
    if m.n_batches_ == 0:
        m.partial_fit(C, modes, complex_g=True); hist.append(dict(n=float(m.n_), preq=None, lam=m.lam)); continue
    g, eps = lam_grad(m, C); p = preq(m, C)
    if not checked and m.n_ >= 1000:                                         # finite-difference check, once
        import copy
        h = 0.2; fd = []
        for sgn in (+1, -1):
            mm = copy.copy(m); mm.lam = m.lam * np.exp(sgn * h); mm.max_outer = 400
            mm._fit_counts(_np(mm.w_), init=m); fd.append(preq(mm, C, eps))
        print(f'FD check at lam {m.lam:.2e}: implicit {g:+.5f}  finite-difference {(fd[0] - fd[1]) / (2 * h):+.5f}', flush=True)
        checked = True
    pool.append(g); m.partial_fit(C)
    if len(pool) >= PE and m.n_ >= 1000:
        G = float(np.mean(pool)); pool = []
        m.lam = float(adam_step(m.lam, G, adam, 'scale', LR, (0.9, 0.999), dlog=True))
    hist.append(dict(n=float(m.n_), preq=p, lam=m.lam, grad=g, fc=[f.tolist() for f in m._fc_now()]))
    print(f'n={m.n_:.0f} preq={p:.4f} dscore/dloglam={g:+.4f} lam={m.lam:.3e} fc={np.round(m._fc_now()[0], 4)} {time.time() - t0:.0f}s', flush=True)
m._fit_counts(_np(m.w_), init=m)                                             # final fit at the learned lam, to convergence
res = dict(fc0=fc, fc=[f.tolist() for f in m._fc_now()], fc_learned=FCUPD, NB=NB, lr=LR, lam0=LAM0, lam_final=m.lam, test_ll=score(m, TE), window_norm=float(m.window_norm_),
           plane_norm=float(m.plane_norm_), history=hist)
res['anscombe'] = bool(int(os.environ.get('ANSC', '0'))); json.dump(res, open(f"{OUT}/electron_empirical_lam{'_anscombe' if res['anscombe'] else ''}{'_fc' + os.environ.get('FC0', 'ml') if FCUPD else ''}.json", 'w'), indent=1)
print({k: v for k, v in res.items() if k != 'history'}, flush=True)
