"""Paper 3: the three-mode partially coherent model of p3_partial3.py (one two-slit mode, two single-slit modes of band
fill fc / 2, the flat background N = 0; against them the one-mode fit with N = 1 and the three-mode fit with N = 1) on a
double-slit record, or on its replica k (data drawn from the record's Fresnel fit, which is partially coherent: V, rho
in its parameters; seed 100 + k, as p3_synth_replica.py; P3_NMULT=m: m times the record's n, file suffix _x<m>; P3_WIDE=E: the
Fresnel truth on a window E px wider on each side with the background's total share kept, suffix _w<E>). Split by seed as p3_slit_main.py.
  python p3_partial3_real.py <set> <seed> [replica]  -> results/p3_partial3_real/<set>[_rep<k>]_s<seed>.json"""
import os, sys, json, time, numpy as np
os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_SOLVER', 'lbfgs')
name, SEED = sys.argv[1], int(sys.argv[2]); REP = int(sys.argv[3]) if len(sys.argv) > 3 else None
sys.argv = sys.argv[:1]; sys.path.insert(0, '.')
import dslit_job as dj, blml_trivial as bt
from blml_binned import BLMLBinned
OUT = 'results/p3_partial3_real'; os.makedirs(OUT, exist_ok=True)
MULT = int(os.environ.get('P3_NMULT', '1'))          # replicas only: draw MULT times the record's n
WIDE = int(os.environ.get('P3_WIDE', '0'))           # replicas only: crop-free window, WIDE px added on each side
fn = f'{OUT}/{name}' + ('' if REP is None else f'_rep{REP}') + (f'_x{MULT}' if MULT > 1 else '') + (f'_w{WIDE}' if WIDE else '') + f'_s{SEED}.json'
if os.path.exists(fn):
    sys.exit(0)
if REP is None:
    from p3_buildup import setup
    x, W = setup(name)[:2]; P = None
else:
    P = np.load(os.path.join(dj.DATA, f'dslit_{name}_fresnel_P.npy')).astype(float); P /= P.sum(); W = len(P)
    n = len(np.load(os.path.join(dj.DATA, f'dslit_{name}_points.npy')))
    if WIDE:                                     # crop-free: the Fresnel signal WIDE px beyond each crop edge, the
        th = np.array(json.load(open(f'results/dslit_1d/{name}/flat/{name}_fresnel.json'))['theta'])   # background's
        dj.BFIX = np.array([np.nan]); beta = float(np.atleast_1d(dj.ref_P(th, np.ones((1, W)) / W, 0, True)[2])[0])
        th[0] += WIDE; W = W + 2 * WIDE; q = dj.ref_parts(th, (W,), 0, True)[0]   # total share kept (not its level)
        P = beta / W + (1 - beta) * q / q.sum(); P /= P.sum()
    x = np.random.default_rng(100 + REP).choice(W, size=n * MULT, p=P)
u = np.random.default_rng(SEED).random(len(x)); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))
fold = np.random.default_rng(1000 + SEED).integers(0, 5, len(x))
cnt = lambda m: np.bincount(x[m], minlength=W).astype(float)
TV, TE = cnt(part < 2), cnt(part == 2)
FOLDS = [(cnt((part < 2) & (fold != k)), cnt((part < 2) & (fold == k))) for k in range(5)]
kl = (lambda Q: float((P * np.log(P / np.maximum(Q / Q.sum(), 1e-300))).sum())) if P is not None else (lambda Q: None)
fr = json.load(open(f'results/dslit_1d/{name}/flat/{name}_fresnel.json'))['params']
fold_of = {id(T): k for k, (T, _) in enumerate(FOLDS)}
FILLS = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.36, 0.33, 0.3, 0.25]
G = list(dj.FC_GRID[dj.FC_GRID >= 2.0 / W]); dj.BFIX = np.array([np.nan])
fc_ref, _ = dj.cv_select(FOLDS, lambda f, T: dj.bg_fit(T, [f], (1,), True, 1, max_sweeps=5000, tol=1e-6)[0], G, patience=6)
Gs = [float(f) for f in G if 0.8 * fc_ref <= f <= 1.6 * fc_ref]
def modes(fc, w, three, nbg=1):
    m = dj.bg_modes([fc], (W,), (nbg,), 1, w)                       # [two-slit mode, background mode]
    return m[:1] + ([bt.make_modes([w * fc / 2], ['sinc'])[0]] * 2 if three else []) + m[1:]
def fit(T, fc, w, three, init=None, nbg=1):
    m = BLMLBinned(tol=1e-6, max_sweeps=5000, ll_tol=1e-10, engine=dj.ENGINE, solver=dj.SOLVER).fit(
        T, modes(fc, w, three, nbg), complex_g=True, init=init)
    U = dj._np(m.Se_); E = dj._np(m.e_hat_); Q = np.array([float(np.real(np.sum(np.conj(E[k]) * U[k]))) for k in range(len(E))])
    Pm = np.maximum(dj._np(m.density_), 1e-300); return Pm / Pm.sum(), m, Q / Q.sum()
res = dict(set=name, seed=SEED, rep=REP, n=len(x), fresnel=dict(V=fr['V'], rho=fr['rho'], a_over_d=fr['a_over_d']), fc_ref=fc_ref); t0 = time.time()
for lab, three, nbg in (('one', False, 1), ('three', True, 1), ('three_flat', True, 0)):
    SL = {}
    for k, (T, _) in enumerate(FOLDS):                              # per fill, increasing fc, warm-started
        for w in FILLS:
            prev = None
            for f in Gs:
                P_, m_, _ = fit(T, f, w, three, prev, nbg); prev = m_; SL[(f, w, k)] = P_
    C_ = sorted(((f, w) for f in Gs for w in FILLS), key=lambda c: (c[0] * c[1], c[0]))
    (fc, w), tab = dj.cv_select(FOLDS, lambda c, T: SL[(c[0], c[1], fold_of[id(T)])], C_)
    Q, m, sh = fit(TV, fc, w, three, None, nbg); a = w * fc / 2
    r = dict(fc=fc, fill=w, a=a, d=fc - a, cv=max(rr['cv'] for rr in tab), test=dj.ll(Q, TE), kl=kl(Q), share=sh.tolist(),
             cv_by_fill={str(v): max(rr['cv'] for rr in tab if rr['cand'][1] == v) for v in FILLS})
    if three:
        r['V_hat'] = float(sh[0] / sh[:3].sum())
    res[lab] = r
res['s'] = time.time() - t0
json.dump(res, open(fn, 'w'), indent=1)
print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != 'cv_by_fill'} if isinstance(v, dict) else v) for k, v in res.items()}))
