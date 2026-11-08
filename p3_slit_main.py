"""Paper 3, the two-slit kernel as the main estimator of the double-slit records (and their replicas): for one record
(or replica) and one random split, complex BLML with the two-slit kernel
  estimated:  cut-off fc, fill w and background harmonics N all by 5-fold CV (per N, (fc, w) by the one-SE rule with
              parsimony fc w over fc in 0.8-2 times the physical band and w on a grid; then the N with the best CV)
  published:  the published slit ratio a/d (fill 2r/(1+r)) and fc = (1 + r) / P from the record's fringe period P;
              only N by CV
and the KDE (flat background by EM, bandwidth by CV); scored on the split's test particles (paired differences).
  [P3_NS=0,1,2 P3_OUT=dir] python p3_slit_main.py <set> <seed> [replica]  -> results/p3_slit_main/<set>[_rep<k>]_s<seed>.json
replica k: data drawn from the record's Fresnel fit (the replicas of p3_synth_replica.py, seed 100 + k), split as the
record; the KL from that truth is also reported."""
import os, sys, json, time, numpy as np
os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_SOLVER', 'lbfgs'); os.environ.setdefault('BLML_COLLAPSE', '1')
name, SEED = sys.argv[1], int(sys.argv[2]); REP = int(sys.argv[3]) if len(sys.argv) > 3 else None
sys.argv = sys.argv[:1]; sys.path.insert(0, '.')
import dslit_job as dj
OUT = os.environ.get('P3_OUT', 'results/p3_slit_main'); os.makedirs(OUT, exist_ok=True)
fn = f'{OUT}/{name}' + ('' if REP is None else f'_rep{REP}') + (f"_w{os.environ['P3_WIDE']}" if os.environ.get('P3_WIDE', '0') != '0' else '') + f'_s{SEED}.json'
if os.path.exists(fn):
    sys.exit(0)
PUB = dict(photon=100 / 500, ghost=100 / 500, photon_mm=100 / 500, electron=62 / 272)
FILLS = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.36, 0.33, 0.3, 0.25]
NS = tuple(int(v) for v in os.environ.get('P3_NS', '1,2,3,5').split(','))   # background harmonics by CV (N >= 3 reaches the fringe band)
th = json.load(open(f'results/dslit_1d/{name}/flat/{name}_fresnel.json'))['params']; Pfr = th['P']
fr = json.load(open(f'results/dslit_1d/{name}/flat/{name}_fraunhofer.json'))['params']; phys = (1 + fr['a_over_d']) / fr['P']
if REP is None:
    _a, sys.argv = sys.argv, sys.argv[:1]
    from p3_buildup import setup
    x, W, bflat, trv, te = setup(name); Ptrue = None
else:
    Ptrue = np.load(os.path.join(dj.DATA, f'dslit_{name}_fresnel_P.npy')).astype(float); Ptrue /= Ptrue.sum(); W = len(Ptrue)
    n = len(np.load(os.path.join(dj.DATA, f'dslit_{name}_points.npy')))
    WIDE = int(os.environ.get('P3_WIDE', '0'))       # crop-free replica: the Fresnel signal WIDE px beyond each crop
    if WIDE:                                         # edge, the background's total share kept; fn suffix _w<WIDE>
        th_ = np.array(json.load(open(f'results/dslit_1d/{name}/flat/{name}_fresnel.json'))['theta'])
        dj.BFIX = np.array([np.nan]); beta_ = float(np.atleast_1d(dj.ref_P(th_, np.ones((1, W)) / W, 0, True)[2])[0])
        th_[0] += WIDE; W = W + 2 * WIDE; q_ = dj.ref_parts(th_, (W,), 0, True)[0]
        Ptrue = beta_ / W + (1 - beta_) * q_ / q_.sum(); Ptrue /= Ptrue.sum()
    x = np.random.default_rng(100 + REP).choice(W, size=n, p=Ptrue); bflat = np.ones((1, W)) / W
u = np.random.default_rng(SEED).random(len(x)); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))
fold = np.random.default_rng(1000 + SEED).integers(0, 5, len(x))
cnt = lambda m: np.bincount(x[m], minlength=W).astype(float)
TV, TE = cnt(part < 2), cnt(part == 2)
FOLDS = [(cnt((part < 2) & (fold != k)), cnt((part < 2) & (fold == k))) for k in range(5)]
nV = sum(V.sum() for _, V in FOLDS); fold_of = {id(T): k for k, (T, _) in enumerate(FOLDS)}
dj.BFIX = np.array([np.nan]); t0 = time.time()
G = [float(f) for f in dj.FC_GRID if 0.8 * phys <= f <= 2.0 * phys]
fit = lambda T, f, w, N, init=None: dj.bg_fit(T, [f], (N,), True, 1, init=init, max_sweeps=5000, tol=1e-6, fill=w)
kl = (lambda Q: float((Ptrue * np.log(Ptrue / np.maximum(Q / Q.sum(), 1e-300))).sum())) if Ptrue is not None else (lambda Q: None)
res = dict(set=name, seed=SEED, rep=REP, phys=phys)
best = None
for N in NS:                                        # estimated geometry: (fc, w) per N, then N
    SL = {}
    for k, (T, _) in enumerate(FOLDS):
        for w in FILLS:
            prev = None
            for f in G:
                P_, prev, _ = fit(T, f, w, N, prev); SL[(f, w, k)] = P_
    C_ = sorted(((f, w) for f in G for w in FILLS), key=lambda c: (c[0] * c[1], c[0]))
    (fc, w), tab = dj.cv_select(FOLDS, lambda c, T: SL[(c[0], c[1], fold_of[id(T)])], C_)
    cvb = max(r['cv'] for r in tab)
    res[f'est_N{N}'] = dict(fc=fc, fill=w, cv=cvb)
    if best is None or cvb > best[0]:
        best = (cvb, N, fc, w)
_, N, fc, w = best
Q = fit(TV, fc, w, N)[0]; res['est'] = dict(N=N, fc=fc, fill=w, ratio=fc / phys, test=dj.ll(Q, TE), kl=kl(Q))
Le = np.log(np.maximum(Q / Q.sum(), 1e-300))
r = PUB[name]; fcp, wp = (1 + r) / Pfr, 2 * r / (1 + r); bestp = None   # published geometry: only N
for N in NS:
    cv = sum(float((V * np.log(np.maximum(fit(T, fcp, wp, N)[0], 1e-300))).sum()) for T, V in FOLDS) / nV
    res[f'pub_N{N}'] = dict(cv=cv)
    if bestp is None or cv > bestp[0]:
        bestp = (cv, N)
Q = fit(TV, fcp, wp, bestp[1])[0]; res['pub'] = dict(N=bestp[1], fc=fcp, fill=wp, test=dj.ll(Q, TE), kl=kl(Q))
Lp = np.log(np.maximum(Q / Q.sum(), 1e-300))
HG = [(h,) for h in (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24)]
h, _ = dj.cv_select(FOLDS, lambda h_, T: dj.em(T, bflat, dj.kde_q(h_))[0], HG[::-1])
Qk = dj.em(TV, bflat, dj.kde_q(h))[0]; res['kde'] = dict(h=h[0], test=dj.ll(Qk, TE), kl=kl(Qk))
Lk = np.log(np.maximum(Qk / Qk.sum(), 1e-300))
for a, La in (('est', Le), ('pub', Lp)):
    d = np.repeat(La - Lk, TE.astype(int)); res[f'{a}_minus_kde'] = dict(diff=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))))
res['s'] = time.time() - t0
json.dump(res, open(fn, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if not k.startswith(('est_N', 'pub_N'))}))
