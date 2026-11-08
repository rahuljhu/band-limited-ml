"""Paper 3: what the periodic two-slit fit (p3_periodic_main.py) recovers from a double-slit record, against the
record's Fresnel fit on the same window. For each split / replica the selected periodic model is refitted on train +
validation (same data and split as p3_periodic_main.py), with its CV-chosen background N and with a flat one (N = 0,
under which the coherent field carries the fringe contrast: the coherence prior). Scores (as p3_cbench_phase.py):
d (|a_k|^2 centroids, cycles/px; reference 1/P), rho (side energies, min(rho, 1/rho)), the envelope centre c_env and
the fringe contrast C = 2 |Q_m| / Q_0 at the fringe harmonic m = round(L/P); for the record's split 0 and the Fresnel
fit, the field's unit-norm Fourier coefficients a_k on the window (|k| <= L/8: the aperture on the window's harmonics). Reference values from the Fresnel fit's
field and density on the window (its own rho and V, separated only through its defocus eps, also reported with
gamma = 2 rho V / (1 + rho^2), the one combination two equal slits' far field determines).
With P3_THREE=1: the periodic three-mode model instead (coherent two-slit mode at the selected harmonics, lo >= 1, two
single-slit modes |k| <= hi - lo, flat background; as p3_cbench.py), scores plus V_hat, its coherent share.
  [P3_THREE=1] python p3_periodic_metrics.py <set>   -> results/p3_periodic_metrics/<set>[_three].json"""
import os, sys, json, glob, numpy as np
os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_SOLVER', 'lbfgs'); os.environ.setdefault('BLML_COLLAPSE', '1')
name = sys.argv[1]; sys.argv = sys.argv[:1]; sys.path.insert(0, '.')
import dslit_job as dj
from blml_binned import BLMLBinned
OUT = 'results/p3_periodic_metrics'; os.makedirs(OUT, exist_ok=True); THREE = os.environ.get('P3_THREE', '0') == '1'
fr = json.load(open(f'results/dslit_1d/{name}/flat/{name}_fresnel.json'))['params']
Pf, c0 = fr['P'], fr['s0']; dj.BFIX = np.array([np.nan])
Ptrue = np.load(os.path.join(dj.DATA, f'dslit_{name}_fresnel_P.npy')).astype(float); Ptrue /= Ptrue.sum(); W = len(Ptrue)
lg = lambda p: float(np.log(p / (1 - p))) if p < 1 else 30.0
th = np.array([c0, Pf, lg(fr['a_over_d']), fr['eps'], np.log(fr['rho']), fr['phi'], lg(fr['V']), 0.0, 0.0])
field = dj.ref_parts(th, (W,), 0, True)[1]
m_ = int(min(c0, W - c0) / Pf - 0.5); a0, a1 = int(round(c0 - (m_ + 0.5) * Pf)), int(round(c0 + (m_ + 0.5) * Pf)); L = a1 - a0
mf = int(round(L / Pf)); k = np.rint(np.fft.fftfreq(L) * L); pos = (k > 0) & (k < L / 4)
def fscores(g):
    a = np.fft.fft(g); ap = {int(kk): a[i] for i, kk in enumerate(k)}; ks = [int(kk) for kk in k[pos]]
    e = np.abs(a) ** 2; neg = (k < 0) & (k > -L / 4)
    cp_, cn = (e[pos] * k[pos]).sum() / e[pos].sum(), (e[neg] * k[neg]).sum() / e[neg].sum(); rho = np.sqrt(e[pos].sum() / e[neg].sum())
    grad = sum(ap[q + 1] * np.conj(ap[q]) + ap[-q] * np.conj(ap[-q - 1]) for q in ks[:-1])
    return dict(d=float((cp_ - cn) / L), rho=float(min(rho, 1 / rho)), c_env=float((-np.angle(grad) * L / (2 * np.pi)) % L))
KS = np.arange(-int(L / 8), int(L / 8) + 1)
coef = lambda g: (lambda a: [[float(v.real), float(v.imag)] for v in a[KS % L] / np.sqrt((np.abs(a) ** 2).sum())])(np.fft.fft(g))
C_ = lambda Q: float(2 * abs(np.fft.fft(Q / Q.sum())[mf]))
ref = dict(fscores(field[a0:a1]), C=C_(Ptrue[a0:a1]), coef=coef(field[a0:a1]), ks=KS.tolist(), d_nominal=1 / Pf, rho_fresnel=min(fr['rho'], 1 / fr['rho']), V_fresnel=fr['V'],
           gamma_fresnel=2 * fr['rho'] * fr['V'] / (1 + fr['rho'] ** 2), eps=fr['eps'], L=L)
line = lambda m: np.asarray(dj._np(m.Se_))[0] / m.n_
def pfit(T, hi, lo, N):
    fc = (2 * hi + 0.5) / L; w = 1.0 if lo == 0 else 1.0 - (lo - 0.5) / (hi + 0.25)
    sig = {0: dict(kind='dirichlet', fc=fc, period=float(L), **({} if w >= 1 else dict(fill=w)))}
    bg = {0: dict(kind='dirichlet', fc=(N + 0.5) * 2 / L, period=float(L))}
    m = BLMLBinned(tol=1e-6, max_sweeps=5000, ll_tol=1e-10, engine=dj.ENGINE, solver=dj.SOLVER).fit(T, [sig, bg], complex_g=True)
    U = dj._np(m.Se_); E = dj._np(m.e_hat_); Qs = np.array([float(np.real(np.sum(np.conj(E[j]) * U[j]))) for j in range(2)])
    P_ = np.maximum(dj._np(m.density_), 1e-300); return P_ / P_.sum(), line(m), float(Qs[1] / Qs.sum())
rows = []
for f in sorted(glob.glob(f'results/p3_periodic_main/{name}_*.json')):
    r = json.load(open(f)); SEED, REP = r['seed'], r['rep']
    if r['set'] != name:                          # photon_* also matches photon_mm_*
        continue
    if REP is None:
        from p3_buildup import setup
        x = setup(name)[0]
    else:
        n = len(np.load(os.path.join(dj.DATA, f'dslit_{name}_points.npy')))
        x = np.random.default_rng(100 + REP).choice(W, size=n, p=Ptrue)
    assert r['window'] == [a0, a1]
    x = x[(x >= a0) & (x < a1)] - a0
    u = np.random.default_rng(SEED).random(len(x)); TV = np.bincount(x[u < 0.80], minlength=L).astype(float)
    p = r['periodic']; row = dict(seed=SEED, rep=REP, hi=p['hi'], lo=p['lo'], N=p['N'])
    if THREE:
        hp, lp = p['hi'], max(p['lo'], 1)
        sig = {0: dict(kind='dirichlet', fc=(2 * hp + 0.5) / L, period=float(L), fill=1.0 - (lp - 0.5) / (hp + 0.25))}
        one = {0: dict(kind='dirichlet', fc=(2 * (hp - lp) + 0.5) / L, period=float(L))}
        bgf = {0: dict(kind='dirichlet', fc=0.5 * 2 / L, period=float(L))}
        m3 = BLMLBinned(tol=1e-6, max_sweeps=5000, ll_tol=1e-10, engine=dj.ENGINE, solver=dj.SOLVER).fit(TV, [sig, one, one, bgf], complex_g=True)
        U = dj._np(m3.Se_); E = dj._np(m3.e_hat_); sh = np.array([float(np.real(np.sum(np.conj(E[j]) * U[j]))) for j in range(4)]); sh /= sh.sum()
        Q3 = np.maximum(dj._np(m3.density_), 1e-300); g3 = line(m3)
        row['three'] = dict(fscores(g3), C=C_(Q3), share=sh.tolist(), V_hat=float(sh[0] / sh[:3].sum()))
        if SEED == 0 and REP is None:
            row['three_coef'] = coef(g3)
        rows.append(row); print(name, SEED, REP, {a: (round(v, 4) if isinstance(v, float) else v) for a, v in row['three'].items()}, flush=True)
        continue
    for tag, N in (('cv', p['N']), ('flat', 0)):
        Q, g, beta = pfit(TV, p['hi'], p['lo'], N); row[tag] = dict(fscores(g), C=C_(Q), beta=beta)
        if SEED == 0 and REP is None:
            row[tag + '_coef'] = coef(g)
    rows.append(row); print(name, SEED, REP, {t: {a: round(v, 4) for a, v in row[t].items()} for t in ('cv', 'flat')}, flush=True)
json.dump(dict(set=name, ref=ref, runs=rows), open(f'{OUT}/{name}' + ('_three' if THREE else '') + '.json', 'w'), indent=1)
