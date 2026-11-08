"""Paper 3, cropped benchmark: the phase scores, re-derived for the models p3_cbench.py selected (same data: the same
seeds and draws; each selected model refitted on train + validation). The slit phase phi is a fringe offset, so it
needs an origin. Three scores, each against the same quantity of the true field on the window:
  phi_axis  the fringe phase about the known optical axis (the window centre c0 = C0 - a0): arg sum_{k>0} a_k
            conj(a_{-k}) e^{4 pi i k c0 / L}; for the KDE (no field) the same from the density's Fourier coefficient
            at the fringe harmonic m = round(d L);
  c_env     the envelope (single-slit) centre, from the phase gradient within each slit's harmonics:
            arg sum a_{k+1} conj(a_k) = -2 pi c / L, both sides averaged (twin-invariant);
  phi_env   the slit phase about the fitted envelope centre (no axis given): phi_axis with c0 -> c_env;
  phi_rel   the relative slit phase, arg sum_{k>0} a_k e^{2 pi i k c_env / L} / sum_{k<0} (same): invariant to the global
            phase, to a far-field shift and to a shift of the harmonics (a_k -> a_{k-s}, i.e. g e^{2 pi i s t / L}, which
            leaves |g|^2 unchanged: the aperture's position is not determined); min over the twin;
  ap_err    the aperture error on the window: ||a_hat - a|| over the window's harmonics, both of unit norm, min over the
            global phase, the twin and integer shifts |s| <= 3, against the truth's a_k (A * sinc_L at k / L);
  width     the rms width of |a_k|^2 on each side (cycles/px), against the truth's (the width of A * sinc_L);
  C         the fringe contrast 2 |Q_m| / Q_0 of the density (Q_m its Fourier coefficient at m), against the window truth's.
  python p3_cbench_phase.py <case> <bg> <n> <rep>   -> adds res['phase'] to results/p3_cbench/<case>_bg<bg>_n<n>_<rep>.json
  python p3_cbench_phase.py <case> <bg> 0 0          -> <CB_OUT>_floor/<case>_bg<bg>.json: the n -> inf fits (to the window's
            density itself) over the periodic bands 1 <= lo < hi <= 8, the floor of the periodic class on the window"""
import os, sys, json, numpy as np
os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_SOLVER', 'lbfgs'); os.environ.setdefault('BLML_COLLAPSE', '1')
case, BG, N_, REP = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); sys.argv = sys.argv[:1]
sys.path.insert(0, '.')
import dslit_job as dj
from blml_binned import BLMLBinned
OUT = os.environ.get('CB_OUT', 'results/p3_cbench'); FLOOR = N_ == 0
fn = f"{OUT}_floor/{case}_bg{BG:g}.json" if FLOOR else f"{OUT}/{case}_bg{BG:g}_n{N_}_{REP}.json"; res = {} if FLOOR else json.load(open(fn))
if 'phi_rel' in res.get('phase', {}).get('periodic', {}):
    sys.exit(0)
RHO, PHI, V = (0.6, 1.0, 1.0) if case == 'asym' else (1.0, 0.0, 0.6) if case == 'partial' else (1.0, 0.0, 1.0)   # as p3_cbench.py
if os.environ.get('CB_TRUTH', 'sharp') == 'tapered':
    D_, A_, NF = 0.017, 0.005, 1 << 16; W, C0 = 2048, 1024.0; Pf, R = 1 / D_, A_ / D_; nu_ = np.fft.fftfreq(NF)
    sl = lambda c: np.where(np.abs(nu_ - c) < A_ / 2, np.cos(np.pi * np.abs(nu_ - c) / A_) ** 2, 0.0)
    fl = lambda A: np.fft.fftshift(np.fft.ifft(A))[NF // 2 - W // 2:NF // 2 + W // 2] * NF
    g1, g2 = fl(sl(-D_ / 2)), fl(sl(D_ / 2)); field = g1 + RHO * np.exp(1j * PHI) * g2
    I = V * np.abs(field) ** 2 + (1 - V) * (np.abs(g1) ** 2 + RHO ** 2 * np.abs(g2) ** 2)
else:
    W, Pf, R, C0 = 512, 56.0, 0.2, 256.0
    lg = lambda p: float(np.log(p / (1 - p))) if p < 1 else 30.0
    th = np.array([C0, Pf, lg(R), 0.0, np.log(RHO), PHI, lg(V), 0.0, 0.0])
    I, field, _ = dj.ref_parts(th, (W,), 0, True)
P = (1 - BG) * I / I.sum() + BG / W; P /= P.sum()
m_ = int(os.environ.get('CB_M', '0')) or int(min(C0, W - C0) / Pf - 0.5); a0, a1 = int(round(C0 - (m_ + 0.5) * Pf)), int(round(C0 + (m_ + 0.5) * Pf)); L = a1 - a0
if not FLOOR:
    rng = np.random.default_rng(1000 * REP + N_ % 997)
    x = rng.choice(W, size=N_, p=P); x = x[(x >= a0) & (x < a1)] - a0
    part = rng.choice(3, size=len(x), p=[0.64, 0.16, 0.20])
    TV = np.bincount(x[part < 2], minlength=L).astype(float)
    assert len(x) == res['n_kept']
gw = field[a0:a1]; c0 = C0 - a0; mf = int(round(L / Pf)); dj.BFIX = np.array([np.nan])
k = np.rint(np.fft.fftfreq(L) * L); pos = (k > 0) & (k < L / 4)
def fscores(g):
    a = np.fft.fft(g); ap = {int(kk): a[i] for i, kk in enumerate(k)}
    ks = [int(kk) for kk in k[pos]]
    pair = lambda c: np.angle(sum(ap[q] * np.conj(ap[-q]) * np.exp(4j * np.pi * q * c / L) for q in ks))
    grad = sum(ap[q + 1] * np.conj(ap[q]) + ap[-q] * np.conj(ap[-q - 1]) for q in ks[:-1])   # k>0 side and k<0 side
    c = (-np.angle(grad) * L / (2 * np.pi)) % L
    side = lambda m_: np.sqrt((e[m_] * k[m_] ** 2).sum() / e[m_].sum() - ((e[m_] * k[m_]).sum() / e[m_].sum()) ** 2) / L
    e = np.abs(a) ** 2; neg = (k < 0) & (k > -L / 4)
    rel = lambda a_: float(np.angle(np.sum(a_[pos] * np.exp(2j * np.pi * k[pos] * c / L)) / np.sum(a_[neg] * np.exp(2j * np.pi * k[neg] * c / L))))
    return dict(phi_axis=float(pair(c0)), c_env=float(c), phi_env=float(pair(c)), width=float((side(pos) + side(neg)) / 2), a=a,
                phi_rel=rel(a), phi_rel_tw=rel(np.conj(a[(-np.arange(L)) % L])))
def iscore(Q):
    a = np.fft.fft(Q / Q.sum())[mf]; return float(np.angle(a * np.exp(2j * np.pi * mf * c0 / L))), float(2 * abs(a))
wrap = lambda a: float(abs(np.angle(np.exp(1j * a))))
T_ = fscores(gw); aT = T_.pop('a'); T_.pop('phi_rel_tw'); T_['phi_I'], T_['C'] = iscore(P[a0:a1])
out = dict(truth=T_)
def add(name, Q, g=None):
    s = dict(zip(('phi_I', 'C'), iscore(Q))); s['phi_I_err'] = wrap(s['phi_I'] - T_['phi_I']); s['C_err'] = abs(s['C'] - T_['C'])
    if g is not None:
        F = fscores(g); ah = F.pop('a'); s.update(F)
        u, v = ah / np.linalg.norm(ah), aT / np.linalg.norm(aT); tw = np.conj(u[(-np.arange(L)) % L])   # twin: a_k -> conj(a_{-k})
        s['ap_err'] = float(min(np.sqrt(max(0.0, 2 - 2 * abs(np.vdot(np.roll(z, sh), v)))) for z in (u, tw) for sh in range(-3, 4)))
        s['phi_rel_err'] = min(wrap(s['phi_rel'] - T_['phi_rel']), wrap(s.pop('phi_rel_tw') - T_['phi_rel']))
        s['width_err'] = abs(s['width'] / T_['width'] - 1)
        s.update(phi_axis_err=wrap(s['phi_axis'] - T_['phi_axis']), phi_env_err=wrap(s['phi_env'] - T_['phi_env']),
                 c_err=float(abs((s['c_env'] - T_['c_env'] + L / 2) % L - L / 2)))
    out[name] = s
line = lambda m: np.asarray(dj._np(m.Se_))[0] / m.n_
def pfit(T, hi, lo, N):
    fc = (2 * hi + 0.5) / L; w = 1.0 if lo == 0 else 1.0 - (lo - 0.5) / (hi + 0.25)
    sig = {0: dict(kind='dirichlet', fc=fc, period=float(L), **({} if w >= 1 else dict(fill=w)))}
    bg = {0: dict(kind='dirichlet', fc=(N + 0.5) * 2 / L, period=float(L))}
    m = BLMLBinned(tol=1e-6, max_sweeps=5000, ll_tol=1e-10, engine=dj.ENGINE, solver=dj.SOLVER).fit(T, [sig, bg], complex_g=True)
    P_ = np.maximum(dj._np(m.density_), 1e-300); return P_ / P_.sum(), m
if FLOOR:
    Pw = P[a0:a1] / P[a0:a1].sum(); F = []
    for hi in range(2, 9):
        for lo in range(1, hi):
            Q, m = pfit(Pw * 1e9, hi, lo, 0); add('p', Q, line(m))
            F.append(dict(hi=hi, lo=lo, kl=float((Pw * np.log(Pw / Q)).sum()), **{a: v for a, v in out.pop('p').items() if isinstance(v, float)}))
    os.makedirs(os.path.dirname(fn), exist_ok=True); json.dump(dict(case=case, bg=BG, L=L, truth=T_, fits=F), open(fn, 'w'), indent=1)
    b = min(F, key=lambda f: f['kl']); print(fn, 'best', {a: round(v, 4) for a, v in b.items()}); sys.exit(0)
r = res['periodic']; Q, m = pfit(TV, r['hi'], r['lo'], r['N']); add('periodic', Q, line(m))
if case == 'partial':
    hp, lp = r['hi'], max(r['lo'], 1)
    sig = {0: dict(kind='dirichlet', fc=(2 * hp + 0.5) / L, period=float(L), fill=1.0 - (lp - 0.5) / (hp + 0.25))}
    one = {0: dict(kind='dirichlet', fc=(2 * (hp - lp) + 0.5) / L, period=float(L))}
    bgf = {0: dict(kind='dirichlet', fc=0.5 * 2 / L, period=float(L))}
    m3 = BLMLBinned(tol=1e-6, max_sweeps=5000, ll_tol=1e-10, engine=dj.ENGINE, solver=dj.SOLVER).fit(TV, [sig, one, one, bgf], complex_g=True)
    Q3 = np.maximum(dj._np(m3.density_), 1e-300); add('periodic_three', Q3 / Q3.sum(), line(m3))
r = res['line']; Q, m, _ = dj.bg_fit(TV, [r['fc']], (r['N'],), True, 1, max_sweeps=5000, tol=1e-6, fill=r['fill']); add('line', Q, line(m))
add('kde', dj.em(TV, np.ones((1, L)) / L, dj.kde_q(res['kde']['h']))[0])
res['phase'] = out; json.dump(res, open(fn, 'w'), indent=1)
print(fn, {e: {a: round(v, 3) for a, v in s.items() if a.endswith('err')} for e, s in out.items() if e != 'truth'})
