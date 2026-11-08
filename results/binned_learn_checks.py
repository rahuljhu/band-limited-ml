"""Checks of kernel-parameter learning and hidden-phase retrieval in blml_binned (BINNED_ENGINE.md, status).
  pgrad    param_grad (the prequential gradient at fixed coefficients) equals the central difference of log_density
           on a held-out batch with the kernel moved and the coefficients fixed: every cell (BLMLBinned real / complex /
           rank 2 per-mode; BLMLEmpirical window / plane, +sqrt h / signed / complex / rank 2)
  adam     BLMLBinnedStreaming and BLMLEmpiricalStreaming recover a known cut-off from above and below (synthetic,
           band-limited truth with sign changes), 1-d sinc and 2-d ball
  hill     a dirichlet unit's harmonics move by the paired-gain rule toward the truth
  retrieve J never rises; signed / complex / rank 2 beat +sqrt h held-out on a sign-changing truth; complex rank 1 and
           real rank 2 (the same model, |a + ib|^2 = a^2 + b^2) score alike
Usage: python results/binned_learn_checks.py [cpu|gpu] [out.json]"""
import sys, json, time, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_trivial import kernel_class
from blml_binned import BLMLBinned, BLMLBinnedStreaming, BLMLEmpirical, BLMLEmpiricalStreaming, _ModeKernel, _np

engine = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
out = sys.argv[2] if len(sys.argv) > 2 else f'results/binned_learn_checks_{engine}.json'
rows = []


def truth(shape, kind, fc0, n_src=6, seed=3, floor=1e-4):
    """A band-limited g with mixed-sign sources (zeros: rings / nodes), kernel `kind` at fc0; returns p on the lattice."""
    r = np.random.default_rng(seed); d = len(shape)
    G = np.meshgrid(*[np.arange(m, dtype=float) for m in shape], indexing='ij')
    g = 0.0
    for _ in range(n_src):
        x0 = [r.uniform(0.3 * m, 0.7 * m) for m in shape]; a = r.choice([-1, 1]) * r.uniform(0.5, 1.0)
        diffs = [G[j] - x0[j] for j in range(d)]
        if kind == 'ball':
            K = kernel_class('ball').eval(dict(fc=np.full(d, fc0), period=None), diffs)
        else:
            K = 1.0
            for j in range(d):
                K = K * kernel_class(kind).eval(dict(fc=np.array([fc0]), period=None), [diffs[j]])
        g = g + a * K
    p = g ** 2 + floor * np.max(g ** 2); return p / p.sum()


def draws(p, n, seed):
    return np.random.default_rng(seed).multinomial(n, p.ravel()).reshape(p.shape).astype(float)


def fd_check(name, m, batch, delta=1e-4):
    """param_grad vs a central difference of log_density(batch) at fixed coefficients, per (mode, unit)."""
    eps = 1e-3 * float(np.median(_np(m.density_)[batch > 0]))   # density units
    G = m.param_grad(batch, eps); worst = 0.0; E0, U0, K0, S0 = m.e_hat_, m.Se_, list(m.kern_), [dict(s) for s in m.specs_]
    for k, kk in enumerate(K0):
        for u, (kind, dims) in enumerate(kk.units):
            if np.isnan(G[k][u]):
                continue
            ll = []
            for sg in (1, -1):
                fc = np.array(S0[k]['fc'], float); fc[list(dims)] *= np.exp(sg * delta)
                sp = dict(S0[k], fc=fc); kc = _ModeKernel(sp, m.shape_, m.h_, m.pad_, m.xp, m.fft, m.rdt_, m.workers)
                U = U0.copy(); U[k] = m._apply(kc.Kf, E0[k]); m.Se_ = U
                ll.append(m.log_density(batch, eps)); m.Se_ = U0
            fd = (ll[0] - ll[1]) / (2 * delta); worst = max(worst, abs(fd - G[k][u]) / max(abs(fd), 1e-3))
    rows.append(dict(check='pgrad', cell=name, grad=[g.tolist() for g in G], relerr=worst))
    print(f'pgrad {name:34s} rel err {worst:.1e}', flush=True)


# ---------------------------------------------------------------- pgrad
p1 = truth((64,), 'sinc', 0.25); p2 = truth((32, 32), 'ball', 0.3)
for name, p, mk, kw, cls in [
        ('binned 1d sinc real', p1, lambda: bt.make_modes([0.3], ['sinc']), {}, BLMLBinned),
        ('binned 1d sinc complex', p1, lambda: bt.make_modes([0.3], ['sinc']), dict(complex_g=True), BLMLBinned),
        ('binned 2d ball complex', p2, lambda: bt.make_modes([0.35, 0.35], ['ball', 'ball']), dict(complex_g=True), BLMLBinned),
        ('binned 1d rank2 per-mode real', p1, lambda: bt.make_modes([[0.3], [0.15]], ['sinc']), {}, BLMLBinned),
        ('binned 2d sinc x dirichlet', truth((24, 20), 'sinc', 0.3), lambda: bt.make_modes([0.35, 3], ['sinc', 'dirichlet'], periods=[None, 20.0]), {}, BLMLBinned)]:
    m = cls(engine=engine, tol=1e-9).fit(draws(p, 4000, 1), mk(), **kw)
    fd_check(name, m, draws(p, 2000, 2))
for name, p, mk, ekw, fkw in [
        ('empirical window +sqrt h', p1, lambda: bt.make_modes([0.3], ['sinc']), {}, {}),
        ('empirical plane +sqrt h', p1, lambda: bt.make_modes([0.3], ['sinc']), dict(normalise='plane'), {}),
        ('empirical window signed', p1, lambda: bt.make_modes([0.3], ['sinc']), dict(retrieve=True), {}),
        ('empirical window complex', p1, lambda: bt.make_modes([0.3], ['sinc']), {}, dict(complex_g=True)),
        ('empirical 2d ball rank2', p2, lambda: bt.make_modes([0.35, 0.35], ['ball', 'ball'], rank=2), {}, {}),
        ('empirical 2d ball complex plane', p2, lambda: bt.make_modes([0.35, 0.35], ['ball', 'ball']), dict(normalise='plane'), dict(complex_g=True))]:
    m = BLMLEmpirical(engine=engine, lam=1e-3, tol=1e-10, **ekw).fit(draws(p, 4000, 1), mk(), **fkw)
    fd_check(name, m, draws(p, 2000, 2))

# ---------------------------------------------------------------- retrieve
for name, p, mk in [('1d sinc', p1, lambda K=1: bt.make_modes([0.25], ['sinc'], rank=K)),
                    ('2d ball', p2, lambda K=1: bt.make_modes([0.3, 0.3], ['ball', 'ball'], rank=K))]:
    tr, te = draws(p, 20000, 4), draws(p, 20000, 5); res = {}
    for lab, ekw, fkw, K in [('+sqrt h', {}, {}, 1), ('signed', dict(retrieve=True), {}, 1),
                             ('complex', {}, dict(complex_g=True), 1), ('real rank2', {}, {}, 2)]:
        m = BLMLEmpirical(engine=engine, lam=1e-4, tol=1e-10, **ekw).fit(tr, mk(K), **fkw)
        J = np.array(m.J_history_); rise = float(np.max(np.diff(J) / np.abs(J[1:]))) if len(J) > 1 else 0.0
        res[lab] = dict(test=m.score(te), outer=m.n_outer_, J_max_rel_rise=rise)
    rows.append(dict(check='retrieve', cell=name, **res))
    print(f'retrieve {name}: ' + '  '.join(f"{k} {v['test']:.5f} (outer {v['outer']}, rise {v['J_max_rel_rise']:.0e})" for k, v in res.items()), flush=True)

# ---------------------------------------------------------------- adam
for name, shape, kind, fc0 in [('1d sinc', (128,), 'sinc', 0.2), ('2d ball', (48, 48), 'ball', 0.3)]:
    p = truth(shape, kind, fc0, n_src=8, floor=0.0); d = len(shape)   # exactly |g|^2: complex fits are well specified
    for cls_name, cls, kw in [('binned real (misspecified)', BLMLBinnedStreaming, dict(complex_g=False)),
                              ('binned complex', BLMLBinnedStreaming, dict(complex_g=True)),
                              ('empirical complex', BLMLEmpiricalStreaming, dict(complex_g=True))]:
        for start in (1.6, 0.6):
            t = time.time()
            ckw = dict(engine=engine, tol=1e-8, lr=0.05, param_every=1)
            s = cls(**ckw, batch_sweeps=50) if cls is BLMLBinnedStreaming else cls(**ckw, lam=1e-4)
            modes = bt.make_modes([fc0 * start] * d, [kind] * d)
            for b in range(150):
                s.partial_fit(draws(p, 2000, 100 + b), modes, **kw)
            fc = s.history_[-1]['fc'][0][0]
            rows.append(dict(check='adam', cell=f'{name} {cls_name} start x{start}', fc_true=fc0, fc_final=fc,
                             fc_path=[h['fc'][0][0] for h in s.history_[::10]], s=time.time() - t))
            print(f'adam {name} {cls_name:26s} start x{start}: fc {fc0 * start:.3f} -> {fc:.3f} (true {fc0})  {time.time() - t:.0f}s', flush=True)

# ---------------------------------------------------------------- hill (dirichlet harmonics)
P = 40
G = np.meshgrid(np.arange(60.0), np.arange(P, dtype=float), indexing='ij')
gt = kernel_class('sinc').eval(dict(fc=np.array([0.25]), period=None), [G[0] - 30]) * np.cos(2 * np.pi * 2 * G[1] / P + 0.4)
p = gt ** 2; p = p / p.sum()
for start in (5, 0):                                              # harmonics N; the truth has N = 2
    s = BLMLBinnedStreaming(engine=engine, tol=1e-8, batch_sweeps=50, param_every=1)
    modes = bt.make_modes([0.25, (start + 0.5) * 2 / P], ['sinc', 'dirichlet'], periods=[None, float(P)])
    for b in range(80):
        s.partial_fit(draws(p, 3000, 300 + b), modes, complex_g=True)
    hp = [int(np.floor(h['fc'][0][1] * P / 2)) for h in s.history_]
    rows.append(dict(check='hill', cell=f'dirichlet start {start}', harmonics_path=hp[::5], final=hp[-1], fc_sinc=s.history_[-1]['fc'][0][0]))
    print(f'hill dirichlet start {start}: harmonics N {hp[0]} -> {hp[-1]} (truth N = 2)  sinc fc {s.history_[-1]["fc"][0][0]:.3f}', flush=True)

json.dump(rows, open(out, 'w'), indent=1)
