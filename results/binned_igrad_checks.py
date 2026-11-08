"""implicit_grad (prequential gradient with the coefficients' refit response) vs the central difference of the held-out
batch score of warm-started refits at fc * exp(+-delta), every cell of BLMLBinned / BLMLEmpirical.
Usage: python results/binned_igrad_checks.py [cpu|gpu] [out.json]"""
import sys, json, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_trivial import kernel_class
from blml_binned import BLMLBinned, BLMLEmpirical, _np
exec(open('results/binned_learn_checks.py').read().split('def fd_check')[0].split('rows = []')[1])   # truth, draws
engine = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
out = sys.argv[2] if len(sys.argv) > 2 else f'results/binned_igrad_checks_{engine}.json'
rows = []


def modes_at(m, F):
    sp = m.specs_[0]
    return bt.make_modes(F, sp['kinds'], periods=[p if p is not None else np.nan for p in sp['periods']] if any(
        p is not None for p in sp['periods']) else None)


def check(name, make, tr, te, fkw, delta=2e-3):
    m = make().fit(tr, modes0, **fkw)
    eps = 1e-3 * float(np.median(_np(m.density_)[te > 0]))   # density units
    Gi = m.implicit_grad(te, eps, tol=1e-10, maxit=2000); Gf = m.param_grad(te, eps)
    F0 = np.array([sp['fc'] for sp in m.specs_], float); worst = 0.0; cells = []
    for k, kk in enumerate(m.kern_):
        for u, (kind, dims) in enumerate(kk.units):
            if np.isnan(Gi[k][u]):
                continue
            ll, refits = [], []
            for sg in (1, -1):
                F = F0.copy(); F[k, list(dims)] *= np.exp(sg * delta)
                r = make().fit(tr, modes_at(m, F), init=m, **fkw); ll.append(r.log_density(te, eps)); refits.append(r)
            fd = (ll[0] - ll[1]) / (2 * delta); err = abs(fd - Gi[k][u]) / max(abs(fd), 1e-3)
            same = not hasattr(m, 'c_raw_') or all(bool(np.array_equal(_np(np.abs(r.c_raw_) > 0), _np(np.abs(m.c_raw_) > 0))) for r in refits[-2:])
            worst = max(worst, err); cells.append(dict(mode=k, unit=u, fd=fd, implicit=Gi[k][u], fixed=Gf[k][u], same_support=same))
    rows.append(dict(cell=name, relerr=worst, units=cells, cg_iters=m.ift_iters_, cg_res=m.ift_res_, delta=delta))
    print(f'{name:34s} rel err {worst:.1e}  ' + '  '.join(f"fd {c['fd']:+.5f} impl {c['implicit']:+.5f} fixed {c['fixed']:+.5f}"
                                                         for c in cells) + f'  cg {m.ift_iters_}' +
          ('' if all(c['same_support'] for c in cells) else '  (support changed in a refit)'), flush=True)


p1 = truth((64,), 'sinc', 0.25); p2 = truth((32, 32), 'ball', 0.3)
tr1, te1, tr2, te2 = draws(p1, 20000, 1), draws(p1, 5000, 2), draws(p2, 20000, 1), draws(p2, 5000, 2)
B = lambda **kw: (lambda: BLMLBinned(engine=engine, tol=1e-11, **kw))
Em = lambda **kw: (lambda: BLMLEmpirical(engine=engine, lam=1e-3, tol=1e-12, outer_tol=1e-10, max_outer=50000, **kw))
for name, make, tr, te, m0, fkw in [
        ('binned 1d sinc real', B(), tr1, te1, bt.make_modes([0.3], ['sinc']), {}),
        ('binned 1d sinc complex', B(), tr1, te1, bt.make_modes([0.3], ['sinc']), dict(complex_g=True)),
        ('binned 2d ball complex', B(), tr2, te2, bt.make_modes([0.35, 0.35], ['ball', 'ball']), dict(complex_g=True)),
        ('binned 1d rank2 per-mode real', B(), tr1, te1, bt.make_modes([[0.3], [0.15]], ['sinc']), {}),
        ('empirical ridge window +sqrt h', Em(), tr1, te1, bt.make_modes([0.3], ['sinc']), {}),
        ('empirical l2 plane +sqrt h', Em(reg='l2', normalise='plane'), tr1, te1, bt.make_modes([0.3], ['sinc']), {}),
        ('empirical l1 window +sqrt h', Em(reg='l1'), tr1, te1, bt.make_modes([0.3], ['sinc']), dict(delta=2e-3)),
        ('empirical l1 window +sqrt h, small step', Em(reg='l1'), tr1, te1, bt.make_modes([0.3], ['sinc']), dict(delta=1e-6)),
        ('empirical ridge window signed', Em(retrieve=True), tr1, te1, bt.make_modes([0.3], ['sinc']), {}),
        ('empirical ridge window complex', Em(), tr1, te1, bt.make_modes([0.3], ['sinc']), dict(complex_g=True)),
        ('empirical 2d ball rank2', Em(), tr2, te2, bt.make_modes([0.35, 0.35], ['ball', 'ball'], rank=2), {})]:
    modes0 = m0; dl = fkw.pop('delta', 2e-3); check(name, make, tr, te, fkw, delta=dl)
json.dump(rows, open(out, 'w'), indent=1)
