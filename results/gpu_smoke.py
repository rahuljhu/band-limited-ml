"""Small GPU smoke test: CPU vs GPU on small problems for every path changed recently (modes API,
max_sweeps, keep_lattice warm starts, counts on a fixed lattice, start search, conditional, select, select_counts,
BLMLRolling, BLMLStreaming). Each check is isolated; prints PASS/FAIL lines. A few minutes on a T4."""
import os, sys, time, traceback, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..'); sys.path.insert(0, '.')
import numpy as np
warnings.filterwarnings('ignore')
import blml_trivial as bt
from blml_trivial import BLMLQuick
from selector import select, select_counts
from blml_rolling import BLMLRolling, BLMLQuickStreaming

rng = np.random.default_rng(0)
X2 = np.c_[rng.normal(size=4000), rng.normal(size=4000) * 0.7]
Q2 = np.c_[rng.normal(size=1000), rng.normal(size=1000) * 0.7]
hr = rng.integers(0, 24, 6000).astype(float)
X3 = np.c_[np.sin(2 * np.pi * hr / 24) + 0.5 * rng.normal(size=6000), hr, rng.normal(size=6000)]
md2 = bt.make_modes([1.5, 1.5], ['sinc', 'sinc'])
md3 = bt.make_modes([1.5, 0.3, 0.5], ['sinc', 'dirichlet', 'sinc'], [None, 24.0, None], dep=0)
F64 = dict(dtype=np.float64)
ll = lambda m, Q: float(np.mean(np.log(np.maximum(m.predict(Q), 1e-300))))
results = []


def check(name, fn):
    t = time.time()
    try:
        msg = fn()
        results.append((name, True)); print(f'PASS {name} ({time.time() - t:.1f}s): {msg}', flush=True)
    except Exception as e:
        results.append((name, False)); print(f'FAIL {name}: {type(e).__name__}: {e}', flush=True)
        traceback.print_exc(limit=4)


def close(a, b, tol, what):
    if not abs(a - b) <= tol:
        raise AssertionError(f'{what}: cpu {a:.5f} vs gpu {b:.5f} (tol {tol})')


def real_fit():
    c = bt.BLMLQuick(**F64).fit(X2, md2); g = BLMLQuick(engine='gpu', **F64).fit(X2, md2)
    close(ll(c, Q2), ll(g, Q2), 2e-3, 'test ll'); return f'll cpu {ll(c, Q2):.4f} gpu {ll(g, Q2):.4f}'


def rank2():
    m = bt.make_modes([1.5, 1.5], ['sinc', 'sinc'], rank=2)
    c = bt.BLMLQuick(**F64).fit(X2, m); g = BLMLQuick(engine='gpu', **F64).fit(X2, m)
    close(ll(c, Q2), ll(g, Q2), 5e-3, 'test ll'); return f'll cpu {ll(c, Q2):.4f} gpu {ll(g, Q2):.4f}'


def complex_starts():
    c = bt.BLMLQuick(**F64).fit(X2, md2, complex_g=True); g = BLMLQuick(engine='gpu', **F64).fit(X2, md2, complex_g=True)
    if not g.starts_ or len(g.starts_) != 5:
        raise AssertionError(f'gpu starts_ {g.starts_}')
    close(ll(c, Q2), ll(g, Q2), 5e-3, 'test ll'); return f'll cpu {ll(c, Q2):.4f} gpu {ll(g, Q2):.4f}, 5 starts'


def conditional():
    c = bt.BLMLQuick(**F64).fit(X3, md3); g = BLMLQuick(engine='gpu', **F64).fit(X3, md3)
    Q = X3[:800]
    a = float(np.mean(np.log(c.predict_ratio(Q)))); b = float(np.mean(np.log(g.predict_ratio(Q))))
    close(a, b, 2e-2, 'cond ll (cpu: two fits, gpu: fused)'); return f'cond ll cpu {a:.4f} gpu {b:.4f}'


def warm_and_cap():
    out = []
    p = BLMLQuick(engine='gpu', **F64).fit(X2[:3000], md2)
    for k in (0, 1, None):
        m = BLMLQuick(engine='gpu', max_sweeps=k, **F64).fit(X2, md2, init=p)
        out.append(f'k={k}: rebin={m.rebinned_} sweeps={m.n_sweeps_} ll {ll(m, Q2):.4f}')
    g = np.linspace(-6, 6, 201); G = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    m = BLMLQuick(engine='gpu', max_sweeps=1, **F64).fit(X2, md2, init=p)
    I = float(m.predict(G).sum() * (g[1] - g[0]) ** 2)
    if abs(I - 1) > 5e-3:
        raise AssertionError(f'k=1 not normalised: int f = {I:.4f}')
    big = BLMLQuick(engine='gpu', **F64).fit(np.r_[X2, X2], md2, init=p)
    if not big.rebinned_:
        raise AssertionError('n x 2.67 did not re-bin')
    return '; '.join(out) + f'; int f (k=1) {I:.4f}; n x2.67 rebinned'


def counts():
    c0 = bt.BLMLQuick(**F64).fit(X2, md2)
    out = []
    for cx in (False, True):
        a = bt.BLMLQuick(**F64).fit(c0.Z_, bt.with_lattice(md2, c0._h_vec), sample_weight=c0.w_, complex_g=cx)
        b = BLMLQuick(engine='gpu', **F64).fit(c0.Z_, bt.with_lattice(md2, c0._h_vec), sample_weight=c0.w_, complex_g=cx)
        close(ll(a, Q2), ll(b, Q2), 5e-3, f'fit_counts complex={cx}')
        bw = BLMLQuick(engine='gpu', max_sweeps=0, **F64).fit(c0.Z_, bt.with_lattice(md2, c0._h_vec), sample_weight=c0.w_, complex_g=cx, init=b)
        close(ll(b, Q2), ll(bw, Q2), 1e-4, 'same-lattice k=0 reuse (renormalised; tol 1e-4 fit)')
        out.append(f'complex={cx}: cpu {ll(a, Q2):.4f} gpu {ll(b, Q2):.4f}')
    return '; '.join(out)


def selectors():
    out = []
    for s in ('shared_hillclimb', 'auto'):
        mc, _ = select(X2, bt.make_modes(np.nan, ['sinc', 'sinc']), search=s, engine='cpu')
        mg, _ = select(X2, bt.make_modes(np.nan, ['sinc', 'sinc']), search=s, engine='gpu')
        fc, fg = bt.modes_fc(mc)[0], bt.modes_fc(mg)[0]
        if not np.allclose(fc, fg, rtol=0.2):
            raise AssertionError(f'{s}: fc cpu {fc} gpu {fg}')
        out.append(f'{s}: cpu {np.round(fc, 3)} gpu {np.round(fg, 3)}')
    mg, _ = select(X3, bt.make_modes(np.nan, ['sinc', 'dirichlet', 'sinc'], [None, 24.0, None], dep=0), search='auto', engine='gpu')
    out.append(f'cond auto gpu {np.round(bt.modes_fc(mg)[0], 3)}')
    return '; '.join(out)


HG = 0.1                                               # detector pixel; full grid, zero counts included
gx, gy = np.arange(-40, 41) * HG, np.arange(-30, 31) * HG
ZG = np.stack(np.meshgrid(gx, gy, indexing='ij'), -1).reshape(-1, 2)


def grid_counts(P):
    ix = np.clip(np.round(P[:, 0] / HG).astype(int) + 40, 0, 80); iy = np.clip(np.round(P[:, 1] / HG).astype(int) + 30, 0, 60)
    return np.bincount(ix * 61 + iy, minlength=81 * 61).astype(float)


def sel_counts():
    wg = grid_counts(X2)
    wtr = np.random.default_rng(2).binomial(wg.astype(int), 0.8).astype(float); wva = wg - wtr
    a, _ = select_counts(ZG, wtr, wva, HG, bt.make_modes(np.nan, ['sinc', 'sinc']), engine='cpu')
    b, _ = select_counts(ZG, wtr, wva, HG, bt.make_modes(np.nan, ['sinc', 'sinc']), engine='gpu')
    fa, fb = bt.modes_fc(a)[0], bt.modes_fc(b)[0]
    if not np.allclose(fa, fb, rtol=0.2):
        raise AssertionError(f'fc cpu {fa} gpu {fb}')
    return f'cpu {np.round(fa, 3)} gpu {np.round(fb, 3)}'


def rolling():
    tmpl = bt.make_modes(np.nan, ['sinc', 'dirichlet', 'sinc'], [None, 24.0, None], dep=0)
    r = BLMLRolling(engine='gpu', search='auto').fit(X3[:3000], tmpl, X3[3000:3750])
    r.update(X3[750:3750], X3[3750:4500])
    h = r.history_[-1]
    return f'update fits={h["n_fits"]} moved={h["moved"]} fc {np.round(h["fc"][0], 3)}'


def streaming():
    tmpl = bt.make_modes(np.nan, ['sinc', 'dirichlet', 'sinc'], [None, 24.0, None], dep=0)
    out = []
    for eng in ('cpu', 'gpu'):
        t = time.time()
        s = BLMLQuickStreaming(engine=eng, search='auto', burn_in=1500, **F64).fit(X3[:4500], tmpl, epochs=2, batch=1500)
        Q = X3[4500:]
        out.append(f'{eng}: {time.time() - t:.1f}s cond ll {np.mean(np.log(np.maximum(s.predict_ratio(Q), 1e-300))):.4f} '
                   f'fc {np.round(bt.modes_fc(s.modes_)[0], 3)}')
    return '; '.join(out)


def binned_streaming():
    out = []
    for eng in ('cpu', 'gpu'):
        s = BLMLQuickStreaming(engine=eng, burn_in=1000, **F64)              # counts on the fixed lattice HG
        for j in range(4):                                                   # 4 batches of 1000 photons
            s.partial_fit(ZG, bt.with_lattice(bt.make_modes(np.nan, ['sinc', 'sinc']), HG),
                          sample_weight=grid_counts(X2[j * 1000:(j + 1) * 1000]))
        out.append(f'{eng}: ll {ll(s, Q2):.4f} fc {np.round(bt.modes_fc(s.modes_)[0], 3)}')
    return '; '.join(out)


def ball():
    import cupy as cp
    from blml_gpu import _J01
    out = []
    for D in (2, 3):
        fc, h = np.full(D, 1.4), np.full(D, 0.01)
        a = bt.radial_table('ball', list(range(D)), fc, h, 40000)
        b = cp.asnumpy(bt.radial_table('ball', list(range(D)), fc, h, 40000, xp=cp, j01=_J01))
        da = bt.radial_dlog_table('ball', list(range(D)), fc, h, 40000)
        db = cp.asnumpy(bt.radial_dlog_table('ball', list(range(D)), fc, h, 40000, xp=cp, j01=_J01))
        if np.abs(a - b).max() > 1e-9 * np.abs(a).max() or np.abs(da - db).max() > 1e-9 * np.abs(da).max():
            raise AssertionError(f'D={D}: device table differs: {np.abs(a - b).max():.2e} / {np.abs(da - db).max():.2e}')
        out.append(f'D={D} tables match')
    mb = bt.make_modes([1.6, 1.6], ['ball', 'ball'])
    c = bt.BLMLQuick(**F64).fit(X2, mb); g = BLMLQuick(engine='gpu', **F64).fit(X2, mb)
    close(ll(c, Q2), ll(g, Q2), 2e-3, 'ball fit')
    gc = c.fc_grad(Q2[:300], [[0, 1]], implicit_grad=True, eps_rel=1e-3).mean(0)
    gg = g.fc_grad(Q2[:300], [[0, 1]], implicit_grad=True, eps_rel=1e-3).mean(0)
    close(float(gc[0]), float(gg[0]), 1e-3 + 1e-2 * abs(float(gc[0])), 'ball fc_grad')
    return '; '.join(out) + f'; fit ll cpu {ll(c, Q2):.4f} gpu {ll(g, Q2):.4f}; grad cpu {float(gc[0]):.4f} gpu {float(gg[0]):.4f}'


def streaming_param_every():
    tmpl = bt.make_modes(np.nan, ['sinc', 'sinc'])
    out = []
    for eng in ('cpu', 'gpu'):
        s = BLMLQuickStreaming(tmpl, engine=eng, search='auto', burn_in=1000, param_every=2, **F64)
        t = time.time()
        for j in range(0, 4000, 500):
            s.partial_fit(X2[j:j + 500])
        out.append(f'{eng}: {time.time() - t:.1f}s ll {ll(s, Q2):.4f} fc {np.round(bt.modes_fc(s.modes_)[0], 3)}')
    return '; '.join(out)


ONLY = os.environ.get('SMOKE_ONLY')                 # e.g. "fit_counts,select_counts"
for name, fn in [('real fit', real_fit), ('rank-2 real', rank2), ('complex + 5 starts', complex_starts),
                 ('conditional (fused gpu)', conditional), ('warm start + max_sweeps', warm_and_cap),
                 ('fit_counts', counts), ('select', selectors), ('select_counts', sel_counts),
                 ('BLMLRolling gpu', rolling), ('BLMLQuickStreaming cpu vs gpu', streaming),
                 ('BLMLBinnedStreaming cpu vs gpu', binned_streaming), ('ball', ball),
                 ('streaming param_every', streaming_param_every)]:
    if ONLY is None or name in ONLY.split(','):
        check(name, fn)
print(f'SUMMARY {sum(ok for _, ok in results)}/{len(results)} passed', flush=True)
