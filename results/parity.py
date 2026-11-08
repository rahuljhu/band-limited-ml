"""CPU/GPU parity: each engine runs the same checks on the same data and saves its numbers; the CPU side runs on
the Mac, the GPU side on Colab, then `compare` checks every number within a tolerance.
    parity.py cpu|gpu [out.json]     run one engine (GPU-only properties are asserted in the gpu run)
    parity.py compare cpu.json gpu.json
SMOKE_ONLY="name1,name2" runs a subset."""
import os, sys, time, json, traceback, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np
warnings.filterwarnings('ignore')
import blml_trivial as bt
from selector import select, select_counts
from blml_rolling import BLMLRolling, BLMLQuickStreaming

TOL = {'ll': 5e-3, 'cond_ll': 2e-2, 'fc': 0.2, 'grad': 2e-2, 'stream_ll': 2e-2, 'period': 0.3}    # abs (fc: relative)

rng = np.random.default_rng(0)
X2 = np.c_[rng.normal(size=4000), rng.normal(size=4000) * 0.7]
Q2 = np.c_[rng.normal(size=1000), rng.normal(size=1000) * 0.7]
hr = rng.integers(0, 24, 6000).astype(float)
X3 = np.c_[np.sin(2 * np.pi * hr / 24) + 0.5 * rng.normal(size=6000), hr, rng.normal(size=6000)]
md2 = bt.make_modes([1.5, 1.5], ['sinc', 'sinc'])
md3 = bt.make_modes([1.5, 0.3, 0.5], ['sinc', 'dirichlet', 'sinc'], [None, 24.0, None], dep=0)
F64 = dict(dtype=np.float64)
HG = 0.1
ZG = np.stack(np.meshgrid(np.arange(-40, 41) * HG, np.arange(-30, 31) * HG, indexing='ij'), -1).reshape(-1, 2)
ll = lambda m, Q: float(np.mean(np.log(np.maximum(m.predict(Q), 1e-300))))


def grid_counts(P):
    ix = np.clip(np.round(P[:, 0] / HG).astype(int) + 40, 0, 80); iy = np.clip(np.round(P[:, 1] / HG).astype(int) + 30, 0, 60)
    return np.bincount(ix * 61 + iy, minlength=81 * 61).astype(float)


def checks(eng):
    if eng == 'gpu':
        import functools; Est = functools.partial(bt.BLMLQuick, engine='gpu')
    else:
        Est = bt.BLMLQuick

    def real_fit():
        return dict(ll=ll(Est(**F64).fit(X2, md2), Q2))

    def rank2():
        return dict(ll=ll(Est(**F64).fit(X2, bt.make_modes([1.5, 1.5], ['sinc', 'sinc'], rank=2)), Q2))

    def complex_starts():
        m = Est(**F64).fit(X2, md2, complex_g=True)
        assert m.starts_ and len(m.starts_) == 5, m.starts_
        return dict(ll=ll(m, Q2))

    def conditional():
        m = Est(**F64).fit(X3, md3)
        return dict(cond_ll=float(np.mean(np.log(m.predict_ratio(X3[:800])))))

    def warm_and_cap():
        p = Est(**F64).fit(X2[:3000], md2)
        out = {}
        for k in (0, 1, None):
            m = Est(max_sweeps=k, **F64).fit(X2, md2, init=p)
            assert not m.rebinned_
            out[f'll_k{k}'] = ll(m, Q2)
        g = np.linspace(-6, 6, 201); G = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
        I = float(Est(max_sweeps=1, **F64).fit(X2, md2, init=p).predict(G).sum() * (g[1] - g[0]) ** 2)
        assert abs(I - 1) < 5e-3, f'k=1 not normalised: {I}'
        assert Est(**F64).fit(np.r_[X2, X2], md2, init=p).rebinned_, 'n x2.67 did not re-bin'
        return {k: v for k, v in out.items()}

    def counts():
        c0 = bt.BLMLQuick(**F64).fit(X2, md2)
        out = {}
        for cx in (False, True):
            a = Est(**F64).fit(c0.Z_, bt.with_lattice(md2, c0._h_vec), sample_weight=c0.w_, complex_g=cx)
            b = Est(max_sweeps=0, **F64).fit(c0.Z_, bt.with_lattice(md2, c0._h_vec), sample_weight=c0.w_, complex_g=cx, init=a)
            assert abs(ll(a, Q2) - ll(b, Q2)) < 1e-4, 'same-lattice zero-sweep reuse changed the fit'
            out[f'll_complex{cx}'] = ll(a, Q2)
        return out

    def selectors():
        out = {}
        for s in ('shared_hillclimb', 'auto'):
            m, _ = select(X2, bt.make_modes(np.nan, ['sinc', 'sinc']), search=s, engine=eng)
            out[f'fc_{s}'] = bt.modes_fc(m)[0].tolist()
        m, _ = select(X3, bt.make_modes(np.nan, ['sinc', 'dirichlet', 'sinc'], [None, 24.0, None], dep=0), search='auto', engine=eng)
        out['fc_cond_auto'] = bt.modes_fc(m)[0].tolist()
        return out

    def sel_counts():
        wg = grid_counts(X2); wtr = np.random.default_rng(2).binomial(wg.astype(int), 0.8).astype(float)
        m, _ = select_counts(ZG, wtr, wg - wtr, HG, bt.make_modes(np.nan, ['sinc', 'sinc']), engine=eng)
        return dict(fc=bt.modes_fc(m)[0].tolist())

    def rolling():
        tmpl = bt.make_modes(np.nan, ['sinc', 'dirichlet', 'sinc'], [None, 24.0, None], dep=0)
        r = BLMLRolling(engine=eng, search='auto').fit(X3[:3000], tmpl, X3[3000:3750])
        r.update(X3[750:3750], X3[3750:4500])
        return dict(fc=r.history_[-1]['fc'][0])

    def quick_streaming():
        tmpl = bt.make_modes(np.nan, ['sinc', 'dirichlet', 'sinc'], [None, 24.0, None], dep=0)
        s = BLMLQuickStreaming(engine=eng, search='auto', burn_in=1500, **F64).fit(X3[:4500], tmpl, epochs=2, batch=1500)
        return dict(stream_ll=float(np.mean(np.log(np.maximum(s.predict_ratio(X3[4500:]), 1e-300)))))

    def touched_bins_exact():
        """Touched-bin streaming at fixed fc (not trainable): 3 epochs reach the exact fit."""
        md = [{j: dict(kind='sinc', fc=dict(value=1.5, trainable=False)) for j in range(2)}]
        s = BLMLQuickStreaming(engine=eng, burn_in=1000, **F64).fit(X2, md, epochs=3, batch=500)
        ex = Est(**F64).fit(X2, md2)
        assert abs(ll(s, Q2) - ll(ex, Q2)) < 2e-3, f'touched-bin {ll(s, Q2):.4f} vs exact {ll(ex, Q2):.4f}'
        return dict(stream_ll=ll(s, Q2))

    def complex_streaming():
        """Complex touched-bin streaming at a fixed fc reaches the exact complex fit (up to 10 epochs or the solver's
        tolerance: with rebin_sweeps=0 the coefficients catch up after each re-bin over epochs, 3 left it 0.012 short)."""
        md = [{j: dict(kind='sinc', fc=dict(value=1.5, trainable=False)) for j in range(2)}]
        s = BLMLQuickStreaming(engine=eng, burn_in=1000, complex_g=True, **F64).fit(X2, md, epochs=10, batch=500, tol=1e-4)
        ex = Est(**F64).fit(X2, md2, complex_g=True)
        # 1e-2: the exact fit keeps the best of 5 complex starts on all the data, the burn-in picks its start on
        # 1000 points; with an arbitrary phase (Gaussian data) the two can end in different local optima
        assert abs(ll(s, Q2) - ll(ex, Q2)) < 1e-2, f'complex streaming {ll(s, Q2):.4f} vs exact {ll(ex, Q2):.4f}'
        return dict(stream_ll=ll(s, Q2))

    def rank2_streaming():
        """Rank-2 touched-bin streaming (two modes, own fixed fc) reaches the exact rank-2 fit (1e-2: rank K is
        non-convex, the burn-in's start and the exact fit's may end in different local optima). Up to 10 epochs or the
        solver's tolerance: with rebin_tol=15 there is no re-bin start to speed it up, and 3 epochs left it 0.011 short."""
        md = [{j: dict(kind='sinc', fc=dict(value=1.5, trainable=False)) for j in range(2)},
              {j: dict(kind='sinc', fc=dict(value=0.7, trainable=False)) for j in range(2)}]
        s = BLMLQuickStreaming(engine=eng, burn_in=1000, **F64).fit(X2, md, epochs=10, batch=500, tol=1e-4)
        ex = Est(**F64).fit(X2, bt.make_modes([[1.5, 1.5], [0.7, 0.7]], ['sinc', 'sinc'], rank=2))
        assert abs(ll(s, Q2) - ll(ex, Q2)) < 1e-2, f'rank-2 streaming {ll(s, Q2):.4f} vs exact {ll(ex, Q2):.4f}'
        return dict(stream_ll=ll(s, Q2), exact_ll=ll(ex, Q2))

    def complex_rank2_streaming():
        """Up to 20 epochs or the solver's tolerance: 10 left it 0.010-0.011 short with either rebin_tol (1 or 15)."""
        md = [{j: dict(kind='sinc', fc=dict(value=1.5, trainable=False)) for j in range(2)},
              {j: dict(kind='sinc', fc=dict(value=0.7, trainable=False)) for j in range(2)}]
        s = BLMLQuickStreaming(engine=eng, burn_in=1000, complex_g=True, **F64).fit(X2, md, epochs=20, batch=500, tol=1e-4)
        ex = Est(**F64).fit(X2, bt.make_modes([[1.5, 1.5], [0.7, 0.7]], ['sinc', 'sinc'], rank=2), complex_g=True)
        assert abs(ll(s, Q2) - ll(ex, Q2)) < 1e-2, f'complex rank-2 streaming {ll(s, Q2):.4f} vs exact {ll(ex, Q2):.4f}'
        return dict(stream_ll=ll(s, Q2), exact_ll=ll(ex, Q2))

    def binned_streaming():
        s = BLMLQuickStreaming(engine=eng, burn_in=1000, **F64)
        for j in range(4):
            s.partial_fit(ZG, bt.with_lattice(bt.make_modes(np.nan, ['sinc', 'sinc']), HG),
                          sample_weight=grid_counts(X2[j * 1000:(j + 1) * 1000]))
        return dict(stream_ll=ll(s, Q2))

    def ball():
        out = {}
        if eng == 'gpu':                                   # device-built radial tables vs host scipy
            import cupy as cp
            from blml_gpu import _J01
            for D in (2, 3):
                fc, h = np.full(D, 1.4), np.full(D, 0.01)
                a = bt.radial_table('ball', list(range(D)), fc, h, 40000)
                b = cp.asnumpy(bt.radial_table('ball', list(range(D)), fc, h, 40000, xp=cp, j01=_J01))
                assert np.abs(a - b).max() <= 1e-9 * np.abs(a).max(), f'D={D} device table differs'
        mb = bt.make_modes([1.6, 1.6], ['ball', 'ball'])
        m = Est(**F64).fit(X2, mb)
        out['ll'] = ll(m, Q2)
        out['grad'] = float(m.fc_grad(Q2[:300], [[0, 1]], implicit_grad=True, eps_rel=1e-3).mean(0)[0])
        return out

    def streaming_param_every():
        s = BLMLQuickStreaming(engine=eng, search='auto', burn_in=1000, param_every=2, **F64).fit(
            X2, bt.make_modes(np.nan, ['sinc', 'sinc']), epochs=2, batch=500)
        return dict(stream_ll=ll(s, Q2))

    def param_grads():
        m = Est(**F64).fit(X3, bt.make_modes([1.5, 0.3, 0.5], ['sinc', 'dirichlet', 'sinc'], [None, 24.0, None]))
        g = m.param_grad(X3[:300] + 0.01, [[0], ([1], 'period'), [2]], implicit_grad=True, eps_rel=1e-3).mean(0)
        return dict(grad_fc0=float(g[0]), grad_period=float(g[1]), grad_fc2=float(g[2]))

    def period_streaming():
        r = np.random.default_rng(5); t = r.uniform(0, 240, 12000)
        X = np.c_[np.sin(2 * np.pi * t / 24) + 0.4 * r.normal(size=12000), t]
        tmpl = [{0: dict(kind='sinc', dep=True), 1: dict(kind='dirichlet', fc=0.2, period=dict(value=22.0, trainable=True))}]
        s = BLMLQuickStreaming(engine=eng, search='auto', burn_in=4000, **F64).fit(X, tmpl, epochs=3, batch=1000)
        assert abs(bt.modes_period(s.modes_)[0][1] - 24) < 1.0, f'period {bt.modes_period(s.modes_)[0][1]:.2f}'
        return dict(period=float(bt.modes_period(s.modes_)[0][1]))

    return [('param_grad', param_grads), ('period streaming', period_streaming), ('real fit', real_fit), ('rank-2 real', rank2), ('complex + 5 starts', complex_starts),
            ('conditional', conditional), ('warm start + max_sweeps', warm_and_cap), ('fit_counts', counts),
            ('select', selectors), ('select_counts', sel_counts), ('BLMLRolling', rolling),
            ('BLMLQuickStreaming', quick_streaming), ('touched-bin = exact', touched_bins_exact),
            ('complex streaming = exact', complex_streaming),
            ('BLMLBinnedStreaming', binned_streaming), ('ball', ball), ('streaming param_every', streaming_param_every),
            ('rank-2 streaming = exact', rank2_streaming), ('complex rank-2 streaming', complex_rank2_streaming)]


def run(eng, path):
    only = os.environ.get('SMOKE_ONLY')
    res = {}
    for name, fn in checks(eng):
        if only and name not in only.split(','):
            continue
        t = time.time()
        try:
            res[name] = dict(ok=True, s=round(time.time() - t, 1), **fn())
            res[name]['s'] = round(time.time() - t, 1)
            print(f'RAN {name} ({res[name]["s"]}s): {res[name]}', flush=True)
        except Exception as e:
            res[name] = dict(ok=False, error=f'{type(e).__name__}: {e}')
            print(f'FAIL {name}: {res[name]["error"]}', flush=True); traceback.print_exc(limit=4)
        json.dump(res, open(path, 'w'), indent=1)


def compare(pa, pb):
    a, b = json.load(open(pa)), json.load(open(pb))
    n_ok = 0
    for name in a:
        if name not in b:
            continue
        ra, rb = a[name], b[name]
        if not (ra['ok'] and rb['ok']):
            print(f'FAIL {name}: cpu {ra.get("error", "ok")} | gpu {rb.get("error", "ok")}'); continue
        bad = []
        for k, va in ra.items():
            if k in ('ok', 's'):
                continue
            vb = rb[k]
            kind = 'fc' if k.startswith('fc') else ('ll' if k.startswith('ll') else ('grad' if k.startswith('grad') else k))
            if kind == 'fc':
                if not np.allclose(va, vb, rtol=TOL['fc']):
                    bad.append(f'{k}: {np.round(va, 3)} vs {np.round(vb, 3)}')
            elif abs(va - vb) > TOL.get(kind, 5e-3):
                bad.append(f'{k}: {va:.4f} vs {vb:.4f}')
        n_ok += not bad
        print(('PASS ' if not bad else 'FAIL ') + f'{name} (cpu {ra["s"]}s, gpu {rb["s"]}s)' + ('' if not bad else ': ' + '; '.join(bad)))
    print(f'SUMMARY {n_ok}/{len([x for x in a if x in b])} passed')


if __name__ == '__main__':
    if sys.argv[1] == 'compare':
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else f'parity_{sys.argv[1]}.json')
