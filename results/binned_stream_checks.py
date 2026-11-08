"""Checks of the streaming functionality of blml_binned (BLMLBinnedStreaming / BLMLEmpiricalStreaming, _KernelLearner),
the BLMLQuickStreaming features on a fixed lattice:
  loo        leave-batch-out: the score and implicit gradient of a held batch from _left_out (coefficients scaled by the
             remaining share) against an exact refit on the held counts less the batch (real, complex)
  epochs     fit(batches, epochs, tol): a cut-off recovered from above and below over epochs, with the residual test
             (converged_, epochs_, residual_), against one pass (real, complex; binned and empirical)
  background rank 2 per-mode: a sinc signal mode plus a fixed dirichlet background mode periodic over the box
             (N = 0: flat; trainable=False): the background fraction Q_bg / Q and the signal fc learned over epochs
             (truth: fraction 0.3, flat)
  burnin     an fc left out of the modes (NaN) selected at the burn-in, with and without split_groups (complex: the
             truth changes sign, so the real root is not band-limited and its held-out score rises to the Nyquist limit)
  roles      bcd_frac < 1 (learn-only counts) and freeze_frac > 0 run, trace_trainable records the trainable values;
             split_groups keep groups contiguous in the epoch order
Usage: python results/binned_stream_checks.py [cpu|gpu] [out.json]"""
import sys, json, time, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_trivial import kernel_class
from blml_binned import BLMLBinned, BLMLBinnedStreaming, BLMLEmpiricalStreaming, _np, _batch_order
engine = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
out = sys.argv[2] if len(sys.argv) > 2 else f'results/binned_stream_checks_{engine}.json'
rows = []


def truth(shape, fc0, n_src=8, seed=3, floor=0.0):
    r = np.random.default_rng(seed); G = np.meshgrid(*[np.arange(m, dtype=float) for m in shape], indexing='ij'); g = 0.0
    for _ in range(n_src):
        x0 = [r.uniform(0.3 * m, 0.7 * m) for m in shape]; a = r.choice([-1, 1]) * r.uniform(0.5, 1.0); K = 1.0
        for j in range(len(shape)):
            K = K * kernel_class('sinc').eval(dict(fc=np.array([fc0]), period=None), [G[j] - x0[j]])
        g = g + a * K
    p = g ** 2; p = p / p.sum(); return (1 - floor) * p + floor / p.size


def draws(p, n, seed):
    return np.random.default_rng(seed).multinomial(n, p.ravel()).reshape(p.shape).astype(float)


# ---------------------------------------------------------------- loo
p = truth((96,), 0.2); batches = [draws(p, 600, 10 + i) for i in range(10)]; tot = sum(batches)
modes = bt.make_modes([0.2], ['sinc'])
for cg in (False, True):
    s = BLMLBinnedStreaming(engine=engine, tol=1e-10, fc_update=None).fit(batches, modes, epochs=1, complex_g=cg)
    s.max_sweeps = 20000; E, U = s._solve(s.e_hat_, None); s._finish(E, U)          # the held fit to convergence
    errs_s, errs_g = [], []
    for j in (0, 4, 8):
        b = batches[j]; eps = 1e-3
        with s._left_out(b):
            sl, gl = s.log_density(b), s.implicit_grad(b, eps)[0][0]
        ex = BLMLBinned(engine=engine, tol=1e-10).fit(tot - b, modes, complex_g=cg, init=s)
        se, ge = ex.log_density(b), ex.implicit_grad(b, eps)[0][0]
        errs_s.append(abs(sl - se)); errs_g.append(abs(gl - ge) / max(abs(ge), 1e-3))
    rows.append(dict(check='loo', cell=f'1d {"complex" if cg else "real"}', score_abs_err=max(errs_s), grad_rel_err=max(errs_g)))
    print(f'loo {"complex" if cg else "real"}: max |score - exact refit| {max(errs_s):.2e} nats, max grad rel err {max(errs_g):.2e}', flush=True)

# ---------------------------------------------------------------- epochs
p = truth((128,), 0.2); batches = [draws(p, 400, 100 + i) for i in range(15)]      # 6000 counts: one pass is short
import selector as sel_
tr_ = sum(batches[:12]); va_ = sum(batches[12:]); ref = {}
for cg in (False, True):                                                          # the batch selector on the same counts
    ref[cg] = float(sel_._search(lambda v: -np.inf if v[0] > 0.5 else BLMLBinned(engine=engine, tol=1e-8).fit(tr_, bt.make_modes([v[0]], ['sinc']), complex_g=cg)._val_score(va_),
                                1, ['sinc'], [None], fc0=0.02, tag='ref')[0][0])
    rows.append(dict(check='epochs', cell=f'batch selector {"complex" if cg else "real"}', fc_selected=ref[cg]))
    print(f'epochs reference: batch selector ({"complex" if cg else "real"}) fc {ref[cg]:.3f}', flush=True)
for cls_name, cls, kw in [('binned real', BLMLBinnedStreaming, dict(batch_sweeps=50)), ('binned complex', BLMLBinnedStreaming, dict(batch_sweeps=50)),
                          ('empirical complex', BLMLEmpiricalStreaming, dict(lam=1e-4, batch_outer=30))]:
    cg = 'complex' in cls_name
    for start in (1.6, 0.6):
        t = time.time(); m0 = bt.make_modes([0.2 * start], ['sinc'])
        one = cls(engine=engine, tol=1e-8, lr=0.05, param_every=1, **kw).fit(batches, m0, epochs=1, complex_g=cg)
        ep = cls(engine=engine, tol=1e-8, lr=0.05, param_every=1, **kw).fit(batches, m0, epochs=12, complex_g=cg, tol=1e-3, tol_check='exact')
        f1, fe = one._fc_now()[0][0], ep._fc_now()[0][0]
        rows.append(dict(check='epochs', cell=f'{cls_name} start x{start}', fc_true=0.2, fc_one_pass=f1, fc_epochs=fe, epochs=ep.epochs_,
                         converged=ep.converged_, residual=ep.residual_, s=time.time() - t))
        print(f'epochs {cls_name:18s} start x{start}: one pass fc {f1:.3f}, epochs fc {fe:.3f} (true 0.2; {ep.epochs_} epochs, '
              f'converged {ep.converged_}, residual {ep.residual_:.1e})  {time.time() - t:.0f}s', flush=True)

# ---------------------------------------------------------------- background mode
shape = (96, 64); p = truth(shape, 0.15, floor=0.3); batches = [draws(p, 3000, 300 + i) for i in range(12)]
sig = bt.make_modes([0.15 * 1.3, 0.15 * 1.3], ['sinc', 'sinc'])[0]
bg = {j: dict(kind='dirichlet', period=float(P), fc=dict(value=0.5 * 2 / P, trainable=False))   # fixed: N = 0, flat
      for j, P in enumerate(shape)}            # (the background's shape is a model choice, not learned)
for cg in (False, True):
    t = time.time()
    s = BLMLBinnedStreaming(engine=engine, tol=1e-6, lr=0.05, param_every=1, batch_sweeps=50).fit(
        batches, [sig, bg], epochs=8, complex_g=cg)
    E, U = _np(s.e_hat_), _np(s.Se_); Q = [float(np.real(np.sum(np.conj(E[k]) * U[k]))) for k in range(2)]
    frac = Q[1] / sum(Q); fcs = s._fc_now(); harm = [int(np.floor(v * P / 2)) for v, P in zip(fcs[1], shape)]
    rows.append(dict(check='background', cell='complex' if cg else 'real', bg_true=0.3, bg_fraction=frac, fc_signal=fcs[0].tolist(),
                     bg_harmonics=harm, s=time.time() - t))
    print(f'background {"complex" if cg else "real"}: fraction {frac:.3f} (true 0.3), signal fc {np.round(fcs[0], 3)} (true 0.15), '
          f'bg harmonics {harm} (fixed)  {time.time() - t:.0f}s', flush=True)

# ---------------------------------------------------------------- burn-in selection
p = truth((128,), 0.2); batches = [draws(p, 1500, 500 + i) for i in range(8)]
for groups in (None, [i // 2 for i in range(8)]):
    s = BLMLBinnedStreaming(engine=engine, tol=1e-8, burn_in=4000, fc_update=None).fit(
        batches, bt.make_modes([np.nan], ['sinc']), epochs=1, split_groups=groups, complex_g=True)
    sel = s.history_[[i for i, h in enumerate(s.history_) if h.get('burn_in')][0]]['fc'][0][0]
    rows.append(dict(check='burnin', cell='groups' if groups else 'binomial', fc_selected=sel, fc_true=0.2))
    print(f'burn-in selection, complex ({"groups" if groups else "binomial"}): fc {sel:.3f} (true 0.2)', flush=True)

# ---------------------------------------------------------------- roles
p = truth((128,), 0.2); batches = [draws(p, 600, 700 + i) for i in range(10)]
for kw in (dict(bcd_frac=0.8), dict(freeze_frac=0.25), dict(trace_trainable=True)):
    s = BLMLBinnedStreaming(engine=engine, tol=1e-8, lr=0.05, param_every=1, batch_sweeps=50, **kw).fit(batches, bt.make_modes([0.3], ['sinc']), epochs=4)
    ok = ('trainable' in s.history_[-1]) if 'trace_trainable' in kw else True
    rows.append(dict(check='roles', cell=str(kw), fc_final=float(s._fc_now()[0][0]), ok=bool(ok), n=float(s.n_)))
    print(f'roles {kw}: fc {s._fc_now()[0][0]:.3f}, held counts {s.n_:.0f} (all: {sum(b.sum() for b in batches):.0f}), ok {ok}', flush=True)
order = _batch_order(9, [0, 0, 1, 1, 1, 2, 2, 3, 3], np.random.default_rng(0)); g = [[0, 0, 1, 1, 1, 2, 2, 3, 3][i] for i in order]
contig = all(g.count(v) == (len(g) - g[::-1].index(v) - g.index(v)) for v in set(g))
rows.append(dict(check='roles', cell='split_groups contiguous', ok=bool(contig))); print('split_groups contiguous:', contig)
json.dump(rows, open(out, 'w'), indent=1)
