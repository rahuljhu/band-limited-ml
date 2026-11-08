"""Paper 5: two synthetic comparisons at d = 4, n in N_LIST (T1, T2 band-limited with fc_true = 0.4; T3 correlated normal),
paper 1's truths and evaluation (relative ISE by importance sampling).
  fixed  (BLML_SYN_PART=fixed): streaming BLML with fc frozen (trainable=False) run to the solver's tolerance
         (fit(tol=TOL), the running check) against the one-shot BCD fit at the same fc and tol: fc_true = 0.4 for T1,
         T2; 1.0 per dimension for T3 (no true cut-off; any fixed fc serves this comparison).
  select (BLML_SYN_PART=select): streaming with Adam on fc (default burn-in, 'auto' search on it, EPOCHS passes)
         against one-shot selection ('auto' search on all points, then the refit).
Both methods of a part on the same engine and dtype (CPU float64, GPU float32 kernels with float64 sums); relative ISE
after every streaming epoch (curve; evaluation time excluded). Usage: p5_synth_figs.py [engine cpu|gpu]
env: BLML_SYN_PART, BLML_SYN_CURVE, BLML_SYN_TRUTHS, BLML_SYN_N, BLML_SYN_REPS, BLML_SYN_D, BLML_SYN_EPOCHS, BLML_SYN_TOL."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')
import numpy as np
import blml_trivial as bt
from blml_trivial import make_modes, modes_fc, BLMLQuick
from blml_rolling import BLMLQuickStreaming
from selector import select
from p1_truth import Truth
import p1_estimators as E
from synth_stream import eval_sets

bt.DIRICHLET_LEGACY = False
ENGINE = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
D = int(os.environ.get('BLML_SYN_D', '4'))
N_LIST = [int(v) for v in os.environ.get('BLML_SYN_N', '20000,50000').split(',')]
TRUTHS = os.environ.get('BLML_SYN_TRUTHS', 'T1,T2,T3').split(',')
REPS = int(os.environ.get('BLML_SYN_REPS', '3'))
DT = np.float32 if ENGINE == 'gpu' else np.float64
PART = os.environ.get('BLML_SYN_PART', 'select')
EPOCHS = int(os.environ.get('BLML_SYN_EPOCHS', '4' if PART == 'select' else '30'))
TOL = float(os.environ.get('BLML_SYN_TOL', '1e-4'))
CURVE = os.environ.get('BLML_SYN_CURVE', '0') == '1'   # relative ISE after every epoch (slow: 3e5 evaluation points)
FC_FIX = {'T1': 0.4, 'T2': 0.4, 'T3': 1.0}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', os.environ.get('BLML_SYN_OUT', 'p5synth'))
os.makedirs(OUT, exist_ok=True)


def sync():
    if ENGINE == 'gpu':
        import cupy as cp
        cp.cuda.Device().synchronize()


def timed(f):
    sync(); t0 = time.time(); r = f(); sync()
    return r, time.time() - t0


def frozen(fc):
    return [{j: dict(kind='sinc', fc=dict(value=float(v), trainable=False)) for j, v in enumerate(fc)}]


def streamed(s, X, modes, bs, ev, **fit_kw):
    """Fit with per-epoch scores; times exclude the evaluations."""
    curve, ev_s = [], [0.0]
    def cb(ep, m):
        sync(); t1 = time.time()
        r = E.metrics(m.predict(ev['Xf']), ev['ff'], m.predict(ev['Xq']), ev['fq'], ev['qq'], ev['int_f2'])
        curve.append(dict(epoch=ep + 1, rel_ise=float(r['rel_ise']), s=t1 - t0 - ev_s[0],
                          fc=[float(v) for v in modes_fc(m.modes_)[0]]))
        sync(); ev_s[0] += time.time() - t1
    sync(); t0 = time.time()
    s.fit(X, modes, epochs=EPOCHS, batch=bs, on_epoch=cb if CURVE else None, **fit_kw)
    if not CURVE:                                  # one score at the end: per-epoch scores cost far more than the fit
        sync(); t = time.time() - t0
        cb(-1, s); curve[-1].update(epoch=s.epochs_, s=t)
    return dict(curve[-1], curve=curve, epochs=s.epochs_, residual=s.residual_, converged=s.converged_)


def score(m, ev, t, fc):
    r = E.metrics(m.predict(ev['Xf']), ev['ff'], m.predict(ev['Xq']), ev['fq'], ev['qq'], ev['int_f2'])
    return dict(rel_ise=float(r['rel_ise']), nll=float(r['nll']), s=t, fc=[float(v) for v in fc])


if __name__ == '__main__':
    for kind in TRUTHS:
        truth = Truth(kind, D); ev = eval_sets(truth, D)
        for n in N_LIST:
            p = os.path.join(OUT, f'{PART}_{kind}_d{D}_n{n}_{ENGINE}.json')
            if os.path.exists(p):
                continue
            bs = max(50, n // 20); rows = []
            for rep in range(REPS):
                X = truth.sample(np.random.default_rng([3, D, n, rep]), n)
                if PART == 'fixed':
                    fc = [FC_FIX[kind]] * D
                    m, t = timed(lambda: BLMLQuick(engine=ENGINE, dtype=DT, tol=TOL).fit(X, make_modes(fc, ['sinc'] * D)))
                    r = dict(bcd_fixed=dict(score(m, ev, t, fc), sweeps=int(m.n_sweeps_), converged=bool(m.converged_)))
                    r['stream_fixed'] = streamed(BLMLQuickStreaming(engine=ENGINE, seed=rep, dtype=DT, rebin_tol=1.0), X, frozen(fc), bs, ev,
                                                 tol=TOL)
                else:
                    tmpl = make_modes(np.nan, ['sinc'] * D)
                    sel, t_sel = timed(lambda: select(X, tmpl, search='auto', seed=rep, engine=ENGINE)[0])
                    fc_os = modes_fc(sel)[0]
                    m, t_fit = timed(lambda: BLMLQuick(engine=ENGINE, dtype=DT).fit(X, make_modes(list(fc_os), ['sinc'] * D)))
                    r = dict(oneshot_select=score(m, ev, t_sel + t_fit, fc_os))
                    r['stream_adam'] = streamed(BLMLQuickStreaming(engine=ENGINE, search='auto', seed=rep, dtype=DT, rebin_tol=1.0), X,
                                                tmpl, bs, ev)
                rows.append(r)
                print(f"{PART} {kind} n={n} rep {rep}: " + ' | '.join(f"{k} {v['rel_ise']:.2e} {v['s']:.1f}s" for k, v in r.items()),
                      flush=True)
            json.dump(dict(part=PART, truth=kind, d=D, n=n, engine=ENGINE, dtype=DT.__name__, epochs=EPOCHS, tol=TOL,
                           reps=rows), open(p, 'w'))
    print('ALL DONE', flush=True)
