"""Paper 2 (and the honest case of paper 1): BLML with a data-driven cut-off against data-driven kernel estimators on
truths whose square root is NOT band-limited and whose spectrum decays slowly:
  L  product Laplace, f = prod 1/2 exp(-|x|): a kink at 0 (spectrum of sqrt f ~ nu^-2)
  B  product of Beta(2, 2) on [-3, 3], f = prod u(1 - u), u = (x + 3)/6: hard edges (sqrt f ~ sqrt(distance))
Same design and metrics as the smooth truths of the streaming paper (p2_selector_synth.py): d in {2, 4, 8},
n in {1e3, 1e4}, 4 replicates; relative ISE by defensive importance sampling, KL on draws from f.
Methods: streaming BLML (burn-in min(n/2, 5000) with the selector's auto search, batches of n/20, Adam on log fc,
scored after 2, 5 and 10 passes (fixed budgets) and at convergence: coefficients at the solver's tolerance 1e-4 and cut-off gradients within one SE of zero, at most
20 passes), one-shot BLML (auto search on all points, refit), and Gaussian KDE with Scott's rule, least-squares CV,
likelihood CV, and Abramson's adaptive KDE (LCV pilot).
Run: ./colab_job.sh start synns syn_nonsmooth_job.py T4   (one JSON per cell in BLML_OUT_DIR/syn; resumable)"""
import os, sys, json, time, subprocess
OUT = os.environ.get('BLML_OUT_DIR', 'outputs_synns'); CODE = os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(OUT, 'syn'), exist_ok=True)
if subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda13x', 'numba'], check=True)
sys.path.insert(0, CODE)
import numpy as np
import cupy as cp
import blml_trivial as bt
from blml_trivial import make_modes, modes_fc, BLMLQuick
from blml_rolling import BLMLQuickStreaming
from p1_truth import DefensiveProposal, NonSmooth
import p1_estimators as E
from selector import select

bt.DIRICHLET_LEGACY = False
M_F, M_Q = 100000, 200000
C_GRID = np.geomspace(0.2, 3.0, 12)
D_LIST = [int(v) for v in os.environ.get('BLML_SYN_D', '2,4,8').split(',')]
N_LIST = [int(v) for v in os.environ.get('BLML_SYN_N', '1000,10000').split(',')]
TRUTHS = os.environ.get('BLML_SYN_TRUTHS', 'L,B').split(',')
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def eval_sets(truth, d):
    rng = np.random.default_rng(20_000 + d)
    Xf = truth.sample(rng, M_F); q = DefensiveProposal(truth, scale=2.0); Xq = q.sample(rng, M_Q); ff = truth.pdf(Xf)
    return dict(Xf=Xf, ff=ff, Xq=Xq, fq=truth.pdf(Xq), qq=q.pdf(Xq), int_f2=float(ff.mean()))


def evaluate(fn, ev):
    t0 = time.time()
    r = E.metrics(E.to_np(fn(ev['Xf'])), ev['ff'], E.to_np(fn(ev['Xq'])), ev['fq'], ev['qq'], ev['int_f2'])
    r['eval_s'] = time.time() - t0
    return r


def lcv_multiplier(X, rng):
    idx = rng.permutation(len(X)); a, b = X[idx[:int(0.8 * len(X))]], X[idx[int(0.8 * len(X)):]]
    h0, ag = E.scott(a), cp.asarray(a)
    ll = [np.log(np.maximum(E.to_np(E.kde_gauss(ag, b, c * h0)), 1e-300)).mean() for c in C_GRID]
    return float(C_GRID[int(np.argmax(ll))])


for kind in TRUTHS:
    for d in D_LIST:
        truth, ev = NonSmooth(kind, d), None
        for n in N_LIST:
            path = os.path.join(OUT, 'syn', f'{kind}_d{d}_n{n}.json')
            if os.path.exists(path):
                continue
            ev = ev or eval_sets(truth, d); rows = []
            for rep in range(4):
                X = truth.sample(np.random.default_rng([5, d, n, rep]), n); Xg = cp.asarray(X); row = {}
                t0 = time.time(); budget, eval_t = {}, [0.0]

                def on_epoch(ep, st):                    # fixed pass budgets, scored along the way
                    if ep + 1 in (2, 5, 10):
                        r = evaluate(st.predict, ev); budget[ep + 1] = r['rel_ise']; eval_t[0] += r['eval_s']
                s = BLMLQuickStreaming(engine='gpu', search='auto', burn_in=min(n // 2, 5000), seed=rep).fit(
                    X[np.random.default_rng(rep).permutation(n)], make_modes(np.nan, ['sinc'] * d), epochs=20, batch=max(n // 20, 1), tol=1e-4, on_epoch=on_epoch)
                row['blml_stream'] = evaluate(s.predict, ev)
                row['blml_stream'].update(fc=[float(v) for v in np.ravel(modes_fc(s.modes_)[0])], fit_s=time.time() - t0 - eval_t[0],
                                           epochs=int(s.epochs_), converged=bool(s.converged_),
                                           rel_ise_at_epoch={str(k): v for k, v in budget.items()})
                t0 = time.time()
                sel, _ = select(X, make_modes(np.nan, ['sinc'] * d), search='auto', seed=rep, engine='gpu'); fc = modes_fc(sel)[0]
                m = BLMLQuick(engine='gpu', dtype=np.float64).fit(X, make_modes(list(np.atleast_1d(fc).astype(float)), ['sinc'] * d, None))
                row['blml_oneshot'] = evaluate(m.predict, ev); row['blml_oneshot'].update(fc=[float(v) for v in fc], fit_s=time.time() - t0)
                h0 = E.scott(X)
                row['kde_scott'] = evaluate(lambda Z: E.kde_gauss(Xg, Z, h0), ev)
                c = E.lscv_gauss(X, C_GRID); row['kde_lscv'] = evaluate(lambda Z: E.kde_gauss(Xg, Z, c * h0), ev); row['kde_lscv']['c'] = c
                c = lcv_multiplier(X, np.random.default_rng(rep))
                row['kde_lcv'] = evaluate(lambda Z: E.kde_gauss(Xg, Z, c * h0), ev); row['kde_lcv']['c'] = c
                row['kde_adaptive'] = evaluate(lambda Z: E.kde_adaptive(Xg, Z, c * h0), ev)
                rows.append(row)
                log(f"{kind} d={d} n={n} rep {rep}: " + ', '.join(f"{k} {v['rel_ise']:.3g}" for k, v in row.items()))
            json.dump(dict(truth=kind, d=d, n=n, reps=rows), open(path, 'w'), indent=1, default=float)
log('done')
