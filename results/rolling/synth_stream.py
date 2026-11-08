"""Paper 5: the synthetic selector study of paper 2 (p2_selector_synth.py) with streaming BLML. Same truths
(T1, T2 band-limited with fc_true = 0.4; T3 correlated normal), the same seeded samples and the same
evaluation (relative ISE by importance sampling, KL on exact samples). Streaming: BLMLQuickStreaming.fit, the
points in a random order (i.i.d. samples), burn-in on min(n/2, 5000) points with the 'auto' search, batches
of n/20, touched-bin updates and Adam on log fc with the analytic gradient, 2 epochs (the second continues
from the first, revisited batches scored leave-batch-out). One-shot selection, the oracle and the KDEs are read from paper 2's
frozen results (results/p2syn5/syn) for the same cell and replicate.
Usage: synth_stream.py [engine cpu|gpu]   env: BLML_SYN_TRUTHS, BLML_SYN_D, BLML_SYN_N"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')
import numpy as np
import blml_trivial as bt
from blml_trivial import make_modes, modes_fc
from blml_rolling import BLMLQuickStreaming
from p1_truth import Truth, DefensiveProposal, FC_TRUE
import p1_estimators as E
RS = dict(rebin_tol=float(os.environ.get('BLML_REBIN_TOL', 1.0)))   # 1: the paper's runs (library default 15 since the paper)
if os.environ.get('BLML_REBIN_SWEEPS') is not None: RS['rebin_sweeps'] = int(os.environ['BLML_REBIN_SWEEPS'])   # 2: the paper's runs
FT = {} if os.environ.get('BLML_STREAM_TOL') is None else dict(tol=float(os.environ['BLML_STREAM_TOL']))   # stop at the solver's test (epochs = cap)

bt.DIRICHLET_LEGACY = False
ENGINE = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
M_F, M_Q = 100000, 200000                          # as p2_selector_synth.py
D_LIST = [int(v) for v in os.environ.get('BLML_SYN_D', '2,4,8').split(',')]
N_LIST = [int(v) for v in os.environ.get('BLML_SYN_N', '1000,10000,100000').split(',')]
TRUTHS = os.environ.get('BLML_SYN_TRUTHS', 'T1,T2,T3').split(',')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', os.environ.get('BLML_SYN_OUT', 'synstream2')); os.makedirs(OUT, exist_ok=True)
FROZEN = os.path.join(HERE, '..', 'p2syn5', 'syn')


def eval_sets(truth, d):                           # identical to p2_selector_synth.eval_sets
    rng = np.random.default_rng(20_000 + d)
    Xf = truth.sample(rng, M_F)
    q = DefensiveProposal(truth, scale=2.0 if truth.kind == 'T3' else 6.0)
    Xq = q.sample(rng, M_Q)
    ff = truth.pdf(Xf)
    return dict(Xf=Xf, ff=ff, Xq=Xq, fq=truth.pdf(Xq), qq=q.pdf(Xq), int_f2=float(ff.mean()))


def stream(X, d, rep, epochs=int(os.environ.get('BLML_STREAM_EPOCHS', 2))):
    n = len(X); burn = min(n // 2, 5000); bs = max(50, n // 20)
    s = BLMLQuickStreaming(engine=ENGINE, search='auto', burn_in=burn, seed=rep, dtype=np.float64, **RS)
    return s.fit(X, make_modes(np.nan, ['sinc'] * d), epochs=epochs, batch=bs, **FT)


if __name__ == '__main__':
    for kind in TRUTHS:
        for d in D_LIST:
            truth = Truth(kind, d); ev = None
            for n in N_LIST:
                nm = f'{kind}_d{d}_n{n}'; p = os.path.join(OUT, nm + '.json')
                if os.path.exists(p):
                    continue
                ev = ev or eval_sets(truth, d)
                frozen = json.load(open(os.path.join(FROZEN, nm + '.json')))
                rows = []
                for rep in range(4 if n < 100000 else 2):
                    X = truth.sample(np.random.default_rng([3, d, n, rep]), n)      # the same sample as paper 2
                    t0 = time.time(); s = stream(X, d, rep); t_fit = time.time() - t0
                    t0 = time.time()
                    r = E.metrics(s.predict(ev['Xf']), ev['ff'], s.predict(ev['Xq']), ev['fq'], ev['qq'], ev['int_f2'])
                    r.update(fc=[float(v) for v in modes_fc(s.modes_)[0]], epochs=s.epochs_, fit_s=t_fit, eval_s=time.time() - t0)
                    fr = frozen['reps'][rep]
                    kdes = {k: fr[k]['rel_ise'] for k in fr if k.startswith('kde_')}
                    rows.append(dict(stream=r, oneshot=fr['blml_auto']['rel_ise'], oneshot_fc=fr['blml_auto']['fc'],
                                     oracle=fr.get('blml_oracle', {}).get('rel_ise'), best_kde=min(kdes.values()),
                                     best_kde_name=min(kdes, key=kdes.get)))
                    print(f"{nm} rep {rep}: stream rel_ise {r['rel_ise']:.2e} (fit {t_fit:.0f}s, eval {r['eval_s']:.0f}s) | "
                          f"one-shot {rows[-1]['oneshot']:.2e} oracle {rows[-1]['oracle'] or float('nan'):.2e} "
                          f"best KDE {rows[-1]['best_kde']:.2e} | fc {np.round(r['fc'], 3)}", flush=True)
                json.dump(dict(truth=kind, d=d, n=n, engine=ENGINE, reps=rows), open(p, 'w'), default=float)
    print('ALL DONE', flush=True)
