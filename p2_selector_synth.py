"""Paper 2: the cutoff selector on truths with known answers (GPU).
Run: ./colab_job.sh start p2syn p2_selector_synth.py   (one JSON per cell; resumable).

Truths: T1, T2 (band-limited, fc_true = 0.4), T3 (correlated normal, no true cutoff) from
Paper 1, and T4 (anisotropic): T1 with dimensions 3..d stretched 4x, so fc_true = 0.4 in the
first two dimensions and 0.1 in the rest -- a shared cutoff cannot be right for both.
Methods, all data-driven except the oracle: BLML with the old shared selector and with the
auto selector (volume grid + per-dimension refinement), BLML at fc_true (oracle), and
Gaussian KDE with Scott's rule, least-squares CV, likelihood CV on the same validation split
as BLML, and Abramson's adaptive KDE with the likelihood-CV pilot.
"""
import os, sys, json, time, subprocess

OUT = os.environ.get('BLML_OUT_DIR', 'outputs_p2syn')
CODE = os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(OUT, 'syn'), exist_ok=True)
if subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda13x', 'numba'], check=True)
sys.path.insert(0, CODE)
import numpy as np
import cupy as cp
import blml_trivial as bt
from blml_trivial import make_modes, modes_fc
from blml_trivial import BLMLQuick
from p1_truth import Truth, DefensiveProposal, FC_TRUE
import p1_estimators as E
from selector import select

bt.DIRICHLET_LEGACY = False
M_F, M_Q = 100000, 200000
C_GRID = np.geomspace(0.2, 3.0, 12)
D_LIST = [int(v) for v in os.environ.get('BLML_SYN_D', '2,4,8').split(',')]
N_LIST = [int(v) for v in os.environ.get('BLML_SYN_N', '1000,10000,100000').split(',')]
TRUTHS = os.environ.get('BLML_SYN_TRUTHS', 'T1,T4,T2,T3').split(',')

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def path(name):
    return os.path.join(OUT, 'syn', name + '.json')


class Stretched:
    """T4: T1 with coordinates scaled by s (pdf by change of variables); fc_true = 0.4 / s."""
    def __init__(self, d, stretch=4.0):
        self.base, self.d, self.kind = Truth('T1', d), d, 'T4'
        self.s = np.where(np.arange(d) < 2, 1.0, stretch)
        self.fc_true = FC_TRUE / self.s
    def sample(self, rng, n):
        return self.base.sample(rng, n) * self.s
    def pdf(self, X):
        return self.base.pdf(np.atleast_2d(X) / self.s) / np.prod(self.s)


def make_truth(kind, d):
    if kind == 'T4':
        return Stretched(d)
    t = Truth(kind, d)
    t.fc_true = None if kind == 'T3' else np.full(d, FC_TRUE)
    return t

def eval_sets(truth, d):
    """Exact samples from f (for KL) and defensive importance samples (for ISE), as in Paper 1."""
    rng = np.random.default_rng(20_000 + d)
    Xf = truth.sample(rng, M_F)
    base = 2.0 if truth.kind == 'T3' else 6.0
    q = DefensiveProposal(truth, scale=base * getattr(truth, 's', 1.0))
    Xq = q.sample(rng, M_Q)
    ff = truth.pdf(Xf)
    return dict(Xf=Xf, ff=ff, Xq=Xq, fq=truth.pdf(Xq), qq=q.pdf(Xq), int_f2=float(ff.mean()))

def evaluate(fn, ev):
    t0 = time.time()
    r = E.metrics(E.to_np(fn(ev['Xf'])), ev['ff'], E.to_np(fn(ev['Xq'])), ev['fq'], ev['qq'], ev['int_f2'])
    r['eval_s'] = time.time() - t0
    return r

def blml(X, fc):
    return BLMLQuick(engine='gpu', dtype=np.float64).fit(X, make_modes(list(np.atleast_1d(fc).astype(float)), ['sinc'] * X.shape[1], None))

def lcv_multiplier(X, rng):
    """KDE bandwidth multiplier by validation log-likelihood on a random 80/20 split -- the same
    criterion and split proportion as the BLML selector."""
    idx = rng.permutation(len(X)); a, b = X[idx[:int(0.8 * len(X))]], X[idx[int(0.8 * len(X)):]]
    h0, ag = E.scott(a), cp.asarray(a)
    ll = [np.log(np.maximum(E.to_np(E.kde_gauss(ag, b, c * h0)), 1e-300)).mean() for c in C_GRID]
    return float(C_GRID[int(np.argmax(ll))])


for kind in TRUTHS:
    for d in D_LIST:
        truth = make_truth(kind, d)
        ev = None
        for n in N_LIST:
            nm = f'{kind}_d{d}_n{n}'
            if os.path.exists(path(nm)):
                continue
            ev = ev or eval_sets(truth, d)
            reps = 4 if n < 100000 else 2
            rows = []
            for rep in range(reps):
                X = truth.sample(np.random.default_rng([3, d, n, rep]), n)
                Xg = cp.asarray(X)
                row = {}
                for mode in ('shared', 'auto'):
                    t0 = time.time()
                    sel, _ = select(X, make_modes(np.nan, ['sinc'] * d), search=mode, seed=rep, engine='gpu'); fc = modes_fc(sel)[0]
                    t_sel = time.time() - t0
                    m = blml(X, fc)
                    row[f'blml_{mode}'] = evaluate(m.predict, ev)
                    row[f'blml_{mode}'].update(fc=[float(v) for v in fc], select_s=t_sel)
                if truth.fc_true is not None:
                    row['blml_oracle'] = evaluate(blml(X, truth.fc_true).predict, ev)
                h0 = E.scott(X)
                row['kde_scott'] = evaluate(lambda Z: E.kde_gauss(Xg, Z, h0), ev)
                c = E.lscv_gauss(X, C_GRID)
                row['kde_lscv'] = evaluate(lambda Z: E.kde_gauss(Xg, Z, c * h0), ev); row['kde_lscv']['c'] = c
                c = lcv_multiplier(X, np.random.default_rng(rep))
                row['kde_lcv'] = evaluate(lambda Z: E.kde_gauss(Xg, Z, c * h0), ev); row['kde_lcv']['c'] = c
                row['kde_adaptive'] = evaluate(lambda Z: E.kde_adaptive(Xg, Z, c * h0), ev)
                rows.append(row)
                log(f"{nm} rep={rep}: " + ' '.join(f"{k}={v['rel_ise']:.2e}/{v['kl']:.3f}" for k, v in row.items())
                    + f" | fc auto={np.round(row['blml_auto']['fc'], 3)}")
            with open(path(nm), 'w') as f:
                json.dump(dict(truth=kind, d=d, n=n, fc_true=None if truth.fc_true is None else list(truth.fc_true),
                               reps=rows), f, default=float)
            log(f'saved {nm}')
log('ALL DONE')
