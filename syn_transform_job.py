"""Transformation BLML (blml_transform.GaussianizedDensity: marginals by 1-D KDE, BLML on the Gaussianised
coordinates, exact change of variables) against plain BLML, both with the cut-off chosen by the selector's auto search,
on the same samples: T1 (band-limited), T3 (Gaussian), L (Laplace, kink), B (Beta(2, 2), edges); d in {2, 4, 8},
n in {1e3, 1e4}, four replicates; relative ISE (defensive importance sampling) and KL, as the other studies.
Run: ./colab_job.sh start syntr syn_transform_job.py T4"""
import os, sys, json, time, subprocess
OUT = os.environ.get('BLML_OUT_DIR', 'outputs_syntr'); CODE = os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(OUT, 'syn'), exist_ok=True)
if subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda13x', 'numba'], check=True)
sys.path.insert(0, CODE)
import numpy as np
from p1_truth import Truth, NonSmooth, DefensiveProposal
import p1_estimators as E
from blml_trivial import make_modes, modes_fc, BLMLQuick
from selector import select
from blml_transform import GaussianizedDensity
M_F, M_Q = 100000, 200000
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def eval_sets(truth, kind, d):
    rng = np.random.default_rng(20_000 + d)
    Xf = truth.sample(rng, M_F); q = DefensiveProposal(truth, scale=6.0 if kind == 'T1' else 2.0); Xq = q.sample(rng, M_Q)
    ff = truth.pdf(Xf)
    return dict(Xf=Xf, ff=ff, Xq=Xq, fq=truth.pdf(Xq), qq=q.pdf(Xq), int_f2=float(ff.mean()))


def blml_auto(X):
    d = X.shape[1]; sel, _ = select(X, make_modes(np.nan, ['sinc'] * d), search='auto', seed=0, engine='gpu')
    m = BLMLQuick(engine='gpu', dtype=np.float64).fit(X, make_modes(list(np.atleast_1d(modes_fc(sel)[0]).astype(float)), ['sinc'] * d, None))
    return m


def score(pdf, ev):
    return E.metrics(E.to_np(pdf(ev['Xf'])), ev['ff'], E.to_np(pdf(ev['Xq'])), ev['fq'], ev['qq'], ev['int_f2'])


for kind in ('L', 'B', 'T1', 'T3'):
    for d in (2, 4, 8):
        truth = NonSmooth(kind, d) if kind in ('L', 'B') else Truth(kind, d); ev = None
        for n in (1000, 10000):
            path = os.path.join(OUT, 'syn', f'{kind}_d{d}_n{n}.json')
            if os.path.exists(path):
                continue
            ev = ev or eval_sets(truth, kind, d); reps = []
            for rep in range(4):
                X = truth.sample(np.random.default_rng([11, d, n, rep]), n); row = {}
                t0 = time.time(); m = blml_auto(X); row['blml'] = score(m.predict, ev); row['blml']['fit_s'] = time.time() - t0
                t0 = time.time(); g = GaussianizedDensity(blml_auto).fit(X); row['blml_gauss'] = score(g.predict, ev)
                row['blml_gauss']['fit_s'] = time.time() - t0
                reps.append(row)
                log(f"{kind} d={d} n={n} rep {rep}: blml {row['blml']['rel_ise']:.4g}  transformed {row['blml_gauss']['rel_ise']:.4g}")
            json.dump(dict(truth=kind, d=d, n=n, reps=reps), open(path, 'w'), indent=1, default=float)
log('done')
