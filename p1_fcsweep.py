"""Paper 1: sensitivity to the cutoff -- error vs fc/fc_true on T1.
Run: ./colab_job.sh start p1fc p1_fcsweep.py   (one JSON per cell; resumable)."""
import os, sys, json, time, subprocess

OUT = os.environ.get('BLML_OUT_DIR', 'outputs_p1fc')
CODE = os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT, exist_ok=True)
if subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda13x', 'numba'], check=True)
sys.path.insert(0, CODE)
import numpy as np
import cupy as cp
from blml_trivial import BLMLQuick
from blml_trivial import make_modes
from p1_truth import Truth, DefensiveProposal, FC_TRUE
import p1_estimators as E

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def path(name):
    return os.path.join(OUT, name + '.json')

MULTS = [float(v) for v in os.environ.get('BLML_FC_MULTS', '0.5,0.7,1,1.4,2,2.8').split(',')]
SUB = os.environ.get('BLML_FC_SUB', 'fcsweep')
M_F, M_Q = 100000, 200000
CELLS_ALL = [('T1', 1, 10000, 4), ('T1', 2, 10000, 4), ('T1', 4, 10000, 4), ('T1', 8, 10000, 4), ('T1', 4, 100000, 3), ('T1', 8, 100000, 3),
         ('T3', 2, 10000, 4), ('T3', 4, 10000, 4), ('T3', 8, 10000, 4), ('T3', 4, 100000, 3), ('T3', 8, 100000, 3)]
CELLS_ADD = [('T1', 1, 100000, 3), ('T1', 2, 100000, 3), ('T3', 1, 10000, 4), ('T3', 1, 100000, 3), ('T3', 2, 100000, 3)]
if SUB == 'fcsweep_add':          # cells added for the d = 1, 2 panels of the sweep figure
    CELLS_ALL = CELLS_ADD
CELLS = [c for c in CELLS_ALL if c[0] in os.environ.get('BLML_FC_TRUTHS', 'T1,T3').split(',')]
for kind, d, n, reps in CELLS:
    nm = f'{SUB}/{kind}_d{d}_n{n}'
    if os.path.exists(path(nm)):
        continue
    truth = Truth(kind, d); q = DefensiveProposal(truth, scale=2.0 if kind == 'T3' else 6.0)
    rng = np.random.default_rng(10_000 + d)
    Xf = truth.sample(rng, M_F); ff = truth.pdf(Xf)
    Xq = q.sample(rng, M_Q); fq, qq = truth.pdf(Xq), q.pdf(Xq)
    int_f2 = float(ff.mean())
    rows = []
    for rep in range(reps):
        X = truth.sample(np.random.default_rng([7, d, n, rep]), n)
        row = {}
        for mult in MULTS:
            t0 = time.time()
            m = BLMLQuick(engine='gpu', dtype=np.float64).fit(X, make_modes(mult * truth.fc_true, ['sinc'] * d, None))
            r = E.metrics(m.predict(Xf), ff, m.predict(Xq), fq, qq, int_f2)
            r.update(fit_s=time.time() - t0, B=m.n_bins_, sweeps=m.n_sweeps_, converged=m.converged_)
            row[f'{mult:g}'] = r
            del m
            cp.get_default_memory_pool().free_all_blocks()
        rows.append(row)
        log(f"fcsweep {kind} d={d} n={n} rep={rep}: relISE by mult " +
            ' '.join(f"{k}:{v['rel_ise']:.1e}" for k, v in row.items()))
    os.makedirs(os.path.dirname(path(nm)), exist_ok=True)
    with open(path(nm) + '.tmp', 'w') as f:
        json.dump(dict(truth=kind, d=d, n=n, mults=MULTS, reps=rows), f, default=float)
    os.replace(path(nm) + '.tmp', path(nm))
    log(f'saved {nm}')
log('ALL DONE')
