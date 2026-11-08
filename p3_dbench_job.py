"""Paper 3, synthetic diffraction benchmark on Colab: every (case, background, n, rep) of p3_dbench.py as one subprocess,
one per CPU core, smallest n first; a progress line per finished data set (job.log).
  Locally: DB_WORKERS=6 BLML_OUT_DIR=results/dbench nice .venv/bin/python p3_dbench_job.py"""
import os, sys, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
CASES = os.environ.get('DB_CASES', 'sym,asym,defocus,partial').split(','); BGS = ['0', '0.1']
NS = [int(v) for v in os.environ.get('DB_NS', '1000,10000,100000').split(',')]; REPS = int(os.environ.get('DB_REPS', '3'))
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', BLML_ENGINE='cpu',
           BLML_SOLVER='lbfgs', BLML_DTYPE='float64')
tasks = [(c, b, str(n), str(r)) for n in sorted(NS, reverse=os.environ.get('DB_FLAT_ONLY') == '1') for r in range(REPS) for c in CASES for b in BGS]
t0 = time.time(); done = 0; WORKERS = int(os.environ.get('DB_WORKERS', os.cpu_count()))
print(f'{len(tasks)} data sets on {WORKERS} workers', flush=True)


def run(t):
    a = time.time(); r = subprocess.run([sys.executable, 'p3_dbench.py', *t], env=env, capture_output=True, text=True)
    return t, r, time.time() - a


with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for fut in as_completed([ex.submit(run, t) for t in tasks]):
        t, r, s = fut.result(); done += 1
        msg = '' if r.returncode == 0 else ' ERR ' + r.stderr.strip().splitlines()[-1][:300]
        print(f"[{done}/{len(tasks)} {(time.time() - t0) / 60:.1f} min] {' '.join(t)} rc={r.returncode} {s:.0f}s{msg}", flush=True)
print('all done', flush=True)
