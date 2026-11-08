"""Paper 3 on Colab (CPU cores in parallel, one fit per core): the synthetic replicas of the four double-slit records
(p3_synth_replica.py, reps 0..P3_REPS-1) and the real records by the same protocol (dslit_job.py: KDE, adaptive KDE,
complex BLML with the EM flat background, with the background mode N = 0, 1, 2, streaming N = 0, 1, and the two-slit
kernel with N = 1), every fit by BLML_SOLVER (default lbfgs), P3_WORKERS processes (default all cores).
Locally: P3_WORKERS=5 BLML_DATA_DIR=~/data_repo/dslit_gpu BLML_OUT_DIR=results/p3synth_lbfgs nice .venv/bin/python p3_synth_job.py Every
task is one subprocess writing its own file under BLML_OUT_DIR; finished ones are skipped.
  BLML_DATA_SRC=~/data_repo/dslit_gpu ./colab_job.sh start p3synth p3_synth_job.py A100"""
import os, sys, json, subprocess, time
from concurrent.futures import ThreadPoolExecutor
OUT = os.environ.get('BLML_OUT_DIR', 'results/p3synth'); os.makedirs(OUT, exist_ok=True)
SETS = ['photon', 'ghost', 'electron', 'photon_mm']; REPS = int(os.environ.get('P3_REPS', '5'))
FILLS = os.environ.get('P3_FILLS', '1.0,0.7,0.5,0.4,0.33,0.25')
base = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', BLML_ENGINE='cpu',
            BLML_COLLAPSE='1', BLML_SELECT='cv', BLML_DTYPE='float64', BLML_SOLVER=os.environ.get('BLML_SOLVER', 'lbfgs'))
WORKERS = int(os.environ.get('P3_WORKERS', os.cpu_count()))


def real(s, sub, **env):
    d = os.path.join(OUT, sub, s); os.makedirs(d, exist_ok=True)
    stub = os.path.join(d, f'{s}_fraunhofer.json')                # the 1-d fits read only t0 from the reference
    if not os.path.exists(stub):
        json.dump(dict(params=dict(t0=0.0)), open(stub, 'w'))
    return [sys.executable, 'dslit_job.py'], dict(base, BLML_SETS=s, BLML_OUT_DIR=d, **env)


tasks = [([sys.executable, 'p3_synth_replica.py', s, str(r)], dict(base)) for r in range(REPS) for s in SETS]
tasks += [real(s, 'flat', BLML_BG_FLAT='1', BLML_PARTS='kde,akde,complex') for s in SETS]       # the real records
tasks += [real(s, f'bg{N}', BLML_BG_HARM=f'{N},{N}', BLML_PARTS='complex_bg') for s in SETS for N in (0, 1, 2)]
tasks += [real(s, f'stream{N}', BLML_BG_HARM=f'{N},{N}', BLML_PARTS='stream_bg') for s in SETS for N in (0, 1)]
tasks += [real(s, 'slits1', BLML_BG_HARM='1,1', BLML_FILLS=FILLS, BLML_PARTS='complex_bg') for s in SETS]
log = sys.stdout                                                   # the launcher appends it to outputs/job.log


def run(t):
    cmd, env = t; t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or [''])[-1][:300]
    print(f"[{time.time() - t0:.0f}s rc={r.returncode}] {' '.join(cmd[1:])} {env.get('BLML_OUT_DIR', '')}: {tail}"
          + ('' if r.returncode == 0 else ' ERR ' + r.stderr[-500:]), file=log, flush=True)


with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    list(ex.map(run, tasks))
print('all done', file=log, flush=True)
