"""Paper 2 batch on one VM (one startup): day split + monthly forecasting (hl, hll; five models),
then the pair densities, then the synthetic selector study.  ./colab_job.sh start p2batch p2_batch.py"""
import os, sys, subprocess, time
CODE = os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__)))
for mod, pkg in [('cupy', 'cupy-cuda13x'), ('numba', 'numba'), ('xgboostlss', 'xgboostlss'), ('tabpfn', 'tabpfn')]:
    if subprocess.run([sys.executable, '-c', f'import {mod}'], capture_output=True).returncode:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=False)
sys.path.insert(0, CODE)
def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] BATCH {m}", flush=True)
os.environ.update(BLML_P2_RUNS='daysplit,rollfwd', BLML_P2_COVSETS='hl,hll', BLML_P2_MODELS='')
log('day split + monthly forecasting')
import p2_experiments
p2_experiments.main()
log('pair densities')
import p2_pairs
p2_pairs.main()
log('synthetic selector study')
subprocess.run([sys.executable, os.path.join(CODE, 'p2_selector_synth.py')], check=False)
log('ALL DONE')
