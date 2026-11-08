"""Streaming speed check on the GPU with the paper's settings (results/bench_stream_gpu_log.py: burn-in search 'shared',
rebin_tol 1 with 2 sweeps, float32, batches of 4000 after a burn-in of 1e4) at n = 2e4, 4e4, 1e5, streaming against
one-shot selection. Paper (T4): 1.3x, 2.5x, 4.2x. Output: RESULT lines in the job log."""
import os, sys, subprocess
CODE = os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__)))
if subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda13x', 'numba'], check=True)
os.makedirs(os.path.join(CODE, 'results'), exist_ok=True)
bench = os.path.join(CODE, 'bench_stream_gpu_log.py')
env = dict(os.environ, BURN_SEARCH='shared', BLML_REBIN_TOL='1', BLML_REBIN_SWEEPS='2')
subprocess.run([sys.executable, bench, '20000', '40000', '100000'], cwd=CODE, env=env, check=False)
print('done', flush=True)
