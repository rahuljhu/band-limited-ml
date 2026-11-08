"""Paper 3 on Colab: the independence test (p3_independence.py) on the heralded, multimode and electron records.
Run: BLML_DATA_SRC=~/data_repo/dslit_gpu ./colab_job.sh start p3indep p3_indep_job.py T4"""
import os, sys, subprocess
CODE = os.environ.get('BLML_CODE_DIR', '.')
if subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda13x', 'numba'], check=True)
subprocess.run([sys.executable, os.path.join(CODE, 'p3_independence.py'), 'photon', 'electron', 'photon_mm'],
               env=dict(os.environ, BLML_ENGINE='gpu', PYTHONUNBUFFERED='1'), check=False)
print('done', flush=True)
