"""Paper 3 on Colab (GPU): the ghost-diffraction record through the 1-d double-slit protocol of
results/dslit_1d_runs.sh, then the build-up study (results/p3_buildup.py) on every record.
Run: BLML_DATA_SRC=~/data_repo/dslit_gpu ./colab_job.sh start p3 p3_colab_job.py T4"""
import os, sys, subprocess, shutil
CODE = os.environ.get('BLML_CODE_DIR', '.'); OUT = os.environ.get('BLML_OUT_DIR', 'results')
if subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda13x', 'numba'], check=True)
base = dict(os.environ, BLML_ENGINE='gpu', BLML_COLLAPSE='1', PYTHONUNBUFFERED='1')
job = os.path.join(CODE, 'dslit_job.py')
F = os.path.join(OUT, 'dslit_1d', 'ghost', 'flat'); os.makedirs(F, exist_ok=True)
subprocess.run([sys.executable, job], env=dict(base, BLML_SETS='ghost', BLML_OUT_DIR=F, BLML_BG_FLAT='1',
                                                BLML_PARTS='fraunhofer,fresnel,kde,akde,hist'), check=False)
for N in (1, 0, 2):
    D = os.path.join(OUT, 'dslit_1d', 'ghost', f'bg{N}'); os.makedirs(D, exist_ok=True)
    if os.path.exists(os.path.join(F, 'ghost_fraunhofer.json')):
        shutil.copy(os.path.join(F, 'ghost_fraunhofer.json'), D)
    subprocess.run([sys.executable, job], env=dict(base, BLML_SETS='ghost', BLML_OUT_DIR=D, BLML_BG_HARM=f'{N},{N}',
                                                    BLML_PARTS='complex_bg' if N else 'complex_bg'), check=False)
subprocess.run([sys.executable, os.path.join(CODE, 'p3_buildup.py'), 'photon', 'ghost', 'electron', 'photon_mm'],
               env=dict(base, BLML_BG_HARM='1,1', P3_OUT=OUT), check=False)
print('done', flush=True)
