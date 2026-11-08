"""Paper 2, second application: GEFCom2014 price track (Hong et al. 2016).
Day-ahead probabilistic forecasts of 24 hourly zonal prices for 15 tasks, each trained on all
data before the target day; scored by pinball loss over the 99 percentiles, the competition's
metric, so results compare directly with the leaderboard. Same models and validation protocol
as CAISO (validation = random 20% of 28-day training blocks).
Run: ./colab_job.sh start p2gef p2_gefcom.py  (data: gefcom_*.npy, see gefcom_data.cache)."""
import os, sys, subprocess
if __name__ == '__main__':
    def pip(*pkgs):
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *pkgs], check=False)
    if subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
        pip('cupy-cuda13x', 'numba')
    for mod, pkg in [('ngboost', 'ngboost'), ('catboost', 'catboost'), ('quantile_forest', 'quantile-forest'),
                     ('lightgbm', 'lightgbm'), ('tabpfn', 'tabpfn')]:
        if subprocess.run([sys.executable, '-c', f'import {mod}'], capture_output=True).returncode:
            pip(pkg)
sys.path.insert(0, os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import p2_experiments as E
import p2_models as M
import gefcom_data

E.TAUS = np.round(np.arange(0.01, 0.995, 0.01), 2)          # 99 percentiles, as scored in GEFCom2014
E.YGRID = np.arange(0.0, 700.5, 0.5)
E.COVSETS = {'hz': ['hour', 'zload'], 'hzl': ['hour', 'zload', 'lag24'],
             'hzll': ['hour', 'zload', 'lag24', 'lag168']}
M.PERIODS.update(gefcom_data.PERIODS)

def main():
    cols, ts, tasks = gefcom_data.load(os.environ.get('BLML_DATA_DIR') or gefcom_data.RAW)
    first_tr = tasks[0][1]
    for c in ('zload', 'tload', 'lag24', 'lag168'):
        M.SCALES[c] = (float(np.median(cols[c][first_tr])), float(np.std(cols[c][first_tr])))
    models = [m for m in E.make_models(None) if m[0] in
              ('BLML-auto', 'CondKDE', 'CondKNN', 'GMM', 'Climatology',
               'LightGBM-quantile', 'QuantileForest', 'CatBoost-Normal', 'MDN', 'NGBoost-Normal', 'TabPFN')]
    only = os.environ.get('BLML_P2_MODELS')
    if only:
        models = [m for m in models if m[0] in only.split(',')]
    for k, tr, te in tasks:
        for covset in E.COVSETS:
            for name, factory in models:
                E.run_one(f'gef{k:02d}', covset, name, factory, cols, tr, te, ts)
    E.log('ALL DONE')

if __name__ == '__main__':
    main()
