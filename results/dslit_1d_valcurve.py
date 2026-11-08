"""Paper 3, double slit, 1-d protocol: the validation curve of complex BLML with the background mode (N = 0, 1, 2)
over dslit_job.FC_GRID, as the selection in dslit_job.bg_search computes it (training fit, validation LL per
particle), cached to results/dslit_1d/<set>/val_curve.json. Usage: python results/dslit_1d_valcurve.py [set ...]"""
import os, sys, json, numpy as np
os.environ.update(BLML_COLLAPSE='1', BLML_ENGINE='cpu', BLML_PARTS='')
os.environ.setdefault('BLML_DATA_DIR', os.path.expanduser('~/data_repo/dslit_gpu'))
SETS = sys.argv[1:] or ['photon', 'electron', 'photon_mm']; sys.argv = sys.argv[:1]; sys.path.insert(0, '.')
import dslit_job as J
for name in SETS:
    D = J.DATA; pts = np.load(f'{D}/dslit_{name}_points.npy'); SH = np.load(f'{D}/dslit_{name}_b.npy').shape[-2:]
    axis = int(np.load(f'{D}/dslit_{name}_axis.npy'))
    x = np.clip(np.floor(pts[:, 1 + axis] + 0.5).astype(int), 0, SH[axis] - 1)
    u = np.random.default_rng(0).random(len(pts)); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))
    TR, VA = [np.bincount(x[part == k], minlength=SH[axis]).astype(float) for k in (0, 1)]
    G = J.FC_GRID[J.FC_GRID >= 2.0 / len(TR)]; out = dict(set=name, n_val=float(VA.sum()), fc=G.tolist(), val={})
    for N in (0, 1, 2):
        out['val'][str(N)] = [J.ll(J.bg_fit(TR, [f], (N,), True, max_sweeps=5000, tol=1e-6)[0], VA) for f in G]
        k = int(np.argmax(out['val'][str(N)])); print(f'{name} N={N}: argmax fc {G[k]:.4f}', flush=True)
    json.dump(out, open(f'results/dslit_1d/{name}/val_curve.json', 'w'), indent=1)
