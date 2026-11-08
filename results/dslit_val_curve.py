"""Paper 3, double slit: the validation log-likelihood against fc along the fringes on a fine grid (factor 2^(1/8)),
collapsed (1-d) sets, for complex / positive with the EM background and complex with the background mode (N = 1).
The selector's bracket doubles, so on a plateau it lands anywhere on it; the grid shows where the plateau begins.
Usage: BLML_COLLAPSE=1 python results/dslit_val_curve.py <set> <out.json>"""
import os, sys, json, numpy as np
os.environ.setdefault('BLML_COLLAPSE', '1'); os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ['BLML_PARTS'] = ''
os.environ.setdefault('BLML_DATA_DIR', os.path.expanduser('~/data_repo/dslit_gpu'))
sys.argv, (name, out) = sys.argv[:1], sys.argv[1:3]; sys.path.insert(0, ".")
import dslit_job as J
D = J.DATA; pts = np.load(f'{D}/dslit_{name}_points.npy'); b = np.load(f'{D}/dslit_{name}_b.npy'); axis = int(np.load(f'{D}/dslit_{name}_axis.npy'))
J.BFIX = np.atleast_1d(np.load(f'{D}/dslit_{name}_beta.npy')).astype(float)
b = b.reshape((-1,) + b.shape[-2:]); SH = b.shape[1:]
ij = np.clip(np.floor(pts[:, 1 + axis] + 0.5).astype(int), 0, SH[axis] - 1)
b = b.sum(2 - axis); b = b / b.sum(1, keepdims=True)
u = np.random.default_rng(0).random(len(pts)); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))
TR, VA = [np.bincount(ij[part == k], minlength=b.shape[1]).astype(float) for k in (0, 1)]
grid = 0.008 * 2.0 ** (np.arange(0, 41) / 8.0); grid = grid[grid <= 0.5]
rows = {}
for lab in ('complex', 'positive', 'complex_bg'):
    v = []
    for fc in grid:
        if lab == 'complex_bg':
            P, m, beta = J.bg_fit(TR, [fc], (1,), True, max_sweeps=5000, tol=1e-6)
        else:
            P = J.em(TR, b, J.blml_q([fc], lab, max_sweeps=3000), tol=1e-8)[0]
        v.append(J.ll(P, VA))
    rows[lab] = v; k = int(np.argmax(v))
    print(f'{lab:11s} best fc {grid[k]:.4f}  val ' + ' '.join(f'{f:.4f}:{x - max(v):+.4f}' for f, x in zip(grid, v) if f <= 0.2), flush=True)
json.dump(dict(set=name, n_val=float(VA.sum()), fc=grid.tolist(), val=rows), open(out, 'w'), indent=1)
