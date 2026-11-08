"""Paper 3: the pattern building up, particle by particle. For each double-slit record, the training + validation
particles of dslit_job.py's split are taken in arrival order; on the first n of them (n = 30, 100, 300, 1000, 3000,
all) complex BLML with the background mode (one harmonic) and the flat-background KDE are fitted by the 1-d protocol
of dslit_job.py (collapsed across the fringes; hyperparameters on a random fifth of the n, refit on all n) and scored
on the same held-out test particles as the main table. Also the selected band relative to the physical band and the
field along the fringes (for |FT g|).
Run from the repo root: BLML_DATA_DIR=~/data_repo/dslit_gpu BLML_ENGINE=cpu BLML_COLLAPSE=1 BLML_BG_HARM=1,1
  .venv/bin/python p3_buildup.py <set> [<set> ...]   ->  results/p3_buildup/<set>.json, <set>_lines.npz"""
import os, sys, json, time, numpy as np
os.environ.setdefault('BLML_COLLAPSE', '1'); os.environ.setdefault('BLML_BG_HARM', '1,1'); os.environ.setdefault('BLML_ENGINE', 'cpu')
sys.path.insert(0, os.environ.get('BLML_CODE_DIR', '.'))
import dslit_job as dj
OUT = os.path.join(os.environ.get('P3_OUT', 'results'), 'p3_buildup'); os.makedirs(OUT, exist_ok=True)
SIZES = [30, 100, 300, 1000, 3000]
PHYS = {'photon': 0.021358537110599967, 'electron': 0.013405756468667592, 'photon_mm': 0.020399879849644066}


def setup(name):
    D = dj.DATA
    pts = np.load(os.path.join(D, f'dslit_{name}_points.npy')); b = np.load(os.path.join(D, f'dslit_{name}_b.npy'))
    axis = int(np.load(os.path.join(D, f'dslit_{name}_axis.npy'))); b = b.reshape((-1,) + b.shape[-2:]); SH = b.shape[1:]
    ij = [np.clip(np.floor(pts[:, k] + 0.5).astype(int), 0, SH[k - 1] - 1) for k in (1, 2)]
    bflat = np.ones((1,) + SH) / np.prod(SH)                       # the references' background: flat, weight fitted
    bflat = bflat.sum(2 - axis); bflat = bflat / bflat.sum(1, keepdims=True); x = ij[axis]; W = bflat.shape[1]
    u = np.random.default_rng(0).random(len(pts)); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))
    order = np.argsort(pts[:, 0], kind='stable')                  # arrival order
    trv = order[part[order] < 2]; te = np.nonzero(part == 2)[0]
    return x, W, bflat, trv, te


def counts(x, idx, W):
    C = np.zeros(W); np.add.at(C, x[idx], 1); return C


def run(name):
    x, W, bflat, trv, te = setup(name); TE = counts(x, te, W); res, lines = [], {}
    phys = PHYS.get(name)
    for n in [s for s in SIZES if s < len(trv)] + [len(trv)]:
        t0 = time.time(); idx = trv[:n]; r = np.random.default_rng(1).random(n) < 0.8
        TR, VA, TV = counts(x, idx[r], W), counts(x, idx[~r], W), counts(x, idx, W)
        fc, nbg, vll, _, _ = dj.bg_search(TR, VA, True, 1, tag=f'{name}:{n}')
        P, m, beta = dj.bg_fit(TV, fc, nbg, True, 1, max_sweeps=5000, tol=1e-5)
        dj.BFIX = np.array([np.nan])
        HG = (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48)
        val = {h: dj.ll(dj.em(TR, bflat, dj.kde_q((h,)))[0], VA) for h in HG}; h = max(val, key=val.get)
        Pk = dj.em(TV, bflat, dj.kde_q((h,)))[0]
        row = dict(n=int(n), blml=dj.ll(P, TE), kde=dj.ll(Pk, TE), hist=dj.ll(TV + 0.5, TE), fc=float(fc[0]),
                   fc_ratio=None if phys is None else float(fc[0] / phys), beta=float(beta), kde_h=float(h), s=time.time() - t0)
        res.append(row); lines[f'n{n}'] = dj.line_of(m, 0, 0.0); lines[f'P{n}'] = P; lines[f'C{n}'] = TV
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        json.dump(dict(set=name, n_test=int(TE.sum()), rows=res), open(f'{OUT}/{name}.json', 'w'), indent=1)
    np.savez_compressed(f'{OUT}/{name}_lines.npz', TE=TE, **lines)


for name in sys.argv[1:]:
    dj.BFIX = np.array([np.nan]); run(name)
