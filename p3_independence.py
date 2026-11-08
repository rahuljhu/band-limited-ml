"""Paper 3: does detecting one particle change where the next one lands? For each double-slit record, take the
positions along the fringes in arrival order (camera frame, then the order within the frame) and form pairs of
detections L camera frames apart (L = 1, 2, 3, 5; within a frame the stored order is the readout order, so a
frame is the time resolution), and pairs within one frame as a separate class, where detector artefacts such as split
clusters would show. Under independence the joint
density of a pair is the product of the marginals. Statistic: the held-out log-likelihood gain per pair of a 2-d
BLML fit of the pairs (product sinc kernel, each axis at the band selected for the 1-d record) over the product of
the 1-d fit with itself, fitted on a random half of the pairs and scored on the other half: an estimate of the
mutual information between successive detections (nats). Null: the same statistic after permuting the arrival
order (B permutations). Run: BLML_DATA_DIR=~/data_repo/dslit_gpu python p3_independence.py <set> ...
-> $P3_OUT/p3_indep/<set>.json"""
import os, sys, json, time, numpy as np
sys.path.insert(0, os.environ.get('BLML_CODE_DIR', '.'))
import blml_trivial as bt
from blml_binned import BLMLBinned, _np
DATA = os.environ.get('BLML_DATA_DIR', '.'); ENGINE = os.environ.get('BLML_ENGINE', 'cpu')
OUT = os.path.join(os.environ.get('P3_OUT', os.environ.get('BLML_OUT_DIR', 'results')), 'p3_indep'); os.makedirs(OUT, exist_ok=True)
B = int(os.environ.get('P3_PERM', '100')); LAGS = (1, 2, 3, 5)
FC = {'photon': 0.022627, 'electron': 0.016, 'photon_mm': 0.053817, 'ghost': None}     # 1-d selections (bg N = 1); ghost: from its run


def positions(name):
    pts = np.load(os.path.join(DATA, f'dslit_{name}_points.npy')); axis = int(np.load(os.path.join(DATA, f'dslit_{name}_axis.npy')))
    b = np.load(os.path.join(DATA, f'dslit_{name}_b.npy')); W = b.shape[-2:][axis]
    o = np.argsort(pts[:, 0], kind='stable'); x = np.clip(np.floor(pts[o, 1 + axis] + 0.5).astype(int), 0, W - 1)
    return x, pts[o, 0], W


def frame_pairs(x, frame, lag, same_frame, rng):
    """Within a frame the stored order is the camera's readout (raster) order, not the arrival order, so the time
    resolution is the frame: each frame's photons are shuffled, and a pair is the first photon of frame t with the
    first of frame t + lag (lag in frames), or, same_frame, the first two photons of one frame."""
    o = np.lexsort((rng.random(len(x)), frame)); x, frame = x[o], frame[o]
    first = np.r_[True, frame[1:] != frame[:-1]]; fi = np.nonzero(first)[0]; fr = frame[fi]
    if same_frame:
        two = fi[(fi + 1 < len(x))]; two = two[frame[np.minimum(two + 1, len(x) - 1)] == frame[two]]
        return x[two], x[two + 1]
    pos = {f: i for f, i in zip(fr, fi)}
    a = [pos[f] for f in fr if f + lag in pos]; c = [pos[f + lag] for f in fr if f + lag in pos]
    return x[a], x[c]


def gain(x, frame, W, fc, lag, same_frame, rng):
    a, c = frame_pairs(x, frame, lag, same_frame, rng)
    if len(a) < 200:
        return None, int(len(a))
    half = rng.random(len(a)) < 0.5
    J = np.zeros((W, W)); np.add.at(J, (a[half], c[half]), 1)
    M1 = np.zeros(W); np.add.at(M1, np.r_[a[half], c[half]], 1)
    mj = BLMLBinned(engine=ENGINE, tol=1e-5, max_sweeps=5000).fit(J, bt.make_modes([fc, fc], ['sinc', 'sinc']), complex_g=True)
    m1 = BLMLBinned(engine=ENGINE, tol=1e-5, max_sweeps=5000).fit(M1, bt.make_modes([fc], ['sinc']), complex_g=True)
    fj = _np(mj.density_); fj = fj / fj.sum(); f1 = _np(m1.density_); f1 = f1 / f1.sum()
    at, ct = a[~half], c[~half]
    g = np.log(np.maximum(fj[at, ct], 1e-300)) - np.log(np.maximum(f1[at], 1e-300)) - np.log(np.maximum(f1[ct], 1e-300))
    return float(g.mean()), int(len(a))


def run(name, fc):
    x, frame, W = positions(name); rng = np.random.default_rng(0); out = dict(set=name, fc=fc, n=int(len(x)), W=int(W), B=B, rows=[])
    for same in (False, True):
        for lag in (LAGS if not same else (1,)):
            t0 = time.time(); g, npair = gain(x, frame, W, fc, lag, same, np.random.default_rng(1))
            if g is None:
                continue
            null = []
            for b in range(B):                       # permute the positions over the record, the frame sequence kept
                p = rng.permutation(len(x)); gb, _ = gain(x[p], frame, W, fc, lag, same, np.random.default_rng(1))   # positions shuffled, frames kept
                null.append(gb)
            null = np.array([v for v in null if v is not None])
            row = dict(lag=lag, same_frame=same, pairs=npair, gain=g, null_mean=float(null.mean()), null_sd=float(null.std()),
                       null_q95=float(np.quantile(null, 0.95)), p=float((1 + (null >= g).sum()) / (1 + len(null))), s=time.time() - t0)
            out['rows'].append(row); print(name, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            json.dump(out, open(f'{OUT}/{name}.json', 'w'), indent=1)


for name in sys.argv[1:]:
    fc = FC.get(name)
    if fc is None:                                   # take the band from the record's own 1-d fit
        fc = json.load(open(os.path.join(os.environ.get('P3_OUT', os.environ.get('BLML_OUT_DIR', 'results')), 'dslit_1d', name, 'bg1', f'{name}_complex_bg.json')))['fc'][0]
    run(name, float(fc))
