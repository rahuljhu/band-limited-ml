"""Are single-particle arrivals exchangeable draws from one |psi|^2?  Double-slit lines (paper 3's records, counts along
the fringes), complex BLML with the background mode N = 1 at the paper's selected fc (results/dslit_1d/<set>/bg1).

(A) Order test. Frames are cut into NB blocks in recording order. Prequential score: each block's arrivals under the
    fit to the blocks before it, two predictors -- all past blocks ('growing'; drift makes the real order score
    LOWER than shuffled orders) and the last WIN blocks ('recent'; drift makes it HIGHER). The real order is compared
    with NPERM random orders of the frames (same arrivals, same model): two-sided permutation p-values.
(B) Pair test. Pairs (x_t, x_{t+1}) of the first arrivals of consecutive frames, and (photons) pairs of arrivals within
    one frame. Statistic: held-out log-likelihood of a joint 2-d complex BLML (sinc x sinc at the line's fc, background
    mode N = 1 x 1) minus that of the product of the 1-d marginal fits, on a 50/50 split of the pairs; null: the same
    statistic with the second member of each pair shuffled (NPERM times). The excess over the null is a held-out
    estimate of the information in one arrival about the other.
Positive controls on synthetic arrivals drawn from the fitted line density with the same frame structure: (A) the
fringe pattern shifted linearly by DRIFT periods over the record; (B) with probability COPY the next arrival repeats
the previous one plus N(0, 2 px).
Usage: python results/exch_test.py <set> [real|drift|copy]  ->  results/exch/<set>_<mode>.json"""
import os, sys, json, time, numpy as np
sys.path.insert(0, '.')
os.environ.setdefault('BLML_COLLAPSE', '1'); os.environ.setdefault('BLML_ENGINE', 'cpu')
import dslit_job as dj
name = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else 'real'
NB, WIN, NPERM = int(os.environ.get('EX_NB', '20')), int(os.environ.get('EX_WIN', '5')), int(os.environ.get('EX_NPERM', '60'))
DRIFT, COPY = 0.25, 0.05
D = os.path.expanduser('~/data_repo/dslit_gpu/'); OUT = 'results/exch'; os.makedirs(OUT, exist_ok=True)
pts = np.load(D + f'dslit_{name}_points.npy'); SH = np.load(D + f'dslit_{name}_b.npy').shape[-2:]
axis = int(np.load(D + f'dslit_{name}_axis.npy')); L = SH[axis]
frame = pts[:, 0].astype(np.int64); x = np.clip(np.floor(pts[:, 1 + axis] + 0.5).astype(int), 0, L - 1)
FC = json.load(open(f'results/dslit_1d/{name}/bg1/{name}_complex_bg.json'))['fc']
rng = np.random.default_rng(7)


def fit_line(counts):
    P, m, beta = dj.bg_fit(counts.astype(float), FC, (1,), True, max_sweeps=3000, tol=1e-6); return P


def hist(xs):
    return np.bincount(xs, minlength=L).astype(float)


if mode != 'real':                                        # synthetic arrivals with the real frame structure
    P0 = fit_line(hist(x)); t = (frame - frame.min()) / max(1, frame.max() - frame.min())
    fr = json.load(open(f'results/dslit_1d/{name}/flat/{name}_fraunhofer.json'))['params']; per = fr['P']
    xs = np.empty_like(x)
    for i in range(len(x)):
        if mode == 'drift':                               # the whole pattern shifted by DRIFT periods over the record
            sh = int(round(DRIFT * per * t[i])); p = np.roll(P0, sh)
            xs[i] = rng.choice(L, p=p / p.sum())
        else:
            xs[i] = rng.choice(L, p=P0)
    if mode == 'copy':                                    # the first arrival of a frame repeats the previous frame's
        first = np.r_[True, frame[1:] != frame[:-1]]; idx = np.flatnonzero(first)
        for a, b in zip(idx[:-1], idx[1:]):
            if rng.random() < COPY:
                xs[b] = int(np.clip(round(xs[a] + rng.normal(0, 2)), 0, L - 1))
    x = xs

# ---------------------------------------------------------------- (A) order test
uf = np.unique(frame)


def prequential(order):
    """order: a permutation of the unique frames. Returns (growing, recent) mean log-prob per arrival over blocks 2..NB."""
    pos = np.empty(uf.max() + 1, np.int64); pos[order] = np.arange(len(order))
    blk = (pos[frame] * NB) // len(order)
    H = np.array([hist(x[blk == k]) for k in range(NB)])
    sg = sr = 0.0; n = 0
    for k in range(1, NB):
        Pg = fit_line(H[:k].sum(0)); Pr = Pg if k <= WIN else fit_line(H[k - WIN:k].sum(0))
        sg += float(H[k] @ np.log(Pg)); sr += float(H[k] @ np.log(Pr)); n += H[k].sum()
    return sg / n, sr / n


t0 = time.time(); real = prequential(uf)
perm = np.array([prequential(rng.permutation(uf)) for _ in range(NPERM)])
pval = lambda r, z: float((1 + np.sum(np.abs(z - z.mean()) >= abs(r - z.mean()))) / (1 + len(z)))
A = dict(real_growing=real[0], real_recent=real[1], perm_mean=perm.mean(0).tolist(), perm_sd=perm.std(0).tolist(),
         z_growing=float((real[0] - perm[:, 0].mean()) / perm[:, 0].std()), z_recent=float((real[1] - perm[:, 1].mean()) / perm[:, 1].std()),
         p_growing=pval(real[0], perm[:, 0]), p_recent=pval(real[1], perm[:, 1]), s=time.time() - t0)

# ---------------------------------------------------------------- (B) pair test
def pair_stat(a, b, idx_tr, idx_te):
    """held-out LL per pair: joint 2-d fit minus the product of the marginal fits (both fitted on idx_tr)."""
    J = np.zeros((L, L)); np.add.at(J, (a[idx_tr], b[idx_tr]), 1)
    Pj, _, _ = dj.bg_fit(J, [FC[0], FC[0]], (1, 1), True, max_sweeps=3000, tol=1e-5)
    Pa, Pb = fit_line(hist(a[idx_tr])), fit_line(hist(b[idx_tr]))
    return float(np.mean(np.log(Pj[a[idx_te], b[idx_te]]) - np.log(Pa[a[idx_te]]) - np.log(Pb[b[idx_te]])))


def pairs_between():
    first = np.r_[True, frame[1:] != frame[:-1]]; i = np.flatnonzero(first)
    return x[i[:-1]], x[i[1:]]


def pairs_within():
    a, b = [], []
    for f in uf:
        j = np.flatnonzero(frame == f)
        if len(j) >= 2:
            k = rng.choice(j, 2, replace=False); a.append(x[k[0]]); b.append(x[k[1]])
    return np.array(a), np.array(b)


B = {}
for lab, (a, b) in (('between_frames', pairs_between()), ('within_frame', pairs_within())):
    if len(a) < 200:
        continue
    t0 = time.time(); sp = rng.permutation(len(a)); tr, te = sp[: len(a) // 2], sp[len(a) // 2:]
    r = pair_stat(a, b, tr, te); NP = max(12, NPERM // 5)
    null = np.array([pair_stat(a, b[rng.permutation(len(b))], tr, te) for _ in range(NP)])
    B[lab] = dict(n_pairs=int(len(a)), stat=r, null_mean=float(null.mean()), null_sd=float(null.std()),
                  excess=float(r - null.mean()), z=float((r - null.mean()) / max(null.std(), 1e-12)), p=float((1 + np.sum(null >= r)) / (1 + NP)),
                  bound95=float(r - null.mean() + 1.645 * null.std()), s=time.time() - t0)

out = dict(set=name, mode=mode, n=int(len(x)), frames=int(len(uf)), NB=NB, WIN=WIN, NPERM=NPERM, fc=FC, order=A, pairs=B,
           control=dict(drift_periods=DRIFT if mode == 'drift' else None, copy_prob=COPY if mode == 'copy' else None))
json.dump(out, open(f'{OUT}/{name}_{mode}.json', 'w'), indent=1)
print(name, mode, 'order p (growing, recent)', round(A['p_growing'], 3), round(A['p_recent'], 3),
      {k: (round(v['excess'], 5), round(v['p'], 3)) for k, v in B.items()}, flush=True)
