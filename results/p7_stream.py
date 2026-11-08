"""Paper 7, streaming: BLMLEmpiricalStreaming (ridge, window) on the double-slit lines (counts along the fringes) in
arrival order, against one-shot selection of the same model. Train + validation stream in batches of NB particles;
the burn-in (BURN particles) selects (fc, lam) on the 2^(1/8) fc grid x LAMS by an 80/20 binomial split of its counts
and fits exactly; then lam stays fixed and fc follows Adam on the leave-batch-out prequential score (implicit
gradient) over epochs, stopping at tol 1e-4 (exact) with the parameters stationary. One-shot: the same grid on train /
validation, refit on train + validation. No background floor in either (the floor is part of sqrt h). Split 64/16/20
(rng 0), test LL per particle, paired.
Usage: python results/p7_stream.py [set ...]   env BLML_BURN (2000), BLML_NB (400), BLML_EPOCHS (20), BLML_TAG
-> results/p7_stream/<set><tag>.json"""
import os, sys, json, time, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_binned import BLMLEmpirical, BLMLEmpiricalStreaming, _np
D = os.path.expanduser('~/data_repo/dslit_gpu/'); OUT = 'results/p7_stream'; os.makedirs(OUT, exist_ok=True)
BURN, NB, EPOCHS = (int(os.environ.get(k, v)) for k, v in (('BLML_BURN', 2000), ('BLML_NB', 400), ('BLML_EPOCHS', 20)))
TAG = os.environ.get('BLML_TAG', '')
GRID = 0.008 * 2.0 ** (np.arange(0, 49) / 8.0); GRID = GRID[GRID <= 0.5]
LAMS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0]
ll = lambda P, N: float((N * np.log(np.maximum(P / P.sum(), 1e-300))).sum() / N.sum())
dens = lambda m: np.maximum(_np(m.density_), 1e-300)


def select(tr, va, G):
    """(fc, lam) maximising the validation log-likelihood of BLMLEmpirical fitted on tr."""
    best = (None, None, -np.inf)
    for lam in LAMS:
        for f in G:
            v = ll(dens(BLMLEmpirical(reg='ridge', lam=lam).fit(tr, bt.make_modes([f], ['sinc']))), va)
            if v > best[2]:
                best = (float(f), lam, v)
    return best


for name in sys.argv[1:] or ['photon', 'electron', 'photon_mm']:
    pts = np.load(D + f'dslit_{name}_points.npy'); L = np.load(D + f'dslit_{name}_b.npy').shape[-2:][int(np.load(D + f'dslit_{name}_axis.npy'))]
    axis = int(np.load(D + f'dslit_{name}_axis.npy'))
    x = np.clip(np.floor(pts[:, 1 + axis] + 0.5).astype(int), 0, L - 1)
    u = np.random.default_rng(0).random(len(pts)); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))
    TR, VA, TE = [np.bincount(x[part == k], minlength=L).astype(float) for k in (0, 1, 2)]
    G = GRID[GRID >= 2.0 / L]; res = dict(set=name, burn=BURN, nb=NB, n_stream=int((part < 2).sum()), n_test=int(TE.sum()))
    t = time.time(); fc, lam, v = select(TR, VA, G)
    m1 = BLMLEmpirical(reg='ridge', lam=lam).fit(TR + VA, bt.make_modes([fc], ['sinc'])); P1 = dens(m1)
    res['oneshot'] = dict(fc=fc, lam=lam, val=v, test=ll(P1, TE), s=time.time() - t)
    xs = x[part < 2]                                            # arrival order
    t = time.time(); W = np.bincount(xs[:BURN], minlength=L).astype(float)
    trb = np.random.default_rng(0).binomial(W.astype(np.int64), 0.8).astype(float)
    fcb, lamb, _ = select(trb, W - trb, G)
    batches = [np.bincount(xs[b0:b0 + NB], minlength=L).astype(float) for b0 in range(0, len(xs), NB)]
    m = BLMLEmpiricalStreaming(reg='ridge', lam=lamb, burn_in=BURN, lr=0.05, param_every=1).fit(
        batches, [{0: dict(kind='sinc', fc=fcb)}], epochs=EPOCHS, tol=1e-4, tol_check='exact')
    P2 = dens(m); fcs = float(np.ravel(m._fc_now()[0])[0])
    res['stream'] = dict(fc_burn=fcb, lam=lamb, fc=fcs, test=ll(P2, TE), epochs=m.epochs_, converged=m.converged_,
                         s=time.time() - t, fc_path=[h['fc'] for h in m.history_ if 'fc' in h][::max(1, len(batches) // 4)])
    d = (np.log(P2 / P2.sum()) - np.log(P1 / P1.sum()))[x[part == 2]]
    res['stream_minus_oneshot'] = dict(mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))))
    json.dump(res, open(f'{OUT}/{name}{TAG}.json', 'w'), indent=1)
    print(name, f"one-shot fc {fc:.4f} lam {lam:g} {res['oneshot']['s']:.0f}s | burn fc {fcb:.4f} lam {lamb:g} -> stream fc {fcs:.4f}",
          f"{m.epochs_} ep conv {m.converged_} {res['stream']['s']:.0f}s | stream-oneshot {d.mean():+.5f}+-{res['stream_minus_oneshot']['se']:.5f}", flush=True)
