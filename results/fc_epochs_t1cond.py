"""Does streaming fc climb with epochs under the conditional score? Two classes, each paper 1's T1 product density in
4-d (fc_true = 0.4), class 1 shifted by SHIFT in every coordinate; the class is a dep column (fc 10000, fixed), the
score the conditional log-likelihood; only the 4 feature fc are trained (burn-in selector, then Adam). Plain vs split
(bcd_frac 0.5, freeze_frac 0). Prints fc per epoch and the final test accuracy / conditional log-likelihood vs the
Bayes-optimal ones from the true class densities. Env: T1C_N (20000), T1C_EPOCHS (20), T1C_SHIFT (3), T1C_CFG.
Writes results/fc_epochs_t1cond_<cfg>.json."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np, warnings
warnings.filterwarnings('ignore')
from p1_truth import sample_f1, f1, FC_TRUE
from blml_rolling import BLMLQuickStreaming

D, N, EP = 4, int(os.environ.get('T1C_N', 20000)), int(os.environ.get('T1C_EPOCHS', 20))
SHIFT = float(os.environ.get('T1C_SHIFT', 3.0))
rng = np.random.default_rng(0)


def draw(m):
    y = np.r_[np.zeros(m // 2), np.ones(m - m // 2)]
    X = np.stack([sample_f1(rng, m) for _ in range(D)], 1) + SHIFT * y[:, None]
    p = rng.permutation(m)
    return X[p], y[p]


X, y = draw(N); Xt, yt = draw(20000)
lf = lambda Z, c: np.sum(np.log(f1(Z - SHIFT * c)), 1)                # true class log densities
L0, L1 = lf(Xt, 0), lf(Xt, 1)
p1_true = 1 / (1 + np.exp(L0 - L1))
bayes_acc = float(np.mean((p1_true > 0.5) == (yt == 1)))
bayes_cll = float(np.mean(np.log(np.where(yt == 1, p1_true, 1 - p1_true))))
modes = [{**{j: dict(kind='sinc') for j in range(D)}, D: dict(kind='sinc', fc=dict(value=10000.0, trainable=False), dep=True)}]
cfgs = {'plain': dict(), 'split': dict(bcd_frac=0.5, freeze_frac=0.0)}
which = os.environ.get('T1C_CFG', 'both')
print(f'Bayes: acc {bayes_acc:.4f}, conditional ll {bayes_cll:.4f}', flush=True)
for name, kw in cfgs.items():
    if which not in ('both', name):
        continue
    t = time.time()
    s = BLMLQuickStreaming(burn_in=1500, lr=0.05, param_every=1, dtype=np.float64, **kw).fit(np.c_[X, y], modes, epochs=EP, batch=500)
    fcs = [np.ravel(h['fc'])[:D] for h in s.history_ if 'fc' in h and 'preq' in h]
    steps_ep = len(fcs) / EP
    per_epoch = [fcs[min(len(fcs) - 1, int(round((e + 1) * steps_ep)) - 1)] for e in range(EP)]
    q1 = s.predict_ratio(np.c_[Xt, np.ones(len(Xt))]); q0 = s.predict_ratio(np.c_[Xt, np.zeros(len(Xt))])
    pp = q1 / np.maximum(q0 + q1, 1e-300)
    acc = float(np.mean((pp > 0.5) == (yt == 1)))
    cll = float(np.mean(np.log(np.maximum(np.where(yt == 1, pp, 1 - pp), 1e-300))))
    out = dict(cfg=name, n=N, epochs=EP, shift=SHIFT, fc_true=FC_TRUE, acc=acc, cll=cll, bayes_acc=bayes_acc,
               bayes_cll=bayes_cll, s=round(time.time() - t, 1), fc_per_epoch=[np.round(f, 4).tolist() for f in per_epoch])
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'fc_epochs_t1cond_{name}.json'), 'w'), indent=1)
    g = [float(np.exp(np.mean(np.log(f)))) for f in per_epoch]
    print(f"{name}: acc {acc:.4f} (Bayes {bayes_acc:.4f}), conditional ll {cll:.4f} (Bayes {bayes_cll:.4f})  ({out['s']:.0f}s)", flush=True)
    print(f"  fc geo-mean by epoch: {' '.join(f'{v:.3f}' for v in g)}", flush=True)
    print(f"  epoch {EP} per dim: {np.round(per_epoch[-1], 3).tolist()}", flush=True)
