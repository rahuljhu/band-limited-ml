"""Does streaming fc keep climbing with epochs? Paper 1's truth T1 (product of f1 per dimension; fc_true = 0.4) in 4-d,
n points, joint log-likelihood score, fc from the burn-in selector then Adam; plain training vs the split
(bcd_frac, freeze_frac 0: fc learns throughout). Prints fc (geometric mean over dims and each dim) at the end of every
epoch and the final held-out log-likelihood vs the truth's. Env: T1_N (20000), T1_EPOCHS (20), T1_CFG (plain|split|both).
Writes results/fc_epochs_t1_<cfg>.json."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np, warnings
warnings.filterwarnings('ignore')
from p1_truth import sample_f1, f1, FC_TRUE
from blml_rolling import BLMLQuickStreaming

D, N, EP = 4, int(os.environ.get('T1_N', 20000)), int(os.environ.get('T1_EPOCHS', 20))
B = 500
rng = np.random.default_rng(0)
X = np.stack([sample_f1(rng, N) for _ in range(D)], 1); Xt = np.stack([sample_f1(rng, 20000) for _ in range(D)], 1)
ll_true = float(np.mean(np.sum(np.log(f1(Xt)), 1)))
modes = [{j: dict(kind='sinc') for j in range(D)}]                   # fc: burn-in selector, then Adam
cfgs = {'plain': dict(), 'split': dict(bcd_frac=0.5, freeze_frac=0.0)}
which = os.environ.get('T1_CFG', 'both')
for name, kw in cfgs.items():
    if which not in ('both', name):
        continue
    t = time.time()
    s = BLMLQuickStreaming(burn_in=1500, lr=0.05, param_every=1, dtype=np.float64, **kw).fit(X, modes, epochs=EP, batch=B)
    fcs = [h['fc'] for h in s.history_ if 'fc' in h and 'preq' in h]
    steps_ep = len(fcs) / EP
    per_epoch = [np.ravel(fcs[min(len(fcs) - 1, int(round((e + 1) * steps_ep)) - 1)]) for e in range(EP)]
    ll = float(np.mean(np.log(np.maximum(s.predict(Xt), 1e-300))))
    out = dict(cfg=name, n=N, epochs=EP, fc_true=FC_TRUE, ll=ll, ll_true=ll_true, s=round(time.time() - t, 1),
               fc_per_epoch=[np.round(f, 4).tolist() for f in per_epoch])
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'fc_epochs_t1_{name}.json'), 'w'), indent=1)
    print(f"{name}: held-out ll {ll:.4f} (truth {ll_true:.4f}, gap {ll_true - ll:.4f})  ({out['s']:.0f}s)", flush=True)
    for e, f in enumerate(per_epoch):
        print(f"  epoch {e + 1:2d}: fc geo-mean {np.exp(np.mean(np.log(f))):.3f}  per dim {np.round(f, 3).tolist()}", flush=True)
