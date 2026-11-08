"""Paper 6: checkerboard and quadrant XOR at n = 20000 (2000 held out), 50-d. LDA, SIR, SAVE, and the
dependence node (memory-light pair sums) from random starts and from SAVE; plus the landscape diagnostic: the
node's prequential d^2 (on a batch the densities never saw) at the true plane, by scale, against random planes. Writes results/node_20k.json."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from blml_node import DependenceNode, save
from node_stage1 import make, angles, bayes_acc, sir, P

N, NT = 20000, 2000
out = {}
for kind in ('checker', 'xor'):
    rng = np.random.default_rng(7)
    X, y, B = make(kind, N, rng); Xt, yt, _ = make(kind, NT, rng)
    mu, sd = X.mean(0), X.std(0); Xs, Xts = (X - mu) / sd, (Xt - mu) / sd
    Bs = np.linalg.qr(B * sd[:, None])[0]
    node = DependenceNode(k=2, fc=1.0, restarts=2, seed=3, verbose=True)
    t = time.time(); b = rng.choice(N, 2000, replace=False); trn = np.setdiff1d(np.arange(N), b)
    pq = lambda W: node.prequential_grad(Xs[trn], y[trn], Xs[b], y[b], W)[0]          # prequential d^2 on one batch
    d2_true = {f: round(pq(Bs * f), 3) for f in (1.0, 2.0, 4.0, 16.0, 256.0)}           # incl. absurd scales
    d2_rand = [round(pq(np.linalg.qr(rng.normal(size=(P, 2)))[0] * f), 3) for f in (1.0, 4.0, 256.0)]
    print(f'{kind}: prequential d^2 at the true plane by scale {d2_true}; random planes (scales 1, 4, 256) {d2_rand} ({time.time() - t:.0f}s)', flush=True)
    w = LDA().fit(Xs, y).scalings_[:, :1]
    cand = {'LDA (+PCA)': np.c_[w, np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)[2][0]], 'SIR': sir(Xs, y, 2), 'SAVE': save(Xs, y, 2)}
    t = time.time(); node.fit_prequential(X, y, init=cand['SAVE'], batch=2000, n_iter=60); tn = time.time() - t
    rnd = max((s_ for s_ in node.starts_ if not s_['init']), key=lambda s_: s_['d2'])
    cand['node (random starts)'] = rnd['W']; cand['node (+SAVE start)'] = node.W_; cand['true plane'] = Bs
    res = dict(d2_true=d2_true, d2_random=d2_rand, node_s=round(tn, 1),
               starts=[dict(init=s_['init'], d2=s_['d2']) for s_ in node.starts_])
    for name, W in cand.items():
        own = name.startswith('node')                                   # the node keeps its learned scales
        res[name] = dict(acc=round(bayes_acc(Xs @ W, y, Xts @ W, yt, scale=not own), 3), angles=[round(float(a), 1) for a in angles(W, Bs)])
        print(f'{kind} n={N} {name:22s} acc {res[name]["acc"]:.3f} angles {res[name]["angles"]}', flush=True)
    out[kind] = res
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'node_20k.json'), 'w'), indent=1)
print('DONE', flush=True)
