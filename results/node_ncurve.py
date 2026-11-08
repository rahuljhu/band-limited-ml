"""Paper 6: stage 1 against n. For n in 500..4000 and the four planted structures of node_stage1.py, held-out
accuracy (BLML Bayes classifier on each method's plane) and principal angles for LDA, SIR, SAVE, the HSIC
projection, and the dependence node from random starts only and with the SAVE start added (best objective).
Writes results/node_ncurve.json."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from blml_node import DependenceNode, save, hsic_projection
from node_stage1 import make, angles, bayes_acc, sir, P

NS, NT = [500, 1000, 2000, 4000], 2000
out = {}
for n in NS:
    for kind in ('blobs', 'rings', 'xor', 'checker'):
        rng = np.random.default_rng(n)
        X, y, B = make(kind, n, rng); Xt, yt, _ = make(kind, NT, rng)
        mu, sd = X.mean(0), X.std(0); Xs, Xts = (X - mu) / sd, (Xt - mu) / sd
        Bs = B * sd[:, None]
        w = LDA().fit(Xs, y).scalings_[:, :1]
        cand = {'LDA (+PCA)': np.c_[w, np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)[2][0]], 'SIR': sir(Xs, y, 2),
                'SAVE': save(Xs, y, 2), 'HSIC (KDR-style)': hsic_projection(Xs, y, 2)}
        t = time.time(); node = DependenceNode(k=2, fc=1.0, restarts=2, n_iter=40).fit(X, y, init=cand['SAVE']); tn = time.time() - t
        rnd = max((s_ for s_ in node.starts_ if not s_['init']), key=lambda s_: s_['d2'])
        cand['node (random starts)'] = rnd['W']; cand['node (+SAVE start)'] = node.W_
        res = {}
        for name, W in cand.items():
            res[name] = dict(acc=round(bayes_acc(Xs @ W, y, Xts @ W, yt), 3), angles=[round(float(a), 1) for a in angles(W, Bs)])
        res['node_s'] = round(tn, 1)
        out[f'{kind}|{n}'] = res
        print(f"n={n:5d} {kind:7s} " + ' '.join(f"{k.split(' ')[0][:6]}{'R' if 'random' in k else ''} {v['acc']:.2f}" for k, v in res.items() if isinstance(v, dict))
              + f"  node {tn:.0f}s", flush=True)
        json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'node_ncurve.json'), 'w'), indent=1)
print('DONE')
