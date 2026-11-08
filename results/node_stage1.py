"""Paper 6, stage 1a: can one dependence node find a planted 2-d class structure in 50-d noise that linear
methods miss? Structures: 'rings' (inner disk vs ring: equal means), 'xor' (y = sign(z1 z2): equal means and
covariances per class... not quite, see below), 'blobs' (shifted means: LDA's case). Compare subspaces from
LDA, SIR, PCA, random, and the node, by the principal angles to the true plane and by the held-out accuracy of
a BLML Bayes classifier on each 2-d projection."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np, warnings
warnings.filterwarnings('ignore')
import blml_trivial as bt
from blml_node import DependenceNode, save, hsic_projection

P, N, NT = 50, 2000, 2000


def make(kind, n, rng):
    Z = rng.normal(size=(n, 2))
    if kind == 'rings':
        y = (np.hypot(Z[:, 0], Z[:, 1]) > 1.18).astype(int)        # median radius: balanced classes
    elif kind == 'xor':
        y = (Z[:, 0] * Z[:, 1] > 0).astype(int)
    elif kind == 'checker':                                       # 4 x 4 cells of width 0.75: no 1st/2nd-order signal
        y = ((np.floor(Z[:, 0] / 0.75) + np.floor(Z[:, 1] / 0.75)) % 2).astype(int)
    else:
        y = rng.integers(0, 2, n); Z[:, 0] += 1.5 * (2 * y - 1)
    X = rng.normal(size=(n, P)); X[:, :2] = Z
    Q = np.linalg.qr(np.random.default_rng(99).normal(size=(P, P)))[0]
    return X @ Q, y, Q[:2].T                                     # true plane = image of the first two coords


def angles(W, B):
    Qw, Qb = np.linalg.qr(W)[0], np.linalg.qr(B)[0]
    return np.degrees(np.arccos(np.clip(np.linalg.svd(Qw.T @ Qb, compute_uv=False), -1, 1)))


def _bayes(Utr, ytr, Ute, fc):
    logp = []
    for c in (0, 1):
        m = bt.BLMLQuick(dtype=np.float64).fit(Utr[ytr == c], bt.make_modes(fc, ['sinc', 'sinc']))
        logp.append(np.log(np.maximum(m.predict(Ute), 1e-300)) + np.log((ytr == c).mean()))
    return logp[1] > logp[0]


def bayes_acc(Utr, ytr, Ute, yte, fcs=(0.5, 1.0, 2.0, 3.0, 4.0), scale=True):
    """Held-out accuracy of a BLML Bayes classifier on a projection; its cut-off chosen on a 20% validation split
    of the training points from fcs (so no method is held back by one fixed resolution). scale=False keeps the
    projection's own scale (a node with learned scales)."""
    if scale:
        sc = Utr.std(0); Utr, Ute = Utr / sc, Ute / sc
    v = np.random.default_rng(1).random(len(Utr)) < 0.2
    best = max(fcs, key=lambda f: np.mean(_bayes(Utr[~v], ytr[~v], Utr[v], f) == ytr[v]))
    return float(np.mean(_bayes(Utr, ytr, Ute, best) == yte))


def sir(X, y, k):
    Xc = X - X.mean(0); L = np.linalg.cholesky(np.cov(Xc.T)); Zs = np.linalg.solve(L, Xc.T).T
    M = sum((y == c).mean() * np.outer(Zs[y == c].mean(0), Zs[y == c].mean(0)) for c in np.unique(y))
    v = np.linalg.eigh(M)[1][:, ::-1][:, :k]
    return np.linalg.solve(L.T, v)


if __name__ == '__main__':
    out, proj = {}, {}
    for kind in ('blobs', 'rings', 'xor', 'checker'):
        rng = np.random.default_rng(0)
        X, y, B = make(kind, N, rng); Xt, yt, _ = make(kind, NT, rng)
        mu, sd = X.mean(0), X.std(0)
        Xs, Xts = (X - mu) / sd, (Xt - mu) / sd
        res = {}
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        w = LDA().fit(Xs, y).scalings_[:, :1]
        cand = {'random': np.linalg.qr(rng.normal(size=(P, 2)))[0], 'PCA': np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)[2][:2].T,
                'LDA (+PCA)': np.c_[w, np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)[2][0]], 'SIR': sir(Xs, y, 2),
                'SAVE': save(Xs, y, 2)}
        t = time.time(); cand['HSIC (KDR-style)'] = hsic_projection(Xs, y, 2); th = time.time() - t
        t = time.time(); node = DependenceNode(k=2, fc=1.0, restarts=2, n_iter=40, verbose=True).fit(X, y, init=cand['SAVE']); tn = time.time() - t
        cand['dependence node'] = node.W_
        Bs = B * sd[:, None]                                         # true plane in standardised coordinates
        for name, W in cand.items():
            a = angles(W, Bs); acc = bayes_acc(Xs @ W, y, Xts @ W, yt)
            res[name] = dict(angles=[round(float(v), 1) for v in a], acc=round(acc, 3))
            proj[f'{kind}|{name}'] = (Xts @ W)[:600]
            print(f'{kind:6s} {name:16s} principal angles {np.round(a, 1)} deg   Bayes acc {acc:.3f}', flush=True)
        res['node_d2'] = node.d2_; res['node_s'] = round(tn, 1); res['hsic_s'] = round(th, 1)
        print(f'{kind:6s} node d^2 {node.d2_:.3f} ({tn:.0f}s)', flush=True)
        out[kind] = res
        proj[f'{kind}|truth'] = (Xts @ np.linalg.qr(Bs)[0])[:600]; proj[f'{kind}|y'] = yt[:600]
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'node_stage1.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'node_stage1_proj.npz'), **proj)
