"""Paper 6: one dependence node on full MNIST (60000 train, 10000 test): blml_net's node (_fit_node) on the whole
image as a single patch (784 pixels, constant ones dropped, projected to k dims from a random W, fc started coarse),
predicting argmax_c of its class evidence. Reference: LDA to the same k dims plus a Gaussian (QDA) classifier.
Env: NODE_K, NODE_EPOCHS, NODE_FC0, NODE_SEED, NODE_ENGINE, NODE_OUT, NODE_N (subsample, smoke tests),
NODE_START (random | lda | ldas | pca: the node's starting directions, blml_net.node_start); NODE_BURN (burn-in points); NODE_PARAM_EVERY (batches per parameter step, default 1); NODE_RANK (modes); NODE_FC0=auto: the
burn-in selector picks the starting fc at the start directions."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from blml_net import _fit_node, _evidence, node_start
import blml_trivial as bt

K, EP = int(os.environ.get('NODE_K', 2)), int(os.environ.get('NODE_EPOCHS', 4))
FC0 = None if os.environ.get('NODE_FC0') == 'auto' else float(os.environ.get('NODE_FC0', 0.5))   # auto: burn-in search
SEED, ENG = int(os.environ.get('NODE_SEED', 1)), os.environ.get('NODE_ENGINE', 'gpu')
OUTF = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get('NODE_OUT', f'node_mnist1_k{K}.json'))
m = np.load(os.environ.get('NODE_MNIST', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mnist.npz')))
X, Xt = m['Xtr'].reshape(-1, 784).astype(np.float64), m['Xte'].reshape(-1, 784).astype(np.float64)
y, yt = m['ytr'].astype(int), m['yte'].astype(int)
if os.environ.get('NODE_N'):
    n = int(os.environ['NODE_N']); X, y, Xt, yt = X[:n], y[:n], Xt[:2000], yt[:2000]
classes = np.unique(y)

keep = X.std(0) > 1e-8
Z = LDA(n_components=K).fit(X[:, keep], y)
lda_acc = float(np.mean(QDA(reg_param=1e-3).fit(Z.transform(X[:, keep]), y).predict(Z.transform(Xt[:, keep])) == yt))
print(f'MNIST D={keep.sum()} k={K}: LDA-{K} + QDA acc {lda_acc:.4f}', flush=True)

t = time.time()
START = os.environ.get('NODE_START', 'random')
BURN = int(os.environ.get('NODE_BURN', 1500))                      # points the burn-in (and its fc search) sees
PE = int(os.environ.get('NODE_PARAM_EVERY', 1))                     # batches per parameter step
RANK = int(os.environ.get('NODE_RANK', 1))                         # modes of the density
node = _fit_node(X, y, classes, K, FC0, 'evidence', EP, 500, BURN, 0.05, ENG, SEED, W0=node_start(X, y, K, START),
                 param_every=PE, rank=RANK)
E = _evidence(node, Xt)                                          # log p(c | u) - log pi_c, centred over classes
acc = float(np.mean(classes[np.argmax(E + np.log(np.bincount(y) / len(y)), 1)] == yt))
s = node['model']
r = dict(D=int(keep.sum()), k=K, n=len(X), epochs=EP, engine=ENG, fc0=FC0, seed=SEED, start=START, param_every=PE, rank=RANK, acc=round(acc, 4),
         lda_qda_acc=round(lda_acc, 4), fc=np.round(bt.modes_fc(s.modes_)[:, :K], 3).tolist(), s=round(time.time() - t, 1))
json.dump(r, open(OUTF, 'w'), indent=1)
print(f"MNIST node k={K} rank={RANK} start={START} fc0={FC0} param_every={PE} epochs={EP} acc {acc:.4f} fc {r['fc']} ({r['s']:.0f}s)", flush=True)
