"""Locally connected networks of dependence nodes (paper 6), trained greedily layer by layer, no backprop.

    net = BLMLNet([LocalLayer(patch=4, stride=2, k=2, pad=1),        # 28x28x1 -> 14x14x2
                   LocalLayer(patch=4, stride=2, k=4, pad=1),        # 14x14x2 -> 7x7x4
                   LocalLayer(patch=4, stride=3, k=8),               # 7x7x4   -> 2x2x8
                   LocalLayer(patch=1, stride=1, k=1)])              # 2x2x8   -> 2x2x1
    net.fit(X, y)            # X (n, H, W) or (n, H, W, C); y integer classes
    net.predict(Xte)         # BLML posterior on the last map

A layer has one node per patch position (no weight sharing): a BLMLQuickStreaming fit of the image label (a dep
column at a fixed integer cut-off, which decouples the classes) on a mapped sinc unit (u = x W) over the patch's inputs
(k outputs, one fc each, W from a random start, fc started coarse). The node's output feeds the next layer:
  output='u'           the projection u = x W (k values; a stack of these is linear in the pixels),
  output='evidence'    log p(c | u) - log pi_c for every class (C values, centred over classes; nonlinear),
  output='evidence_k'  that evidence reduced to k values by its principal components on the training data.
Nodes of a layer are independent: n_jobs fits them in parallel processes. The head is an exact BLML fit of the
label on the flattened last map (fc chosen by the selector), predicting argmax_c p(c | features)."""
import os
import numpy as np


def patches(Z, patch, stride, pad):
    """Z (n, H, W, C) -> (n, H', W', patch * patch * C) patch vectors (zero padding), H' = (H + 2 pad - patch) // stride + 1."""
    if pad:
        Z = np.pad(Z, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    n, H, W, C = Z.shape
    Ho, Wo = (H - patch) // stride + 1, (W - patch) // stride + 1
    out = np.empty((n, Ho, Wo, patch * patch * C), Z.dtype)
    for i in range(Ho):
        for j in range(Wo):
            out[:, i, j] = Z[:, i * stride:i * stride + patch, j * stride:j * stride + patch, :].reshape(n, -1)
    return out


def node_start(P, y, k, start):
    """A node's starting directions (p_kept, k), orthonormal, on its varying inputs: 'random' (None: the fit's own
    seeded start), 'lda' (the top k discriminant directions, then PCA if there are fewer than k), 'ldas' (the same
    with shrinkage: stable when inputs outnumber what the data pin down), or 'pca'."""
    if start in (None, 'random'):
        return None
    P = np.asarray(P, np.float64); P = P[:, P.std(0) > 1e-8]; Pc = P - P.mean(0)
    pcs = np.linalg.svd(Pc, full_matrices=False)[2]
    if start == 'pca':
        return pcs[:k].T
    if start not in ('lda', 'ldas'):
        raise ValueError(f"start {start!r}: 'random', 'lda', 'ldas' or 'pca'")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    m = min(k, len(np.unique(y)) - 1)
    lda = LDA(n_components=m) if start == 'lda' else LDA(solver='eigen', shrinkage='auto')
    W = lda.fit(P, y).scalings_[:, :m]
    return np.linalg.qr(np.c_[W, pcs[:k - m].T])[0] if m < k else np.linalg.qr(W)[0]


def _fit_node(P, y, classes, k, fc0, output, epochs, batch, burn_in, lr, engine, seed, W0=None, param_every=1, rank=1):
    """One node on patch vectors P (n, p): returns (W, fitted streaming model or None, PCA basis or None).
    W0: starting directions on the varying inputs (node_start), else a random start. fc0: every output's starting
    fc, or None for the burn-in selector's pick at the start W (scale-aware: the right choice for a W0 whose
    directions differ in spread, e.g. discriminant ones). param_every: batches per parameter step (W and fc).
    rank: modes of the density (f = sum_k |g_k|^2; the map is shared, each mode has its own fc)."""
    import warnings; warnings.filterwarnings('ignore')
    from blml_rolling import BLMLQuickStreaming
    P = np.asarray(P, np.float64); p = P.shape[1]
    keep = P.std(0) > 1e-8                                          # constant inputs (e.g. image borders) carry nothing
    if keep.sum() == 0:
        return dict(keep=keep, W=None, model=None, pca=None)
    Pk = P[:, keep]; k = min(k, Pk.shape[1])                       # few varying inputs: fewer outputs (padded later)
    unit = dict(kind='sinc', k=k) if W0 is None else dict(kind='sinc', W=W0[:, :k])
    if fc0 is not None:
        unit['fc'] = [fc0] * k
    modes = [{tuple(range(Pk.shape[1])): dict(unit),
              Pk.shape[1]: dict(kind='sinc', fc=dict(value=10000.0, trainable=False), dep=True)} for _ in range(rank)]
    s = BLMLQuickStreaming(burn_in=burn_in, lr=lr, engine=engine, seed=seed, param_every=param_every,
                           dtype=np.float32 if engine == 'gpu' else np.float64).fit(
        np.c_[Pk, y], modes, epochs=epochs, batch=batch)
    node = dict(keep=keep, W=s.W_, model=s if output != 'u' else None, pca=None, classes=classes)
    if output == 'evidence_k':
        E = _evidence(node, P)
        mu = E.mean(0); V = np.linalg.svd(E - mu, full_matrices=False)[2][:k].T
        node['pca'] = (mu, V)
    return node


def _evidence(node, P):
    """log p(c | u) - log pi_c per class, centred over classes (the node's class evidence at patch vectors P)."""
    s, Pk = node['model'], np.asarray(P, np.float64)[:, node['keep']]
    L = np.stack([np.log(np.maximum(s.predict_ratio(np.c_[Pk, np.full(len(Pk), c)]), 1e-300)) for c in node['classes']], 1)
    return L - L.mean(1, keepdims=True)


def _node_out(node, P, output, k, n_classes):
    width = n_classes if output == 'evidence' else k
    out = np.zeros((len(P), width))
    if node['W'] is None:                                           # a node with no varying input: zeros
        return out
    if output == 'u':
        o = np.asarray(P, np.float64)[:, node['keep']] @ node['W']
    else:
        o = _evidence(node, P)
        if output == 'evidence_k':
            mu, V = node['pca']; o = (o - mu) @ V
    out[:, :o.shape[1]] = o
    return out


def _single_threaded():
    """Worker processes inherit these: one thread each (the parallelism is across nodes)."""
    for v in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v] = '1'


class LocalLayer:
    """One node per patch position (see the module docstring). patch, stride, pad: the patch grid on the input map;
    k: outputs per node; output: 'u' | 'evidence' | 'evidence_k'; fc0: every output's starting fc (coarse);
    epochs, batch, burn_in, lr: the nodes' streaming fit; n_jobs: parallel node fits; engine: 'cpu' | 'gpu'."""

    def __init__(self, patch, stride, k, pad=0, output='u', fc0=0.5, epochs=4, batch=2000, burn_in=2000, lr=0.05,
                 n_jobs=1, engine='cpu', seed=0):
        self.patch, self.stride, self.k, self.pad, self.output = patch, stride, k, pad, output
        self.fc0, self.epochs, self.batch, self.burn_in, self.lr = fc0, epochs, batch, burn_in, lr
        self.n_jobs, self.engine, self.seed = n_jobs, engine, seed

    def fit(self, Z, y):
        from joblib import Parallel, delayed
        P = patches(Z, self.patch, self.stride, self.pad); self.grid_ = P.shape[1:3]
        self.classes_ = np.unique(y)
        pos = [(i, j) for i in range(P.shape[1]) for j in range(P.shape[2])]
        args = lambda i, j: (P[:, i, j], y, self.classes_, self.k, self.fc0, self.output, self.epochs, self.batch,
                             self.burn_in, self.lr, self.engine, self.seed + 1000 * i + j)
        _single_threaded()
        first = _fit_node(*args(*pos[0]))                         # serially first: compiles and caches the kernels once
        rest = Parallel(n_jobs=self.n_jobs)(delayed(_fit_node)(*args(i, j)) for i, j in pos[1:])
        self.nodes_ = dict(zip(pos, [first] + rest))
        return self

    def transform(self, Z):
        from joblib import Parallel, delayed
        P = patches(Z, self.patch, self.stride, self.pad)
        pos = list(self.nodes_); _single_threaded()
        outs = Parallel(n_jobs=self.n_jobs)(delayed(_node_out)(self.nodes_[q], P[:, q[0], q[1]], self.output, self.k,
                                                               len(self.classes_))
                                            for q in pos)
        H, W = self.grid_
        out = np.empty((len(Z), H, W, outs[0].shape[1]))
        for (i, j), o in zip(pos, outs):
            out[:, i, j] = o
        return out


class BLMLNet:
    """Greedy stack of LocalLayers and a BLML posterior head on the flattened last map (module docstring)."""

    def __init__(self, layers, head_search='shared', verbose=True):
        self.layers, self.head_search, self.verbose = layers, head_search, verbose

    @staticmethod
    def _as_maps(X):
        X = np.asarray(X, np.float64)
        return X[..., None] if X.ndim == 3 else X

    def features(self, X):
        Z = self._as_maps(X)
        for L in self.layers:
            Z = L.transform(Z)
        return Z.reshape(len(Z), -1)

    def fit(self, X, y):
        import time, warnings; warnings.filterwarnings('ignore')
        from blml_trivial import BLMLQuick
        from selector import select
        y = np.asarray(y); Z = self._as_maps(X)
        for li, L in enumerate(self.layers):
            t = time.time(); L.fit(Z, y); Z = L.transform(Z)
            if self.verbose:
                print(f"layer {li}: {len(L.nodes_)} nodes -> map {Z.shape[1:]} ({time.time() - t:.0f}s)", flush=True)
        F = Z.reshape(len(Z), -1); self.classes_ = np.unique(y)
        keep = F.std(0) > 1e-8; self.keep_ = keep; F = F[:, keep]
        d = F.shape[1]
        tmpl = [{**{j: {} for j in range(d)}, d: dict(kind='sinc', fc=dict(value=10000.0, trainable=False), dep=True)}]
        modes, _ = select(np.c_[F, y], tmpl, search=self.head_search)
        self.head_ = BLMLQuick().fit(np.c_[F, y], modes)
        return self

    def predict_proba(self, X):
        F = self.features(X)[:, self.keep_]
        R = np.stack([self.head_.predict_ratio(np.c_[F, np.full(len(F), c)]) for c in self.classes_], 1)
        return R / np.maximum(R.sum(1, keepdims=True), 1e-300)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]
