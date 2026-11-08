"""Paper 6 prototype: a supervised dependence node. A node projects x in R^p to u = W'x in R^k (W orthonormal,
p x k) and maximises the mutual dependence between u and the class c,

    d^2(u, c) = 1 - sum_c pi_c <g_c, g> / (||g_c|| ||g||),

with g_c, g the BLML square-root densities of the class-c and pooled projections. Each BLML fit is turned into
per-point coefficients a_i = e_{bin(i)} / count_{bin(i)} (g(u) ~ (1/n) sum_i a_i K(u - u_i)), so every inner
product is an exact pairwise kernel sum (reproducing property), and its gradient in W is
sum_ij alpha_i beta_j (x_i - x_j) grad K(u_i - u_j)'. The node alternates BLML fits at fixed W with Stiefel
gradient steps in W at fixed coefficients (QR retraction). O(n^2 k) per step: for prototypes (n up to a few 1e3).
"""
import numpy as np
import blml_trivial as bt
from numba import njit, prange


@njit(parallel=True, cache=True)
def _pair_sums(U, Wt, fc):
    """For each point i and weight column q: S[i, q] = sum_j Wt[j, q] K(u_i - u_j) and
    D[i, q, m] = sum_j Wt[j, q] dK/dt_m (u_i - u_j), sinc product kernel K(t) = prod_m fc_m sinc(fc_m t_m)
    (prefix / suffix products: O(k) per pair). O(n^2 k) time, O(n k) memory."""
    n, k = U.shape; Q = Wt.shape[1]
    S = np.zeros((n, Q)); D = np.zeros((n, Q, k))
    for i in prange(n):
        kv = np.empty(k); dk = np.empty(k); pre = np.empty(k + 1); suf = np.empty(k + 1)
        for j in range(n):
            pre[0] = 1.0
            for m in range(k):
                t = U[i, m] - U[j, m]; pt = np.pi * fc[m] * t
                if abs(pt) < 1e-12:
                    kv[m] = fc[m]; dk[m] = 0.0
                elif abs(pt) < 0.05:                       # series: pt cos pt - sin pt cancels to -pt^3/3
                    p2 = pt * pt
                    kv[m] = fc[m] * (1.0 - p2 / 6.0 + p2 * p2 / 120.0 - p2 * p2 * p2 / 5040.0)
                    dk[m] = np.pi * fc[m] * fc[m] * pt * (-1.0 / 3.0 + p2 / 30.0 - p2 * p2 / 840.0 + p2 * p2 * p2 / 45360.0)
                else:
                    kv[m] = np.sin(pt) / (np.pi * t)
                    dk[m] = (pt * np.cos(pt) - np.sin(pt)) / (np.pi * t * t)
                pre[m + 1] = pre[m] * kv[m]
            suf[k] = 1.0
            for m in range(k - 1, -1, -1):
                suf[m] = suf[m + 1] * kv[m]
            for q in range(Q):
                w = Wt[j, q]
                if w == 0.0:
                    continue
                S[i, q] += w * pre[k]
                for m in range(k):
                    D[i, q, m] += w * dk[m] * pre[m] * suf[m + 1]
    return S, D


from blml_local import cross_sums as _cross_sums


def _kernel_and_grad(U, fc):          # dense O(n^2 k) memory: reference for small n
    """K(u_i - u_j) and its gradient in u_i (n, n, k) for the sinc product kernel at cut-off fc (per dim)."""
    n, k = U.shape
    T = U[:, None, :] - U[None, :, :]                       # (n, n, k)
    f = np.asarray(fc, float)[None, None, :]
    pt = np.pi * f * T
    with np.errstate(divide='ignore', invalid='ignore'):
        Kd = np.where(np.abs(pt) < 1e-12, f, np.sin(pt) / (np.pi * T))
        dKd = np.where(np.abs(pt) < 1e-12, 0.0, (pt * np.cos(pt) - np.sin(pt)) / (np.pi * T * T))
    K = np.prod(Kd, axis=2)
    G = np.empty_like(T)
    for m in range(k):
        G[..., m] = dKd[..., m] * np.prod(np.delete(Kd, m, axis=2), axis=2)
    return K, G


def _point_coefficients(U, fc):
    """BLML fit of U at cut-off fc, as per-point coefficients a_i (sum_i a_i K(u - u_i) / n ~ g)."""
    k = U.shape[1]
    m = bt.BLMLQuick(dtype=np.float64).fit(U, bt.make_modes(fc, ['sinc'] * k))
    Zi = np.round(m._fit_kernel.quantize(U) / m._h_vec).astype(np.int64)
    key = {tuple(z): i for i, z in enumerate(map(tuple, np.asarray(m._Zi)))}
    b = np.array([key[tuple(z)] for z in map(tuple, Zi)])
    return np.asarray(m.e_hat_, float)[b] / np.asarray(m.w_, float)[b]


class DependenceNode:
    """W (p x k, orthonormal) maximising the class mutual dependence of u = X W.
    fc: BLML cut-off on the (standardised) projection, per dim or scalar; lr: Stiefel step; restarts: random
    initialisations, the best objective kept."""

    def __init__(self, k=2, fc=1.0, lr=0.5, n_iter=60, restarts=4, seed=0, verbose=False):
        self.k, self.fc, self.lr, self.n_iter, self.restarts, self.seed, self.verbose = k, fc, lr, n_iter, restarts, seed, verbose

    def objective_grad(self, X, y, W):
        """d^2 and its gradient in W (fixed BLML coefficients), from per-point kernel and kernel-gradient sums
        (_pair_sums): O(n^2 k) time, O(n) memory."""
        U = np.ascontiguousarray(X @ W); n, k = U.shape
        fc = np.broadcast_to(np.asarray(self.fc, float), (k,)).copy()
        classes, counts = np.unique(y, return_counts=True)
        cols = [_point_coefficients(U, fc) / n]                       # pooled
        for c in classes:
            ac = np.zeros(n); idx = y == c; ac[idx] = _point_coefficients(U[idx], fc) / idx.sum(); cols.append(ac)
        A = np.stack(cols, 1)                                          # (n, 1 + C)
        S, D = _pair_sums(U, A, fc)
        a = A[:, 0]; gg = a @ S[:, 0]
        pg = lambda q1, q2: np.stack([X.T @ (A[:, q1] * D[:, q2, m] + A[:, q2] * D[:, q1, m]) for m in range(k)], 1)
        dgg = pg(0, 0)
        d2, grad = 1.0, np.zeros_like(W)
        for q, p_c in enumerate(counts / n, start=1):
            ac = A[:, q]; cg, cc = ac @ S[:, 0], ac @ S[:, q]
            cos = cg / np.sqrt(cc * gg)
            d2 -= p_c * cos
            grad -= p_c * (pg(q, 0) / np.sqrt(cc * gg) - 0.5 * cos * (pg(q, q) / cc + dgg / gg))
        return float(d2), grad

    def prequential_grad(self, Xtr, ytr, Xb, yb, W, eps_rel=1e-3):
        """Prequential dependence on a batch (Xb) the class densities never saw: f_c fitted on (Xtr, ytr) at W,
        f = sum_c pi_c f_c, d2_pre = 1 - sum_c pi_c mean_b sqrt((f_c(u_b) + eps) / (f(u_b) + eps)), and its gradient
        in W at fixed coefficients (through both the batch points and the training points). An overfitted (spiky)
        fit puts no mass on new points and scores ~0, so the objective cannot be gamed by resolution."""
        Utr = np.ascontiguousarray(Xtr @ W); Ub = np.ascontiguousarray(Xb @ W); k = W.shape[1]
        fc = np.broadcast_to(np.asarray(self.fc, float), (k,)).copy()
        classes, counts = np.unique(ytr, return_counts=True); pi = counts / counts.sum()
        A = np.zeros((len(Utr), len(classes)))
        for q, c in enumerate(classes):
            idx = ytr == c; A[idx, q] = _point_coefficients(Utr[idx], fc) / idx.sum()
        Sg, Dg = _cross_sums(Ub, Utr, A, fc)                      # g_c(u_b) and grad_u g_c(u_b)
        fcb = Sg ** 2; f = fcb @ pi
        eps = eps_rel * float(np.median(f))
        r = np.sqrt((fcb + eps) / (f + eps)[:, None])              # (nb, C)
        mb = len(Ub)
        d2 = 1.0 - float(pi @ r.mean(0))
        # d r_c / d g_c' = [delta_cc' g_c / (f_c + eps) - pi_c' g_c' / (f + eps)] r_c   (f_c = g_c^2)
        dr_dg = np.zeros((mb, len(classes), len(classes)))
        for c in range(len(classes)):
            for c2 in range(len(classes)):
                dr_dg[:, c, c2] = r[:, c] * ((c == c2) * Sg[:, c] / (fcb[:, c] + eps) - pi[c2] * Sg[:, c2] / (f + eps))
        wq = -(dr_dg * pi[None, :, None]).sum(1) / mb               # d d2 / d g_c'(u_b): (nb, C)
        # g_c(u_b) = sum_i a_ic K(u_b - u_i): d/dW = sum_i a_ic grad K(u_b - u_i) (x_b - x_i)'
        G = np.zeros_like(W)
        Sr, Dr = _cross_sums(Utr, Ub, wq, fc)                      # sum_b wq_bc grad K(u_i - u_b) at training points
        for m in range(k):
            G[:, m] += Xb.T @ (wq * Dg[:, :, m]).sum(1)                       # through the batch points
            G[:, m] += Xtr.T @ (A * Dr[:, :, m]).sum(1)                       # through the training points (odd grad)
        return d2, G

    def fit_prequential(self, X, y, init=None, batch=2000, n_iter=60, lr=0.05, betas=(0.9, 0.999)):
        """Streaming-style fit: each step draws a fresh batch, fits the class densities on the other points at the
        current W = Q diag(s), and takes an Adam step on (Q, log s) from the prequential dependence of the batch.
        The resolution s is free: the batch penalises any scale the data do not support."""
        X = np.asarray(X, float); y = np.asarray(y)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-12
        Xs = (X - self.mu_) / self.sd_; n = len(Xs)
        rng = np.random.default_rng(self.seed)
        starts = ([np.linalg.qr(np.asarray(init, float))[0]] if init is not None else [])
        starts += [np.linalg.qr(rng.normal(size=(X.shape[1], self.k)))[0] for _ in range(self.restarts)]
        best = None; self.starts_ = []
        for r, Q in enumerate(starts):
            ls = np.zeros(self.k); mQ = np.zeros_like(Q); vQ = 0.0; ms = np.zeros(self.k); vs = np.zeros(self.k); hist = []
            for it in range(n_iter):
                b = rng.choice(n, batch, replace=False); tr = np.setdiff1d(np.arange(n), b)
                sc = np.exp(ls); d2, G = self.prequential_grad(Xs[tr], y[tr], Xs[b], y[b], Q * sc)
                hist.append(d2)
                gs = sc * np.einsum('pk,pk->k', Q, G); Gq = G * sc
                Gt = Gq - Q @ ((Q.T @ Gq + Gq.T @ Q) / 2)
                t = it + 1; b1, b2 = betas
                mQ = b1 * mQ + (1 - b1) * Gt; vQ = b2 * vQ + (1 - b2) * float(np.sum(Gt * Gt))
                ms = b1 * ms + (1 - b1) * gs; vs = b2 * vs + (1 - b2) * gs * gs
                Q = np.linalg.qr(Q + lr * (mQ / (1 - b1 ** t)) / (np.sqrt(vQ / (1 - b2 ** t)) + 1e-12))[0]
                ls = ls + 2 * lr * (ms / (1 - b1 ** t)) / (np.sqrt(vs / (1 - b2 ** t)) + 1e-12)
            d2 = float(np.mean(hist[-10:]))
            if self.verbose:
                print(f'  start {r}: prequential d^2 {hist[0]:.3f} -> {d2:.3f} (last 10 batches)  scales {np.round(np.exp(ls), 2)}', flush=True)
            W = Q * np.exp(ls)
            self.starts_.append(dict(init=(init is not None and r == 0), d2=d2, W=W))
            if best is None or d2 > best[0]:
                best = (d2, W, hist)
        self.d2_, self.W_, self.history_ = best
        return self

    def fit(self, X, y, init=None):
        """W = Q diag(s): Q orthonormal (Stiefel steps, QR retraction) and s > 0 per-direction scales (steps on
        log s). Scaling u_m by s_m is the same as cut-off fc s_m, so the node learns its resolution per output
        direction. Each step is accepted only if d^2 improves (backtracking). init: optional starting Q (p x k)."""
        X = np.asarray(X, float); y = np.asarray(y)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-12
        Xs = (X - self.mu_) / self.sd_
        rng = np.random.default_rng(self.seed)
        best = None; self.starts_ = []
        starts = ([np.linalg.qr(np.asarray(init, float))[0]] if init is not None else [])
        starts += [np.linalg.qr(rng.normal(size=(X.shape[1], self.k)))[0] for _ in range(self.restarts)]
        for r, Q in enumerate(starts):
            ls = np.zeros(self.k); step_q, step_s = self.lr, 0.3
            d2, G = self.objective_grad(Xs, y, Q * np.exp(ls)); hist = [d2]
            for it in range(self.n_iter):
                sc = np.exp(ls)
                gs = sc * np.einsum('pk,pk->k', Q, G)                 # d d^2 / d log s
                Gq = G * sc                                           # d d^2 / dQ
                Gt = Gq - Q @ ((Q.T @ Gq + Gq.T @ Q) / 2)
                for _ in range(6):                                    # backtracking
                    Qn = np.linalg.qr(Q + step_q * Gt / (np.linalg.norm(Gt) + 1e-12))[0]
                    lsn = ls + step_s * np.clip(gs / (np.abs(gs).max() + 1e-12), -1, 1)
                    d2n, Gn = self.objective_grad(Xs, y, Qn * np.exp(lsn))
                    if d2n > d2:
                        Q, ls, d2, G = Qn, lsn, d2n, Gn; step_q, step_s = min(step_q * 1.5, 1.0), min(step_s * 1.5, 0.6)
                        break
                    step_q, step_s = step_q / 2, step_s / 2
                else:
                    break
                hist.append(d2)
            if self.verbose:
                print(f'  start {r}: d^2 {hist[0]:.3f} -> {d2:.3f}  scales {np.round(np.exp(ls), 2)}', flush=True)
            W = Q * np.exp(ls)
            self.starts_.append(dict(init=(init is not None and r == 0), d2=d2, W=W))
            if best is None or d2 > best[0]:
                best = (d2, W, hist)
        self.d2_, self.W_, self.history_ = best
        return self

    def transform(self, X):
        return ((np.asarray(X, float) - self.mu_) / self.sd_) @ self.W_


def save(X, y, k):
    """Sliced average variance estimation (classes as slices): the top-k eigenvectors of
    sum_c pi_c (I - Cov(z | c))^2 in whitened coordinates, mapped back; finds class differences in covariance
    (rings, quadrant XOR) that mean-based methods (LDA, SIR) miss. Returns W (p x k) in X's coordinates."""
    X = np.asarray(X, float); Xc = X - X.mean(0)
    L = np.linalg.cholesky(np.cov(Xc.T) + 1e-9 * np.eye(X.shape[1]))
    Z = np.linalg.solve(L, Xc.T).T
    I = np.eye(X.shape[1]); M = np.zeros_like(I)
    for c in np.unique(y):
        D = I - np.cov(Z[y == c].T); M += (y == c).mean() * D @ D
    v = np.linalg.eigh(M)[1][:, ::-1][:, :k]
    return np.linalg.solve(L.T, v)


def hsic_projection(X, y, k=2, sigma=1.0, lr=0.5, n_iter=60, restarts=3, seed=0):
    """KDR-style baseline: W (orthonormal) maximising HSIC(Xw, y) with a Gaussian kernel on u and a class kernel on
    y, by the same Stiefel steps. X standardised by the caller."""
    n = len(X); H = np.eye(n) - 1.0 / n
    Ly = (y[:, None] == y[None, :]).astype(float); Lc = H @ Ly @ H
    rng = np.random.default_rng(seed); best = None
    for r in range(restarts):
        W = np.linalg.qr(rng.normal(size=(X.shape[1], k)))[0]
        for it in range(n_iter):
            U = X @ W; T = U[:, None, :] - U[None, :, :]
            Ku = np.exp(-(T ** 2).sum(2) / (2 * sigma ** 2))
            A = Lc * Ku                                      # d HSIC / d u_i = sum_j A_ij (-(u_i - u_j)/sigma^2)
            G = np.zeros_like(W)
            for m in range(k):
                Mm = -A * T[..., m] / sigma ** 2
                G[:, m] = X.T @ (Mm.sum(1) - Mm.sum(0))
            Gt = G - W @ ((W.T @ G + G.T @ W) / 2)
            W = np.linalg.qr(W + lr * Gt / (np.linalg.norm(Gt) + 1e-12))[0]
        U = X @ W; h = float(np.sum(Lc * np.exp(-((U[:, None] - U[None]) ** 2).sum(2) / (2 * sigma ** 2))))
        if best is None or h > best[0]:
            best = (h, W)
    return best[1]
