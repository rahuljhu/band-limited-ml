"""Nonparametric baseline density estimators and error metrics (GPU via CuPy,
NumPy fallback for local testing). All estimators return density values at
query points; evaluation is chunked so memory stays bounded.

Baselines: Gaussian product KDE (Scott rule, least-squares CV, oracle
bandwidth), 6th-order Gaussian KDE, sinc KDE (same fc as BLML, as in the base
paper), Abramson adaptive KDE, and the k-NN density estimator.
"""
import math
import numpy as np

try:
    import cupy as xp
    GPU = True
except ImportError:
    xp = np
    GPU = False

CHUNK = 2 ** 24      # elements per (query x train x d) chunk


def to_np(a):
    return xp.asnumpy(a) if GPU else np.asarray(a)


def _chunks(m, n, d):
    qm = max(1, min(m, CHUNK // max(1, n * d)))
    if qm == m or n * d <= CHUNK:
        return qm, n
    tn = max(1, CHUNK // max(1, qm * d))
    return qm, tn


def _pairwise_sum(Xtr, Xq, fn, weights=None):
    """sum_i w_i * fn(Xq - Xtr_i) over training points, chunked; fn maps
    (qm, tn, d) differences to (qm, tn) values."""
    Xtr, Xq = xp.asarray(Xtr), xp.asarray(Xq)
    m, d = Xq.shape
    n = Xtr.shape[0]
    w = xp.ones(n) if weights is None else xp.asarray(weights)
    qm, tn = _chunks(m, n, d)
    out = xp.zeros(m)
    for i in range(0, m, qm):
        for j in range(0, n, tn):
            diff = Xq[i:i+qm, None, :] - Xtr[None, j:j+tn, :]
            out[i:i+qm] += fn(diff) @ w[j:j+tn]
    return out


def kde_gauss(Xtr, Xq, h):
    """Gaussian product KDE with per-dimension bandwidths h (d,)."""
    Xtr, Xq = xp.asarray(Xtr), xp.asarray(Xq)
    h = xp.asarray(h, dtype=float)
    n, d = Xtr.shape
    A, Bq = Xtr / h, Xq / h
    na = (A ** 2).sum(1)
    m = Xq.shape[0]
    qm = max(1, min(m, CHUNK // max(1, n)))
    out = xp.empty(m)
    norm = 1.0 / (n * (2 * math.pi) ** (d / 2) * float(xp.prod(h)))
    for i in range(0, m, qm):
        b = Bq[i:i+qm]
        d2 = (b ** 2).sum(1)[:, None] + na[None, :] - 2.0 * b @ A.T
        out[i:i+qm] = xp.exp(-0.5 * xp.maximum(d2, 0)).sum(1) * norm
    return out


def kde6(Xtr, Xq, h):
    """6th-order Gaussian product KDE: K6(u) = (15/8 - 5/4 u^2 + 1/8 u^4) phi(u)."""
    h = xp.asarray(h, dtype=float)
    n = Xtr.shape[0]
    def fn(diff):
        u = diff / h
        k = (15/8 - 5/4 * u**2 + 1/8 * u**4) * xp.exp(-0.5 * u**2) / math.sqrt(2 * math.pi)
        return xp.prod(k, axis=2)
    return _pairwise_sum(Xtr, Xq, fn) / (n * float(xp.prod(h)))


def kde_sinc(Xtr, Xq, fc):
    """Sinc KDE: (1/n) sum_i prod_k fc_k sinc(fc_k (x_k - x_ik)). Can go negative."""
    fc = xp.asarray(np.broadcast_to(fc, (Xtr.shape[1],)), dtype=float)
    n = Xtr.shape[0]
    fn = lambda diff: xp.prod(fc * xp.sinc(fc * diff), axis=2)
    return _pairwise_sum(Xtr, Xq, fn) / n


def kde_adaptive(Xtr, Xq, h, alpha=0.5):
    """Abramson adaptive KDE: pilot Gaussian KDE with bandwidth h, local factors
    lambda_i = (pilot(x_i) / geomean)^(-alpha)."""
    Xtr = xp.asarray(Xtr)
    h = xp.asarray(h, dtype=float)
    n, d = Xtr.shape
    pilot = xp.maximum(kde_gauss(Xtr, Xtr, h), 1e-300)
    lam = (pilot / xp.exp(xp.log(pilot).mean())) ** (-alpha)
    A = Xtr / h
    na = (A ** 2).sum(1)
    Bq = xp.asarray(Xq) / h
    m = Bq.shape[0]
    qm = max(1, min(m, CHUNK // max(1, n)))
    w = lam ** (-d) / (n * (2 * math.pi) ** (d / 2) * float(xp.prod(h)))
    out = xp.empty(m)
    for i in range(0, m, qm):
        b = Bq[i:i+qm]
        d2 = (b ** 2).sum(1)[:, None] + na[None, :] - 2.0 * b @ A.T
        out[i:i+qm] = xp.exp(-0.5 * xp.maximum(d2, 0) / lam[None, :] ** 2) @ w
    return out


def knn_density(Xtr, Xq, k=None):
    """k-NN density k / (n V_d r_k^d), Euclidean on standardized coordinates."""
    Xtr, Xq = xp.asarray(Xtr), xp.asarray(Xq)
    n, d = Xtr.shape
    k = k or max(1, int(round(math.sqrt(n))))
    s = Xtr.std(0)
    A, Bq = Xtr / s, Xq / s
    na = (A ** 2).sum(1)
    m = Bq.shape[0]
    qm = max(1, min(m, CHUNK // max(1, n)))
    rk = xp.empty(m)
    for i in range(0, m, qm):
        b = Bq[i:i+qm]
        d2 = xp.maximum((b ** 2).sum(1)[:, None] + na[None, :] - 2.0 * b @ A.T, 0)
        rk[i:i+qm] = xp.sqrt(xp.partition(d2, k - 1, axis=1)[:, k - 1])
    vd = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
    return k / (n * vd * xp.maximum(rk, 1e-300) ** d * float(xp.prod(s)))


def scott(Xtr, order=2):
    """Rule-of-thumb bandwidth sigma_k n^{-1/(d+2*order)} (order 2: Scott)."""
    n, d = Xtr.shape
    return to_np(xp.asarray(Xtr).std(0)) * n ** (-1.0 / (d + 2 * order))


def lscv_gauss(Xtr, grid, n_max=5000, seed=0):
    """Least-squares CV bandwidth multiplier c for h = c * scott(X).

    LSCV(h) = int f_hat^2 - (2/n) sum_i f_hat_{-i}(x_i); for Gaussian kernels
    int f_hat^2 = (1/n^2) sum_ij phi_{sqrt(2) h}(x_i - x_j). O(n^2), so it runs
    on a subsample of n_max points and the chosen c is reused at full n
    (c multiplies the n-dependent Scott rule).
    """
    X = np.asarray(Xtr)
    if len(X) > n_max:
        X = X[np.random.default_rng(seed).choice(len(X), n_max, replace=False)]
    n = len(X)
    h0 = scott(X)
    best = (np.inf, None)
    for c in grid:
        h = c * h0
        int_f2 = to_np(kde_gauss(X, X, np.sqrt(2) * h)).mean()
        self_k = 1.0 / ((2 * math.pi) ** (X.shape[1] / 2) * np.prod(h))
        loo = (to_np(kde_gauss(X, X, h)) * n - self_k) / (n - 1)
        score = int_f2 - 2 * loo.mean()
        if score < best[0]:
            best = (score, c)
    return best[1]


# ---------------------------------------------------------------- metrics
def metrics(fhat_f, f_f, fhat_q, f_q, q_q, int_f2):
    """Errors of an estimate from values on two evaluation sets.

    fhat_f, f_f: estimate and truth at points drawn from f.
    fhat_q, f_q, q_q: estimate, truth and proposal density at points drawn from q.
    int_f2: int f^2 (for relative ISE).
    """
    fhat_f, f_f = np.asarray(fhat_f, float), np.asarray(f_f, float)
    fhat_q, f_q, q_q = np.asarray(fhat_q, float), np.asarray(f_q, float), np.asarray(q_q, float)
    terms = (fhat_q - f_q) ** 2 / q_q
    ise = float(np.mean(terms))
    ise_se = float(np.std(terms) / np.sqrt(len(terms)))      # Monte Carlo standard error
    pos = np.maximum(fhat_f, 1e-300)
    return {
        'ise': ise,
        'rel_ise': ise / int_f2,
        'rel_ise_mc_se': ise_se / int_f2,
        'ess_frac': float(np.mean(terms) ** 2 / np.mean(terms ** 2)) if np.any(terms) else 1.0,
        'hellinger2': float(1.0 - np.mean(np.sqrt(np.maximum(fhat_f, 0) / f_f))),
        'nll': float(-np.mean(np.log(pos))),
        'kl': float(np.mean(np.log(f_f) - np.log(pos))),
        'frac_nonpos': float(np.mean(fhat_f <= 0)),
    }
