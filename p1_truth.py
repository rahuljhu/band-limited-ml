"""Band-limited, strictly positive truth densities with exact i.i.d. samplers.

T1 (product): f(x) = prod_k f1(x_k), the base paper's 1-D density
    f1(t) = 0.15 * (s(0.2 t)^4 + s(0.2 t + 0.1)^4),   s = numpy sinc,
which is band-limited to |nu| <= 0.4 and has no real zeros (the two terms'
zeros never coincide).

T2 (non-separable, exactly in the BLML model class): f = h^2 / ||h||^2 with
    h(x) = sum_j a_j prod_k s(b (x_k - mu_jk)),   s(t) = sinc(t)^2 + sinc(t + 0.5)^2,
a_j > 0, b = 0.2, centers on the diagonal. s is strictly positive (its two terms
have disjoint zero sets) and band-limited to |nu| <= 1, so h > 0 is band-limited to
b = 0.2 per dimension: sqrt(f) = h/||h|| has band fc/2 and f has band fc = 0.4 =
fc_true. Tails decay like |x|^-4 (finite variance), as in T1. Unlike a mixture, the
square root of f is itself band-limited, so BLML at fc_true is well specified.

Sampling is exact (no MCMC):
  * s(v)^2 is a density; rejection from a Cauchy(0, 1/pi) proposal with
    constant 2 is exact since s(v)^2 <= min(1, 1/(pi v)^2) <= 2/(1 + pi^2 v^2).
  * s(v)^4 ~ s(v)^2 * s(v)^2: draw from s^2, accept with probability s(v)^2.
  * h^2: by Cauchy-Schwarz h^2 <= (sum a) * sum_j a_j K_j^2, and each K_j^2 is a
    product of s^2 densities; s^2 <= 2 (sinc^4(t) + sinc^4(t+.5)) gives an exact
    per-coordinate rejection sampler (acceptance ~0.9), so no dimension curse.

In the estimator's convention (kernel fc*sinc(fc t): sqrt(f) band fc/2, f band
fc) both truths have fc_true = 0.4 per dimension.
"""
import numpy as np

FC_TRUE = 0.4
_A1 = 0.2          # f1 = 0.15 (s(0.2 t)^4 + s(0.2 t + 0.1)^4)
_B2 = 0.2          # T2 kernel scale


def _sinc2_draws(rng, size):
    """Exact draws from the density s(v)^2 (numpy sinc)."""
    out = np.empty(size)
    filled = 0
    while filled < size:
        m = int(1.3 * (size - filled)) + 16
        v = np.tan(np.pi * (rng.random(m) - 0.5)) / np.pi
        keep = rng.random(m) * 2.0 / (1.0 + (np.pi * v) ** 2) <= np.sinc(v) ** 2
        v = v[keep][:size - filled]
        out[filled:filled + len(v)] = v
        filled += len(v)
    return out


def _sinc4_draws(rng, size):
    """Exact draws from the density proportional to s(v)^4."""
    out = np.empty(size)
    filled = 0
    while filled < size:
        m = int(2.0 * (size - filled)) + 16
        v = _sinc2_draws(rng, m)
        v = v[rng.random(m) <= np.sinc(v) ** 2][:size - filled]
        out[filled:filled + len(v)] = v
        filled += len(v)
    return out


def _h_sinc2(u):
    """h(u) = int s(t)^2 s(t-u)^2 dt, elementwise. The integrand is band-limited,
    so a Riemann sum with spacing 0.25 is exact up to the |t|^-4 tails (< 1e-11)."""
    t = np.arange(-3000.0, 3000.0, 0.25)
    u = np.asarray(u, dtype=float)
    out = np.empty(u.shape)
    for idx in np.ndindex(u.shape):
        out[idx] = 0.25 * np.sum(np.sinc(t) ** 2 * np.sinc(t - u[idx]) ** 2)
    return out


def _s(t):
    return np.sinc(t) ** 2 + np.sinc(t + 0.5) ** 2


def _H_s(w):
    """H(w) = int s(t) s(t-w) dt, elementwise (Riemann sum: integrand band-limited,
    tails ~|t|^-4 < 1e-11 beyond +-3000)."""
    t = np.arange(-3000.0, 3000.0, 0.25)
    st = _s(t)
    w = np.asarray(w, dtype=float)
    out = np.empty(w.shape)
    for idx in np.ndindex(w.shape):
        out[idx] = 0.25 * np.sum(st * _s(t - w[idx]))
    return out


def _s2_draws(rng, size):
    """Exact draws from the density proportional to s(t)^2."""
    out = np.empty(size)
    filled = 0
    while filled < size:
        m = int(1.2 * (size - filled)) + 16
        t = _sinc4_draws(rng, m)
        t = np.where(rng.random(m) < 0.5, t, t - 0.5)       # density ~ sinc^4(t) + sinc^4(t+.5)
        u, v = np.sinc(t) ** 2, np.sinc(t + 0.5) ** 2
        keep = rng.random(m) * 2.0 * (u * u + v * v) <= (u + v) ** 2
        t = t[keep][:size - filled]
        out[filled:filled + len(t)] = t
        filled += len(t)
    return out


def f1(t):
    t = np.asarray(t, dtype=float)
    return 0.15 * (np.sinc(_A1 * t) ** 4 + np.sinc(_A1 * t + 0.1) ** 4)


def sample_f1(rng, size):
    t = _sinc4_draws(rng, size) / _A1
    shift = rng.random(size) < 0.5          # second term: s(0.2 (t + 0.5))^4
    return np.where(shift, t - 0.5, t)


class Truth:
    """Truth density in d dims: 'T1' (product), 'T2' (non-separable), or 'T3'
    (correlated normal, unit variances, correlation 0.5).

    T3 is not exactly band-limited, but sqrt(f) is Gaussian with spectrum
    ~ exp(-pi^2 nu^2 sigma^2 * 2), so cutting at |nu| <= fc/2 leaves a relative
    bias ~ exp(-pi^2 fc^2 / 2): ~1e-2 at fc = 1, ~3e-9 at fc = 2."""

    def __init__(self, kind, d):
        self.kind, self.d = kind, d
        self.fc_true = 2.0 if kind == 'T3' else FC_TRUE
        if kind == 'T3':
            self.cov = 0.5 * np.eye(d) + 0.5
            self.prec = np.linalg.inv(self.cov)
            self.lognorm = -0.5 * (d * np.log(2 * np.pi) + np.linalg.slogdet(self.cov)[1])
            self.chol = np.linalg.cholesky(self.cov)
        if kind == 'T2':
            self.a = np.array([1.0, 0.7, 0.7, 1.0])        # positive: h > 0
            self.mu = np.outer([-9.0, -3.0, 3.0, 9.0], np.ones(d))   # (J, d)
            dmu = self.mu[:, None, :] - self.mu[None, :, :]
            gram = np.prod(_H_s(_B2 * dmu) / _B2, axis=2)             # <K_i, K_j>
            self.Z = float(self.a @ gram @ self.a)

    def _g(self, X):
        K = np.prod(_s(_B2 * (X[:, None, :] - self.mu[None, :, :])), axis=2)  # (m, J)
        return K @ self.a, K

    def pdf(self, X):
        X = np.atleast_2d(X)
        if self.kind == 'T3':
            return np.exp(self.lognorm - 0.5 * np.einsum('ij,jk,ik->i', X, self.prec, X))
        p1 = np.prod(f1(X), axis=1)
        if self.kind == 'T1':
            return p1
        g, _ = self._g(X)
        return g ** 2 / self.Z

    def sample(self, rng, n):
        if self.kind == 'T3':
            return rng.standard_normal((n, self.d)) @ self.chol.T
        if self.kind == 'T1':
            return sample_f1(rng, n * self.d).reshape(n, self.d)
        return self._sample_g2(rng, n)

    def _sample_g2(self, rng, n):
        A = np.abs(self.a).sum()
        out = np.empty((n, self.d))
        filled = 0
        while filled < n:
            m = 2 * (n - filled) + 16
            j = rng.choice(len(self.a), size=m, p=np.abs(self.a) / A)
            X = self.mu[j] + _s2_draws(rng, m * self.d).reshape(m, self.d) / _B2
            g, K = self._g(X)
            accept = rng.random(m) * A * (K ** 2 @ np.abs(self.a)) <= g ** 2
            X = X[accept][:n - filled]
            out[filled:filled + len(X)] = X
            filled += len(X)
        return out


class DefensiveProposal:
    """q = 0.5 f + 0.5 c, c = product Cauchy(0, scale). Used for importance-
    sampled ISE: (f_hat - f)^2 / q stays bounded both where f has mass and in
    the tails (f ~ |x|^-4 per coordinate, Cauchy ~ |x|^-2)."""

    def __init__(self, truth, scale=6.0):
        self.t, self.s = truth, scale

    def _cpdf(self, X):
        return np.prod(1.0 / (np.pi * self.s * (1 + (X / self.s) ** 2)), axis=1)

    def sample(self, rng, m):
        half = rng.random(m) < 0.5
        X = np.empty((m, self.t.d))
        X[half] = self.t.sample(rng, int(half.sum()))
        k = int((~half).sum())
        X[~half] = self.s * np.tan(np.pi * (rng.random((k, self.t.d)) - 0.5))
        return X

    def pdf(self, X):
        return 0.5 * self.t.pdf(X) + 0.5 * self._cpdf(X)


class NonSmooth:
    """Truths whose square root is not band-limited, with slowly decaying spectra: 'L' product Laplace (a kink at
    the mode), 'B' product Beta(2, 2) on [-3, 3] (hard edges). No true cutoff."""
    def __init__(self, kind, d):
        self.kind, self.d, self.fc_true = kind, d, None
    def sample(self, rng, n):
        return rng.laplace(size=(n, self.d)) if self.kind == 'L' else rng.beta(2, 2, size=(n, self.d)) * 6 - 3
    def pdf(self, X):
        X = np.atleast_2d(X)
        if self.kind == 'L':
            return np.exp(-np.abs(X).sum(1)) / 2 ** self.d
        u = (X + 3) / 6
        return np.prod(np.where((u > 0) & (u < 1), u * (1 - u), 0.0), axis=1)
