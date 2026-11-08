"""Transformation density estimation: estimate each marginal in one dimension, map every coordinate to a standard
normal through its estimated CDF, fit a density estimator (BLML) to the transformed points, and map back exactly:

    z_k = Phi^{-1}(F_k(x_k)),      f(x) = f_Z(z) * prod_k f_k(x_k) / phi(z_k).

The marginals carry the features a band limit cannot represent compactly (kinks, edges, heavy tails); the transformed
density carries only the dependence between coordinates, which is smooth whenever the copula is. With independent
coordinates f_Z is exactly the standard normal. This is the transformation kernel density estimator of Wand, Marron and
Ruppert (1991) with BLML in place of the kernel estimator in the transformed space.

The marginals are one-dimensional Gaussian KDEs with the bandwidth multiplier of Silverman's rule chosen by held-out
likelihood; their CDF and density are tabulated on a grid of GRID points over the data's range (plus a margin of ten
bandwidths) and interpolated, so a fit costs O(n * GRID) per coordinate. Outside the grid the marginal falls back to
the exact kernel sum."""
import numpy as np
from scipy.special import ndtr, ndtri

LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)
GRID = 4096


class Marginal:
    """A one-dimensional Gaussian KDE with its CDF, tabulated for fast evaluation."""

    def fit(self, x, mults=np.geomspace(0.25, 2.0, 12), seed=0):
        x = np.sort(np.asarray(x, float)); n = len(x)
        h0 = 0.9 * min(x.std(), (np.percentile(x, 75) - np.percentile(x, 25)) / 1.34) * n ** -0.2
        h0 = h0 if h0 > 0 else x.std() * n ** -0.2 + 1e-12
        rng = np.random.default_rng(seed); idx = rng.permutation(n); a, b = np.sort(x[idx[: int(0.8 * n)]]), x[idx[int(0.8 * n):]]
        score = [np.mean(np.log(np.maximum(self._pdf_exact(a, b, m * h0), 1e-300))) for m in mults]
        self.h = float(mults[int(np.argmax(score))] * h0); self.x = x
        lo, hi = x[0] - 10 * self.h, x[-1] + 10 * self.h
        self.grid = np.linspace(lo, hi, GRID); self.pdf_tab = self._pdf_exact(x, self.grid, self.h)
        self.cdf_tab = self._cdf_exact(x, self.grid, self.h)
        return self

    @staticmethod
    def _pdf_exact(x, q, h, chunk=2048):
        out = np.empty(len(q))
        for i in range(0, len(q), chunk):
            z = (q[i:i + chunk, None] - x[None, :]) / h
            out[i:i + chunk] = np.exp(-0.5 * z * z - LOG_SQRT_2PI).mean(1) / h
        return out

    @staticmethod
    def _cdf_exact(x, q, h, chunk=2048):
        out = np.empty(len(q))
        for i in range(0, len(q), chunk):
            out[i:i + chunk] = ndtr((q[i:i + chunk, None] - x[None, :]) / h).mean(1)
        return out

    def pdf(self, q):
        q = np.asarray(q, float); inside = (q >= self.grid[0]) & (q <= self.grid[-1]); out = np.empty(len(q))
        out[inside] = np.interp(q[inside], self.grid, self.pdf_tab)
        if (~inside).any():
            out[~inside] = self._pdf_exact(self.x, q[~inside], self.h)
        return out

    def cdf(self, q):
        q = np.asarray(q, float); inside = (q >= self.grid[0]) & (q <= self.grid[-1]); out = np.empty(len(q))
        out[inside] = np.interp(q[inside], self.grid, self.cdf_tab)
        if (~inside).any():
            out[~inside] = self._cdf_exact(self.x, q[~inside], self.h)
        return out


class GaussianizedDensity:
    """f(x) = f_Z(z) prod_k f_k(x_k) / phi(z_k), z_k = Phi^{-1}(F_k(x_k)), with f_Z fitted by `fit_inner(Z)`, a callable
    returning an object with predict(Z) (e.g. a BLML fit with its cut-off selected on Z)."""

    def __init__(self, fit_inner, eps=1e-12):
        self.fit_inner, self.eps = fit_inner, eps

    def _z(self, X):
        U = np.column_stack([m.cdf(X[:, k]) for k, m in enumerate(self.marg)])
        return ndtri(np.clip(U, self.eps, 1 - self.eps))

    def fit(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        self.marg = [Marginal().fit(X[:, k], seed=k) for k in range(X.shape[1])]
        self.inner = self.fit_inner(self._z(X))
        return self

    def logpdf(self, Xq):
        Xq = np.atleast_2d(np.asarray(Xq, float)); Z = self._z(Xq)
        log_inner = np.log(np.maximum(np.asarray(self.inner.predict(Z), float), 1e-300))
        log_marg = sum(np.log(np.maximum(m.pdf(Xq[:, k]), 1e-300)) for k, m in enumerate(self.marg))
        log_phi = (-0.5 * Z * Z - LOG_SQRT_2PI).sum(1)
        return log_inner + log_marg - log_phi

    def predict(self, Xq):
        return np.exp(self.logpdf(Xq))
