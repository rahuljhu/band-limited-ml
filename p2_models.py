"""Conditional density / quantile models for p(price | calendar covariates).

Common interface (cov = dict of covariate name -> array, y = price):
    m.fit(cov_tr, y_tr, cov_va, y_va)          (validation points for tuning)
    m.grid_pdf(cov, ygrid) -> (m, G) density on a price grid, or None
    m.logpdf(cov, y)       -> (m,) log density at the observations, or None
    m.quantiles(cov, taus) -> (m, T)   (density models derive them from the grid CDF)

Periodic covariates live on [0, P) (see caiso_data.PERIODS).
"""
import math
import time
import os
import numpy as np
from caiso_data import PERIODS

try:
    import cupy as xp
    GPU = True
except ImportError:
    xp = np
    GPU = False

def to_np(a):
    return xp.asnumpy(a) if GPU and isinstance(a, xp.ndarray) else np.asarray(a)

# Non-periodic covariates (e.g. GEFCom load forecasts): (centre, scale) set by the experiment
# from training data; they enter every baseline standardized instead of on a circle.
SCALES = {}

def std(name, v):
    c, s = SCALES[name]
    return (np.asarray(v, float) - c) / s

def features(cov):
    """Raw covariates plus sin/cos encodings (for tree/NN baselines); standardized value for
    non-periodic covariates."""
    cols = []
    for name, v in cov.items():
        if name not in PERIODS:
            cols.append(std(name, v)); continue
        ang = 2 * np.pi * v / PERIODS[name]
        cols += [v, np.sin(ang), np.cos(ang)]
    return np.stack(cols, axis=1)

def cdf_quantiles(pdf, ygrid, taus):
    """Quantiles from a density on a grid (CDF renormalized to the grid mass)."""
    dy = np.gradient(ygrid)
    F = np.cumsum(pdf * dy[None, :], axis=1)
    F = F / np.maximum(F[:, -1:], 1e-300)
    out = np.empty((pdf.shape[0], len(taus)))
    for i in range(pdf.shape[0]):
        out[i] = np.interp(taus, F[i], ygrid)
    return out


class DensityModel:
    def update(self, cov_tr, y_tr, cov_va, y_va):
        """Daily refit with the hyperparameters chosen by the last fit() (default: a full fit)."""
        return self.fit(cov_tr, y_tr, cov_va, y_va)

    def quantiles(self, cov, taus, ygrid):
        return cdf_quantiles(self.grid_pdf(cov, ygrid), ygrid, taus)

    YQ = np.arange(-150.0, 1300.25, 0.5)                 # grid for medians / sampling of closed-form densities

    def _cdf(self, cov):
        p = self.grid_pdf(cov, self.YQ)
        F = np.cumsum(p, axis=1) - 0.5 * p                  # CDF at the grid points (midpoint rule)
        return F / np.maximum(p.sum(1, keepdims=True), 1e-300)

    def _rows(self, cov, f, chunk=5000):         # apply f to row chunks of cov (bounded grid memory)
        m = len(next(iter(cov.values())))
        return np.concatenate([f({k: np.asarray(v)[i:i + chunk] for k, v in cov.items()}) for i in range(0, m, chunk)])

    def _icdf(self, F, u):
        """Row-wise inverse CDF, vectorized: F (m, G) on the grid YQ, u (m, n) -> y (m, n)."""
        m, G = F.shape
        off = np.arange(m)[:, None] * 2.0
        k = np.searchsorted((F + off).ravel(), (u + off).ravel()).reshape(u.shape) - np.arange(m)[:, None] * G
        k = np.clip(k, 1, G - 1)
        r = np.arange(m)[:, None]
        F0, F1 = F[r, k - 1], F[r, k]
        t = np.clip((u - F0) / np.maximum(F1 - F0, 1e-300), 0.0, 1.0)
        return self.YQ[k - 1] + t * (self.YQ[k] - self.YQ[k - 1])

    def median(self, cov):
        """Median of p(y | x): the point that minimizes the expected absolute error."""
        return self._rows(cov, lambda c: self._icdf(self._cdf(c), np.full((len(next(iter(c.values()))), 1), 0.5))[:, 0])

    def sample(self, cov, n, rng):
        """n draws of y per row of cov (inverse CDF on the grid)."""
        return self._rows(cov, lambda c: self._icdf(self._cdf(c), rng.random((len(next(iter(c.values()))), n))))


# ------------------------------------------------------------------ BLML
class BLML(DensityModel):
    """BLML conditional density with the shared+hillclimb (or other) selector."""
    name = 'BLML'

    def __init__(self, mode='shared_hillclimb', cond_method='ratio', rule=None, shared_step='double',
                 ytrans=None, yscales=(1.0, 3.0, 10.0, np.inf), complex_g=False, rank=1):
        self.mode, self.cond_method, self.rule, self.shared_step = mode, cond_method, rule, shared_step
        # complex_g / rank: f = sum_k |g_k|^2 with rank K (complex) band-limited g_k, for selection and
        # the final fit; the joint and the marginal are then two separate fits (no fused dep= fit).
        self.complex_g, self.rank = bool(complex_g), int(rank)
        self.psd = dict(complex_g=self.complex_g, rank=self.rank) if (complex_g or rank > 1) else {}
        # ytrans='asinh': fit BLML to z = asinh((y - m)/s), m = median and s = c * MAD of the training
        # prices, c chosen from `yscales` on the validation rows; the density maps back exactly.
        self.ytrans, self.yscales = ytrans, yscales

    def _z(self, y):
        y = np.asarray(y, float)
        if self.ytrans == 'asinh' and np.isfinite(self.ys):     # ys = inf: no transform
            u = (y - self.ym) / self.ys
            return np.arcsinh(u), -np.log(self.ys) - 0.5 * np.log1p(u * u)   # z, log dz/dy
        return y, np.zeros_like(y)

    def _X(self, cov, y):
        return np.stack([self._z(y)[0]] + [cov[c] for c in self.covs], axis=1)

    def fit(self, cov_tr, y_tr, cov_va=None, y_va=None):
        if self.ytrans == 'asinh' and len(self.yscales) > 1:
            best = None
            for c in self.yscales:
                m = BLML(self.mode, self.cond_method, self.rule, self.shared_step, 'asinh', (c,),
                         self.complex_g, self.rank)
                m._fit_selected(cov_tr, y_tr, cov_va, y_va, refit=False)
                ll = float(np.mean(m.logpdf(cov_va, y_va)))
                if best is None or ll > best[0]:
                    best = (ll, c, m.fc)
            self.yscale, self.fc_given = best[1], best[2]
        return self._fit_selected(cov_tr, y_tr, cov_va, y_va)

    def _fit_selected(self, cov_tr, y_tr, cov_va=None, y_va=None, refit=True):
        from selector import select
        from blml_trivial import make_modes, modes_fc, BLMLQuick
        engine = os.environ.get('BLML_ENGINE', 'gpu')     # 'cpu': BLMLQuick (e.g. on a laptop), same models
        if engine == 'gpu':
            import functools; from blml_trivial import BLMLQuick; Est = functools.partial(BLMLQuick, engine='gpu')
        else:
            Est = BLMLQuick
        self.covs = list(cov_tr)
        if self.ytrans == 'asinh':           # robust centre and scale from the training targets
            self.ym = float(np.median(y_tr))
            c = getattr(self, 'yscale', self.yscales[0])
            self.ys = c * (float(np.median(np.abs(np.asarray(y_tr) - self.ym))) or 1.0)
        periods = [None] + [PERIODS.get(c) for c in self.covs]
        kinds = ['sinc'] + ['dirichlet' if p else 'sinc' for p in periods[1:]]
        # Tune on the same validation rows as every other model (whole held-out
        # days), then refit on train + validation.
        Xa, Xb = self._X(cov_tr, y_tr), self._X(cov_va, y_va)
        X = np.concatenate([Xa, Xb])
        if getattr(self, 'fc_given', None) is not None:    # chosen with the target scale
            self.fc = list(self.fc_given)
        elif self.rule == 'inv_sigma':                     # base paper's fc = 1/sigma rule
            self.fc = list(1.0 / X.std(0))
        else:
            t0 = time.time()
            tmpl = make_modes(np.nan, kinds, periods, rank=self.psd.get('rank', 1), dep=0)
            sel, _ = select(Xa, tmpl, search=self.mode, seed=0, engine=engine, cond_method=self.cond_method, X_val=Xb,
                            shared_step=self.shared_step, complex_g=self.psd.get('complex_g', False))
            self.fc = [float(f) for f in modes_fc(sel)[0]]
            self.select_s = time.time() - t0
        Xf = X if refit else Xa
        t0 = time.time()
        # conditional = joint / marginal of the covariates (same cutoffs, one fused fit): O(B) per point
        if self.psd:                     # separate joint and marginal fits
            K, cg = self.psd.get('rank', 1), self.psd.get('complex_g', False)
            self.m = Est(seed=0).fit(Xf, make_modes(self.fc, kinds, periods, rank=K), complex_g=cg)
            self.mm = Est(seed=0).fit(Xf[:, 1:], make_modes(self.fc[1:], kinds[1:], periods[1:], rank=K), complex_g=cg)
        else:
            self.m = Est(seed=0).fit(Xf, make_modes(self.fc, kinds, periods, dep=0))
        self.final_fit_s = time.time() - t0
        return self

    def logpdf(self, cov, y):
        X = self._X(cov, y)
        if self.psd:
            return (np.log(np.maximum(self.m.predict(X), 1e-300)) - np.log(np.maximum(self.mm.predict(X[:, 1:]), 1e-300))
                    + self._z(y)[1])
        return np.log(np.maximum(self.m.predict_ratio(X), 1e-300)) + self._z(y)[1]

    def sample(self, cov, n, rng=None):
        """Draws from the BLML conditional (importance resampling, no grid)."""
        Xrest = np.stack([np.asarray(cov[c], float) for c in self.covs], axis=1)
        z, self.ess_frac = self.m.sample_conditional(Xrest, n_samples=n,
                                                         seed=0 if rng is None else int(rng.integers(1 << 31)))
        return z[:, :, 0]                                  # no transform: z = y

    def median(self, cov, n=100):
        return np.median(self.sample(cov, n), axis=1)

    def grid_pdf(self, cov, ygrid):
        z, logjac = self._z(ygrid)
        Xrest = np.stack([np.asarray(cov[c], float) for c in self.covs], axis=1)
        m, G = len(Xrest), len(z)
        pts = np.column_stack([np.tile(z, m), np.repeat(Xrest, G, axis=0)])
        joint = self.m.predict(pts).reshape(m, G)
        marg = self.mm.predict(Xrest) if self.psd else self.m.predict_marginal(Xrest)
        return joint / np.maximum(marg, 1e-300)[:, None] * np.exp(logjac)[None, :]


# ------------------------------------------------------------------ change from yesterday
class BLMLStream(DensityModel):
    """Streaming BLML (blml_rolling.BLMLQuickStreaming): the training rows (train + validation, the data the
    one-shot BLML sees) arrive in batches of whole days, in a random day order (split_groups = day), so that
    every prequential batch holds days the model has not seen (rows of one day are strongly correlated;
    batches that split days would reward memorizing them), and the burn-in selection validates on whole days.
    Burn-in: the first burn_in rows get the selector's 'auto' search and an exact fit; then each batch updates
    the bins it touches and Adam steps the cut-offs (analytic gradient of the conditional log-likelihood of
    each new batch). Later epochs continue from the current fit (revisited days scored leave-batch-out).
    Day boundaries come from the hour column."""
    name = 'BLML-stream'

    def __init__(self, batch_days=None, burn_in=5000, param_every=None, seed=None, rank=1, fc_init=None, freeze=()):
        env = os.environ.get
        # rank K: f = sum_k g_k^2, each mode its own cut-offs; fc_init (K x d) starts them apart (no burn-in search);
        # freeze: covariate names whose cut-off stays at fc_init in every mode
        self.rank, self.fc_init, self.freeze = int(rank), fc_init, tuple(freeze)
        self.epoch_eval, self.epoch_ll = None, []    # epoch_eval = (cov, y): test LL after every epoch
        self.batch_days = int(batch_days or env('BLML_STREAM_BATCH_DAYS', 42))
        # pool each fc step over ~1000 rows (~42 days) whatever the batch size
        self.param_every = int(param_every or env('BLML_STREAM_PARAM_EVERY', max(1, round(126 / self.batch_days))))
        self.seed = int(seed if seed is not None else env('BLML_STREAM_SEED', 0))
        self.epochs = int(env('BLML_STREAM_EPOCHS', 6))
        self.burn_in = burn_in

    def _X(self, cov, y):
        return np.stack([np.asarray(y, float)] + [np.asarray(cov[c], float) for c in self.covs], axis=1)

    def fit(self, cov_tr, y_tr, cov_va=None, y_va=None):
        from blml_trivial import make_modes, modes_fc
        from blml_rolling import BLMLQuickStreaming
        self.covs = list(cov_tr)
        periods = [None] + [PERIODS.get(c) for c in self.covs]
        kinds = ['sinc'] + ['dirichlet' if p else 'sinc' for p in periods[1:]]
        parts = [(cov_tr, y_tr)] + ([(cov_va, y_va)] if cov_va is not None else [])
        Xs, days, off = [], [], 0
        for cov, y in parts:                     # day id: the hour wraps around at each new day
            hr = np.asarray(cov['hour'], float)
            day = np.r_[0, np.cumsum(np.diff(hr) < 0)] + off
            off = day.max() + 1
            Xs.append(self._X(cov, y)); days.append(day)
        X, day = np.concatenate(Xs), np.concatenate(days)
        t0 = time.time()
        self.s = BLMLQuickStreaming(engine=os.environ.get('BLML_ENGINE', 'gpu'), search='auto', burn_in=self.burn_in,
                                    param_every=self.param_every, seed=self.seed)
        batch = int(round(self.batch_days * len(X) / (day.max() + 1)))     # = batch_days whole days per batch
        fc0 = np.nan if self.fc_init is None else self.fc_init
        modes = make_modes(fc0, kinds, periods, rank=self.rank, dep=0)
        for md in modes:
            for c in self.freeze:
                j = 1 + self.covs.index(c)
                md[j]['fc'] = dict(value=md[j]['fc'], trainable=False)

        def on_epoch(ep, s):
            if self.epoch_eval is None:
                return
            ll = float(np.mean(self.logpdf(*self.epoch_eval)))
            self.epoch_ll.append(dict(epoch=ep + 1, test_ll=round(ll, 5), elapsed_s=round(time.time() - t0, 1),
                                      fc=[[round(float(f), 4) for f in m] for m in modes_fc(s.modes_)]))
            print(f'    epoch {ep + 1}: test LL {ll:.4f} ({time.time() - t0:.0f}s) fc {self.epoch_ll[-1]["fc"]}', flush=True)
        self.s.fit(X, modes, epochs=self.epochs, batch=batch, split_groups=day, on_epoch=on_epoch)
        self.fit_stream_s = time.time() - t0
        self.fc = [float(f) for f in modes_fc(self.s.modes_)[0]]
        self.fc_modes = [[float(f) for f in m] for m in modes_fc(self.s.modes_)]
        return self

    def logpdf(self, cov, y):
        return np.log(np.maximum(self.s.predict_ratio(self._X(cov, y)), 1e-300))


class Diff(DensityModel):
    """Wraps a density model to fit z = y - lag24 given the same covariates; p(y|x) = p_z(y - lag24 | x).
    Removes slow level drift, so a joint-density model can carry the price relation to new levels."""
    ZGRID = np.arange(-1300.0, 1300.5, 1.0)

    def __init__(self, base):
        self.base = base

    def fit(self, cov_tr, y_tr, cov_va, y_va):
        self.base.fit(cov_tr, np.asarray(y_tr) - cov_tr['lag24'], cov_va, np.asarray(y_va) - cov_va['lag24'])
        for attr in ('fc', 'K', 'select_s', 'final_fit_s'):
            if hasattr(self.base, attr):
                setattr(self, attr, getattr(self.base, attr))
        return self

    def logpdf(self, cov, y):
        return self.base.logpdf(cov, np.asarray(y) - cov['lag24'])

    def median(self, cov):
        return self.base.median(cov) + np.asarray(cov['lag24'])

    def sample(self, cov, n, rng):
        return self.base.sample(cov, n, rng) + np.asarray(cov['lag24'])[:, None]

    def grid_pdf(self, cov, ygrid):
        pz = self.base.grid_pdf(cov, self.ZGRID)
        return np.stack([np.interp(ygrid - l, self.ZGRID, row, left=0.0, right=0.0)
                         for l, row in zip(cov['lag24'], pz)])


# ------------------------------------------------------------------ conditional KDE
class CondKDE(DensityModel):
    """Nadaraya-Watson conditional KDE: von Mises kernels on periodic covariates,
    Gaussian kernel in price; bandwidths by coordinate ascent on validation LL."""
    name = 'CondKDE'

    def _w(self, cov, kappa):
        logw = 0
        for c, k in zip(self.covs, kappa):
            if c not in PERIODS:              # Gaussian kernel on the standardized covariate
                dz = (xp.asarray(cov[c])[:, None] - self.ctr[c][None, :]) / SCALES[c][1]
                logw = logw - 0.5 * k * dz * dz; continue
            ang = 2 * math.pi * (xp.asarray(cov[c])[:, None] - self.ctr[c][None, :]) / PERIODS[c]
            logw = logw + k * (xp.cos(ang) - 1.0)
        return xp.exp(logw - logw.max(axis=1, keepdims=True))

    def _logpdf(self, cov, y, kappa, hy):
        W = self._w(cov, kappa)
        z = (xp.asarray(y)[:, None] - self.ytr[None, :]) / hy
        num = (W * xp.exp(-0.5 * z * z)).sum(1) / (hy * math.sqrt(2 * math.pi))
        return xp.log(xp.maximum(num / W.sum(1), 1e-300))

    def fit(self, cov_tr, y_tr, cov_va, y_va):
        self.covs = list(cov_tr)
        self.ctr = {c: xp.asarray(cov_tr[c]) for c in self.covs}
        self.ytr = xp.asarray(y_tr)
        kgrid = np.geomspace(0.3, 3000, 18)
        hgrid = np.std(y_tr) * len(y_tr) ** -0.2 * np.geomspace(0.03, 2, 14)
        kappa, hy = [10.0] * len(self.covs), hgrid[6]
        score = lambda kap, h: float(to_np(self._logpdf(cov_va, y_va, kap, h)).mean())
        best = score(kappa, hy)
        for _ in range(3):
            for i in range(len(self.covs)):
                for k in kgrid:
                    kap = list(kappa); kap[i] = k
                    s = score(kap, hy)
                    if s > best:
                        best, kappa = s, kap
            for h in hgrid:
                s = score(kappa, h)
                if s > best:
                    best, hy = s, h
        self.kappa, self.hy = kappa, hy
        # Refit on train + validation with the chosen bandwidths.
        self.ctr = {c: xp.asarray(np.concatenate([cov_tr[c], cov_va[c]])) for c in self.covs}
        self.ytr = xp.asarray(np.concatenate([y_tr, y_va]))
        return self

    def logpdf(self, cov, y):
        return to_np(self._logpdf(cov, y, self.kappa, self.hy))

    def grid_pdf(self, cov, ygrid):
        W = self._w(cov, self.kappa)
        W = W / W.sum(1, keepdims=True)
        z = (xp.asarray(ygrid)[None, :] - self.ytr[:, None]) / self.hy
        Phi = xp.exp(-0.5 * z * z) / (self.hy * math.sqrt(2 * math.pi))
        return to_np(W @ Phi)


# ------------------------------------------------------------------ conditional k-NN
class CondKNN(DensityModel):
    """k nearest neighbors in (circular, period-normalized) covariate space, then a
    Gaussian KDE of their prices; k and the price bandwidth tuned on validation."""
    name = 'CondKNN'

    def _nbrs(self, cov, k):
        d2 = 0
        for c in self.covs:
            if c not in PERIODS:              # standardized distance, ~ same range as circular
                dd = xp.abs(xp.asarray(cov[c])[:, None] - self.ctr[c][None, :]) / (4 * SCALES[c][1])
            else:
                dd = xp.abs(xp.asarray(cov[c])[:, None] - self.ctr[c][None, :]) / PERIODS[c]
                dd = xp.minimum(dd, 1.0 - dd)
            d2 = d2 + dd * dd
        idx = xp.argpartition(d2, k - 1, axis=1)[:, :k]
        return self.ytr[idx]                                  # (m, k)

    def _logpdf(self, cov, y, k, c):
        Y = self._nbrs(cov, k)
        h = c * Y.std(1, keepdims=True) * k ** -0.2 + 1e-6
        z = (xp.asarray(y)[:, None] - Y) / h
        return xp.log(xp.maximum((xp.exp(-0.5 * z * z) / (h * math.sqrt(2 * math.pi))).mean(1), 1e-300))

    def fit(self, cov_tr, y_tr, cov_va, y_va):
        self.covs = list(cov_tr)
        self.ctr = {c: xp.asarray(cov_tr[c]) for c in self.covs}
        self.ytr = xp.asarray(y_tr)
        best = (-np.inf, None)
        for k in [10, 25, 50, 100, 200, 400]:
            for c in [0.25, 0.5, 1.0, 1.5]:
                s = float(to_np(self._logpdf(cov_va, y_va, k, c)).mean())
                if s > best[0]:
                    best = (s, (k, c))
        self.k, self.c = best[1]
        self.ctr = {c: xp.asarray(np.concatenate([cov_tr[c], cov_va[c]])) for c in self.covs}
        self.ytr = xp.asarray(np.concatenate([y_tr, y_va]))
        return self

    def logpdf(self, cov, y):
        return to_np(self._logpdf(cov, y, self.k, self.c))

    def grid_pdf(self, cov, ygrid):
        Y = self._nbrs(cov, self.k)
        h = self.c * Y.std(1, keepdims=True) * self.k ** -0.2 + 1e-6
        out = xp.zeros((Y.shape[0], len(ygrid)))
        g = xp.asarray(ygrid)
        for j in range(Y.shape[1]):
            z = (g[None, :] - Y[:, j:j+1]) / h
            out += xp.exp(-0.5 * z * z) / (h * math.sqrt(2 * math.pi))
        return to_np(out / Y.shape[1])


# ------------------------------------------------------------------ GMM
class CondGMM(DensityModel):
    """Gaussian mixture on (price, sin/cos covariates); analytic conditional in price.
    Number of components chosen by validation conditional LL."""
    name = 'GMM'

    def _z(self, cov):
        cols = []
        for c in self.covs:
            if c not in PERIODS:
                cols.append(std(c, cov[c])); continue
            ang = 2 * np.pi * cov[c] / PERIODS[c]
            cols += [np.sin(ang), np.cos(ang)]
        return np.stack(cols, axis=1)

    def _cond(self, g, Z):
        """Per-row mixture over price: weights (m,K), means (m,K), sds (K,)."""
        mu, S, pi = g.means_, g.covariances_, g.weights_
        K = len(pi)
        logw = np.empty((len(Z), K)); cm = np.empty((len(Z), K)); sd = np.empty(K)
        for k in range(K):
            Szz, Syz = S[k][1:, 1:], S[k][0, 1:]
            Sinv = np.linalg.inv(Szz)
            dz = Z - mu[k, 1:]
            _, logdet = np.linalg.slogdet(Szz)
            logw[:, k] = np.log(pi[k]) - 0.5 * (np.einsum('ij,jk,ik->i', dz, Sinv, dz) + logdet)
            cm[:, k] = mu[k, 0] + dz @ (Sinv @ Syz)
            sd[k] = math.sqrt(max(S[k][0, 0] - Syz @ Sinv @ Syz, 1e-12))
        w = np.exp(logw - logw.max(1, keepdims=True))
        return w / w.sum(1, keepdims=True), cm, sd

    def _logpdf(self, g, cov, y):
        w, cm, sd = self._cond(g, self._z(cov))
        dens = (w * np.exp(-0.5 * ((y[:, None] - cm) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))).sum(1)
        return np.log(np.maximum(dens, 1e-300))

    def fit(self, cov_tr, y_tr, cov_va, y_va):
        from sklearn.mixture import GaussianMixture
        self.covs = list(cov_tr)
        A = np.column_stack([y_tr, self._z(cov_tr)])
        best = (-np.inf, None)
        for K in [2, 4, 8, 12, 16, 24, 32]:
            g = GaussianMixture(K, covariance_type='full', reg_covar=1e-4, random_state=0, max_iter=500).fit(A)
            s = self._logpdf(g, cov_va, y_va).mean()
            if s > best[0]:
                best = (s, K)
        self.K = best[1]
        A = np.column_stack([np.concatenate([y_tr, y_va]),
                             self._z({c: np.concatenate([cov_tr[c], cov_va[c]]) for c in self.covs})])
        self.g = GaussianMixture(self.K, covariance_type='full', reg_covar=1e-4, random_state=0, max_iter=500).fit(A)
        return self

    def update(self, cov_tr, y_tr, cov_va, y_va):
        """Daily refit: one EM run at the month's K, started from the current mixture."""
        from sklearn.mixture import GaussianMixture
        A = np.column_stack([np.concatenate([y_tr, y_va]),
                             self._z({c: np.concatenate([cov_tr[c], cov_va[c]]) for c in self.covs})])
        g = self.g
        self.g = GaussianMixture(self.K, covariance_type='full', reg_covar=1e-4, random_state=0, max_iter=500,
                                 weights_init=g.weights_, means_init=g.means_,
                                 precisions_init=g.precisions_).fit(A)
        return self

    def logpdf(self, cov, y):
        return self._logpdf(self.g, cov, y)

    def grid_pdf(self, cov, ygrid):
        w, cm, sd = self._cond(self.g, self._z(cov))
        out = np.zeros((len(w), len(ygrid)))
        for k in range(len(sd)):
            out += w[:, k:k+1] * np.exp(-0.5 * ((ygrid[None, :] - cm[:, k:k+1]) / sd[k]) ** 2) / (sd[k] * math.sqrt(2 * math.pi))
        return out


# ------------------------------------------------------------------ climatology / cell models
class Climatology(DensityModel):
    """Empirical price distribution per hour x month cell (Gaussian KDE in price,
    Silverman bandwidth); cells with < 20 points fall back to the hour alone."""
    name = 'Climatology'

    def __init__(self, family='kde'):
        self.family = family
        self.use_month = True       # False for the hour-only covariate set
        self.name = {'kde': 'Climatology', 'normal': 'Normal(hour×month)', 't': 'Student-t(hour×month)'}[family]

    def _cell(self, extra):
        month = np.floor(extra['moy']).astype(int) % 12 if self.use_month else 0
        return np.floor(extra['hour']).astype(int) * 12 + month

    def fit(self, cov_tr, y_tr, cov_va, y_va, extra_tr=None, extra_va=None):
        from scipy import stats
        y = np.concatenate([y_tr, y_va])
        cell = np.concatenate([self._cell(extra_tr), self._cell(extra_va)])
        hour = cell // 12
        self.params = {}
        for c in range(24 * 12):
            yc = y[cell == c]
            if len(yc) < 20:
                yc = y[hour == c // 12]
            if self.family == 'kde':
                self.params[c] = (yc, 1.06 * yc.std() * len(yc) ** -0.2 + 1e-6)
            elif self.family == 'normal':
                self.params[c] = (yc.mean(), yc.std() + 1e-6)
            else:
                self.params[c] = stats.t.fit(yc)
        return self

    def _pdf_row(self, c, yv):
        from scipy import stats
        p = self.params[c]
        if self.family == 'kde':
            ys, h = p
            return (np.exp(-0.5 * ((yv[:, None] - ys[None, :]) / h) ** 2).sum(1)
                    / (len(ys) * h * math.sqrt(2 * math.pi)))
        if self.family == 'normal':
            return stats.norm.pdf(yv, *p)
        return stats.t.pdf(yv, *p)

    def logpdf(self, cov, y, extra=None):
        cell = self._cell(extra)
        out = np.empty(len(y))
        for c in np.unique(cell):
            sel = cell == c
            out[sel] = np.log(np.maximum(self._pdf_row(c, y[sel]), 1e-300))
        return out

    def grid_pdf(self, cov, ygrid, extra=None):
        cell = self._cell(extra)
        rows = {c: self._pdf_row(c, ygrid) for c in np.unique(cell)}
        return np.stack([rows[c] for c in cell])

    def quantiles(self, cov, taus, ygrid, extra=None):
        return cdf_quantiles(self.grid_pdf(cov, ygrid, extra), ygrid, taus)


# ------------------------------------------------------------------ parametric predictive (NGBoost, CatBoost)
class NGBoost(DensityModel):
    def __init__(self, dist='Normal'):
        self.dist = dist
        self.name = f'NGBoost-{dist}'

    def fit(self, cov_tr, y_tr, cov_va, y_va):
        from ngboost import NGBRegressor
        from ngboost import distns
        D = getattr(distns, self.dist)
        self.m = NGBRegressor(Dist=D, n_estimators=1500, learning_rate=0.02, verbose=False,
                              random_state=0)
        self.m.fit(features(cov_tr), y_tr, X_val=features(cov_va), Y_val=y_va, early_stopping_rounds=50)
        best = getattr(self.m, 'best_val_loss_itr', None)
        self.n_rounds = int(best) + 1 if best is not None else len(self.m.base_models)
        return self

    def update(self, cov_tr, y_tr, cov_va, y_va):
        """Daily refit with the month's number of boosting rounds (no early-stopping search)."""
        from ngboost import NGBRegressor
        from ngboost import distns
        self.m = NGBRegressor(Dist=getattr(distns, self.dist), n_estimators=self.n_rounds, learning_rate=0.02,
                              verbose=False, random_state=0)
        self.m.fit(features(cov_tr), y_tr)
        return self

    def logpdf(self, cov, y):
        return self.m.pred_dist(features(cov)).logpdf(y)

    def grid_pdf(self, cov, ygrid):
        dist = self.m.pred_dist(features(cov))
        return np.stack([np.exp(dist.logpdf(np.full(len(cov[next(iter(cov))]), g))) for g in ygrid], axis=1)


class XGBoostLSS(DensityModel):
    """XGBoostLSS (distributional gradient boosting, GPU) with a Gaussian predictive. The target is
    standardized for training (unstandardized it fails to train); densities are mapped back with 1/sd."""
    name = 'XGBoostLSS'

    def _dm(self, cov, y=None):
        import xgboost as xgb
        return xgb.DMatrix(features(cov), label=None if y is None else (np.asarray(y) - self.ym) / self.ys)

    def _train(self, dtr, rounds, evals=None):
        from xgboostlss.model import XGBoostLSS as LSS
        from xgboostlss.distributions.Gaussian import Gaussian
        self.m = LSS(Gaussian())
        params = dict(eta=0.03, max_depth=4, device='cuda', tree_method='hist')
        self.m.train(params, dtr, num_boost_round=rounds, evals=evals,
                     early_stopping_rounds=50 if evals else None, verbose_eval=False)

    def fit(self, cov_tr, y_tr, cov_va, y_va):
        self.ym, self.ys = float(np.median(y_tr)), float(np.std(y_tr))
        dtr = self._dm(cov_tr, y_tr)
        self._train(dtr, 2000, evals=[(dtr, 'train'), (self._dm(cov_va, y_va), 'val')])
        self.n_rounds = int(self.m.booster.best_iteration) + 1
        return self

    def update(self, cov_tr, y_tr, cov_va, y_va):
        """Daily refit with the month's number of boosting rounds (no early-stopping search)."""
        self._train(self._dm(cov_tr, y_tr), self.n_rounds)
        return self

    def _ms(self, cov):
        P = self.m.predict(self._dm(cov), pred_type='parameters')
        return P['loc'].values * self.ys + self.ym, P['scale'].values * self.ys

    def logpdf(self, cov, y):
        mu, sd = self._ms(cov)
        return -0.5 * ((y - mu) / sd) ** 2 - np.log(sd * math.sqrt(2 * math.pi))

    def grid_pdf(self, cov, ygrid):
        mu, sd = self._ms(cov)
        return np.exp(-0.5 * ((ygrid[None, :] - mu[:, None]) / sd[:, None]) ** 2) / (sd[:, None] * math.sqrt(2 * math.pi))

    def median(self, cov):
        return self._ms(cov)[0]

    def sample(self, cov, n, rng):
        mu, sd = self._ms(cov)
        return mu[:, None] + sd[:, None] * rng.standard_normal((len(mu), n))


class CatBoostUnc(DensityModel):
    name = 'CatBoost-Normal'

    def fit(self, cov_tr, y_tr, cov_va, y_va):
        from catboost import CatBoostRegressor
        self.m = CatBoostRegressor(loss_function='RMSEWithUncertainty', iterations=3000, learning_rate=0.03,
                                   depth=6, random_seed=0, verbose=False, od_type='Iter', od_wait=100)
        self.m.fit(features(cov_tr), y_tr, eval_set=(features(cov_va), y_va), use_best_model=True)
        return self

    def _ms(self, cov):
        p = np.asarray(self.m.predict(features(cov), prediction_type='RMSEWithUncertainty'))
        return p[:, 0], np.sqrt(np.maximum(p[:, 1], 1e-12))

    def logpdf(self, cov, y):
        mu, sd = self._ms(cov)
        return -0.5 * ((y - mu) / sd) ** 2 - np.log(sd * math.sqrt(2 * math.pi))

    def grid_pdf(self, cov, ygrid):
        mu, sd = self._ms(cov)
        return np.exp(-0.5 * ((ygrid[None, :] - mu[:, None]) / sd[:, None]) ** 2) / (sd[:, None] * math.sqrt(2 * math.pi))


# ------------------------------------------------------------------ mixture density network
class MDN(DensityModel):
    name = 'MDN'

    def __init__(self, K=8, hidden=128):
        self.K, self.hidden = K, hidden

    def _net(self, din):
        import torch.nn as nn
        return nn.Sequential(nn.Linear(din, self.hidden), nn.SiLU(), nn.Linear(self.hidden, self.hidden),
                             nn.SiLU(), nn.Linear(self.hidden, 3 * self.K))

    def _params(self, Fx):
        import torch
        out = self.net(Fx)
        logit, mu, logsd = out[:, :self.K], out[:, self.K:2*self.K], out[:, 2*self.K:]
        return torch.log_softmax(logit, 1), mu, logsd.clamp(-7, 5)

    def _lp(self, Fx, ys):
        import torch
        lw, mu, ls = self._params(Fx)
        z = (ys[:, None] - mu) / ls.exp()
        return torch.logsumexp(lw - 0.5 * z * z - ls - 0.5 * math.log(2 * math.pi), 1)

    def fit(self, cov_tr, y_tr, cov_va, y_va):
        import torch
        torch.manual_seed(0)
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        Ftr, Fva = features(cov_tr), features(cov_va)
        self.fm, self.fs = Ftr.mean(0), Ftr.std(0) + 1e-9
        self.ym, self.ys = y_tr.mean(), y_tr.std()
        T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
        Xt, Yt = T((Ftr - self.fm) / self.fs), T((y_tr - self.ym) / self.ys)
        Xv, Yv = T((Fva - self.fm) / self.fs), T((y_va - self.ym) / self.ys)
        self.net = self._net(Xt.shape[1]).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        best, best_state, bad = np.inf, None, 0
        for epoch in range(400):
            perm = torch.randperm(len(Xt), device=dev)
            self.net.train()
            for i in range(0, len(Xt), 256):
                b = perm[i:i+256]
                loss = -self._lp(Xt[b], Yt[b]).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                v = -self._lp(Xv, Yv).mean().item()
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= 25:
                    break
        self.net.load_state_dict(best_state)
        self.dev = dev
        return self

    def update(self, cov_tr, y_tr, cov_va, y_va, max_epochs=20, patience=3):
        """Daily refit: continue training from the current weights (month's input/output
        scaling kept), early-stopped on the validation days."""
        import torch
        dev = self.dev
        T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
        Xt, Yt = T((features(cov_tr) - self.fm) / self.fs), T((y_tr - self.ym) / self.ys)
        Xv, Yv = T((features(cov_va) - self.fm) / self.fs), T((y_va - self.ym) / self.ys)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.net.eval()
        with torch.no_grad():
            best = -self._lp(Xv, Yv).mean().item()
        best_state, bad = {k: t.clone() for k, t in self.net.state_dict().items()}, 0
        for epoch in range(max_epochs):
            perm = torch.randperm(len(Xt), device=dev)
            self.net.train()
            for i in range(0, len(Xt), 256):
                b = perm[i:i+256]
                loss = -self._lp(Xt[b], Yt[b]).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                v = -self._lp(Xv, Yv).mean().item()
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    def logpdf(self, cov, y):
        import torch
        Fx = torch.tensor((features(cov) - self.fm) / self.fs, dtype=torch.float32, device=self.dev)
        ys = torch.tensor((y - self.ym) / self.ys, dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            return self._lp(Fx, ys).cpu().numpy().astype(float) - math.log(self.ys)

    def grid_pdf(self, cov, ygrid):
        import torch
        Fx = torch.tensor((features(cov) - self.fm) / self.fs, dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            lw, mu, ls = self._params(Fx)
            g = torch.tensor((ygrid - self.ym) / self.ys, dtype=torch.float32, device=self.dev)
            z = (g[None, :, None] - mu[:, None, :]) / ls.exp()[:, None, :]
            lp = torch.logsumexp(lw[:, None, :] - 0.5 * z * z - ls[:, None, :] - 0.5 * math.log(2 * math.pi), 2)
        return np.exp(lp.cpu().numpy().astype(float)) / self.ys


# ------------------------------------------------------------------ neural spline flow
class NSF(DensityModel):
    """Conditional neural spline flow (zuko NSF) for p(y | covariates): one feature, the covariate features as
    context, standardised; 5 spline transforms, two hidden layers of 128, Adam, early stopping on the validation
    rows (as the MDN)."""
    name = 'NSF'

    def fit(self, cov_tr, y_tr, cov_va, y_va):
        import torch, zuko
        torch.manual_seed(0); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        Ftr, Fva = features(cov_tr), features(cov_va)
        self.fm, self.fs = Ftr.mean(0), Ftr.std(0) + 1e-9
        self.ym, self.ys = y_tr.mean(), y_tr.std()
        T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
        Xt, Yt = T((Ftr - self.fm) / self.fs), T((y_tr - self.ym) / self.ys)[:, None]
        Xv, Yv = T((Fva - self.fm) / self.fs), T((y_va - self.ym) / self.ys)[:, None]
        self.flow = zuko.flows.NSF(features=1, context=Xt.shape[1], transforms=5, hidden_features=[128, 128]).to(dev)
        opt = torch.optim.Adam(self.flow.parameters(), lr=1e-3); best, best_state, bad = np.inf, None, 0
        for epoch in range(400):
            perm = torch.randperm(len(Xt), device=dev)
            for i in range(0, len(Xt), 256):
                b = perm[i:i + 256]; loss = -self.flow(Xt[b]).log_prob(Yt[b]).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                v = -self.flow(Xv).log_prob(Yv).mean().item()
            if v < best - 1e-4:
                best, bad = v, 0; best_state = {k: t.clone() for k, t in self.flow.state_dict().items()}
            else:
                bad += 1
                if bad >= 25:
                    break
        self.flow.load_state_dict(best_state); self.dev = dev
        return self

    def logpdf(self, cov, y):
        import torch
        Fx = torch.tensor((features(cov) - self.fm) / self.fs, dtype=torch.float32, device=self.dev)
        ys = torch.tensor((y - self.ym) / self.ys, dtype=torch.float32, device=self.dev)[:, None]
        with torch.no_grad():
            return self.flow(Fx).log_prob(ys).cpu().numpy().astype(float) - math.log(self.ys)

    def grid_pdf(self, cov, ygrid):
        import torch
        Fx = torch.tensor((features(cov) - self.fm) / self.fs, dtype=torch.float32, device=self.dev)
        g = torch.tensor((ygrid - self.ym) / self.ys, dtype=torch.float32, device=self.dev)
        out = np.empty((len(Fx), len(g)))
        with torch.no_grad():
            for i in range(len(Fx)):                       # one conditional density per row, on the whole grid
                out[i] = self.flow(Fx[i:i + 1].expand(len(g), -1)).log_prob(g[:, None]).cpu().numpy()
        return np.exp(out) / self.ys


# ------------------------------------------------------------------ TabPFN
class TabPFN(DensityModel):
    name = 'TabPFN'

    def fit(self, cov_tr, y_tr, cov_va, y_va):
        import tabpfn
        from tabpfn import TabPFNRegressor
        self.tabpfn_version = getattr(tabpfn, '__version__', 'unknown')   # default checkpoint follows the package
        F = np.concatenate([features(cov_tr), features(cov_va)])
        y = np.concatenate([y_tr, y_va])
        if len(y) > 10000:                       # TabPFN's supported context size
            keep = np.random.default_rng(0).choice(len(y), 10000, replace=False)
            F, y = F[keep], y[keep]
        if getattr(self, 'm', None) is None:          # load the pretrained model once
            self.m = TabPFNRegressor(random_state=0, ignore_pretraining_limits=True)
        self.m.fit(F, y)                              # new context; same model object
        return self

    def _full(self, cov, chunk=8192):            # batched: bounded GPU memory on large test sets
        import torch
        F = features(cov); logits = []
        for i in range(0, len(F), chunk):
            out = self.m.predict(F[i:i + chunk], output_type='full')
            logits.append(out['logits'])
        return torch.cat(logits), out['criterion']

    def logpdf(self, cov, y):
        import torch
        logits, crit = self._full(cov)
        # The bar-distribution criterion indexes into y with its own borders, so y must have the
        # borders' dtype/device (float64 on current TabPFN), not the logits'.
        yt = torch.as_tensor(y, dtype=crit.borders.dtype, device=crit.borders.device)
        return (-crit(logits, yt)).detach().cpu().numpy().astype(float)

    def median(self, cov, chunk=8192):
        F = features(cov)
        return np.concatenate([np.asarray(self.m.predict(F[i:i + chunk], output_type='median'), float)
                               for i in range(0, len(F), chunk)])

    def sample(self, cov, n, rng):
        """Draws from the bar distribution: a bin by its probability, uniform within the bin."""
        import torch
        def draw(c_):
            logits, crit = self._full(c_)
            bo = crit.borders.detach().cpu().numpy().astype(float)
            p = torch.softmax(logits.float(), -1).detach().cpu().numpy().astype(float)
            c = np.cumsum(p, 1); c /= c[:, -1:]
            k = np.minimum(np.array([np.searchsorted(ci, ui) for ci, ui in zip(c, rng.random((len(p), n)))]), len(bo) - 2)
            return bo[k] + (bo[k + 1] - bo[k]) * rng.random(k.shape)
        return self._rows(cov, draw, chunk=20000)

    def grid_pdf(self, cov, ygrid):
        import torch
        logits, crit = self._full(cov)
        cols = []
        for g in ygrid:
            yt = torch.full((logits.shape[0],), float(g), dtype=crit.borders.dtype, device=crit.borders.device)
            cols.append(torch.exp(-crit(logits, yt)).detach().cpu().numpy())
        return np.stack(cols, axis=1).astype(float)


# ------------------------------------------------------------------ quantile models
class QuantileModel:
    """Base for models that only give quantiles (no density, no NLL)."""
    def grid_pdf(self, cov, ygrid):
        return None
    def logpdf(self, cov, y):
        return None


class LGBMQuantile(QuantileModel):
    name = 'LightGBM-quantile'

    def fit(self, cov_tr, y_tr, cov_va, y_va, taus=None):
        import lightgbm as lgb
        self.taus = taus
        self.models = []
        for t in taus:
            m = lgb.LGBMRegressor(objective='quantile', alpha=float(t), n_estimators=2000, learning_rate=0.03,
                                  num_leaves=31, min_child_samples=20, verbose=-1, random_state=0)
            m.fit(features(cov_tr), y_tr, eval_set=[(features(cov_va), y_va)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            self.models.append(m)
        return self

    def quantiles(self, cov, taus, ygrid=None):
        Q = np.stack([m.predict(features(cov)) for m in self.models], axis=1)
        return np.sort(Q, axis=1)


class QRF(QuantileModel):
    name = 'QuantileForest'

    def fit(self, cov_tr, y_tr, cov_va, y_va, taus=None):
        from quantile_forest import RandomForestQuantileRegressor
        F = np.concatenate([features(cov_tr), features(cov_va)])
        y = np.concatenate([y_tr, y_va])
        self.m = RandomForestQuantileRegressor(n_estimators=300, min_samples_leaf=10, random_state=0, n_jobs=-1).fit(F, y)
        return self

    def quantiles(self, cov, taus, ygrid=None):
        return np.sort(np.asarray(self.m.predict(features(cov), quantiles=list(taus))), axis=1)
