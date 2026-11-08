"""Rolling and streaming BLML.

BLMLRolling: one estimator refitted (to tol) on successive data windows (sliding or expanding), CPU or
GPU; the caller passes each window and every fit runs to tol (accuracy at every step). BLMLQuickStreaming
(raw points, or counts on a fixed lattice: h in the modes): online learning for speed -- each batch
updates only the bins it touches, the parameters move by Adam (or a hill-climb) on the prequential score,
and revisiting the data is controlled by epochs (see _StreamingBase).

    r = BLMLRolling(engine='cpu')
    r.fit(X_train, make_modes(np.nan, kinds, periods, dep=0), X_val)            # full fc selection (selector.select), then a fit on train + val
    for each new window:
        r.update(X_train, X_val)     # local fc refinement from the current fc, warm-started fits
        r.predict_ratio(Xq)          # (or predict / predict_marginal / sqrt_density)

The caller passes the current window every time (nothing is stored), so the window policy is the
caller's. The objective follows the modes, as in select: dep columns -> conditional log-likelihood
log p(x_dep | x_rest) = log joint - log marginal; none -> joint log-likelihood.

update():
  * refine (default): from the current fc, per-block steps of sqrt(2) up and down (each column, or a
    whole radial group, a block; per (mode, block) when the modes' fc differ), each candidate fitted
    on X_train warm-started from the current best fit and scored on X_val; a move is accepted only
    if its mean per-point validation gain exceeds z_se standard errors; sweeps until no block moves
    (at most max_refine_sweeps). Every fit runs to tol.
  * then the final fit on X_train + X_val (refit=True) warm-started from the previous final fit.
  * every `reselect_every` updates (None: never) a full cold fit() instead, e.g. to let a rank > 1
    fit leave the previous window's split between its modes.
"""
import time
import numpy as np
from blml_trivial import (BLMLQuick, make_modes, modes_fc, with_fc, with_lattice, strip_lattice, modes_lattice,
                          strip_dep, marginal_modes, modes_trainable, modes_period, modes_period_trainable, with_period)
from selector import select, _template, _blocks, _move
from blml_trivial import (kernel_units, kernel_class, projected_modes, unprojected_modes, lattice_fixed, param_value,
                          KERNEL_PARAMS, LATTICE_PARAMS, lattice_fold)
from blml_local import FusedFit, Grown, grow
from blml_kernels import ProductKernel


class BLMLRolling:
    def __init__(self, engine='cpu', search='shared_hillclimb', complex_g=False, refine=True, z_se=1.0,
                 max_refine_sweeps=6, reselect_every=None, refit=True, rebin_tol=1.0, seed=0, **est_kw):
        """engine: 'cpu' or 'gpu' (BLMLQuick(engine=...)). search: select's search pattern for fit().
        est_kw: estimator settings (tol, dtype, bin_scale, batch_size / block_size, ...). The modes template
        (fc optional, ignored by fit's selection) is passed to fit(), as BLMLQuick.fit(X, modes)."""
        self.template, self.engine, self.search, self.complex_g = None, engine, search, complex_g
        self.refine, self.z_se, self.max_refine_sweeps, self.reselect_every = refine, z_se, max_refine_sweeps, reselect_every
        self.refit, self.rebin_tol, self.seed, self.est_kw = refit, rebin_tol, seed, est_kw
        self.model_ = self.modes_ = None
        self.history_ = []

    # ---------------------------------------------------------------- fits
    def _est(self):
        return BLMLQuick(engine=self.engine, seed=self.seed, **self.est_kw)

    def _fit(self, X, modes, init=None):
        return self._est().fit(X, modes, init=init, rebin_tol=self.rebin_tol, complex_g=self.complex_g)

    def _score(self, m, X):
        """Per-point validation log-likelihood: conditional if the modes have dep columns."""
        p = m.predict_ratio(X) if self._dep is not None else m.predict(X)
        return np.log(np.maximum(p, 1e-300))

    def _final(self, Xtr, Xva, init):
        X = Xtr if (Xva is None or not self.refit) else np.concatenate([Xtr, Xva])
        return self._fit(X, self.modes_, init=init)

    # ---------------------------------------------------------------- public
    def fit(self, X_train, modes=None, X_val=None):
        """Full fc selection (select with self.search; X_val None: select's random 20% split), then
        the final fit (cold). modes: the template (needed on the first call)."""
        t0 = time.time()
        if modes is not None:
            self.template = modes
        if self.template is None:
            raise ValueError("pass modes on the first fit")
        X_train = np.atleast_2d(np.asarray(X_train, float))
        self._kinds, self._periods, self._dep, self._K, _ = _template(self.template, X_train.shape[1])
        self.modes_, _ = select(X_train, self.template, search=self.search, seed=self.seed, engine=self.engine,
                                X_val=X_val, complex_g=self.complex_g)
        self._warm = None
        self.model_ = self._final(X_train, X_val, None)
        self.n_updates_ = 0
        self.history_.append(dict(kind='select', fc=modes_fc(self.modes_).tolist(), s=time.time() - t0))
        return self

    def update(self, X_train, X_val=None):
        """New window: local fc refinement (needs X_val) and a warm-started final fit; a full fit()
        every reselect_every updates."""
        if self.model_ is None:
            return self.fit(X_train, None, X_val)
        self.n_updates_ += 1
        if self.reselect_every and self.n_updates_ % self.reselect_every == 0:
            n = self.n_updates_
            self.fit(X_train, None, X_val); self.n_updates_ = n
            return self
        t0 = time.time()
        X_train = np.atleast_2d(np.asarray(X_train, float))
        n_fits, moved_any = 0, False
        if self.refine and X_val is not None:
            n_fits, moved_any = self._refine(X_train, np.atleast_2d(np.asarray(X_val, float)))
        self.model_ = self._final(X_train, X_val, self.model_)
        self.history_.append(dict(kind='update', fc=modes_fc(self.modes_).tolist(), moved=moved_any,
                                  n_fits=n_fits + 1, s=time.time() - t0))
        return self

    def _refine(self, Xa, Xb):
        """Per-block sqrt(2) steps from the current fc, accepted on a z_se standard-error gain."""
        F = modes_fc(self.modes_); K, d = F.shape
        same = np.allclose(F, F[:1])           # modes share fc: search one row, every mode gets it
        R = 1 if same else K
        blocks = [[k * d + j for j in b] for k in range(R) for b in _blocks(self._kinds, d)]
        kinds, periods = list(self._kinds) * R, list(self._periods) * R     # flat (R * d) view for _move
        f = F[0].copy() if same else F.ravel()
        full = (lambda v: np.tile(v, (K, 1))) if same else (lambda v: v.reshape(K, d))
        warm = self._warm if self._warm is not None else self.model_
        cur = self._fit(Xa, self.modes_, init=warm); best = self._score(cur, Xb); n_fits = 1
        moved_any, moved = False, True
        for _ in range(self.max_refine_sweeps):
            if not moved:
                break
            moved = False
            for blk in blocks:
                for fac in (np.sqrt(2.0), 1 / np.sqrt(2.0)):
                    while True:
                        cand = _move(f, blk, fac, kinds, periods)
                        if cand is None:
                            break
                        cm = with_fc(self.modes_, full(cand))
                        m = self._fit(Xa, cm, init=cur); lp = self._score(m, Xb); n_fits += 1
                        diff = lp - best
                        if diff.mean() > self.z_se * diff.std(ddof=1) / np.sqrt(len(diff)):
                            f, self.modes_, cur, best, moved, moved_any = cand, cm, m, lp, True, True
                        else:
                            break
        self._warm = cur
        return n_fits, moved_any

    # ---------------------------------------------------------------- prediction (current fit)
    def predict(self, Xq):
        return self.model_.predict(Xq)

    def predict_ratio(self, Xq):
        return self.model_.predict_ratio(Xq)

    def predict_marginal(self, Xrest):
        return self.model_.predict_marginal(Xrest)

    def sqrt_density(self, Xq):
        return self.model_.sqrt_density(Xq)

    def __repr__(self):
        fc = None if self.modes_ is None else np.round(modes_fc(self.modes_), 4).tolist()
        return f"BLMLRolling(engine={self.engine!r}, search={self.search!r}, fc={fc})"


def _burnin_split(G, n, seed, frac=0.2):
    """(train rows, validation rows) for the burn-in selection: with no groups, all rows and None (the
    selector then holds out random rows); with groups (n, 1), 20% of whole groups for validation; with
    (n, 2) labels (e.g. charge and discharge day of a pair, consecutive integers), validation is the rows
    whose first label is a chosen group, and training drops every row that touches a chosen group or the
    one after it (pairs spanning two days share prices with both)."""
    if G is None:
        return np.arange(n), None
    G = np.asarray(G)[:n]
    u = np.unique(G[:, 0])
    V = np.random.default_rng(seed).choice(u, max(1, int(round(frac * len(u)))), replace=False)
    va = np.isin(G[:, 0], V)
    if G.shape[1] == 1:
        return np.where(~va)[0], np.where(va)[0]
    Vh = np.union1d(V, V + 1)
    tr = ~np.isin(G[:, 0], Vh) & ~np.isin(G[:, 1], Vh)
    return np.where(tr)[0], np.where(va)[0]


def _hold_periods(modes):
    """Copy of modes with every period fixed at its value (for a selector that does not search periods)."""
    from blml_trivial import param_value
    out = []
    for m in modes:
        out.append({j: (dict(c, period=dict(value=param_value(c['period'])[0], trainable=False)) if 'period' in c else dict(c))
                    for j, c in m.items()})
    return out


class _StreamingBase:
    """Streaming BLML: online learning, for speed. The burn-in (the first >= burn_in points or counts) gets an
    exact fit with the cut-offs selected (select's `search`; every fc given: no selection). Then each batch:

      1. Prequential score: the batch is scored by the current model before it is used (out of sample, like
         SGD's loss on the next batch): log f(x), or log f(x_dep | x_rest) = log joint - log marginal when the
         modes have dep columns. self.preq_ keeps the means.
      2. Gradient parts of that score for every trainable parameter -- as its owner declares it (each mode's kernel
         units: KERNEL_PARAMS; input maps: InputMap.params; _registry) -- at fixed coefficients, smoothed as
         log(f + eps), eps = grad_eps * the median density (log f is singular where g crosses zero); pooled over
         param_every batches. implicit_grad=True adds the
         coefficients' one-sweep response dL/de de/dtheta (real rank 1); off by default: it showed no gain
         (paper 5's ablation; the hidden-plane runs, D = 5 / 20: the same fc paths and accuracy) and costs a pass.
      3. The batch's bins get one coordinate update (touched bins only: blml_local.FusedFit.cd); the
         normaliser e'Se is tracked through the updates and corrected from each batch's rows.
      4. Every param_every batches every parameter moves by its declared type (blml_maps.adam_step: 'scale' by
         Adam on log theta, 'stiefel' on the tangent space, 'free' plainly); fc_update='hillclimb', the lattice
         parameters by one multiplicative step (`step` up or down) once their pooled paired gain on the batches
         exceeds z_se standard errors; 'integer' parameters (dirichlet harmonics) always by that rule.
         The coefficients are not refitted when the parameters move; later batches and epochs catch up.

    Every term of a batch is O(d |I| B) (I: its bins, B: the bins held), so an epoch is O(d n^2) -- about two
    sweeps of the one-shot solver -- in place of a fit per batch; see blml_local. Rank K (each mode its own kernel:
    kinds, periods, cut-offs, on one shared lattice), g real or complex (complex_g: the burn-in fit runs the solver's
    complex starts; the gradient has no one-sweep response).
    score: the prequential score the parameters ascend. 'cond_loglik' (default): log f(x) without dep columns,
    log f(y | x) = log joint - log marginal with them (mutual information, a proper scoring rule).
    'dependence' (dep columns required): the mutual dependence of the dep columns and the rest (Agarwal, Sacre &
    Sarma 2015), d^2 = 1 - mean_b sqrt(f_X(x_b) f_Y(y_b) / f_J(x_b, y_b)) on the batch's labelled points (a Hellinger
    distance between the joint and the product of marginals; bounded, robust to a few points in low-density
    regions, not proper); it adds a fit of the dep columns' marginal f_Y, and its gradient is the conditional one
    reweighted by s_b / 2, s_b = sqrt(f_X f_Y / f_J), with the -log f_Y term. A binary dep column with a fixed
    integer cut-off (e.g. 10000) decouples the classes exactly (K(1) = 0): the joint is then the class-conditional
    densities and f_Y the priors.
    Mapped units (a tuple key with W= / k= (u = x W) or map= any blml_maps.InputMap; any kernel kind): the map's
    declared parameters are in the registry; their gradient is the score's derivative in the coordinates
    (FusedFit.batch_stats) pulled back by the map's vjp through the batch points and the held points (bins move
    with their points); then the held points are re-mapped and only those whose cell changed move
    (FusedFit.move_points; no re-bin). The outputs' kernel parameters move as any. Trainable maps hold the raw
    inputs. CPU or GPU.
    trace_trainable=True keeps the trainable parameters after every step in history_['trainable'] ({unit key:
    {name: value}}, keys as in the input's modes: fc, period, W); off by default.
    search: the burn-in selector's search (default 'shared': one common cut-off, a few fits; Adam then refines
    each column, so a finer burn-in search buys little and on a GPU costs about a full one-shot search).
    dtype (est_kw): float32 on the GPU runs the touched-bin kernels in float32 arithmetic with float64 sums (fast
    on GPUs with slow float64, e.g. the T4); float64 for A100 / H100 or exact parity; the CPU kernels use float64.
    Template parameters (blml_trivial.KERNEL_PARAMS): fixed (trainable=False) never move; trainable with a
    value start there. A trainable period puts its column on an unfolded lattice (raw coordinate; the Dirichlet
    row over raw offsets is P-periodic by itself), so a period step only rebuilds that row. A period is learned
    only for a conditioning column of a conditional (the marginal of t mod P, which carries -log P, cancels),
    and needs data spanning several periods: on one period (e.g. hour of day in [0, 24)) any P beyond the data's
    range wraps nothing and fits about as well, so P is not identified."""
    _X, _Xraw, _w, _G, _ids = Grown(), Grown(), Grown(), Grown(), Grown()   # held data: grow per batch without copying


    def __init__(self, burn_in=2000, fc_update='adam', lr=0.1, betas=(0.9, 0.999), step=1.05, z_se=1.0,
                 param_every=3, grad_eps=1e-3, search='shared', complex_g=False, engine='cpu', seed=0,
                 score='cond_loglik', trace_trainable=False, map_norm='off', implicit_grad=False, **est_kw):
        if fc_update not in ('adam', 'hillclimb'):
            raise ValueError("fc_update: 'adam' or 'hillclimb'")
        if score not in ('cond_loglik', 'dependence'):
            raise ValueError("score: 'cond_loglik' or 'dependence'")
        self.score = score
        self.proj_, self.trace_trainable = None, trace_trainable
        self.burn_in, self.fc_update, self.lr, self.betas, self.step, self.z_se = burn_in, fc_update, lr, betas, step, z_se
        self.param_every, self.grad_eps, self.search, self.complex_g = max(1, int(param_every)), grad_eps, search, complex_g
        if map_norm not in ('sampled', 'exact', 'off'):
            raise ValueError("map_norm: 'off' (the point part only; default: single-pass fits), 'sampled' (O(b B) per "
                             "step, pooled) or 'exact' (O(B^2) per map step)")
        self.map_norm, self.implicit_grad = map_norm, implicit_grad
        self.engine, self.seed, self.est_kw = engine, seed, est_kw
        self.template = self.modes_ = self.ff = None
        self.preq_, self.history_ = [], []
        self._n_batches, self._pool, self._hc, self._adam, self._mpool, self._epoch_g = 0, [], {}, {}, [], []
        self._rng = np.random.default_rng([seed, 7])

    # ---------------------------------------------------------------- setup
    def _setup(self, modes, d):
        if self.template is not None:
            return
        if modes is None:
            raise ValueError("pass modes on the first call")
        from blml_trivial import parse_modes
        self.template, self._d = modes, d
        specs, self._dep = parse_modes(modes, d, need_fc=False)
        self._K = len(specs)
        self._kindsK = [list(sp['kinds']) for sp in specs]
        self._rest = [j for j in range(d) if j not in (self._dep or [])]
        ptr = modes_period_trainable(modes)
        for k in range(self._K):
            for j in range(d):
                if ptr[k][j] and (not self._dep or j in self._dep):
                    raise ValueError(f"column {j}: a trainable period needs that column to be a conditioning column of a "
                                     "conditional (dep set on other columns): the joint likelihood of t mod P carries "
                                     "-log P (a density on a circle of length P), so it always prefers a shorter period")
            if self._dep:
                for _, dims in kernel_units(self._kindsK[k], d):
                    if 0 < sum(j in self._rest for j in dims) < len(dims):
                        raise ValueError("streaming: a radial group straddles dep and conditioning columns")

    def _registry(self):
        """Every trainable parameter, as its owner declares it (a kernel class: KERNEL_PARAMS; an input map:
        InputMap.params), blind to its meaning: [dict(src, key, name, ptype, rule, ...)] with src 'kernel' (mode,
        dims: each mode's units own their parameters) or 'map' (unit: the mapped unit's index), rule 'adam' (Adam by
        declared type: blml_maps.adam_step) or 'hill' (an 'integer' parameter, e.g. dirichlet harmonics, or with
        fc_update='hillclimb' the lattice parameters: a paired-gain hill-climb). Built once at the burn-in."""
        from blml_trivial import _percol
        reg = []
        for k, kinds in enumerate(self._kindsK):
            cols = _percol(self.modes_[k])
            for kind, dims in kernel_units(kinds, self._d):
                for name, (ptype, default) in kernel_class(kind).params.items():
                    tr = param_value(cols[dims[0]].get(name))[1]
                    if not (default if tr is None else bool(tr)):
                        continue
                    hill = ptype == 'integer' or (self.fc_update == 'hillclimb' and name in LATTICE_PARAMS)
                    reg.append(dict(src='kernel', key=(k, tuple(dims), name), mode=k, dims=list(dims), name=name,
                                    ptype=ptype, rule='hill' if hill else 'adam'))
        if self.proj_ is not None:
            ui = -1
            for u in self.proj_.units:
                if not u['trainable']:
                    continue
                ui += 1
                for name, (ptype, tr) in u['map'].params.items():
                    reg.append(dict(src='map', key=('map', ui, name), unit=ui, name=name, ptype=ptype, rule='adam'))
        return reg

    def _theta(self, k, dims, name):
        """A kernel parameter's current value (the per-column arrays the kernel tables take: fc, period)."""
        return float(self._fc[k][dims[0]]) if name == 'fc' else float(self._P[k][dims[0]])

    def _put(self, fcK, PK, k, dims, name, value):
        """fcK, PK with a kernel parameter set (the one place mapping a declared name to those arrays)."""
        if name == 'fc':
            fcK[k][dims] = value
        elif name == 'period':
            PK[k][dims[0]] = float(value)
        else:
            raise NotImplementedError(f"streaming: kernel parameter {name!r} (no per-column storage)")

    def _masks(self):
        """The densities of the fused fit: [(columns, sign)] -- the joint (+1); with dep columns the marginal of the
        rest (-1); with score='dependence' also the marginal of the dep columns (-1)."""
        out = [(list(range(self._d)), +1.0)]
        if self._dep:
            out.append((list(self._rest), -1.0))
            if self.score == 'dependence':
                out.append((list(self._dep), -1.0))
        return out

    def _start(self, modes):
        """Burn-in: exact fits of the joint and its marginals on the held data at `modes`, loaded into one FusedFit
        (the marginals mapped onto the joint's bins)."""
        from blml_trivial import parse_modes, lattice_fold_specs, is_periodic_kind
        self.modes_ = modes
        specs = parse_modes(modes, self._d)[0]
        self._fc = modes_fc(modes).copy()                                   # (K, d): each mode's
        self._P = [[p if is_periodic_kind(kd) else None for kd, p in zip(sp['kinds'], sp['periods'])] for sp in specs]
        self._fold = lattice_fold_specs(specs)            # a trainable or mode-dependent period: unfolded
        if self.score == 'dependence' and not self._dep:
            raise ValueError("score='dependence' needs dep columns")
        self._pk = ProductKernel(self._kindsK[0], self._P[0], self._fold)   # binning (the fit's own copies hold the tables)
        self._lattice(first=True)
        masks = self._masks(); self._signs = np.array([s_ for _, s_ in masks])
        dt = self.est_kw.get('dtype', np.float64)
        self.ff = FusedFit(self._kindsK, self._P, self._fc, self._h, [c for c, _ in masks], self._bin_centres,
                           fold=self._fold, engine=self.engine, dtype=dt, implicit_grad=self.implicit_grad)
        models = []
        for cols, _ in masks:                               # the solver's exact fit (its lattice folds as ours)
            Z, w = self._counts_cols(cols)
            mc = strip_dep(modes) if len(cols) == self._d else marginal_modes(modes, cols)
            models.append(self._est().fit(Z, with_lattice(strip_lattice(mc), self._h[cols]), sample_weight=w,
                                          complex_g=self.complex_g))
        self.ff.load(models)
        self._reg = self._registry()
        if self._learn_W():                                # held points' bins (a trained map moves them)
            self.ff.track_points(self._X, self._w)

    def _est(self):
        return BLMLQuick(engine=self.engine, seed=self.seed, **self.est_kw)

    # ---------------------------------------------------------------- one batch
    def _weights(self, wt):
        """The score's per-point gradient weights from the batch's log densities lp (b, M): w = wt for the
        conditional log-likelihood; for the dependence, wt s_b / 2 with s_b = sqrt(f_X f_Y / f_J) (smoothed)."""
        if self.score != 'dependence':
            return None
        def fn(lp):
            f = np.exp(lp); fe = np.log(f + self.grad_eps * np.median(f, axis=0)[None, :] + 1e-300)
            logr = -(fe * self._signs[None, :]).sum(1)
            self._sb = np.exp(np.clip(0.5 * logr, -30, np.log(1e3)))
            return 0.5 * self._sb * wt
        return fn

    def _step(self, X, wt, revisit, learn=True, fit=True):
        """One batch: learn -- its gradient (every registered parameter) goes into the pools and the parameters move
        every param_every batches; fit -- its counts are added (unless revisit) and its touched bins get the BCD
        update. Both by default; fit.(bcd_frac, freeze_frac) splits them (a learn-only batch is scored without being
        added)."""
        t0 = time.time(); ff = self.ff
        grad = [(e['mode'], tuple(e['dims']), e['name']) for e in self._reg if e['src'] == 'kernel' and e['rule'] == 'adam']
        hc = [e for e in self._reg if e['rule'] == 'hill']
        if not learn:
            grad, hc = [], []
        idx = ff.bins(X, add=not revisit)
        I, inv = np.unique(idx, return_inverse=True); bc = np.bincount(np.ravel(inv), weights=wt)
        wsh = None
        if learn and self._learn_W() and getattr(self, '_braw', None) is not None:   # maps: their gradient in the same passes
            wsh = dict(cols=self._shift_cols())
        if not (grad or hc or wsh is not None):              # nothing learns from this batch: its touched rows only --
            if fit:                                          # no score, normaliser draws or batch removal (cd keeps Q
                if not revisit:                              # exact while the parameters stay put)
                    ff.add_counts(idx, wt)
                ff.cd(I, ff._rmK(ff.Zi[I], ff.Zi, ff.e)[0])
                if not revisit:
                    self._lattice()
            self.preq_.append(np.nan)
            self.history_.append(dict(n=ff.n, preq=np.nan, fc=self._fc.tolist(), moved=[],
                                      periods=[p for P in self._P for p in P if p is not None], s=time.time() - t0,
                                      **(dict(trainable=self._trainable()) if self.trace_trainable else {})))
            return
        ig = ff.implicit_grad
        ff.implicit_grad = ig and fit                         # a batch not added moves no coefficient: no response term
        try:
            lp, SeI, st = ff.batch_stats(X, idx, bc, I, wt, grad, revisit, self.grad_eps, bool(grad) or wsh is not None,
                                         self._rng, weight_fn=self._weights(wt), wsh=wsh)
        finally:
            ff.implicit_grad = ig
        lpc = lp[:, 0] - (lp[:, 1] if self._dep else 0.0)    # the conditional (or joint) log density
        W = float(wt.sum())
        if self.score == 'dependence':
            if not grad:                                     # the weights were not needed for a gradient: compute s_b
                self._weights(wt)(lp)
            self.preq_.append(1.0 - float(wt @ self._sb) / W)
        else:
            self.preq_.append(float(wt @ lpc) / W)
        if st is not None:
            if wsh is not None:
                self._mpool.append(self._map_grads(st, self._braw))
            if grad:
                self._pool.append(st)
                self._epoch_g.append((float(sum(s_['Wn'] for s_ in st)), self._pooled_grad([st], grad)))
        if hc:
            self._hc_accumulate(hc, X, I, bc, lpc, wt, revisit)
        if fit:
            if not revisit:
                ff.add_counts(idx, wt)
            ff.cd(I, SeI)
        moved = []
        if learn:
            self._n_batches += 1
            if self._n_batches % self.param_every == 0:
                moved = self._param_step(grad, hc)
        if fit and not revisit:
            self._lattice()
        self.history_.append(dict(n=ff.n, preq=self.preq_[-1], fc=self._fc.tolist(), moved=moved,
                                  periods=[p for P in self._P for p in P if p is not None], s=time.time() - t0,
                                  **(dict(trainable=self._trainable()) if self.trace_trainable else {})))

    def _learn_W(self):
        """Whether any mapped unit is trained (then the raw inputs are held)."""
        return self.proj_ is not None and any(u['trainable'] for u in self.proj_.units)

    def _shift_cols(self):
        """The lattice columns of every trained map, in lattice order (their coordinate derivatives)."""
        return [L + j for u, L, k in self.proj_.spans() if u['trainable'] for j in range(k)]

    def _map_grads(self, st, Xb_raw):
        """This batch's parts of every trained map's gradient, per density: (W, Wn, Qs, [({parameter: point part},
        {parameter: normaliser part}) per unit]) -- the batch points' vb and the held points' share of their bins' Vc
        (point part) and of the sampled rows' VR (normaliser part, FusedFit.batch_stats) pulled back by each map's vjp,
        O(n cost(map)). Pooled over param_every batches and combined as the kernel parameters' gradient at the step."""
        ff = self.ff; held = self._Xraw[:len(ff.pbin)]              # tracked points (the batch joins after)
        share = ff.pw / np.maximum(ff.cnt[ff.pbin], 1e-300)
        out = []
        for m, s_ in enumerate(st):
            dm = s_.pop('map'); per, c0 = [], 0
            inR = np.zeros(len(ff.cnt), bool); inR[dm['R']] = True
            VR = np.zeros((len(ff.cnt), dm['VR'].shape[1])); VR[dm['R']] = dm['VR']
            hit = inR[ff.pbin]
            for u, L, k in self.proj_.spans():
                if not u['trainable']:
                    continue
                sl = slice(c0, c0 + k); c0 += k; mp = u['map']; cols = u['cols']
                cont = mp.vjp(Xb_raw[:, cols], dm['vb'][:, sl])
                hv = mp.vjp(held[:, cols], share[:, None] * dm['Vc'][ff.pbin, sl])
                nq = mp.vjp(held[hit][:, cols], share[hit][:, None] * VR[ff.pbin[hit], sl])
                per.append(({n_: cont[n_] + hv[n_] for n_ in cont}, nq))
            out.append((s_['W'], s_['Wn'], s_['Qs'], per))
        return out

    def _pooled_grad(self, pool, grad):
        """The kernel parameters' gradient from batch statistics pooled over `pool`: per density, the mean over points
        of w_b d log f_b, point parts / n - (sum w / n) d log Q (w = 1: the usual mean), combined by the densities'
        signs (the conditional: joint - marginal)."""
        G = np.zeros(len(grad))
        for m, sgn in enumerate(self._signs):
            sts = [p[m] for p in pool]
            Wn = sum(s_['Wn'] for s_ in sts); Ww = sum(s_['W'] for s_ in sts)
            g = sum(s_['cont'] + s_['resp'] for s_ in sts) / Wn - (Ww / Wn) * sum(s_['dQ'] for s_ in sts) / sum(s_['Qs'] for s_ in sts)
            G += sgn * g
        return G

    def _param_stationary(self):
        """Whether the epoch's parameters are stationary (as blml_binned._KernelLearner): for each Adam parameter the
        count-weighted mean of the epoch's per-batch gradients within z_se standard errors of 0, and no pending
        hill-climb move (a pooled gain beyond z_se standard errors). Returns (bool, max |mean| / se)."""
        ok, zmax = True, 0.0
        if self._epoch_g:
            if len(self._epoch_g) < 2:
                return False, np.inf
            W = np.array([w for w, _ in self._epoch_g]); G = np.array([g for _, g in self._epoch_g])
            mu = (W[:, None] * G).sum(0) / W.sum(); var = (W[:, None] * (G - mu) ** 2).sum(0) / W.sum()
            se = np.sqrt(var / (len(W) - 1))
            z = np.where(se > 0, np.abs(mu) / np.where(se > 0, se, 1.0), np.where(mu == 0, 0.0, np.inf))
            ok = bool(np.all(z <= self.z_se)); zmax = float(z.max()) if not ok else 0.0
        for acc in self._hc.values():
            if acc['W'] > 1:
                m_ = acc['s'] / acc['W']; se_ = np.sqrt(max(acc['ss'] / acc['W'] - m_ * m_, 0.0) / (acc['W'] - 1))
                if m_ > 0 and m_ > self.z_se * se_:
                    ok = False
        return ok, zmax

    def _param_step(self, grad, hc):
        """One step of every registered parameter (_registry), each by its declared type (blml_maps.adam_step) from the
        gradient of each density pooled over param_every batches -- point part / n minus (sum w / n) d log Q, combined
        over the densities by their signs (the conditional: joint - marginal) -- and the hill-climb blocks (a block
        moves once its pooled paired gain exceeds z_se standard errors; one move per step). Kernel parameters move at
        fixed coefficients (FusedFit.set_params; later batches catch up); a map's move re-maps the held points and moves
        those whose cell changed (FusedFit.move_points: the lattice carries forward). Map gradients: map_norm 'off'
        (the point part only: in one pass every batch is out of sample and the normaliser adds nothing -- XOR, n = 20k,
        one epoch: off / sampled / exact 0.813 / 0.788 / 0.807), 'sampled' (dQ, Q from each batch's sampled rows) or
        'exact' (over all bin pairs at the step, O(B^2 k)); over repeated epochs the normaliser keeps the map from
        fitting the revisited points (n = 5k, 24 epochs: off fails a seed, exact 3 / 3). Returns what moved."""
        from blml_maps import adam_step
        moved = []
        fcK, PK = self._fc.copy(), [list(P) for P in self._P]
        if grad and self._pool:                            # kernel parameters
            G = self._pooled_grad(self._pool, grad)
            if not np.all(np.isfinite(G)):
                raise FloatingPointError(f"streaming: non-finite gradient {G}")
            for q, (k, dims, name) in enumerate(grad):
                e = next(x for x in self._reg if x['key'] == (k, dims, name))
                st = self._adam.setdefault(e['key'], [0.0, 0.0, 0])
                old = self._theta(k, list(dims), name)
                new = float(adam_step(old, G[q], st, e['ptype'], self.lr, self.betas, dlog=True))
                self._put(fcK, PK, k, list(dims), name, new)
                moved.append((e['key'], round(float(np.log(new / old)) if e['ptype'] == 'scale' else new - old, 5)))
        self._pool = []
        for blk in hc:                                     # hill-climb: pooled paired gain over z_se SEs
            for sgn in (+1, -1):
                acc = self._hc.get((blk['key'], sgn))
                if not acc or acc['W'] <= 0:
                    continue
                m_ = acc['s'] / acc['W']; se = np.sqrt(max(acc['ss'] / acc['W'] - m_ * m_, 0.0) / max(acc['W'] - 1, 1))
                if m_ > 0 and m_ > self.z_se * se:            # only this block's change (the others' moves stand)
                    k, dims = blk['mode'], blk['dims']
                    fcK[k][dims] = acc['cand'][k][dims]; self._hc = {}
                    moved.append((blk['key'], sgn)); break
            else:
                continue
            break
        if moved:
            self.ff.set_params(fcK, PK, self._rng)        # tables only (a period moves no bins: unfolded)
            self._fc, self._P = fcK, PK
            self.modes_ = with_fc(self.modes_, self._fc)
            if any(p is not None for P in self._P for p in P):
                self.modes_ = with_period(self.modes_, [[p if p is not None else np.nan for p in P] for P in self._P])
        if self._mpool:                                    # input maps
            moved.append(('maps',) + self._map_step())
        return moved

    def _map_step(self):
        """The maps' part of _param_step: one Adam step of every trained map's registered parameters, then the held
        points re-mapped and moved. Returns (|G|, points moved)."""
        from blml_maps import adam_step
        ff = self.ff
        if self.map_norm == 'exact':
            shift = [(None, j, 'shift') for j in self._shift_cols()]
            S, dS = ff._rmK(ff.Zi, ff.Zi, ff.e, shift, True)   # (B, M, K), (B, M, K, k)
            Qx = np.real(np.einsum('bmk,bmk->m', np.conj(ff.e), S))
            V = 2 * np.real(np.conj(ff.e)[:, :, :, None] * dS).sum(2)    # dQ / dz_a = 2 sum_k e_ak sum_c e_ck dK(z_a - z_c)
            held = self._Xraw[:len(ff.pbin)]; share = ff.pw / np.maximum(ff.cnt[ff.pbin], 1e-300)
        norm2, c0 = 0.0, 0
        spans = [(u, L, k) for u, L, k in self.proj_.spans() if u['trainable']]
        for e in [x for x in self._reg if x['src'] == 'map']:
            u, L, k = spans[e['unit']]; mp = u['map']; cols = u['cols']
            sl = slice(sum(s[2] for s in spans[:e['unit']]), sum(s[2] for s in spans[:e['unit'] + 1]))
            name, ui = e['name'], e['unit']
            G = 0.0
            for m, sgn in enumerate(self._signs):
                Ww = sum(p[m][0] for p in self._mpool); Wn = sum(p[m][1] for p in self._mpool)
                point = sum(p[m][3][ui][0][name] for p in self._mpool)
                if self.map_norm == 'off':
                    G = G + sgn * point / Wn
                    continue
                if self.map_norm == 'exact':
                    dQ = mp.vjp(held[:, cols], share[:, None] * V[ff.pbin, m, sl])[name]; Q = Qx[m]
                else:
                    dQ = sum(p[m][3][ui][1][name] for p in self._mpool); Q = sum(p[m][2] for p in self._mpool)
                G = G + sgn * (point / Wn - (Ww / Wn) * dQ / Q)
            if not np.all(np.isfinite(G)):
                raise FloatingPointError(f"streaming: non-finite gradient of map parameter {name}")
            st = self._adam.setdefault(e['key'], [0.0, 0.0, 0])
            mp.set(name, adam_step(mp.get(name), G, st, e['ptype'], self.lr, self.betas))
            norm2 += float(np.sum(np.asarray(G) ** 2))
        self._mpool = []
        if self.map_norm == 'exact':
            ff.Q = np.maximum(Qx, 1e-300); ff.renorm()                 # the exact normaliser, as computed
        self._X = self.proj_(self._Xraw)
        moved = ff.move_points(self._X, self._rng)
        return round(norm2 ** 0.5, 5), int(len(moved))

    def _hc_accumulate(self, hc, X, I, bc, lpc, wt, revisit):
        """Paired per-point gains in the conditional (or joint) log density of one step up / down of each hill-climb
        block (a mode's unit) on this batch, at fixed coefficients (Q(candidate) / Q estimated on draws from the held
        data, as FusedFit.set_params; leave-batch-out on a revisit, as the score)."""
        ff = self.ff
        R, cR = ff.sample_bins(len(X), self._rng)
        wR = (cR / ff.cnt[R])[:, None, None] * np.conj(ff.e[R])
        q0 = np.real(np.einsum('rmk,rmk->m', wR, ff._rmK(ff.Zi[R], ff.Zi, ff.e)[0]))
        for blk in hc:
            k, dims = blk['mode'], blk['dims']
            for sgn, fac in ((+1, self.step), (-1, 1.0 / self.step)):
                cand = _move(self._fc[k], dims, fac, self._kindsK[k], self._P[k])
                if cand is None:
                    continue
                candK = self._fc.copy(); candK[k] = cand
                g = ff.g(X, fcK=candK)[0]
                if revisit:
                    g = g - ff._pmK(np.asarray(X, float), ff.centres(I), ff._share(I, bc), fcK=candK)[0]
                ratio = np.real(np.einsum('rmk,rmk->m', wR, ff._rmK(ff.Zi[R], ff.Zi, ff.e, fcK=candK)[0])) / np.maximum(q0, 1e-300)
                lgm = np.log(np.maximum((np.abs(g) ** 2).sum(2), 1e-300)) - np.log(ratio)[None, :]
                lc = lgm[:, 0] - (lgm[:, 1] if self._dep else 0.0)
                gain = lc - lpc
                acc = self._hc.setdefault((blk['key'], sgn), dict(s=0.0, ss=0.0, W=0.0, cand=candK))
                acc['s'] += float(wt @ gain); acc['ss'] += float(wt @ gain ** 2); acc['W'] += float(wt.sum())
                acc['cand'] = candK

    # ---------------------------------------------------------------- prediction
    def _model_cols(self, Xq):
        Xq = np.atleast_2d(np.asarray(Xq, float))
        return Xq if self.proj_ is None else self.proj_(Xq)

    def predict(self, Xq):
        """The density at raw inputs (mapped units are mapped onto the lattice columns)."""
        return self.ff.predict(self._model_cols(Xq), 0)

    def predict_marginal(self, Xrest):
        """The marginal density of the conditioning columns, at points in those columns."""
        if not self._dep:
            raise RuntimeError("no dep columns in the modes")
        Xrest = np.atleast_2d(np.asarray(Xrest, float))
        if self.proj_ is not None:                       # input non-dep columns -> lattice rest columns
            Xrest = self.proj_.rest(Xrest, [self.proj_.others[j - self.proj_.k] for j in self._dep])
        Xf = np.zeros((len(Xrest), self._d)); Xf[:, self._rest] = Xrest          # the other columns are masked out
        return self.ff.predict(Xf, 1)

    def predict_ratio(self, Xq):
        Xm = self._model_cols(Xq)
        Xf = np.array(Xm, float); Xf[:, list(self._dep)] = 0.0
        return self.ff.predict(Xm, 0) / np.maximum(self.ff.predict(Xf, 1), 1e-300)

    def sqrt_density(self, Xq):
        if self.ff.K > 1:
            raise ValueError("sqrt_density: rank K has no single g (predict gives f = sum_k |g_k|^2)")
        return self.ff.g(self._model_cols(Xq))[0][:, 0, 0]

    def _trainable(self):
        """The trainable parameters now, {unit key: {name: value}} in the input's form (rank 1: the first mode): the
        kernel's (fc, period) and a mapped unit's (W= / k=: W; map=: every trainable parameter its map declares)."""
        out = {}
        maps = {tuple(u['cols']): u for u in self.proj_.units} if self.proj_ is not None else {}
        for key, c in self.modes_in_[0].items():
            ps = {}
            if c.get('map') is not None and maps.get(key, {}).get('trainable'):      # a map= unit: its declared params
                mp = maps[key]['map']
                ps.update({name: np.array(mp.get(name), float).copy() for name, (_, tr) in mp.params.items() if tr})
            for name, v in c.items():
                if name in ('kind', 'k', 'dep', 'h', 'map', 'trainable'):
                    continue
                val, tr = param_value(v)
                kind = c.get('kind', 'sinc')
                default = KERNEL_PARAMS.get(kind, {}).get(name, (None, False))[1]
                if (default if tr is None else tr) and val is not None:
                    ps[name] = np.array(val, float).copy()
            if ps:
                out[key] = ps
        return out

    @property
    def W_(self):
        """The first mapped unit's current W (D, k) (a LinearMap: its directions), or None."""
        if self.proj_ is None:
            return None
        return getattr(self.proj_.units[0]['map'], 'W', None)

    @property
    def modes_in_(self):
        """The fitted modes in the input's form (tuple keys, current W and fc)."""
        return self.modes_ if self.proj_ is None else unprojected_modes(self.proj_, self.modes_, self._modes_in)

    @property
    def lj(self):                                      # (the joint's state, for code that inspects it)
        return getattr(self, 'ff', None)

    def _tx(self, X):
        return X

    def __repr__(self):
        fc = None if self.modes_ is None else np.round(modes_fc(self.modes_), 4).tolist()
        return f"{type(self).__name__}(engine={self.engine!r}, fc_update={self.fc_update!r}, score={self.score!r}, fc={fc})"




def _group_batches(n, groups, batch, rng):
    """Index arrays of whole groups (first label) in random group order, round(batch / mean group size) groups
    per batch; no groups: a random permutation cut every `batch` rows."""
    if groups is None:
        p = rng.permutation(n)
        return [p[i:i + batch] for i in range(0, n, batch)]
    g = np.asarray(groups); g = g[:, 0] if g.ndim == 2 else g
    u, inv = np.unique(g, return_inverse=True)
    members = [[] for _ in u]
    for r, k in enumerate(np.ravel(inv)):
        members[k].append(r)
    per = max(1, int(round(batch * len(u) / n)))
    order = rng.permutation(len(u))
    return [np.array([r for k in order[i:i + per] for r in members[k]]) for i in range(0, len(u), per)]


class BLMLQuickStreaming(_StreamingBase):
    """Streaming BLML on raw points.

        s = BLMLQuickStreaming().fit(X, modes, epochs=2, batch=2000, split_groups=day)
        s = BLMLQuickStreaming(); s.partial_fit(X0, modes); s.partial_fit(X1); ...   # pure online (epoch 1)

    fit: the burn-in once, then `epochs` passes over X in batches (whole split_groups, random group order);
    later epochs continue from the current coefficients and parameters, the counts are not added again and a
    revisited batch is scored leave-batch-out (its share of each touched coefficient removed), so the
    gradient stays out of sample. partial_fit: one new batch (epoch 1); modes on the first call.
    split_groups: a label per row, (n,) or (n, 2); rows sharing a label are never separated -- not between
    batches (fit) nor between the burn-in selection's training and validation sides (20% of whole groups held
    out; with two labels, two-day blocks; see _burnin_split). Same-day rows would otherwise leak into
    validation and drive the selected cut-offs up.
    The lattice h = bin_scale n^-1/4 / fc follows n and fc; it is rebuilt when its target moves by more than
    (1 + rebin_tol)^(1/4) (default 15: a factor 2, n grows 16x or fc doubles; 1: n doubles, or fc moves ~19%,
    the streaming paper's runs), O(log n) times: each new bin starts from the current g
    (LocalFit.rebin), then rebin_sweeps full sweeps on the new lattice (default 0: the touched-bin updates carry the
    start forward). Each sweep is O(d B^2): the re-bins as n doubles sum to ~4/3 of a full-size sweep each, a re-bin for
    a moved fc costs one at full size. Without sweeps the coefficients lag after a re-bin and can bias the parameter
    gradient (2-D heavy-tailed benchmark, rebin_tol=1: -0.016 nats per point with 0 against 2; the streaming paper's
    runs used 2). With rebin_tol=15 that benchmark (n up to 8x its burn-in) never re-bins and matches rebin_tol=1 with
    2 sweeps, 7-18% faster.
    A mapped unit's parameters move with the others (native gradient; _map_step): only points whose cell
    changes move, the lattice stays."""

    _bin_centres = False                               # inputs are points (the units bin them); counts: True

    def __init__(self, rebin_tol=15.0, bin_scale=0.1, rebin_sweeps=0, bcd_frac=1.0, freeze_frac=0.0, **kw):
        super().__init__(**kw)
        self.rebin_tol, self.bin_scale, self.rebin_sweeps = rebin_tol, bin_scale, rebin_sweeps
        if not (0.0 < bcd_frac <= 1.0 and 0.0 <= freeze_frac < 1.0):
            raise ValueError("bcd_frac in (0, 1], freeze_frac in [0, 1)")
        self.bcd_frac, self.freeze_frac = float(bcd_frac), float(freeze_frac)
        self._X = None; self._G = None

    def fit(self, X, modes, epochs=2, batch=None, split_groups=None, sample_weight=None, on_epoch=None, tol=None,
            tol_check='running'):
        """Stopping: tol None runs `epochs` passes; tol > 0 stops after the first epoch whose coefficients meet the
        one-shot solver's test, residual max_b |(S e)_b e_b / (n w_b) - 1| < tol (complex / rank K: its row form;
        FusedFit.residual, one full row pass per epoch, about a sweep), at most `epochs` passes; converged_ says whether
        it was met, epochs_ how many passes ran, residual_ the last residual (history_: one entry per epoch, with
        params_stationary and grad_z). The trainable parameters must be stationary too (_param_stationary: the epoch's
        per-batch gradients within z_se standard errors of 0, no pending hill-climb move), as in blml_binned. tol_check: 'running'
        (default) -- the maximum over the epoch of each touched bin's residual, from the fresh row the batch computes for
        its update anyway, just before that update (no extra pass; from the second epoch, when every occupied bin is
        touched with a coefficient already set; it lags the end-of-epoch residual by at most an epoch) -- or 'exact', the
        residual of every bin after the epoch (one full row pass, about a sweep, per epoch).
        sample_weight: (n,) positive weights (default 1): point i counts w_i times in the counts, the scores
        and the gradients (e.g. bin centres weighted by their counts). on_epoch(epoch, self): called after each
        epoch (epoch 0 first; e.g. a held-out score or progress).
        bcd_frac < 1 (constructor): a fixed share 1 - bcd_frac of the points (the learn points) drives the parameters
        (fc, periods, maps) and is scored without being fitted, the rest gets the BCD updates -- the parameters' gradient
        stays out of sample at every epoch (without it, revisited points are in the coefficients and the parameters
        overfit them). freeze_frac > 0: the last freeze_frac of the batches (epochs x batches per epoch) fit every point
        by BCD at frozen parameters, the learn points joining the coefficients then. Defaults (1, 0): every batch
        learns and is fitted."""
        X = np.atleast_2d(np.asarray(X, float)); n = len(X)
        sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
        batch = int(batch or max(100, n // 20))
        grp = None if split_groups is None else np.asarray(split_groups)
        if tol is not None and not tol > 0:
            raise ValueError("tol: None (fixed epochs) or > 0")
        if tol is not None and (self.bcd_frac < 1.0 or self.freeze_frac > 0.0):
            raise ValueError("tol needs bcd_frac = 1 and freeze_frac = 0: the learn points stay out of the coefficients "
                             "until the frozen batches, whose number is set by epochs")
        if tol_check not in ('running', 'exact'):
            raise ValueError("tol_check: 'running' or 'exact'")
        self.converged_, self.epochs_, self.residual_ = None, 0, None
        if self.bcd_frac >= 1.0 and self.freeze_frac <= 0.0:   # every batch learns and is fitted
            for ep in range(epochs):
                rng = np.random.default_rng(self.seed + ep); self._epoch_g = []
                if self.ff is not None:
                    self.ff.rho_run = 0.0 if (tol is not None and tol_check == 'running') else None
                for b in _group_batches(n, split_groups, batch, rng):
                    if ep == 0:
                        self._add(X, b, sw, grp, modes)
                    else:
                        self._again(X, b, sw)
                self.epochs_ = ep + 1
                if tol is not None and self.ff is not None and (tol_check == 'exact' or ep > 0):
                    self.residual_ = self.ff.residual() if tol_check == 'exact' else self.ff.rho_run
                    still, z = self._param_stationary() if self.fc_update is not None else (True, 0.0)
                    self.converged_ = bool(self.residual_ < tol and still)
                    self.history_.append(dict(epoch=ep + 1, residual=self.residual_, params_stationary=still, grad_z=z))
                if on_epoch is not None:
                    on_epoch(ep, self)
                if tol is not None and self.converged_:
                    break
            return self
        # split roles: a fixed share 1 - bcd_frac of the points (the learn points) drives the parameters and is not
        # fitted while they learn -- so their gradient stays out of sample at every epoch -- the rest is fitted (BCD);
        # the last freeze_frac of the batches fit every point at frozen parameters (the learn points join then)
        nb = sum(1 for _ in _group_batches(n, split_groups, batch, np.random.default_rng(self.seed)))
        T_learn = int(round((1.0 - self.freeze_frac) * epochs * nb))
        lpt = np.random.default_rng([self.seed, 11]).random(n) >= self.bcd_frac
        held = np.zeros(n, bool); t = 0
        for ep in range(epochs):
            rng = np.random.default_rng(self.seed + ep)
            for b in _group_batches(n, split_groups, batch, rng):
                frozen = t >= T_learn; t += 1
                if self.lj is None:                           # burn-in: whole batches are held -- in the
                    self._add(X, b, sw, grp, modes); held[b] = True     # coefficients, so never learn points
                    lpt[b] = False
                    continue
                both = self.bcd_frac >= 1.0 and not frozen    # no learn points: the batch does both
                F, L = (b, b[:0]) if (frozen or self.bcd_frac >= 1.0) else (b[~lpt[b]], b[lpt[b]])
                new, old = F[~held[F]], F[held[F]]
                if len(new):
                    self._add(X, new, sw, grp, modes, learn=both); held[new] = True
                if len(old):
                    self._again(X, old, sw, learn=both)
                if len(L):                                    # scored, gradient pooled, not added
                    if self._learn_W():
                        self._braw = X[L]
                    self._step(self._tx(X[L]), sw[L], revisit=False, learn=True, fit=False)
            self.epochs_ = ep + 1
            if on_epoch is not None:
                on_epoch(ep, self)
        return self

    def _add(self, X, rows, sw, grp, modes, learn=True):
        """New points: held, then a step (learn: also its gradient)."""
        grow(self, '_ids', rows)                              # held row -> input row
        self.partial_fit(X[rows], modes, None if grp is None else grp[rows], sample_weight=sw[rows], _learn=learn)

    def _again(self, X, rows, sw, learn=True):
        """Held points revisited (leave-batch-out)."""
        if self._learn_W():
            self._braw = X[rows]
        self._step(self._tx(X[rows]), sw[rows], revisit=True, learn=learn)

    def _tx(self, X):
        return X if self.proj_ is None else self.proj_(X)

    def partial_fit(self, X, modes=None, split_groups=None, sample_weight=None, _learn=True):
        X = np.atleast_2d(np.asarray(X, float))
        w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, float)
        if self.template is None and modes is not None:
            self._fixed_h = modes_lattice(modes, X.shape[1]) if not projected_modes(modes)[0] else None
            if self._fixed_h is not None:                    # counts on a fixed lattice: inputs are bin centres
                lattice_fixed(modes, 'a fixed lattice (h in the modes)')
                self._bin_centres = True
        fixed = getattr(self, '_fixed_h', None) is not None
        if w.shape != (len(X),) or not (np.all(w >= 0) if fixed else np.all(w > 0)):
            raise ValueError("sample_weight: (n,) " + ("non-negative counts" if fixed else "positive weights"))
        if self.template is None and modes is not None:      # mapped units: the model sees the lattice columns
            self._modes_in = modes
            self.proj_, modes = projected_modes(modes, self.seed)
        if self._learn_W():                                  # raw inputs held: W moves the points
            grow(self, '_Xraw', X)
            self._braw = X
        X = self._tx(X)
        if self.template is None and fixed:
            self._setup(modes, X.shape[1])
        self._setup(modes, X.shape[1])
        grow(self, '_X', X); grow(self, '_w', w)
        if self.lj is None:
            if split_groups is not None:
                g = np.asarray(split_groups); g = g[:, None] if g.ndim == 1 else g
                grow(self, '_G', g)
            if float(self._w.sum()) >= self.burn_in:
                t0 = time.time()
                self._start(self._select())
                self.history_.append(dict(n=len(self._X), burn_in=True, fc=self._fc.tolist(), s=time.time() - t0))
            return self
        self._step(X, w, revisit=False, learn=_learn)
        return self

    def _select(self):
        if np.all(np.isfinite(modes_fc(self.template))):      # every fc given (start or fixed): no selection
            return self.template
        if getattr(self, '_fixed_h', None) is not None:      # counts: select_counts on an 80/20 binomial split
            from selector import select_counts
            h = self._fixed_h
            u, inv = np.unique(np.round(self._X / h).astype(np.int64), axis=0, return_inverse=True)
            Z, w = u * h, np.bincount(np.ravel(inv), weights=self._w, minlength=len(u))     # every pixel, zeros too
            wtr = np.random.default_rng(self.seed).binomial(w.astype(np.int64), 0.8).astype(float)
            modes, _ = select_counts(Z, wtr, w - wtr, h, strip_lattice(_hold_periods(self.template)), engine=self.engine,
                                     complex_g=self.complex_g)
            return with_lattice([{j: dict(c, period=self.template[k][j]['period']) if 'period' in c else c
                                  for j, c in m.items()} for k, m in enumerate(modes)], h)
        held = _hold_periods(self.template)                   # the burn-in search holds trainable periods
        tr, va = _burnin_split(self._G, len(self._X), self.seed)
        unit = bool(np.all(self._w == 1))                    # weights only when there are any (exact as before)
        modes, _ = select(self._X[tr], held, search=self.search, seed=self.seed, engine=self.engine,
                          X_val=None if va is None else self._X[va], complex_g=self.complex_g,
                          sample_weight=None if unit else self._w[tr],
                          val_weight=None if (unit or va is None) else self._w[va])
        return [{j: dict(c, period=self.template[k][j]['period']) if 'period' in c else c for j, c in m.items()}
                for k, m in enumerate(modes)]

    def _lattice(self, first=False, force=False):
        """The lattice for the held weight n: the units' steps for bin_scale n^-1/4 / fc (ProductKernel.step);
        re-binned when the target moved by more than rebin_tol (or force). A fixed lattice never moves."""
        if getattr(self, '_fixed_h', None) is not None:
            if first:
                self._h = self._fixed_h.copy(); self._h_t = self._h.copy()
            return False
        n = float(self._w.sum())
        from blml_trivial import lattice_target
        h_t = lattice_target(self.bin_scale * n ** -0.25 / np.max(self._fc, axis=0), self._kindsK)   # fine enough for every mode
        h = self._pk.steps(h_t, self._P[0])
        if not (first or force) and np.all(np.abs(np.log(h_t / self._h_t)) <= 0.25 * np.log1p(self.rebin_tol) + 1e-12):
            return False
        self._h, self._h_t = h, h_t
        if not first:
            self.ff.rebin(self._X, self._w, self._h, self.rebin_sweeps)               # tracks the points' bins (pbin)
            self.history_.append(dict(n=n, rebinned=True, h=self._h.tolist()))
        return True

    def _counts_cols(self, cols):
        Xq = np.ascontiguousarray(self._pk.quantize(self._X[:, cols], self._h[cols], cols, self._P[0]))
        u, inv = np.unique(Xq, axis=0, return_inverse=True)
        return u, np.bincount(np.ravel(inv), weights=self._w, minlength=len(u))

    def _counts_all(self):
        Zj, wj = self._counts_cols(list(range(self._d)))
        Zm, wm = self._counts_cols(self._rest) if self._dep else (None, None)
        return Zj, wj, Zm, wm
