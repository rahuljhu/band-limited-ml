"""Unified selector.
Public API: select(X, modes, ...) and select_counts(Z, w_tr, w_va, h, modes, ...) take a modes template
(blml_trivial.parse_modes; fc optional) and return the modes with the selected fc. Internally the search
runs on per-dim fc vectors.
Selector owns IQR scaling, works in scaled units, returns fc in original units.

search (select's search pattern; not related to the rank-K `modes`):
  'shared_hillclimb' (default): shared first, then hillclimb from the shared optimum.
  'shared': 1-D doubling selector, single fc across all scaled dims (fc0=0.2).
  'hillclimb': coordinate-wise doubling, per-dim fc (up first, then down).
  'auto': shared doubling, then harmonic-aware block refinement (_select_auto_flat).

The objective follows the modes: no dep columns -> joint validation LL; dep columns -> conditional
validation LL = mean log p(x_dep | x_rest). cond_method='ratio' (default, the original objective) scores
mean[log p_joint - log p_marginal] with a separately fitted marginal sharing the
joint's fcs: one extra cheap fit, any number of dep dims, but not an exactly
normalized density.

engine='cpu' (default) uses BLMLQuick; engine='gpu' uses BLMLQuick(engine='gpu') (CuPy).
"""
import functools
import numpy as np, warnings
warnings.filterwarnings('ignore')
from blml_trivial import projected_modes, unprojected_modes
from blml_trivial import (BLMLQuick, radial_groups, is_radial, parse_modes, make_modes, modes_fc, with_fc, modes_trainable,
                          modes_period_trainable)


def _fit(Est, X, fc, kinds, periods, rank=1, complex_g=False, signs=None, dep=None, sw=None):
    """One fit at cut-offs fc (scalar, (d,) or (K, d)) with the template's per-dim kinds / periods; sw: point
    weights (None: unit)."""
    kw = {} if sw is None else dict(sample_weight=sw)
    return Est().fit(X, make_modes(fc, kinds, periods, rank, dep, d=X.shape[1]), signs=signs, complex_g=complex_g, **kw)


def _wmean(v, w):
    """Mean of v, weighted by w (None: plain mean)."""
    return float(np.mean(v)) if w is None else float(np.average(v, weights=w))


def _wquantile(X, q, w):
    """Per-column quantiles q (in %) of X, weighted by w (None: np.percentile)."""
    if w is None or np.all(w == w[0]):              # constant weights: the plain quantiles
        return np.percentile(X, q, axis=0)
    out = np.empty((len(q), X.shape[1])); qs = np.asarray(q, float) / 100.0
    for j in range(X.shape[1]):                      # np.percentile's linear rule on the points repeated w times
        o = np.argsort(X[:, j], kind='stable'); xs = X[o, j]; cw = np.cumsum(w[o]); N = cw[-1]
        h = qs * (N - 1); lo = np.floor(h); hi = np.minimum(lo + 1, np.floor(N - 1))
        at = lambda k: xs[np.minimum(np.searchsorted(cw, k, side='right'), len(xs) - 1)]
        out[:, j] = at(lo) + (h - lo) * (at(hi) - at(lo))
    return out


def _template(modes, d):
    """kinds, periods, dep, rank and fc (K, d; NaN where missing) of a modes template. The selector
    needs the modes to share kinds, periods and groups; they may differ in fc."""
    specs, dep = parse_modes(modes, d, need_fc=False)
    if any(sp['kinds'] != specs[0]['kinds'] or sp['periods'] != specs[0]['periods'] for sp in specs):
        raise ValueError("selector: modes must share kind, period and group per column (fc may differ)")
    return specs[0]['kinds'], specs[0]['periods'], dep, len(specs), modes_fc(modes)


def _blocks(kinds, d):
    """Search coordinates: each radial group is one block (its fc moves together, so the band
    stays round), every other dim is its own block."""
    groups = [dims for _, dims in radial_groups(kinds)]
    grouped = {k for dims in groups for k in dims}
    blocks, seen = [], set()
    for k in range(d):
        if k in grouped:
            g = next(dims for dims in groups if k in dims)
            if g[0] not in seen:
                blocks.append(list(g)); seen.add(g[0])
        else:
            blocks.append([k])
    return blocks


def _start_free(start, free, iqr, d):
    """Scaled start (NaN where none), free mask (trainable), and the columns the shared bracket grows
    (free and without a start). start, free: per column in data units / bool, or None."""
    st = np.full(d, np.nan) if start is None else np.asarray(start, float) * iqr
    fr = np.ones(d, bool) if free is None else np.asarray(free, bool)
    if start is not None and np.any(~fr & ~np.isfinite(st)):
        raise ValueError("a fixed (non-trainable) fc needs a value")
    return st, fr, fr & ~np.isfinite(st)


def _free_blocks(kinds, d, free):
    return [b for b in _blocks(kinds, d) if all(free[b])]


def _tie(fc, kinds):
    """Equalise fc within each radial group (geometric mean)."""
    fc = np.array(fc, dtype=float)
    for _, dims in radial_groups(kinds):
        fc[dims] = np.exp(np.log(fc[dims]).mean())
    return fc

def _estimator_class(engine):
    """The estimator constructor for engine='cpu'|'gpu' (BLMLQuick(engine=...))."""
    if engine not in ('cpu', 'gpu'):
        raise ValueError(f"engine must be 'cpu' or 'gpu', got {engine}")
    return functools.partial(BLMLQuick, engine=engine)

def _scaled_signs(signs, iqr, n_raw, X_val):
    """Signs for fits on scaled data. A callable s(X) in the data's units is wrapped to take scaled
    coordinates. A per-sample array is only usable when every training point is fitted
    (X_val given); with the internal 80/20 split pass a callable instead."""
    if signs is None:
        return None
    if callable(signs):
        return lambda Zs: signs(np.asarray(Zs) * iqr)
    v = np.asarray(signs).reshape(-1)
    if len(v) != n_raw:
        raise ValueError("per-sample signs must have one entry per training point")
    if X_val is None:
        raise ValueError("per-sample signs need X_val (all training points are fitted); "
                         "otherwise pass a callable signs(X)")
    return v

def _no_signs_with_dep(signs, dep):
    if signs is not None and dep is not None:
        raise ValueError("signs with dep=... (the marginal's g has its own signs)")

def _split(Xtr, iqr, seed, X_val, sw=None, sw_val=None):
    """Inner train/validation data in scaled units, with their point weights (None: unit). If X_val is given
    (e.g. a held-out block for forecasting), train on all of Xtr and validate on X_val, scaled by the
    training IQR; otherwise validate on a random 20% of the training points (weights follow their points)."""
    if X_val is not None:
        return Xtr, np.asarray(X_val, dtype=float) / iqr, sw, sw_val
    rng = np.random.default_rng(seed); n = len(Xtr)
    idx = rng.permutation(n); n2 = int(0.8*n)
    sub = lambda a, i: None if a is None else a[i]
    return Xtr[idx[:n2]], Xtr[idx[n2:]], sub(sw, idx[:n2]), sub(sw, idx[n2:])

def _scale_data(Xtr_raw, kinds, periods, sw=None):
    """Compute IQR (weighted by sw), scale data and periods. Returns (Xtr_scaled, periods_scaled, iqr)."""
    q75, q25 = _wquantile(np.asarray(Xtr_raw, float), [75, 25], sw)
    iqr = np.maximum(q75 - q25, 1e-10)
    # radial groups share one scale (geometric mean IQR), so a round band in scaled units is
    # round in the data's units
    groups = radial_groups(kinds)
    for _, dims in groups:
        iqr[dims] = np.exp(np.log(iqr[dims]).mean())
    if groups:
        print("info: selector picks one fc per radial group (" + ", ".join(f"{n} over dims {d}" for n, d in groups)
              + "); the group is rescaled by one common factor, so its dims must already be on a common "
              "scale in the input data.", flush=True)
    Xtr = Xtr_raw / iqr
    periods_scaled = None
    if periods is not None:
        periods_scaled = []
        for k, p in enumerate(periods):
            if p is not None and kinds is not None and k < len(kinds) and str(kinds[k]).partition(':')[0] in ('dirichlet', 'pball'):
                periods_scaled.append(p / iqr[k])
            else:
                periods_scaled.append(p)
    return Xtr, periods_scaled, iqr

def _shared_step(shared_step, d):
    """Multiplicative step of the shared stage. 'double' (default): 2 per dimension, which
    multiplies the effective parameter count prod(fc) by 2^d. 'volume': 2^(1/d), which doubles
    prod(fc) per step at any dimension (error constants scale like prod(fc))."""
    return 2.0 ** (1.0 / d) if shared_step == 'volume' else float(shared_step) if not isinstance(shared_step, str) else 2.0


def _select_shared(Xtr_raw, kinds, periods, seed=0, engine='cpu', X_val=None, shared_step='double', signs=None,
                   complex_g=False, rank=1, start=None, free=None, sw=None, sw_val=None):
    """Mode 'shared': single fc across all scaled dims.
    Doubling from coarsest, stop at first strict vll decrease, keep peak. start / free (per column, data
    units / trainable): fixed columns and trainable ones with a start stay at their value; the shared
    doubling grows the others. Returns (fc_original_per_dim, best_vll).
    """
    Est = _estimator_class(engine)
    Xtr, periods_scaled, iqr = _scale_data(Xtr_raw, kinds, periods, sw)
    Xa, Xb, wa, wb = _split(Xtr, iqr, seed, X_val, sw, sw_val)
    sgs = _scaled_signs(signs, iqr, len(Xtr), X_val)
    step = _shared_step(shared_step, Xa.shape[1])
    jac = np.log(iqr).sum()  # report joint LL in original units
    def vll(fc_s):
        m = _fit(Est, Xa, fc_s, kinds, periods_scaled, rank, complex_g, sgs, sw=wa)
        return _wmean(np.log(np.maximum(m.predict(Xb).ravel(), 1e-300)), wb) - jac
    st, fr, grow = _start_free(start, free, iqr, Xa.shape[1])
    vec = lambda c: np.where(grow, c, st)          # shared value on the growing columns
    # Start at fc=0.2 in scaled space (user-directed 10-13; was 0.8, 0.4)
    fc = 0.2
    best_ll = vll(vec(fc))
    if not grow.any():
        return vec(fc) / iqr, best_ll
    print(f"  [shared] fc={fc:.4f} (scaled), vll={best_ll:.4f}", flush=True)
    prev_ll = best_ll
    while True:
        fc *= step
        ll = vll(vec(fc))
        print(f"  [shared] fc={fc:.4f} (scaled), vll={ll:.4f}", flush=True)
        if ll < prev_ll:
            fc /= step
            break
        if ll > best_ll:
            best_ll = ll
        prev_ll = ll
    fc_orig = vec(fc) / iqr  # per-dim in original units
    return fc_orig, best_ll

def _select_hillclimb(Xtr_raw, kinds, periods, seed=0, init_fc=None, engine='cpu', X_val=None, signs=None,
                      complex_g=False, rank=1, start=None, free=None, sw=None, sw_val=None):
    """Mode 'hillclimb': per-dim fc via coordinate-wise doubling.
    If init_fc (scaled) provided, start from there; else start from coarsest. start / free (per column,
    data units / trainable): given values replace the start there; only trainable blocks move.
    Returns (fc_original_per_dim, best_vll).
    """
    Est = _estimator_class(engine)
    Xtr, periods_scaled, iqr = _scale_data(Xtr_raw, kinds, periods, sw)
    Xa, Xb, wa, wb = _split(Xtr, iqr, seed, X_val, sw, sw_val)
    n2, d = Xa.shape
    jac = np.log(iqr).sum()  # report joint LL in original units
    sgs = _scaled_signs(signs, iqr, len(Xtr), X_val)
    def vll(fc_vec):
        m = _fit(Est, Xa, fc_vec, kinds, periods_scaled, rank, complex_g, sgs, sw=wa)
        return _wmean(np.log(np.maximum(m.predict(Xb).ravel(), 1e-300)), wb) - jac
    # Initialize: from init_fc if provided, else coarsest per-dim
    if init_fc is not None:
        fc = np.array(init_fc, dtype=float)
    else:
        ranges = Xa.max(axis=0) - Xa.min(axis=0)
        ranges = np.maximum(ranges, 1e-10)
        fc = 0.5 * 0.1 * (n2 ** -0.25) / ranges
    st, fr, _ = _start_free(start, free, iqr, d)
    fc = _tie(np.where(np.isfinite(st), st, fc), kinds)
    best_ll = vll(fc)
    print(f"  [hill] init fc={fc} (scaled), vll={best_ll:.4f}", flush=True)
    # Coordinate sweeps until one full sweep accepts no move (user-directed 10-13).
    # Per dim: try 2x exhaustively; if no up-move improved, try 0.5x exhaustively.
    sweep = 0
    while True:
        sweep += 1
        moved = False
        for k in _free_blocks(kinds, d, fr):
            # Try up first
            improved = False
            while True:
                fc_try = fc.copy(); fc_try[k] *= 2.0
                ll = vll(fc_try)
                if ll > best_ll:
                    print(f"  [hill] sweep {sweep} dim {k} x2: vll={ll:.4f} (up)", flush=True)
                    best_ll, fc = ll, fc_try
                    improved = True
                else:
                    break
            # If up didn't improve, try down
            if not improved:
                while True:
                    fc_try = fc.copy(); fc_try[k] *= 0.5
                    ll = vll(fc_try)
                    if ll > best_ll:
                        print(f"  [hill] sweep {sweep} dim {k} x0.5: vll={ll:.4f} (down)", flush=True)
                        best_ll, fc = ll, fc_try
                        improved = True
                    else:
                        break
            moved = moved or improved
        if not moved:
            break
    fc_orig = fc / iqr
    return fc_orig, best_ll

def _select_flat(Xtr_raw, kinds, periods, search='shared_hillclimb', seed=0, init_fc=None, dep=None,
           engine='cpu', cond_method='ratio', X_val=None, shared_step='double', signs=None, complex_g=False, rank=1,
           factors=(np.sqrt(2.0),), separate_modes=False, start=None, free=None, sw=None, sw_val=None):
    """Select fc (start / free: per-column start values in data units, NaN where none, and the trainable
    mask; fixed columns keep their value, trainable ones with a value start there). If dep is given, maximizes conditional validation LL
    (joint minus marginal, shared fcs); otherwise maximizes joint validation LL.
    The objective is inferred from dep — no separate search needed.

    dep: None (joint), an int column index, or a list of int column indices
    for multiple dependent variables. The marginal is fit on the remaining
    dims with the same fc (dep dims removed).
    engine: 'cpu' or 'gpu' (BLMLQuick(engine=...); gpu needs CuPy).
    cond_method: 'ratio' (joint / marginal BLML fits; the only option)
    (exact conditional from the joint fit).
    X_val: optional validation points (same columns as Xtr_raw) for choosing fc, e.g.
    a held-out block when forecasting; if None, a random 20% of the training points.
    signs: optional fixed sign of g (callable s(X) in the data's units, or a per-sample array
    when X_val is given); joint selection only.
    factors: refinement step(s) of the 'auto' search (default sqrt(2)); e.g. (2**0.5, 2**0.25)
    refines further after the sqrt(2) pass.
    separate_modes: with rank K > 1 and search='auto', select one fc per dim per mode.
    complex_g, rank: fit f = sum_k |g_k|^2 with rank K band-limited (complex) g_k (joint or
    conditional; on the GPU the conditional's joint and marginal are then two separate fits).
    """
    Xtr_raw = np.asarray(Xtr_raw)
    d = Xtr_raw.shape[1]
    cond = dep is not None
    if cond:
        dep_idx = sorted(set(int(i) for i in np.atleast_1d(dep)))
        if not dep_idx or any(i < 0 or i >= d for i in dep_idx):
            raise ValueError(f"dep indices {dep_idx} out of range for {d}-dim data")
    _no_signs_with_dep(signs, dep)
    if search == 'auto':
        return _select_auto_flat(Xtr_raw, kinds, periods, dep=dep, seed=seed, engine=engine,
                           cond_method=cond_method, X_val=X_val, signs=signs,
                           complex_g=complex_g, rank=rank, factors=factors, separate_modes=separate_modes,
                           start=start, free=free, sw=sw, sw_val=sw_val)
    sf = dict(start=start, free=free, sw=sw, sw_val=sw_val)
    if search == 'shared':
        if cond:
            return _select_conditional_shared(Xtr_raw, kinds, periods, dep, seed, engine,
                                             cond_method, X_val, shared_step, complex_g, rank, **sf)
        return _select_shared(Xtr_raw, kinds, periods, seed, engine, X_val, shared_step, signs, complex_g, rank, **sf)
    elif search == 'hillclimb':
        if cond:
            return _select_conditional_hillclimb(Xtr_raw, kinds, periods, dep, seed, init_fc,
                                                engine, cond_method, X_val, complex_g, rank, **sf)
        return _select_hillclimb(Xtr_raw, kinds, periods, seed, init_fc, engine, X_val, signs, complex_g, rank, **sf)
    elif search == 'shared_hillclimb':
        # Two-stage: shared first (fixed / started columns held), then hillclimb from it on the free blocks
        if cond:
            fc_shared_orig, _ = _select_conditional_shared(Xtr_raw, kinds, periods, dep, seed,
                                                          engine, cond_method, X_val, shared_step,
                                                          complex_g, rank, **sf)
        else:
            fc_shared_orig, _ = _select_shared(Xtr_raw, kinds, periods, seed, engine, X_val, shared_step, signs,
                                               complex_g, rank, **sf)
        _, _, iqr = _scale_data(Xtr_raw, kinds, periods, sw)
        fc_init_scaled = fc_shared_orig * iqr
        if cond:
            return _select_conditional_hillclimb(Xtr_raw, kinds, periods, dep, seed,
                                                init_fc=fc_init_scaled, engine=engine,
                                                cond_method=cond_method, X_val=X_val,
                                                complex_g=complex_g, rank=rank, free=free, sw=sw, sw_val=sw_val)
        return _select_hillclimb(Xtr_raw, kinds, periods, seed, init_fc=fc_init_scaled,
                                engine=engine, X_val=X_val, signs=signs, complex_g=complex_g, rank=rank, free=free,
                                sw=sw, sw_val=sw_val)
    else:
        raise ValueError(f"search must be 'shared_hillclimb', 'shared', 'hillclimb' or 'auto', got {search}")


def _cond_vll_fn(Xa, Xb, kinds, periods, periods_scaled, dep, iqr, engine='cpu',
                 cond_method='ratio', complex_g=False, rank=1, wa=None, wb=None):
    """Return cvll(fc_vec): mean conditional log-likelihood on validation,
    in original units (subtracts the dep-dim log-Jacobian).

    'ratio': mean[log p_joint - log p_marginal], marginal fit on the other
    dims with the same fc (legacy; not a normalized density).
    complex_g / rank: both fits are rank-K (complex) BLML; the joint and the marginal are then
    fitted separately on either engine (the GPU's fused dep= fit is real rank 1 only).
    """
    Est = _estimator_class(engine)
    d = Xa.shape[1]
    dep_idx = sorted(set(int(i) for i in np.atleast_1d(dep)))
    if not dep_idx or any(i < 0 or i >= d for i in dep_idx):
        raise ValueError(f"dep indices {dep_idx} out of range for {d} dims")
    keep = [k for k in range(d) if k not in dep_idx]
    if not keep:
        raise ValueError("dep covers all dims; need at least one conditioning dim for the marginal")
    if cond_method != 'ratio':
        raise ValueError(f"cond_method must be 'ratio', got {cond_method}")
    kinds_m = [kinds[k] for k in keep] if kinds else None
    periods_m = [periods[k] for k in keep] if periods else None
    periods_scaled_m = [periods_scaled[k] for k in keep] if periods_scaled else None
    jac = np.log(iqr[dep_idx]).sum()  # conditional density Jacobian -> original units
    def cvll(fc_vec):
        fc_vec = list(fc_vec)
        if engine == 'gpu' and not (complex_g or rank > 1):   # one fused fit: joint and marginal share the kernel evaluations
            mj = _fit(Est, Xa, fc_vec, kinds, periods_scaled, dep=dep_idx, sw=wa)
            lj = np.log(np.maximum(mj.predict(Xb).ravel(), 1e-300))
            lm = np.log(np.maximum(mj.predict_marginal(Xb[:, keep]).ravel(), 1e-300))
            return _wmean(lj - lm, wb) - jac
        mj = _fit(Est, Xa, fc_vec, kinds, periods_scaled, rank, complex_g, sw=wa)
        mm = _fit(Est, Xa[:, keep], [fc_vec[k] for k in keep], kinds_m, periods_scaled_m, rank, complex_g, sw=wa)
        lj = np.log(np.maximum(mj.predict(Xb).ravel(), 1e-300))
        lm = np.log(np.maximum(mm.predict(Xb[:, keep]).ravel(), 1e-300))
        return _wmean(lj - lm, wb) - jac
    return cvll


def _select_conditional_shared(Xtr_raw, kinds, periods, dep=0, seed=0, engine='cpu',
                              cond_method='ratio', X_val=None, shared_step='double',
                              complex_g=False, rank=1, start=None, free=None, sw=None, sw_val=None):
    """Shared doubling on conditional validation LL (start / free as _select_shared).
    Returns (fc_orig, best_cvll)."""
    Xtr, periods_scaled, iqr = _scale_data(Xtr_raw, kinds, periods, sw)
    Xa, Xb, wa, wb = _split(Xtr, iqr, seed, X_val, sw, sw_val)
    step = _shared_step(shared_step, Xa.shape[1])
    cvll = _cond_vll_fn(Xa, Xb, kinds, periods, periods_scaled, dep, iqr, engine,
                        cond_method, complex_g, rank, wa, wb)
    # Start at fc=0.2 in scaled space, same as joint shared (vector: shared across dims)
    d = Xa.shape[1]
    st, fr, grow = _start_free(start, free, iqr, d)
    vec = lambda c: np.where(grow, c, st)
    fc = 0.2
    best_ll = cvll(vec(fc))
    if not grow.any():
        return vec(fc) / iqr, best_ll
    print(f"  [cond-shared] fc={fc:.4f} (scaled), cvll={best_ll:.4f}", flush=True)
    prev_ll = best_ll
    while True:
        fc = fc * step
        ll = cvll(vec(fc))
        print(f"  [cond-shared] fc={fc:.4f} (scaled), cvll={ll:.4f}", flush=True)
        if ll < prev_ll:
            fc = fc / step
            break
        if ll > best_ll:
            best_ll = ll
        prev_ll = ll
    fc_orig = vec(fc) / iqr  # per-dim in original units
    return fc_orig, best_ll


def _select_conditional_hillclimb(Xtr_raw, kinds, periods, dep=0, seed=0, init_fc=None,
                                engine='cpu', cond_method='ratio', X_val=None,
                                complex_g=False, rank=1, start=None, free=None, sw=None, sw_val=None):
    """Per-dim doubling hillclimb (2x first, then 0.5x) on conditional validation LL.
    Returns (fc_original_per_dim, best_cvll).
    """
    Xtr, periods_scaled, iqr = _scale_data(Xtr_raw, kinds, periods, sw)
    Xa, Xb, wa, wb = _split(Xtr, iqr, seed, X_val, sw, sw_val)
    n2, d = Xa.shape
    cvll = _cond_vll_fn(Xa, Xb, kinds, periods, periods_scaled, dep, iqr, engine,
                        cond_method, complex_g, rank, wa, wb)
    if init_fc is not None:
        fc = np.array(init_fc, dtype=float)
    else:
        fc = np.full(d, 0.2)
    st, fr, _ = _start_free(start, free, iqr, d)
    fc = _tie(np.where(np.isfinite(st), st, fc), kinds)
    best_ll = cvll(fc)
    print(f"  [cond-hill] init fc={fc} (scaled), cvll={best_ll:.4f}", flush=True)
    # Coordinate sweeps until one full sweep accepts no move (user-directed 10-13).
    # Per dim: try 2x exhaustively; if no up-move improved, try 0.5x exhaustively.
    sweep = 0
    while True:
        sweep += 1
        moved = False
        for k in _free_blocks(kinds, d, fr):
            improved = False
            while True:
                fc_try = fc.copy(); fc_try[k] *= 2.0
                ll = cvll(fc_try)
                if ll > best_ll:
                    print(f"  [cond-hill] sweep {sweep} dim {k} x2: cvll={ll:.4f} (up)", flush=True)
                    best_ll, fc = ll, fc_try
                    improved = True
                else:
                    break
            if not improved:
                while True:
                    fc_try = fc.copy(); fc_try[k] *= 0.5
                    ll = cvll(fc_try)
                    if ll > best_ll:
                        print(f"  [cond-hill] sweep {sweep} dim {k} x0.5: cvll={ll:.4f} (down)", flush=True)
                        best_ll, fc = ll, fc_try
                        improved = True
                    else:
                        break
            moved = moved or improved
        if not moved:
            break
    fc_orig = fc / iqr
    return fc_orig, best_ll


# ---------------------------------------------------------------- 'auto' (default for Paper 2)
def _harmonics(fc_s, Ps):
    return int(np.floor(fc_s * Ps / 2.0))

def _move(fc, block, factor, kinds, periods_scaled):
    """fc (scaled) with the dims in `block` (one dim, or a whole radial group) multiplied by factor.
    A periodic dimension's density only changes when its harmonic count N = floor(fc P / 2) does,
    so a move that leaves N unchanged is pushed to the next N in that direction (flat steps would
    stop any local search)."""
    new = fc.copy(); new[block] *= factor
    if len(block) > 1:
        return new
    k = block[0]
    if kinds is not None and str(kinds[k]).partition(':')[0] == 'dirichlet':
        Ps = periods_scaled[k]
        n_old, n_new = _harmonics(fc[k], Ps), _harmonics(new[k], Ps)
        if n_new == n_old:
            n_new = n_old + 1 if factor > 1 else n_old - 1
            if n_new < 0:
                return None
            new[k] = (n_new + 0.5) * 2.0 / Ps
    return new

def _search(obj, d, kinds, periods_scaled, fc0=0.2, patience=2, max_sweeps=6, tag='auto',
            factors=(np.sqrt(2.0),), start=None, free=None):
    """Stage 1: shared search. Bracket the optimum by doubling every fc (continuing `patience`
    steps past a non-improvement, for validation noise), then refine on the volume grid: every
    fc times 2**(j/d), so prod(fc), which sets the error constant, changes by a factor 2 per
    grid step at any d. Stage 2: coordinate refinement
    from there, per-dimension steps of sqrt(2), until a full sweep improves nothing.
    start (scaled, NaN where none) / free (trainable): stage 1 grows only free dims without a start
    (skipped if there are none); stage 2 moves only free blocks; fixed dims stay at their value."""
    cache = {}
    def f(fc):
        key = tuple(np.round(np.log(fc), 6))
        if key not in cache:
            cache[key] = obj(fc)
        return cache[key]
    st = np.full(d, np.nan) if start is None else np.asarray(start, float)
    fr = np.ones(d, bool) if free is None else np.asarray(free, bool)
    grow = fr & ~np.isfinite(st)
    dg = int(grow.sum())
    # 1a: bracket with doublings of every growing fc; 1b: refine on the volume grid 2**(j/dg) in the bracket
    fc = np.where(grow, float(fc0), st); best_fc, best = fc.copy(), f(fc); bad = 0
    while dg and bad <= patience:
        fc = np.where(grow, fc * 2.0, fc)
        ll = f(fc)
        if ll > best:
            best, best_fc, bad = ll, fc.copy(), 0
        else:
            bad += 1
    # Volume grid centre * 2**(j/dg), walked outward from the best doubling and stopped at the
    # first step that does not improve: up first, and down only if no up-step improved.
    centre = best_fc.copy()
    for sign in (1, -1):
        improved = False
        for j in range(1, dg):
            cand = np.where(grow, centre * 2.0 ** (sign * j / dg), centre)
            ll = f(cand)
            if ll > best:
                best, best_fc, improved = ll, cand, True
            else:
                break
        if improved:
            break
    if dg:
        print(f"  [{tag}] shared best fc={best_fc[grow][0]:.3f} ll={best:.4f}", flush=True)
    fc = best_fc
    for factor in factors:
        for sweep in range(max_sweeps):
            moved = False
            for blk in _free_blocks(kinds, d, fr):
                for fac in (factor, 1.0 / factor):
                    stepped = False
                    while True:
                        cand = _move(fc, blk, fac, kinds, periods_scaled)
                        if cand is None:
                            break
                        ll = f(cand)
                        if ll > best + 1e-6:
                            best, fc, moved, stepped = ll, cand, True, True
                        else:
                            break
                    if stepped:          # moved up: no need to try down this sweep
                        break
            if not moved:
                break
    print(f"  [{tag}] final fc={np.round(fc, 4)} ll={best:.4f} ({len(cache)} fits)", flush=True)
    return fc, best

def _select_auto_flat(Xtr_raw, kinds, periods, dep=None, seed=0, engine='cpu', cond_method='ratio',
                X_val=None, fc0=0.2, signs=None, complex_g=False, rank=1, factors=(np.sqrt(2.0),),
                separate_modes=False, start=None, free=None, sw=None, sw_val=None):
    """Default selector: volume-scale shared search + harmonic-aware coordinate refinement,
    on the joint (dep=None) or conditional (dep=int) validation log-likelihood.
    separate_modes=True (rank K > 1): every mode gets its own fc per dim; the search runs on the
    flattened K x d vector (stage 1 moves all modes together, stage 2 one (mode, dim) block at a
    time). Returns fc of shape (d,), or (K, d) for separate modes, in the data's units."""
    Xtr_raw = np.asarray(Xtr_raw, dtype=float)
    Xtr, periods_scaled, iqr = _scale_data(Xtr_raw, kinds, periods, sw)
    Xa, Xb, wa, wb = _split(Xtr, iqr, seed, X_val, sw, sw_val)
    d = Xa.shape[1]
    _no_signs_with_dep(signs, dep)
    kinds_l = list(kinds) if kinds is not None else ['sinc'] * d
    ps_l = list(periods_scaled) if periods_scaled is not None else [None] * d
    K = int(rank) if (separate_modes and int(rank) > 1) else 1
    if K > 1:
        if signs is not None:
            raise ValueError("separate modes estimate the phases; do not pass signs")
        Est = _estimator_class(engine)
        dep_idx = None if dep is None else sorted(set(int(i) for i in np.atleast_1d(dep)))
        keep = None if dep_idx is None else [k for k in range(d) if k not in dep_idx]
        jac = np.log(iqr[dep_idx]).sum() if dep_idx is not None else np.log(iqr).sum()
        def obj(fcv):
            F = np.asarray(fcv, float).reshape(K, d)
            mj = _fit(Est, Xa, F, kinds_l, ps_l, complex_g=complex_g, sw=wa)   # one fc row per mode
            lj = np.log(np.maximum(mj.predict(Xb).ravel(), 1e-300))
            if dep_idx is None:
                return _wmean(lj, wb) - jac
            mm = _fit(Est, Xa[:, keep], F[:, keep], [kinds_l[q] for q in keep], [ps_l[q] for q in keep],
                      complex_g=complex_g, sw=wa)
            lm = np.log(np.maximum(mm.predict(Xb[:, keep]).ravel(), 1e-300))
            return _wmean(lj - lm, wb) - jac
        kinds_all = sum((_mode_kinds(kinds_l, k) for k in range(K)), [])
        st = None if start is None else (np.broadcast_to(np.asarray(start, float), (K, d)) * iqr[None]).ravel()
        fr = None if free is None else np.broadcast_to(np.asarray(free, bool), (K, d)).ravel()
        fc, best = _search(obj, d * K, kinds_all, ps_l * K, fc0=fc0, factors=factors, start=st, free=fr)
        return fc.reshape(K, d) / iqr[None], best
    if dep is not None:
        obj = _cond_vll_fn(Xa, Xb, kinds, periods, periods_scaled, dep, iqr, engine, cond_method,
                           complex_g, rank, wa, wb)
    else:
        Est = _estimator_class(engine); jac = np.log(iqr).sum()
        sgs = _scaled_signs(signs, iqr, len(Xtr), X_val)
        def obj(fc_vec):
            m = _fit(Est, Xa, fc_vec, kinds, periods_scaled, rank, complex_g, sgs, sw=wa)
            return _wmean(np.log(np.maximum(m.predict(Xb).ravel(), 1e-300)), wb) - jac
    st = None if start is None else np.asarray(start, float) * iqr
    fc, best = _search(obj, d, kinds, periods_scaled, fc0=fc0, factors=factors, start=st, free=free)
    return fc / iqr, best


# ---------------------------------------------------------------- binned counts (e.g. camera pixels)
def _mode_kinds(kinds, k):
    """Radial group tags made unique per mode, so each mode's group moves as its own block."""
    out = []
    for kd in kinds:
        if is_radial(kd):
            base, _, tag = kd.partition(':')
            out.append(f"{base}:{tag or 'g'}_m{k}")
        else:
            out.append(kd)
    return out

def _select_counts_flat(Z, w_tr, w_va, h, kinds=None, periods=None, rank=1, complex_g=False,
                  separate_modes=False, fc0=None, fc_max=None, tag='counts',
                  factors=(np.sqrt(2.0),), engine='cpu', start=None, free=None, dep=None, **fit_kw):
    """Select cut-offs for counts on a fixed lattice (BLMLQuick with h in the modes) : fit on (Z, w_tr), score the mean log-probability of
    the validation counts w_va, with bin probabilities f(z_b) normalised over the bins given (so Z
    should be the full pixel grid, zero counts included: over occupied bins only, a flat f looks best). dep columns:
    the conditional -- the fit's joint / marginal ratio at each bin, normalised over the bins sharing its other
    columns. Same search
    as select_auto (shared doubling, then per-block refinement), no grid. fc is capped at
    fc_max (default 1/h per dim: above it the bin lattice aliases the band).
    separate_modes=True (rank K > 1): each mode gets its own fc per dim; the search runs on the
    flattened K x d vector, one block per (mode, dim or radial group).
    Returns (fc, best) with fc of shape (d,) or, for separate modes, (K, d)."""
    from blml_trivial import with_lattice
    Z = np.atleast_2d(np.asarray(Z, float)); d = Z.shape[1]
    kinds = list(kinds) if kinds is not None else ['sinc'] * d
    periods = list(periods) if periods is not None else [None] * d
    h_vec = np.broadcast_to(np.asarray(h, float), (d,)).copy()
    cap = np.broadcast_to(np.asarray(fc_max if fc_max is not None else 1.0 / h_vec, float), (d,))
    w_tr = np.asarray(w_tr, float).ravel(); w_va = np.asarray(w_va, float).ravel()
    K = int(rank) if separate_modes else 1
    rest = None if dep is None else [j for j in range(d) if j not in dep]
    kinds_all = sum((_mode_kinds(kinds, k) for k in range(K)), []) if K > 1 else kinds
    periods_all = periods * K
    def obj(fcv):
        F = np.asarray(fcv, float).reshape(K, d)
        if np.any(F > cap[None] * (1 + 1e-9)):
            return -np.inf
        m = (BLMLQuick(dtype=np.float64, tol=fit_kw.get('tol', 1e-5)) if engine == 'cpu'
             else _estimator_class(engine)(tol=fit_kw.get('tol', 1e-5)))
        m.fit(Z, with_lattice(make_modes(F if K > 1 else F[0], kinds, periods, rank, dep=dep, d=d), h_vec),
              sample_weight=w_tr, complex_g=complex_g)
        if dep is None:
            P = np.maximum(m.predict(Z), 1e-300); P = P / P.sum()
        else:
            P = np.maximum(m.predict_ratio(Z), 1e-300)
            grp = np.unique(np.ascontiguousarray(Z[:, rest]), axis=0, return_inverse=True)[1].ravel()
            P = P / np.bincount(grp, weights=P)[grp]
        return float(w_va @ np.log(P) / w_va.sum())
    fc0 = float(fc0) if fc0 is not None else 0.1 * float(cap.min())
    # refinement steps sqrt(2), then 2^(1/4), 2^(1/8): resolves fc to about 9 %
    # factors: refinement steps (default sqrt(2)); e.g. (2**0.5, 2**0.25, 2**0.125) resolves fc to ~9 %
    fc, best = _search(obj, d * K, kinds_all, periods_all, fc0=fc0, tag=tag, factors=factors, start=start, free=free)
    return (fc.reshape(K, d) if K > 1 else fc), best


# ---------------------------------------------------------------- public API: modes in, modes out
def select(Xtr_raw, modes, search='shared_hillclimb', seed=0, init_fc=None, engine='cpu', cond_method='ratio', X_val=None,
           shared_step='double', signs=None, complex_g=False, factors=(np.sqrt(2.0),), separate_modes=False,
           sample_weight=None, val_weight=None):
    """Select fc for a modes template (blml_trivial.parse_modes; rank K
    = len(modes), dep columns give the conditional objective). The modes must share kind, period and
    group per column. separate_modes (K > 1, search='auto'): one fc per column per mode; otherwise every
    mode gets the same fc. search: 'shared_hillclimb' (default), 'shared', 'hillclimb', 'auto'; dep columns in modes switch to the
    conditional objective
    (init_fc in scaled units for the hillclimb searches). Returns (modes with fc in the data's units, best
    validation log-likelihood).
    Template fc (blml_trivial.KERNEL_PARAMS): missing -> chosen from the default start; a trainable value
    -> the search starts there; fixed (trainable=False) -> kept as given. The search is over the lattice
    parameters (LATTICE_PARAMS: fc); every other parameter (e.g. a trainable period) is held at its value and
    returned as given (BLMLQuickStreaming trains it).
    sample_weight / val_weight: point weights of Xtr_raw / X_val (None: unit) -- in the fits, the validation
    means and the IQR scaling; integer weights equal repeating the points."""
    proj, lmodes = projected_modes(modes)                 # mapped units: search fc at the given map
    if proj is not None:
        sel, best = select(proj(Xtr_raw), lmodes, search=search, seed=seed, init_fc=init_fc, engine=engine,
                           cond_method=cond_method, X_val=None if X_val is None else proj(X_val), shared_step=shared_step,
                           signs=signs, complex_g=complex_g, factors=factors, separate_modes=separate_modes,
                           sample_weight=sample_weight, val_weight=val_weight)
        return unprojected_modes(proj, sel, modes), best
    Xtr_raw = np.asarray(Xtr_raw)
    sw = None if sample_weight is None else np.asarray(sample_weight, float)
    sw_val = None if val_weight is None else np.asarray(val_weight, float)
    kinds, periods, dep, K, F = _template(modes, Xtr_raw.shape[1])
    T = modes_trainable(modes)
    if separate_modes and K > 1:
        start, free = F, T
    else:
        if not (np.allclose(F, F[:1], equal_nan=True) and (T == T[:1]).all()):
            raise ValueError("selector: without separate_modes the modes share fc (values and trainability)")
        start, free = F[0], T[0]
    start = None if np.all(np.isnan(start)) else start
    free = None if np.all(free) else free
    if search == 'auto':
        fc, best = _select_auto_flat(Xtr_raw, kinds, periods, dep=dep, seed=seed, engine=engine,
                                     cond_method=cond_method, X_val=X_val, signs=signs, complex_g=complex_g,
                                     rank=K, factors=factors, separate_modes=separate_modes, start=start, free=free,
                                     sw=sw, sw_val=sw_val)
    else:
        if separate_modes:
            raise ValueError("separate_modes needs search='auto'")
        fc, best = _select_flat(Xtr_raw, kinds, periods, search=search, seed=seed, init_fc=init_fc, dep=dep,
                                engine=engine, cond_method=cond_method, X_val=X_val, shared_step=shared_step,
                                signs=signs, complex_g=complex_g, rank=K, start=start, free=free, sw=sw, sw_val=sw_val)
    return with_fc(modes, fc), best


def select_counts(Z, w_tr, w_va, h, modes, complex_g=False, separate_modes=False, fc0=None, fc_max=None,
                  tag='counts', factors=(np.sqrt(2.0),), engine='cpu', **fit_kw):
    """Select fc for counts on a fixed lattice (see _select_counts_flat) from a modes template (rank K =
    len(modes); dep columns give the conditional objective; template fc as in select: missing, a trainable start, or fixed; counts are
    in the data's units, so no IQR scaling). Returns (modes with the selected fc, best)."""
    from blml_trivial import lattice_fixed
    lattice_fixed(modes, 'select_counts')                # counts keep their lattice: no mapped units
    Z = np.atleast_2d(np.asarray(Z, float))
    kinds, periods, dep, K, F = _template(modes, Z.shape[1])
    T = modes_trainable(modes)
    rows = (F, T) if (separate_modes and K > 1) else (F[0], T[0])
    start = None if np.all(np.isnan(rows[0])) else np.asarray(rows[0], float).ravel()
    free = None if np.all(rows[1]) else np.asarray(rows[1], bool).ravel()
    fc, best = _select_counts_flat(Z, w_tr, w_va, h, kinds=kinds, periods=periods, rank=K, complex_g=complex_g,
                                   separate_modes=separate_modes, fc0=fc0, fc_max=fc_max, tag=tag,
                                   factors=factors, engine=engine, start=start, free=free, dep=dep, **fit_kw)
    return with_fc(modes, fc), best
