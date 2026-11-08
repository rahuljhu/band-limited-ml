"""The product kernel on a lattice. Each unit (a sinc or dirichlet column, a radial group; blml_trivial's kernel
classes) owns its lattice step, its bin centres and its value block (step / quantize / block / dblock);
ProductKernel puts them together for the row kernels (blml_local.rows, blml_local_gpu.rows) and caches them.

  ProductKernel(kinds, periods, fold, engine, f32)
    step(fc, periods, n, bin_scale)   per-column lattice steps for n points: bin_scale n^-1/4 / fc, a folded
                                      period tiled by whole bins (steps: for given target widths)
    binned(fc, h)                     the kernel on the lattice h as a Gram function of bin centres
    quantize(X, h, cols) / coords     bin centres / integer lattice coordinates of the columns cols
    tables(fc, periods, h, Zi, want)  the flat value arrays at (fc, periods) for the bins Zi, cached: rebuilt
                                      when a parameter moves, the bins outgrow the span the rows were built for
                                      (padded 25%), or a rank column meets a value its table lacks (then built
                                      with that value added to the ones it had); on the GPU built on the device
    k0(fc, periods)                   K(0) per column and of the product (S_bb)
  rank_coords(Z, tab)                 coordinates as the row kernels read them (rank columns: ranks)

Every column has its own block in one flat array (base[k], no padding to the longest column); a radial group's
row is on its own r^2 grid (R). The derivative blocks (dT) follow the same layout for the wanted units."""
import numpy as np
from blml_trivial import kernel_class, kernel_units, unit_theta, radial_groups, is_radial, is_periodic_group, shift_index

KIND_CODES = dict(offset=0, circular=1, radial=2, rank=3, torus=4)


def _rankable(kind):
    """Columns whose block may be a rank table over their occupied values (sinc, slits)."""
    return str(kind).partition(':')[0] in ('sinc', 'slits')


class ProductKernel:
    def __init__(self, kinds, periods=None, fold=None, engine='cpu', f32=False):
        self.kinds = list(kinds); self.d = len(self.kinds)
        self.periods = list(periods) if periods is not None else [None] * self.d
        self.fold = [True] * self.d if fold is None else list(fold)
        self.units = kernel_units(self.kinds, self.d)
        self.groups = radial_groups(self.kinds)
        self.engine, self.f32 = engine, f32
        self._cache = {}
        self.cache_size = 3                                # current, current with derivatives, one candidate (per mode)

    # ---- lattice
    def _col(self, k, periods):
        """(kernel, th, fold kwargs) of column k for its 1-D methods (radial columns bin as sinc)."""
        kind = self.kinds[k]
        if is_periodic_group(kind):                        # a periodic ball's column bins as a dirichlet column
            kind = 'dirichlet'
        th = unit_theta(kind, np.zeros(self.d), periods, [k])
        if is_radial(kind):
            kind = 'sinc'
        kern = kernel_class(kind)
        return kern, th, (dict(fold=self.fold[k]) if str(kind).partition(':')[0] == 'dirichlet' else {})

    def step(self, fc, periods, n, bin_scale):
        """(h, h_target): the target widths bin_scale n^-1/4 / fc and the steps the units make of them."""
        h_t = bin_scale * float(n) ** -0.25 / np.asarray(fc, float)
        return self.steps(h_t, periods), h_t

    def steps(self, h_target, periods=None):
        """The units' lattice steps for target widths (a folded period tiled by whole bins)."""
        periods = self.periods if periods is None else periods
        h_t = np.broadcast_to(np.asarray(h_target, float), (self.d,))
        if not np.all(h_t > 0):
            raise ValueError("bin width h must be positive")
        h = np.empty(self.d)
        for k in range(self.d):
            kern, th, kw = self._col(k, periods)
            h[k] = kern.step(th, h_t[k], **kw)
        return h

    def binned(self, fc, h):
        """The kernel at cut-offs fc on the lattice of steps h, as a Gram function of bin centres (BinnedGram)."""
        return BinnedGram(self, fc, self.steps(h))

    def quantize(self, X, h, cols=None, periods=None):
        periods = self.periods if periods is None else periods
        cols = list(range(self.d)) if cols is None else list(cols)
        X = np.atleast_2d(np.asarray(X, float)); h = np.asarray(h, float)
        Xq = np.empty_like(X)
        for i, k in enumerate(cols):
            kern, th, kw = self._col(k, periods)
            Xq[:, i] = kern.quantize(th, X[:, i], float(h[i] if len(h) == len(cols) else h[k]), **kw)
        return Xq + 0.0                                    # -0.0 -> +0.0 (bins are unique by their bytes)

    def coords(self, X, h, cols=None, periods=None, centres=False):
        """Integer lattice coordinates; centres: X are bin centres already (taken as they are)."""
        cols = list(range(self.d)) if cols is None else list(cols)
        hh = np.asarray(h, float); hh = hh if len(hh) == len(cols) else hh[cols]
        Xq = np.asarray(X, float) if centres else self.quantize(X, hh, cols, periods)
        return np.round(Xq / hh).astype(np.int64)

    # ---- value tables
    def tables(self, fc, periods, h, Zi, want=()):
        periods = self.periods if periods is None else periods
        span = Zi.max(0) - Zi.min(0)
        key = (tuple(np.round(np.asarray(fc, float), 12)), tuple(periods), tuple(want), tuple(self.fold),
               tuple(np.round(np.asarray(h, float), 12)))
        hit = self._cache.get(key)
        if hit is not None and np.all(span <= hit[0]) and all(      # rank columns: every bin value in the table
                v is None or np.isin(Zi[:, k], v).all() for k, v in enumerate(hit[1]['vals'])):
            return hit[1]
        occ = [np.unique(Zi[:, k]) if _rankable(kd) else None for k, kd in enumerate(self.kinds)]
        if hit is not None:                                # a miss on new values: added to the cached ones
            occ = [o if v is None else np.union1d(o, v) for o, v in zip(occ, hit[1]['vals'])]
        alloc = np.ceil(np.maximum(span, 1) * 1.25).astype(np.int64)
        if self.engine == 'gpu':                          # built on the device, stays there
            import cupy as cp
            from blml_gpu import _J01
            tab = self._assemble(fc, periods, h, alloc, want, occ, cp, _J01)
            if self.f32:                                   # the kernels read float32 tables
                tab = {k: (v.astype(cp.float32) if isinstance(v, cp.ndarray) else v) for k, v in tab.items()}
        else:
            tab = self._assemble(fc, periods, h, alloc, want, occ, np, None)
        while len(self._cache) >= self.cache_size:          # oldest out
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (alloc, tab)
        return tab

    def clear(self):
        self._cache = {}

    def k0(self, fc, periods=None):
        """(diag, sbb): K(0) per column (radial columns 1; their group's K(0) enters sbb) and S_bb = K(0) of the
        product kernel, from the units' exact formulas (float64)."""
        periods = self.periods if periods is None else periods
        diag = np.ones(self.d); sbb = 1.0
        for kind, dims in self.units:
            v = float(np.ravel(kernel_class(kind).eval(unit_theta(kind, fc, periods, dims), [np.zeros(1) for _ in dims]))[0])
            if not is_radial(kind):
                diag[dims[0]] = v
            sbb *= v
        return diag, sbb

    def _assemble(self, fc, periods, h, span, want, vals, xp, j01):
        """The arrays the row kernels take: T (flat), base / size per column, Jk (offset rows' half width), mk
        (rank tables' size; circular rows' bins in nbins), kinds codes, radial rows R (groups x r grid; a periodic
        ball's group: its torus table, index row-major over its columns' circular offsets, bins in nbins), and one
        derivative block per entry q of want = [(unit index, parameter or 'shift' / ('shift', i))]: a column's row
        at dT[dbase[q]:] (wcol[q] its column), or a radial group's row dR[q] (wgrp[q]; a shift row multiplied by the
        offset of column wcol[q]); wunit[q] the unit. A unit may appear in several entries (e.g. its fc and the
        coordinate shift of an input map)."""
        d = self.d; fc = np.asarray(fc, float); h = np.asarray(h, float)
        span = np.maximum(np.asarray(span, np.int64), 1)
        units = self.units; U = len(units); G = len(self.groups)
        kinds_arr = np.zeros(d, np.int64); nbins = np.zeros(d, np.int64); grp = np.full(d, -1, np.int64)
        dim_unit = np.full(d, -1, np.int64); grp_unit = np.zeros(max(G, 1), np.int64)
        Jk = np.zeros(d, np.int64); mk = np.zeros(d, np.int64); rvals = [None] * d
        rows, drows, rrows, drr = {}, {}, {}, {}
        for ui, (kind, dims) in enumerate(units):
            th = unit_theta(kind, fc, periods, dims); kern = kernel_class(kind)
            if is_radial(kind):
                g = next(i for i, (_, gd) in enumerate(self.groups) if list(gd) == list(dims))
                code, row, ms = kern.block(th, h[dims], span[dims], xp=xp, j01=j01)
                grp[dims] = g; grp_unit[g] = ui; kinds_arr[dims] = code; rrows[g] = row
                if code == KIND_CODES['torus']:            # a periodic ball: its columns' bins per period
                    if not all(self.fold[k] for k in dims):
                        raise ValueError(f"pball over columns {dims}: needs a folded lattice (fixed periods)")
                    nbins[dims] = ms
                continue
            k = dims[0]; dim_unit[k] = ui; J = int(span[k]); Jk[k] = J
            kw = dict(fold=self.fold[k]) if str(kind).partition(':')[0] == 'dirichlet' else {}
            code, row, m = kern.block(th, float(h[k]), J, vals[k] if _rankable(kind) else None, xp=xp, **kw)
            kinds_arr[k] = code; rows[k] = row
            if code == KIND_CODES['circular']:
                nbins[k] = m
            elif code == KIND_CODES['rank']:
                mk[k] = m; rvals[k] = np.asarray(vals[k], np.int64)
        nW = len(want)                                     # one derivative block per wanted entry (a unit may have several)
        wunit = np.zeros(max(nW, 1), np.int64); wcol = np.full(max(nW, 1), -1, np.int64)
        wgrp = np.full(max(nW, 1), -1, np.int64)
        for q, (ui, p) in enumerate(want):
            kind, dims = units[ui]; wunit[q] = ui
            th = unit_theta(kind, fc, periods, dims); kern = kernel_class(kind)
            if is_radial(kind):
                g = int(grp[dims[0]]); wgrp[q] = g
                i = shift_index(p)
                if i is not None:                          # times the offset of that column (the row kernels; a
                    wcol[q] = dims[i]                      # torus row is the derivative itself, read as is)
                drr[q] = kern.dblock(th, p, h[dims], span[dims], xp=xp, j01=j01)[1]
            else:
                k = dims[0]; wcol[q] = k
                kw = dict(fold=self.fold[k]) if str(kind).partition(':')[0] == 'dirichlet' else {}
                drows[q] = kern.dblock(th, p, float(h[k]), int(Jk[k]), rvals[k], xp=xp, **kw)[1]

        def flat(rw, n):                                   # blocks 0..n-1 -> one array + start offsets
            base = np.zeros(max(n, 1), np.int64); parts = []; at = 0
            for k in range(n):
                if k in rw:
                    base[k] = at; parts.append(xp.asarray(rw[k], dtype=float)); at += len(rw[k])
            return (xp.concatenate(parts) if parts else xp.zeros(1)), base
        T, base = flat(rows, d); dT, dbase = flat(drows, nW)
        LR = max([len(v) for v in list(rrows.values()) + list(drr.values())] or [2])
        R = xp.zeros((max(G, 1), LR)); dR = xp.zeros((max(nW, 1), LR))
        for g, v in rrows.items():
            R[g, :len(v)] = v
        for q, v in drr.items():
            dR[q, :len(v)] = v
        size = np.array([len(rows[k]) if k in rows else 0 for k in range(d)], np.int64)
        return dict(T=T, base=base, size=size, Jk=Jk, mk=mk, vals=rvals, kinds=kinds_arr, nbins=nbins, grp=grp,
                    R=R, dim_unit=dim_unit, grp_unit=grp_unit, U=U, dT=dT, dbase=dbase, dR=dR, wunit=wunit,
                    wcol=wcol, wgrp=wgrp, nW=nW)


def rank_coords(Z, tab):
    """Lattice coordinates as the row kernels read them: a rank column's coordinate replaced by its rank among the
    table's values. A coordinate not among them means the tables predate the bin (fetch tables after adding
    bins): an error, never a wrong value."""
    rc = [k for k, v in enumerate(tab['vals']) if v is not None]
    Z = np.asarray(Z, np.int64)
    if not rc:
        return np.ascontiguousarray(Z)
    Z = Z.copy()
    for k in rc:
        v = tab['vals'][k]; r = np.minimum(np.searchsorted(v, Z[:, k]), len(v) - 1)
        if len(Z) and not np.array_equal(v[r], Z[:, k]):
            raise RuntimeError(f"rank column {k}: a bin value is not in the kernel table (stale tables)")
        Z[:, k] = r
    return np.ascontiguousarray(Z)


class BinnedGram:
    """The product kernel at cut-offs fc on a lattice of steps h (ProductKernel.binned): K(X, Y) between the bin
    centres of X and Y, (len X, len Y), from the units' exact formulas at lattice differences (so exactly the Gram
    of the binned data the solvers' tables hold); quantize / coords snap points to the lattice; k0 = K(0) = S_bb.
    A Gram function in the sense of BLMLTrivial's kernel= (with _overlap = 'gram')."""
    _overlap = "gram"

    def __init__(self, pk, fc, h):
        self.pk, self.fc, self.h = pk, np.broadcast_to(np.asarray(fc, float), (pk.d,)).copy(), np.asarray(h, float)
        self.kinds, self.periods, self.d = pk.kinds, pk.periods, pk.d

    def quantize(self, X):
        return self.pk.quantize(X, self.h)

    def coords(self, X):
        return self.pk.coords(X, self.h)

    def k0(self):
        return self.pk.k0(self.fc)[1]

    def __call__(self, X, Y):
        return self.gram_int(self.coords(X), self.coords(Y))

    def gram_int(self, Xi, Yi):
        """The Gram from integer lattice coordinates."""
        Xi, Yi = np.atleast_2d(Xi), np.atleast_2d(Yi)
        G = np.ones((len(Xi), len(Yi)))
        for kind, dims in self.pk.units:
            th = unit_theta(kind, self.fc, self.periods, dims)
            diffs = [(Xi[:, k, None] - Yi[None, :, k]) * self.h[k] for k in dims]
            G *= kernel_class(kind).eval(th, diffs)
        return G

    def __repr__(self):
        return f"BinnedGram(fc={self.fc}, h={self.h}, kinds={self.kinds})"
