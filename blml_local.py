"""Touched-bin machinery for streaming BLML (blml_rolling.BLMLQuickStreaming).

A batch updates only the coefficients of the bins it falls in (I, |I| <= batch size), never the whole lattice.
Every term of a batch is a sum over the B bins held, for each touched bin or batch point: O(d |I| B) time.
Value and every unit's d/dlog theta come from one pass over the unit factors with prefix / suffix products
(O(d) per pair, not O(d^2)), and nothing of size |I| x B is stored by the CPU path (memory O(B + d |I|)).
Summed over an epoch (sum_t |I_t| B_t ~ n^2 / 2 when every point has its own bin): O(d n^2), about two sweeps
of the one-shot solver.

  (tables and binning: blml_kernels.ProductKernel)
  rows           : s_r = sum_j S(Z_r, Z_j) w_j and d s_r / d log theta_u on the lattice (numba, parallel)
  point_rows     : the same with the continuous kernel at arbitrary points (kernel classes; numpy or cupy)
  FusedFit       : the densities' state -- bins, coefficients, counts, the normalisers Q = e'Se -- and their
                   touched-bin update, parameter steps, re-binning and prediction (each mode its own kernel).
"""
import numpy as np
from blml_kernels import ProductKernel, rank_coords
from blml_trivial import (cd_root, cd_line_search, SincKernel, DirichletKernel, RadialKernel, kernel_class, kernel_units, unit_theta,
                          radial_groups, is_radial, RADIAL_Q)

try:
    from numba import njit, prange
except ImportError:                                   # pragma: no cover
    njit, prange = None, range

STEPS = 0.5 ** np.arange(0, 31)


class GrowBuf:
    """Rows appended in amortised O(rows) (capacity doubles), read as a view of the first n rows (the same view
    object until the next append or replacement, so identity-keyed caches keep working)."""

    def __init__(self, a):
        a = np.asarray(a); self.n = len(a)
        self.buf = np.empty((max(16, 2 * self.n),) + a.shape[1:], a.dtype); self.buf[:self.n] = a; self._v = None

    def view(self):
        if self._v is None:
            self._v = self.buf[:self.n]
        return self._v

    def append(self, a):
        a = np.asarray(a)
        if self.n + len(a) > len(self.buf):
            nb = np.empty((max(2 * len(self.buf), self.n + len(a)),) + self.buf.shape[1:], np.result_type(self.buf, a))
            nb[:self.n] = self.buf[:self.n]; self.buf = nb
        elif not np.can_cast(a.dtype, self.buf.dtype, 'same_kind'):
            self.buf = self.buf.astype(np.result_type(self.buf, a))
        self.buf[self.n:self.n + len(a)] = a; self.n += len(a); self._v = None


class Grown:
    """An attribute held in a GrowBuf: reading gives its rows (a view; None if unset), assigning replaces them
    (O(n), rare: re-bins, map steps), grow(obj, name, rows) appends (amortised O(rows): no per-batch copying)."""

    def __set_name__(self, owner, name):
        self.key = '_buf_' + name

    def __get__(self, obj, typ=None):
        if obj is None:
            return self
        b = obj.__dict__.get(self.key)
        return None if b is None else b.view()

    def __set__(self, obj, val):
        obj.__dict__[self.key] = None if val is None else GrowBuf(val)


def grow(obj, name, rows):
    """Append rows to obj's Grown attribute name (or set it when unset)."""
    b = obj.__dict__.get('_buf_' + name)
    if b is None:
        setattr(obj, name, rows)
    else:
        b.append(rows)


                      # block line search (the solver's steps)


# ------------------------------------------------------------------------------------------------ row kernels
# Every kernel below reads the tables of blml_kernels.ProductKernel through _col_index: the index of column k's
# factor for lattice coordinates za, zb (rank columns: their ranks) -- one lookup for rows, rows_multi and cols.
if njit is not None:
    @njit(inline='always')
    def _col_index(kd, za, zb, k, base, Jk, mk, nbins):
        if kd == 0:
            return base[k] + za - zb + Jk[k]
        if kd == 3:                                        # rank column: an m x m table
            return base[k] + za * mk[k] + zb
        nb = nbins[k]; idx = (za - zb) % nb                # circular: min circular distance
        if idx < 0:
            idx += nb
        if idx > nb - idx:
            idx = nb - idx
        return base[k] + idx

    @njit(inline='always')
    def _dval(q, ZA, ZB, r, j, kinds, Jk, mk, dT, dbase, dR, wcol, wgrp, r2, tor):
        """Derivative entry q's factor for the pair (r, j): a column's row (raw offsets; rank columns: their table),
        or a radial group's row on the r grid, times the offset of column wcol[q] for a shift row; a periodic ball's
        group (tor): its torus table at the group index r2[g], read as is."""
        g = wgrp[q]; k = wcol[q]
        if g >= 0 and tor[g]:
            return dR[q, r2[g]]
        if g < 0:
            za = ZA[r, k]; zb = ZB[j, k]
            if kinds[k] == 3:
                return dT[dbase[q] + za * mk[k] + zb]
            return dT[dbase[q] + za - zb + Jk[k]]          # raw offsets (sinc, and every periodic derivative)
        t = np.sqrt(r2[g] * 1.0) * RADIAL_Q; i0 = int(t); f = t - i0
        v = dR[q, i0] * (1.0 - f) + dR[q, i0 + 1] * f
        if k >= 0:
            v *= ZA[r, k] - ZB[j, k]
        return v

    @njit(parallel=True, cache=True)
    def _cols_numba(ZA, ZB, T, base, Jk, mk, kinds, nbins, grp, R, out):
        """out[r, j] = S(ZA_r, ZB_j): an explicit block (the one-shot solvers' Gauss-Seidel and GEMM paths)."""
        n_r, d = ZA.shape; M = ZB.shape[0]; G = R.shape[0]
        for r in prange(n_r):
            r2 = np.zeros(G, np.int64); tor = np.zeros(G, np.int64)
            for j in range(M):
                v = 1.0
                for g in range(G):
                    r2[g] = 0; tor[g] = 0
                for k in range(d):
                    kd = kinds[k]
                    if kd == 2:
                        off = ZA[r, k] - ZB[j, k]; r2[grp[k]] += off * off
                    elif kd == 4:                          # torus: row-major circular offset over the group
                        g_ = grp[k]; nb = nbins[k]; r2[g_] = r2[g_] * nb + (ZA[r, k] - ZB[j, k]) % nb; tor[g_] = 1
                    else:
                        v *= T[_col_index(kd, ZA[r, k], ZB[j, k], k, base, Jk, mk, nbins)]
                for g in range(G):
                    if tor[g]:
                        v *= R[g, r2[g]]
                        continue
                    t = np.sqrt(r2[g] * 1.0) * RADIAL_Q; i0 = int(t); f = t - i0
                    v *= R[g, i0] * (1.0 - f) + R[g, i0 + 1] * f
                out[r, j] = v


if njit is not None:
    @njit(parallel=True, cache=True)
    def _rows_numba(ZA, ZB, w, T, base, Jk, mk, kinds, nbins, grp, R, dim_unit, grp_unit, U,
                    dT, dbase, dR, wunit, wcol, wgrp, nW, grad):
        n_r, d = ZA.shape; M = ZB.shape[0]; G = R.shape[0]
        s = np.zeros(n_r); ds = np.zeros((n_r, max(nW, 1)))
        for r in prange(n_r):
            fac = np.ones(U); pre = np.empty(U + 1); suf = np.empty(U + 1)
            r2 = np.zeros(max(G, 1), np.int64); tor = np.zeros(max(G, 1), np.int64); acc = 0.0; dacc = np.zeros(max(nW, 1))
            for j in range(M):
                wj = w[j]
                if wj == 0.0:
                    continue
                for g in range(G):
                    r2[g] = 0; tor[g] = 0
                for k in range(d):
                    off = ZA[r, k] - ZB[j, k]
                    kd = kinds[k]
                    if kd == 2:
                        r2[grp[k]] += off * off
                        continue
                    if kd == 4:                            # torus: row-major circular offset over the group
                        g_ = grp[k]; nb = nbins[k]; r2[g_] = r2[g_] * nb + off % nb; tor[g_] = 1
                        continue
                    fac[dim_unit[k]] = T[_col_index(kd, ZA[r, k], ZB[j, k], k, base, Jk, mk, nbins)]
                for g in range(G):
                    if grp_unit[g] < 0:
                        continue
                    if tor[g]:
                        fac[grp_unit[g]] = R[g, r2[g]]
                        continue
                    t = np.sqrt(r2[g] * 1.0) * RADIAL_Q; i0 = int(t); f = t - i0
                    fac[grp_unit[g]] = R[g, i0] * (1.0 - f) + R[g, i0 + 1] * f
                pre[0] = 1.0
                for u in range(U):
                    pre[u + 1] = pre[u] * fac[u]
                acc += pre[U] * wj
                if grad:
                    suf[U] = 1.0
                    for u in range(U - 1, -1, -1):
                        suf[u] = suf[u + 1] * fac[u]
                    for q in range(nW):
                        u = wunit[q]
                        dacc[q] += _dval(q, ZA, ZB, r, j, kinds, Jk, mk, dT, dbase, dR, wcol, wgrp, r2, tor) * pre[u] * suf[u + 1] * wj
            s[r] = acc
            for q in range(max(nW, 1)):
                ds[r, q] = dacc[q]
        return s, ds


def _row_args(tab):
    """The table arrays the row kernels take, after the coordinates and weights."""
    G = tab['R'].shape[0] if (tab['grp'] >= 0).any() else 0
    grp_unit = tab['grp_unit'] if G else np.full(1, -1, np.int64)
    return (tab['T'], tab['base'], tab['Jk'], tab['mk'], tab['kinds'], tab['nbins'], tab['grp'], tab['R'][:max(G, 1)],
            tab['dim_unit'], grp_unit, tab['U'], tab['dT'], tab['dbase'], tab['dR'], tab['wunit'], tab['wcol'],
            tab['wgrp'], tab['nW'])


def rows(ZA, ZB, w, tab, grad=False):
    """s_r = sum_j S(ZA_r, ZB_j) w_j on the lattice (integer coordinates), and with grad ds (R, nW): d s_r / d theta_q
    for the tables' `want` entries (d/dlog of a parameter, d/dt of a coordinate shift). O(d + nW) per pair; memory
    O(R nW)."""
    return _rows_numba(rank_coords(ZA, tab), rank_coords(ZB, tab), np.ascontiguousarray(w, np.float64),
                       *_row_args(tab), bool(grad))


if njit is not None:
    @njit(parallel=True, cache=True)
    def _rows_numba_m(ZA, ZB, Wt, umask, T, base, Jk, mk, kinds, nbins, grp, R, dim_unit, grp_unit, U,
                      dT, dbase, dR, wunit, wcol, wgrp, nW, grad):
        """_rows_numba for several densities at once (the fused joint and marginals): unit factors computed once
        per pair; density m multiplies the units with umask[m, u] = 1 (the others count as 1), weights Wt[:, m]."""
        n_r, d = ZA.shape; M = ZB.shape[0]; G = R.shape[0]; Mm = Wt.shape[1]
        s = np.zeros((n_r, Mm)); ds = np.zeros((n_r, Mm, max(nW, 1)))
        for r in prange(n_r):
            fac = np.ones(U); pre = np.empty(U + 1); suf = np.empty(U + 1); dv = np.zeros(max(nW, 1))
            r2 = np.zeros(max(G, 1), np.int64); tor = np.zeros(max(G, 1), np.int64)
            for j in range(M):
                anyw = False
                for m in range(Mm):
                    if Wt[j, m] != 0.0:
                        anyw = True
                if not anyw:
                    continue
                for g in range(G):
                    r2[g] = 0; tor[g] = 0
                for k in range(d):
                    off = ZA[r, k] - ZB[j, k]
                    kd = kinds[k]
                    if kd == 2:
                        r2[grp[k]] += off * off
                        continue
                    if kd == 4:                            # torus: row-major circular offset over the group
                        g_ = grp[k]; nb = nbins[k]; r2[g_] = r2[g_] * nb + off % nb; tor[g_] = 1
                        continue
                    fac[dim_unit[k]] = T[_col_index(kd, ZA[r, k], ZB[j, k], k, base, Jk, mk, nbins)]
                for g in range(G):
                    if grp_unit[g] < 0:
                        continue
                    if tor[g]:
                        fac[grp_unit[g]] = R[g, r2[g]]
                        continue
                    t = np.sqrt(r2[g] * 1.0) * RADIAL_Q; i0 = int(t); f = t - i0
                    fac[grp_unit[g]] = R[g, i0] * (1.0 - f) + R[g, i0 + 1] * f
                if grad:
                    for q in range(nW):
                        dv[q] = _dval(q, ZA, ZB, r, j, kinds, Jk, mk, dT, dbase, dR, wcol, wgrp, r2, tor)
                for m in range(Mm):
                    wj = Wt[j, m]
                    if wj == 0.0:
                        continue
                    pre[0] = 1.0
                    for u in range(U):
                        pre[u + 1] = pre[u] * (fac[u] if umask[m, u] else 1.0)
                    s[r, m] += pre[U] * wj
                    if grad:
                        suf[U] = 1.0
                        for u in range(U - 1, -1, -1):
                            suf[u] = suf[u + 1] * (fac[u] if umask[m, u] else 1.0)
                        for q in range(nW):
                            u = wunit[q]
                            if umask[m, u]:
                                ds[r, m, q] += dv[q] * pre[u] * suf[u + 1] * wj
        return s, ds


def cols(ZA, ZB, tab, out=None, ranked=False):
    """The explicit block S(ZA_r, ZB_j) (R, M), float64 (or into out, any float dtype). ranked: ZA, ZB are
    rank_coords already (a solver ranks its bins once, not per block)."""
    G = tab['R'].shape[0] if (tab['grp'] >= 0).any() else 0
    A, B_ = (ZA, ZB) if ranked else (rank_coords(ZA, tab), rank_coords(ZB, tab))
    out = np.empty((len(A), len(B_))) if out is None else out
    _cols_numba(A, B_, tab['T'], tab['base'], tab['Jk'], tab['mk'], tab['kinds'], tab['nbins'], tab['grp'],
                tab['R'][:max(G, 1)] if G else np.zeros((0, 2)), out)
    return out


def rows_multi(ZA, ZB, Wt, tab, umask, grad=False):
    """Fused rows: s (R, M) and ds (R, M, nW) for M densities (columns of Wt) whose kernels are the products over the
    units marked in umask (M, U) -- the joint (all units) and marginals (subsets), one table pass."""
    return _rows_numba_m(rank_coords(ZA, tab), rank_coords(ZB, tab), np.ascontiguousarray(Wt, np.float64),
                         np.ascontiguousarray(umask, np.int64), *_row_args(tab), bool(grad))


RADIAL_DS = 2e-3                                      # continuous radial profile grid (linear interpolation)
# point-kernel derivative codes (_point_args): d/dlog fc of a sinc column, d/dt of a sinc column, d/dlog P of a
# dirichlet column, d/dt of a dirichlet column, d/dlog fc of a radial group, d/dt_k of column k of a radial group,
# d/dlog fc and d/dt of a slits column (point kinds: 0 sinc, 1 dirichlet, 2 radial, 3 slits; FI the slits' inner fc,
# on a dirichlet column the inner harmonic count M_i of a periodic two-slit band, 0 for the plain band)
PD_SINC_FC, PD_SINC_SHIFT, PD_DIR_PERIOD, PD_DIR_SHIFT, PD_RAD_FC, PD_RAD_SHIFT, PD_SLITS_FC, PD_SLITS_SHIFT, PD_TOR_SHIFT = range(9)

if njit is not None:
    @njit(inline='always')
    def _psinc(f, x):
        t = np.pi * f * x
        return f if abs(t) < 1e-12 else np.sin(t) / (np.pi * x)

    @njit(inline='always')
    def _psinc_dt(f, x):
        """d/dt of f sinc(f t): (pt cos pt - sin pt) / (pi x^2), series near 0."""
        pt = np.pi * f * x
        if abs(pt) < 0.05:
            p2 = pt * pt
            return np.pi * f * f * pt * (-1.0 / 3.0 + p2 / 30.0 - p2 * p2 / 840.0 + p2 * p2 * p2 / 45360.0)
        return (pt * np.cos(pt) - np.sin(pt)) / (np.pi * x * x)

    @njit(inline='always')
    def _pfac(k, x, kinds, fc, FI, P, M, scale):
        """K of a sinc (kind 0), dirichlet (1: scale (D_M - D_FI), FI = M_i) or slits (3: fc sinc(fc x) - FI sinc(FI x))
        column at the difference x."""
        if kinds[k] == 0:
            return _psinc(fc[k], x)
        if kinds[k] == 3:
            return _psinc(fc[k], x) - (_psinc(FI[k], x) if FI[k] > 0.0 else 0.0)
        Pk = P[k]; tt = abs(x) % Pk
        if tt > Pk - tt:
            tt = Pk - tt
        if tt < 1e-10:
            return scale[k] * (M[k] - FI[k])
        return scale[k] * (np.sin(np.pi * M[k] * tt / Pk) - np.sin(np.pi * FI[k] * tt / Pk)) / np.sin(np.pi * tt / Pk)

    @njit(inline='always')
    def _ptor(g, xg, Hh, nh, gsc, i):
        """A periodic ball's group at the differences xg[g]: gsc sum_h cos(phase_h) (i < 0), or its d/dt of slot i,
        -gsc sum_h Hh[g, h, i] sin(phase_h); phase_h = sum_c Hh[g, h, c] xg[g, c]."""
        v = 0.0; D = Hh.shape[2]
        for h in range(nh[g]):
            ph = 0.0
            for c in range(D):
                ph += Hh[g, h, c] * xg[g, c]
            if i < 0:
                v += np.cos(ph)
            else:
                v -= Hh[g, h, i] * np.sin(ph)
        return gsc[g] * v

    @njit(inline='always')
    def _pdval(code, k, g, x, s2g, fc, FI, P, M, scale, dprof, dsprof, ds, prodfc):
        """Derivative entry of one pair (codes PD_*): x the column's difference, s2g its radial group's s^2."""
        if code == 0:
            return fc[k] * np.cos(np.pi * fc[k] * x)
        if code == 1:
            return _psinc_dt(fc[k], x)
        if code == 6:
            return fc[k] * np.cos(np.pi * fc[k] * x) - FI[k] * np.cos(np.pi * FI[k] * x)
        if code == 7:
            return _psinc_dt(fc[k], x) - (_psinc_dt(FI[k], x) if FI[k] > 0.0 else 0.0)
        if code == 2 or code == 3:                         # D_M - D_{M_i} (M_i = FI[k], 0: plain band)
            Pk = P[k]; uu = 2 * np.pi * x / Pk; sn = np.sin(uu / 2); Mi = FI[k]
            if abs(sn) < 1e-9:
                Dv = M[k] - Mi; dD = 0.0
            else:
                Dv = (np.sin(M[k] * uu / 2) - np.sin(Mi * uu / 2)) / sn
                dD = (0.5 * (M[k] * np.cos(M[k] * uu / 2) - Mi * np.cos(Mi * uu / 2)) * sn
                      - 0.5 * (np.sin(M[k] * uu / 2) - np.sin(Mi * uu / 2)) * np.cos(uu / 2)) / sn ** 2
            if code == 2:
                return -(Dv + uu * dD) / Pk
            return scale[k] * dD * (2 * np.pi / Pk)
        tq = np.sqrt(s2g) / ds; i0 = int(tq); f = tq - i0
        if code == 4:
            return prodfc[g] * (dprof[g, i0] * (1.0 - f) + dprof[g, i0 + 1] * f)
        return prodfc[g] * fc[k] * fc[k] * x * (dsprof[g, i0] * (1.0 - f) + dsprof[g, i0 + 1] * f)

    @njit(parallel=True, cache=True)
    def _point_rows_numba(A, Bp, w, kinds, dim_unit, grp, grp_unit, U, fc, FI, P, M, scale,
                          prof, dprof, dsprof, ds, prodfc, wunit, wcode, wcol, wgrp, nW, slot, Hh, nh, gtor, gsc, grad):
        """s_r = sum_j K(A_r - Bp_j) w_j, continuous product kernel: sinc (kind 0), dirichlet (1) columns and
        radial groups (2; profile tables on the grid s = j ds, interpolated); with grad, the derivative of each wanted
        entry q (codes PD_*: a parameter's d/dlog, a coordinate's d/dt) by prefix / suffix products: O(d + nW) per
        pair, memory O(R nW)."""
        n_r, d = A.shape; m = Bp.shape[0]; G = prof.shape[0]
        s = np.zeros(n_r); ds_out = np.zeros((n_r, max(nW, 1)))
        for r in prange(n_r):
            fac = np.ones(U); pre = np.empty(U + 1); suf = np.empty(U + 1)
            s2 = np.zeros(max(G, 1)); acc = 0.0; dacc = np.zeros(max(nW, 1)); xg = np.zeros((max(G, 1), Hh.shape[2]))
            for j in range(m):
                wj = w[j]
                if wj == 0.0:
                    continue
                for g in range(G):
                    s2[g] = 0.0
                for k in range(d):
                    x = A[r, k] - Bp[j, k]
                    if kinds[k] == 2:
                        s2[grp[k]] += (fc[k] * x) ** 2
                        continue
                    if kinds[k] == 4:                      # a periodic ball's column: its slot in the group
                        xg[grp[k], slot[k]] = x
                        continue
                    fac[dim_unit[k]] = _pfac(k, x, kinds, fc, FI, P, M, scale)
                for g in range(G):
                    if gtor[g]:
                        fac[grp_unit[g]] = _ptor(g, xg, Hh, nh, gsc, -1)
                        continue
                    tq = np.sqrt(s2[g]) / ds; i0 = int(tq); f = tq - i0
                    fac[grp_unit[g]] = prodfc[g] * (prof[g, i0] * (1.0 - f) + prof[g, i0 + 1] * f)
                pre[0] = 1.0
                for u_ in range(U):
                    pre[u_ + 1] = pre[u_] * fac[u_]
                acc += pre[U] * wj
                if grad:
                    suf[U] = 1.0
                    for u_ in range(U - 1, -1, -1):
                        suf[u_] = suf[u_ + 1] * fac[u_]
                    for q in range(nW):
                        k = wcol[q]; g = wgrp[q]; u_ = wunit[q]
                        x = A[r, k] - Bp[j, k] if k >= 0 else 0.0
                        if wcode[q] == 8:
                            dv = _ptor(g, xg, Hh, nh, gsc, slot[k])
                        else:
                            dv = _pdval(wcode[q], k, g, x, s2[g] if g >= 0 else 0.0, fc, FI, P, M, scale, dprof, dsprof, ds, prodfc)
                        dacc[q] += dv * pre[u_] * suf[u_ + 1] * wj
            s[r] = acc
            for q in range(max(nW, 1)):
                ds_out[r, q] = dacc[q]
        return s, ds_out


def _point_args(A, Bp, kinds, fc, periods, want):
    """Arrays for _point_rows_numba (radial profile tables over the s range of these points). want = [(unit, p)]:
    a declared parameter (sinc / radial fc, dirichlet period) or a coordinate shift ('shift' / ('shift', i))."""
    from blml_trivial import dirichlet_modes_scale, dirichlet_inner_M, RADIAL_KERNELS, RADIAL_DLOG, shift_index
    d = len(kinds); units = kernel_units(kinds, d); U = len(units); fc = np.asarray(fc, float)
    from blml_trivial import kernel_class
    name = [str(k).partition(':')[0] for k in kinds]
    kd = np.array([4 if nm == 'pball' else 2 if is_radial(k) else 0 if nm == 'sinc' else 3 if nm == 'slits' else 1
                   for k, nm in zip(kinds, name)], np.int64)
    FIv = np.array([(1.0 - kernel_class(k).fill) * fc[j] if nm == 'slits' else 0.0 for j, (k, nm) in enumerate(zip(kinds, name))], float)
    dim_unit = np.full(d, -1, np.int64); grp = np.full(d, -1, np.int64)
    groups = radial_groups(kinds); G = len(groups); grp_unit = np.zeros(max(G, 1), np.int64)
    for ui, (kind, dims) in enumerate(units):
        if is_radial(kind):
            g = next(i for i, (_, gd) in enumerate(groups) if list(gd) == list(dims)); grp[dims] = g; grp_unit[g] = ui
        else:
            dim_unit[dims[0]] = ui
    Pv = np.array([periods[k] if kd[k] == 1 else 1.0 for k in range(d)], float)
    Ms = [dirichlet_modes_scale(fc[k], Pv[k]) if kd[k] == 1 else (0.0, 1.0) for k in range(d)]
    for k in range(d):                                     # a periodic two-slit band: FI holds its M_i
        if kd[k] == 1:
            FIv[k] = float(dirichlet_inner_M(fc[k], Pv[k], kernel_class(kinds[k]).fill))
    ext = np.maximum(A.max(0), Bp.max(0)) - np.minimum(A.min(0), Bp.min(0))
    L = 2
    for name, dims in groups:
        if name != 'pball':
            L = max(L, int(np.sqrt(((fc[dims] * ext[dims]) ** 2).sum()) / RADIAL_DS) + 3)
    prof = np.zeros((max(G, 1), L)); dprof = np.zeros((max(G, 1), L)); dsprof = np.zeros((max(G, 1), L))
    prodfc = np.ones(max(G, 1))
    sg = np.arange(L) * RADIAL_DS
    from blml_trivial import pball_harmonics
    slot = np.zeros(d, np.int64); gtor = np.zeros(max(G, 1), np.int64); gsc = np.ones(max(G, 1))
    harm = {}
    for g, (name, dims) in enumerate(groups):
        for i, k in enumerate(dims):
            slot[k] = i
        if name == 'pball':                                # its harmonic phases 2 pi k_c / P_c per column
            Pg = np.array([periods[k] for k in dims], float)
            Kv, sc = pball_harmonics(fc[dims[0]], Pg); harm[g] = 2 * np.pi * Kv / Pg; gtor[g] = 1; gsc[g] = sc
    Dm = max([len(dims) for _, dims in groups] or [1]); Hm = max([len(v) for v in harm.values()] or [1])
    Hh = np.zeros((max(G, 1), Hm, Dm)); nh = np.zeros(max(G, 1), np.int64)
    for g, v in harm.items():
        Hh[g, :len(v), :v.shape[1]] = v; nh[g] = len(v)
    for g, (name, dims) in enumerate(groups):
        if name == 'pball':
            continue
        D = len(dims); prof[g] = RADIAL_KERNELS[name](sg, D); dl = RADIAL_DLOG[name](sg, D)
        dprof[g] = D * prof[g] + dl
        dsprof[g, 1:] = dl[1:] / sg[1:] ** 2; dsprof[g, 0] = dsprof[g, 1]        # p'(s) / s
        prodfc[g] = float(np.prod(fc[dims]))
    nW = len(want)
    wunit = np.zeros(max(nW, 1), np.int64); wcode = np.zeros(max(nW, 1), np.int64)
    wcol = np.full(max(nW, 1), -1, np.int64); wgrp = np.full(max(nW, 1), -1, np.int64)
    for q, (ui, p) in enumerate(want):
        kind, dims = units[ui]; wunit[q] = ui; i = shift_index(p)
        if is_radial(kind) and str(kind).partition(':')[0] == 'pball':
            if i is None:
                raise ValueError("pball: no parameter gradient (fc in whole harmonics, fixed periods); shifts only")
            wgrp[q] = grp[dims[0]]; wcode[q], wcol[q] = PD_TOR_SHIFT, dims[i]
        elif is_radial(kind):
            wgrp[q] = grp[dims[0]]
            wcode[q], wcol[q] = (PD_RAD_FC, -1) if i is None else (PD_RAD_SHIFT, dims[i])
        elif kind == 'sinc':
            wcode[q], wcol[q] = (PD_SINC_FC if i is None else PD_SINC_SHIFT), dims[0]
        elif kd[dims[0]] == 3:
            wcode[q], wcol[q] = (PD_SLITS_FC if i is None else PD_SLITS_SHIFT), dims[0]
        else:
            wcode[q], wcol[q] = (PD_DIR_PERIOD if i is None else PD_DIR_SHIFT), dims[0]
    return (kd, dim_unit, grp, grp_unit, U, fc, FIv, Pv, np.array([x[0] for x in Ms], float),
            np.array([x[1] for x in Ms], float), prof[:G], dprof[:G], dsprof[:G],
            RADIAL_DS, prodfc, wunit, wcode, wcol, wgrp, nW, slot, Hh, nh, gtor[:max(G, 1)], gsc)


if njit is not None:
    @njit(parallel=True, cache=True)
    def _point_rows_numba_m(A, Bp, Wt, umask, kinds, dim_unit, grp, grp_unit, U, fc, FI, P, M, scale,
                            prof, dprof, dsprof, ds, prodfc, wunit, wcode, wcol, wgrp, nW, slot, Hh, nh, gtor, gsc, grad):
        """_point_rows_numba for several densities (umask (Mm, U), weights Wt (m, Mm)), factors computed once."""
        n_r, d = A.shape; m_ = Bp.shape[0]; G = prof.shape[0]; Mm = Wt.shape[1]
        s = np.zeros((n_r, Mm)); ds_out = np.zeros((n_r, Mm, max(nW, 1)))
        for r in prange(n_r):
            fac = np.ones(U); pre = np.empty(U + 1); suf = np.empty(U + 1); dv = np.zeros(max(nW, 1))
            s2 = np.zeros(max(G, 1)); xg = np.zeros((max(G, 1), Hh.shape[2]))
            for j in range(m_):
                anyw = False
                for mm in range(Mm):
                    if Wt[j, mm] != 0.0:
                        anyw = True
                if not anyw:
                    continue
                for g in range(G):
                    s2[g] = 0.0
                for k in range(d):
                    x = A[r, k] - Bp[j, k]
                    if kinds[k] == 2:
                        s2[grp[k]] += (fc[k] * x) ** 2
                        continue
                    if kinds[k] == 4:                      # a periodic ball's column: its slot in the group
                        xg[grp[k], slot[k]] = x
                        continue
                    fac[dim_unit[k]] = _pfac(k, x, kinds, fc, FI, P, M, scale)
                for g in range(G):
                    if gtor[g]:
                        fac[grp_unit[g]] = _ptor(g, xg, Hh, nh, gsc, -1)
                        continue
                    tq = np.sqrt(s2[g]) / ds; i0 = int(tq); f = tq - i0
                    fac[grp_unit[g]] = prodfc[g] * (prof[g, i0] * (1.0 - f) + prof[g, i0 + 1] * f)
                if grad:
                    for q in range(nW):
                        k = wcol[q]; g = wgrp[q]
                        x = A[r, k] - Bp[j, k] if k >= 0 else 0.0
                        dv[q] = (_ptor(g, xg, Hh, nh, gsc, slot[k]) if wcode[q] == 8 else
                                 _pdval(wcode[q], k, g, x, s2[g] if g >= 0 else 0.0, fc, FI, P, M, scale, dprof, dsprof, ds, prodfc))
                for mm in range(Mm):
                    wj = Wt[j, mm]
                    if wj == 0.0:
                        continue
                    pre[0] = 1.0
                    for u_ in range(U):
                        pre[u_ + 1] = pre[u_] * (fac[u_] if umask[mm, u_] else 1.0)
                    s[r, mm] += pre[U] * wj
                    if grad:
                        suf[U] = 1.0
                        for u_ in range(U - 1, -1, -1):
                            suf[u_] = suf[u_ + 1] * (fac[u_] if umask[mm, u_] else 1.0)
                        for q in range(nW):
                            u_ = wunit[q]
                            if umask[mm, u_]:
                                ds_out[r, mm, q] += dv[q] * pre[u_] * suf[u_ + 1] * wj
        return s, ds_out


def point_rows_multi(A, Bp, Wt, kinds, fc, periods, umask, want=(), grad=False):
    """Fused point sums: s (m, M), ds (m, M, nW) for the M densities of umask (as rows_multi), continuous kernel."""
    A = np.ascontiguousarray(A, float); Bp = np.ascontiguousarray(Bp, float)
    args = _point_args(A, Bp, kinds, fc, periods, want)
    return _point_rows_numba_m(A, Bp, np.ascontiguousarray(Wt, float), np.ascontiguousarray(umask, np.int64), *args,
                               bool(grad))


def point_rows(A, Bp, w, kinds, fc, periods, want=(), grad=False, xp=np, max_elems=2e7):
    """s_r = sum_j K(A_r - Bp_j) w_j with the continuous product kernel (kernel classes), and with grad ds
    (R, len(want)) = the derivative of s_r for each wanted entry (unit, p) -- d/dlog of a parameter, d/dt of a
    coordinate shift; a unit may appear more than once -- by prefix / suffix products over the units (O(d) per
    pair). Rows in chunks of ~max_elems kernel entries (memory O(max_elems))."""
    d = A.shape[1]; units = kernel_units(kinds, d); U = len(units)
    if xp is np and njit is not None:                      # numba (CPU)
        A = np.ascontiguousarray(A, float); Bp = np.ascontiguousarray(Bp, float)
        args = _point_args(A, Bp, kinds, fc, periods, want)
        return _point_rows_numba(A, Bp, np.ascontiguousarray(w, float), *args, bool(grad))
    m, B = A.shape[0], Bp.shape[0]
    s = xp.zeros(m); ds = xp.zeros((m, max(len(want), 1)))
    step = max(1, int(max_elems // max(1, B * (U + 2))))
    for i0 in range(0, m, step):
        Ac = A[i0:i0 + step]
        diffs = [Ac[:, k, None] - Bp[None, :, k] for k in range(d)]
        F = [kernel_class(kind).eval(unit_theta(kind, fc, periods, dims), [diffs[k] for k in dims], xp)
             for kind, dims in units]
        pre = [None] * (U + 1); pre[0] = 1.0
        for u in range(U):
            pre[u + 1] = pre[u] * F[u]
        s[i0:i0 + step] = pre[U] @ w
        if grad and want:
            suf = [None] * (U + 1); suf[U] = 1.0
            for u in range(U - 1, -1, -1):
                suf[u] = suf[u + 1] * F[u]
            for q, (ui, p) in enumerate(want):
                kind, dims = units[ui]
                dF = kernel_class(kind).eval_dlog(unit_theta(kind, fc, periods, dims), p, [diffs[k] for k in dims], xp)
                ds[i0:i0 + step, q] = (dF * pre[ui] * suf[ui + 1]) @ w
    return s, ds


if njit is not None:
    @njit(parallel=True, cache=True)
    def cross_sums(UA, UB, Wt, fc):
        """Sinc product kernel only. S[a, q] = sum_b Wt[b, q] K(uA_a - uB_b) and D[a, q, m] = sum_b Wt[b, q] dK/dt_m (uA_a - uB_b), sinc product
        kernel; O(nA nB k) time, O(nA k) memory."""
        nA, k = UA.shape; nB = UB.shape[0]; Q = Wt.shape[1]
        S = np.zeros((nA, Q)); D = np.zeros((nA, Q, k))
        for i in prange(nA):
            kv = np.empty(k); dk = np.empty(k); pre = np.empty(k + 1); suf = np.empty(k + 1)
            for j in range(nB):
                pre[0] = 1.0
                for m in range(k):
                    t = UA[i, m] - UB[j, m]; pt = np.pi * fc[m] * t
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


# ------------------------------------------------------------------------------------------------ state
def _complex_ok(fn):
    """A row / point kernel that also takes complex weights (real and imaginary parts in two passes)."""
    def f(A, B, w, *a, **k):
        if np.iscomplexobj(w):
            r, i = fn(A, B, np.ascontiguousarray(w.real), *a, **k), fn(A, B, np.ascontiguousarray(w.imag), *a, **k)
            return r[0] + 1j * i[0], r[1] + 1j * i[1]
        return fn(A, B, w, *a, **k)
    return f


def _unit_k0(kinds, fc, periods):
    """K_u(0) per kernel unit (the factor a masked-out unit is divided by)."""
    d = len(kinds); out = []
    for kind, dims in kernel_units(kinds, d):
        th = unit_theta(kind, fc, periods, dims)
        out.append(float(np.ravel(kernel_class(kind).eval(th, [np.zeros(1) for _ in dims]))[0]))
    return np.array(out)


def _per_mode(v, K, d):
    """A per-column list (d,) for every mode, or K per-mode lists, as K lists."""
    v = list(v)
    if len(v) == K and K and all(isinstance(x, (list, tuple, np.ndarray)) for x in v) and \
            all(len(x) == d for x in v) and not (K == d and all(isinstance(x, str) for x in v)):
        return [list(x) for x in v]
    return [list(v) for _ in range(K)]


class FusedFit:
    """Several densities on one lattice, updated by touched bins (the streaming joint and its marginals), each of
    rank K: density m is f_m = sum_k |g_mk|^2 with g_mk(x) = (1/n) sum_b e_bmk K_mk(x - z_b), K_mk the product kernel
    of mode k -- its own kinds, periods and cut-offs -- over the units of mask m (the joint: all units; a marginal:
    the units of its columns). All densities and modes share the joint's bins and counts -- fitting a marginal on the
    joint's bins gives the same estimate as on its own (bins differing only outside its columns have identical rows;
    their coefficients split in proportion to the counts) -- so one table pass per mode serves all densities
    (rows_multi / point_rows_multi). masks: list of column lists, the first all columns. kinds, periods: (d,) or one
    list per mode; fc: (d,) or (K, d); fold: the shared lattice's fold per column (blml_trivial.lattice_fold_specs).
    State e (B, M, K); the plane normalisation sum_k e_mk^H S_k e_mk = n^2 per density; the stationarity
    (S_k e_mk)_b = n w_b e_bmk / ||e_bm||^2 (rank 1: (S e)_b e_b = n w_b).
    Wanted derivatives: (mode, dims, name) -- a declared parameter of the unit over dims of that mode (mode None: of
    every mode) -- or (None, j, 'shift'), the coordinate derivative of column j (an input map), each mode's unit
    holding j. engine 'gpu': the same kernels in CUDA (blml_local_gpu; dtype float32: float32 kernel arithmetic with
    float64 sums); the state stays on the host. Complex g: the solver's complex row update. implicit_grad: add the
    coefficients' one-sweep response (dL/de de/dtheta, real rank 1) to the gradient; off by default."""
    Zi, e, cnt, pbin, pw = Grown(), Grown(), Grown(), Grown(), Grown()   # grow per batch without copying

    def __init__(self, kinds, periods, fc, h, masks, centres=False, q_rate=0.1, fold=None, engine='cpu', dtype=np.float64,
                 implicit_grad=False):
        self._dev, self.implicit_grad = None, implicit_grad
        self._f32 = engine == 'gpu' and np.dtype(dtype) == np.float32
        if engine == 'gpu':                               # the same kernels in CUDA (blml_local_gpu)
            import functools
            import blml_local_gpu as G
            import cupy as cp
            self._rows, self._pts = functools.partial(G.rows, f32=self._f32), functools.partial(G.point_rows, f32=self._f32)
            self._dev = cp.asarray
        else:
            self._rows, self._pts = rows, point_rows
        self._rows, self._pts = _complex_ok(self._rows), _complex_ok(self._pts)
        fcK = np.atleast_2d(np.asarray(fc, float)); self.K, d = fcK.shape
        self.fcK, self.d = fcK.copy(), d
        self.kindsK, self.periodsK = _per_mode(kinds, self.K, d), _per_mode(periods, self.K, d)
        self.fold = [True] * d if fold is None else list(fold)
        self.h, self.centres_in, self.q_rate = np.asarray(h, float), centres, q_rate
        self.unitsK = [kernel_units(k, d) for k in self.kindsK]
        self.pkK = [ProductKernel(k, p, self.fold, engine, self._f32) for k, p in zip(self.kindsK, self.periodsK)]
        self.pk = self.pkK[0]                             # binning: the shared lattice (folded columns: every mode's period)
        self.masks = [list(m) for m in masks]; self.M = len(self.masks)
        self.umaskK = [np.array([[int(all(c in m for c in dims)) for _, dims in units] for m in self.masks], np.int64)
                       for units in self.unitsK]
        for units in self.unitsK:
            for cols in self.masks:
                for _, dims in units:
                    if 0 < sum(c in cols for c in dims) < len(dims):
                        raise ValueError("a kernel unit (radial group) straddles a density's columns")

    @property
    def cx(self):
        return np.iscomplexobj(getattr(self, 'e', 0.0))

    @property
    def fc(self):
        return self.fcK[0]

    # ---- tables and the fused kernels of one mode (CPU: one pass; GPU: the same fused pass in CUDA)
    def tables_k(self, k, want=(), fc=None, periods=None):
        """Mode k's value tables at its parameters (or fc / periods given) for the current bins; want = [(unit, p)]."""
        return self.pkK[k].tables(self.fcK[k] if fc is None else fc, self.periodsK[k] if periods is None else periods,
                                  self.h, self.Zi, want)

    def _rm(self, ZA, ZB, Wt, tab, k, grad=False, fcK=None, periodsK=None):
        if not grad:                                       # values only: the explicit block S(ZA, ZB) and a matrix product
            return self._rm_block(ZA, ZB, Wt, tab, k, fcK=fcK, periodsK=periodsK)
        if np.iscomplexobj(Wt):
            a, b = self._rm(ZA, ZB, np.ascontiguousarray(Wt.real), tab, k, grad), self._rm(ZA, ZB, np.ascontiguousarray(Wt.imag), tab, k, grad)
            return a[0] + 1j * b[0], a[1] + 1j * b[1]
        if self._dev is None:
            return rows_multi(ZA, ZB, Wt, tab, self.umaskK[k], grad)
        import blml_local_gpu as G                         # the same fused pass in CUDA (no masked table copies)
        return G.rows_multi(ZA, ZB, np.ascontiguousarray(Wt), tab, self.umaskK[k], grad, f32=self._f32)

    def _rm_block(self, ZA, ZB, Wt, tab, k, max_elems=1 << 23, fcK=None, periodsK=None):
        """rows_multi without derivatives, ~3x faster: the block S(ZA, ZB) by the solver's kernel (cols; GPU: its CUDA
        kernel), then S @ Wt (BLAS; GPU: cuBLAS, float64 sums), in row chunks of <= max_elems entries. A density whose
        mask leaves units out (a marginal) takes the block with those units' columns at offset 0 in both inputs, divided
        by their K(0) = S_bb(joint) / S_bb(m) at the parameters tab was built for (fcK, periodsK; default the current
        ones) (CPU), or the kernel's marginal output (GPU, depmask). Returns (s (R, M), zero ds) as rows_multi; Wt real
        or complex."""
        umask = self.umaskK[k]; units = self.unitsK[k]; R, M = len(ZA), Wt.shape[1]
        s = np.zeros((R, M), np.result_type(Wt, np.float64)); ds = np.zeros((R, M, 1), s.dtype)
        if not (R and len(ZB)):
            return s, ds
        groups = {}                                        # one block per distinct mask
        for m in range(M):
            groups.setdefault(tuple(umask[m]), []).append(m)
        A, Bc = rank_coords(ZA, tab), rank_coords(ZB, tab)
        off = {key: [c for ui, (_, dims) in enumerate(units) if not key[ui] for c in dims] for key in groups}
        rows = max(1, max_elems // len(ZB))
        if self._dev is None:
            sbb = self._diag_m((), fcK, periodsK)[0][:, k]
            for key, ms in groups.items():
                A2, B2 = A, Bc
                if off[key]:
                    A2, B2 = A.copy(), Bc.copy(); A2[:, off[key]] = 0; B2[:, off[key]] = 0
                div = sbb[0] / sbb[ms[0]] if off[key] else 1.0
                W_ = np.ascontiguousarray(Wt[:, ms]); buf = np.empty((min(rows, R), len(ZB)))
                for r0 in range(0, R, rows):
                    blk = cols(A2[r0:r0 + rows], B2, tab, buf[:min(rows, R - r0)], ranked=True)
                    s[r0:r0 + rows][:, ms] = (blk @ W_) / div
            return s, ds
        import cupy as cp
        import blml_local_gpu as G
        dt = cp.float32 if self._f32 else cp.float64
        Ad, Bd = cp.asarray(A, cp.int64), cp.asarray(Bc, cp.int64)
        for key, ms in groups.items():
            dep = sum(1 << c for c in off[key])
            W_ = cp.asarray(np.ascontiguousarray(Wt[:, ms]))
            for r0 in range(0, R, rows):
                r1 = min(R, r0 + rows)
                out = cp.empty((r1 - r0, len(ZB)), dt)
                if dep:
                    G.cols(Ad[r0:r1], Bd, tab, cp.empty_like(out), out, dep, f32=self._f32)
                else:
                    G.cols(Ad[r0:r1], Bd, tab, out, f32=self._f32)
                s[r0:r1][:, ms] = cp.asnumpy(out.astype(cp.float64) @ W_)
        return s, ds

    def _pm(self, A, Bp, Wt, k, fc=None, want=(), grad=False):
        fc = self.fcK[k] if fc is None else fc; kinds, periods = self.kindsK[k], self.periodsK[k]
        if np.iscomplexobj(Wt):
            a, b = self._pm(A, Bp, np.ascontiguousarray(Wt.real), k, fc, want, grad), self._pm(A, Bp, np.ascontiguousarray(Wt.imag), k, fc, want, grad)
            return a[0] + 1j * b[0], a[1] + 1j * b[1]
        if self._dev is None:
            return point_rows_multi(A, Bp, Wt, kinds, fc, periods, self.umaskK[k], want, grad)
        units, umask = self.unitsK[k], self.umaskK[k]
        k0 = _unit_k0(kinds, fc, periods); out_s, out_d = [], []
        for m in range(self.M):                             # GPU: masked columns' differences set to 0, / K_u(0)
            off = [c for ui, (_, dims) in enumerate(units) if not umask[m, ui] for c in dims]
            A2, B2 = np.array(A, float), np.array(Bp, float); A2[:, off] = 0.0; B2[:, off] = 0.0
            s_, d_ = self._pts(A2, B2, np.ascontiguousarray(Wt[:, m]), kinds, fc, periods, want, grad)
            div = float(np.prod([k0[ui] for ui in range(len(units)) if not umask[m, ui]]))
            d_ = d_ / div
            for q, (ui, _) in enumerate(want):
                if not umask[m, ui]:
                    d_[:, q] = 0.0
            out_s.append(s_ / div); out_d.append(d_)
        return np.stack(out_s, 1), np.stack(out_d, 1)

    # ---- every mode: want = [(mode, dims, name)] or (None, j, 'shift')
    def _mode_want(self, want, k):
        """(positions in want, [(unit index, parameter)]) of the entries that are mode k's."""
        units = self.unitsK[k]; pos, out = [], []
        for q, (kk, dims, p) in enumerate(want):
            if kk is not None and kk != k:
                continue
            if p == 'shift':                               # column j's coordinate: the unit holding it
                j = int(dims); ui = next(i for i, (_, ud) in enumerate(units) if j in ud)
                pos.append(q); out.append((ui, ('shift', units[ui][1].index(j))))
                continue
            ui = next((i for i, (_, ud) in enumerate(units) if list(ud) == list(dims)), None)
            if ui is None:
                if kk is None:                             # a unit some mode lacks: no derivative there
                    continue
                raise ValueError(f"mode {k}: columns {dims} are not one of its kernel units")
            pos.append(q); out.append((ui, p))
        return pos, out

    def _rmK(self, ZA, ZB, E, want=(), grad=False, fcK=None, periodsK=None):
        """S (R, M, K) = each mode's fused rows with E[:, :, k], and dS (R, M, K, len(want)) (zero where an entry is not
        the mode's)."""
        fcK = self.fcK if fcK is None else fcK; periodsK = self.periodsK if periodsK is None else periodsK
        S = np.zeros((len(ZA), self.M, self.K), np.result_type(E, np.float64)); dS = np.zeros(S.shape + (max(len(want), 1),), S.dtype)
        for k in range(self.K):
            pos, wk = self._mode_want(want, k)
            s_, d_ = self._rm(ZA, ZB, np.ascontiguousarray(E[:, :, k]),
                              self.tables_k(k, wk if grad else (), fcK[k], periodsK[k]), k, grad and len(wk) > 0, fcK, periodsK)
            S[:, :, k] = s_
            if grad and pos:
                dS[:, :, k, pos] = d_[:, :, :len(pos)]
        return S, dS

    def _pmK(self, A, Bp, E, want=(), grad=False, fcK=None):
        """The point kernel of every mode: (len A, M, K) and its derivatives (.., len(want)) as _rmK."""
        fcK = self.fcK if fcK is None else fcK
        G_ = np.zeros((len(A), self.M, self.K), np.result_type(E, np.float64)); dG = np.zeros(G_.shape + (max(len(want), 1),), G_.dtype)
        for k in range(self.K):
            pos, wk = self._mode_want(want, k)
            s_, d_ = self._pm(A, Bp, np.ascontiguousarray(E[:, :, k]), k, fcK[k], wk if grad else (), grad and len(wk) > 0)
            G_[:, :, k] = s_
            if grad and pos:
                dG[:, :, k, pos] = d_[:, :, :len(pos)]
        return G_, dG

    # ---- state
    def load(self, models):
        """models: the joint's and each marginal's burn-in fit (fitted on their own bins, rank K: e_hat_ (B,) or
        (B, K)); the marginals are mapped onto the joint's bins, each joint bin taking its marginal bin's coefficients
        times count_joint / count_marg."""
        get = lambda a: a.get() if hasattr(a, 'get') else np.asarray(a)
        ek = lambda a: (get(a)[:, None] if get(a).ndim == 1 else get(a))
        mj = models[0]
        self.Zi = np.ascontiguousarray(get(mj._Zi).astype(np.int64)); self.cnt = get(mj.w_).astype(np.float64).copy()
        self.n = float(mj.n_samples_)
        cx = any(np.iscomplexobj(get(m.e_hat_)) for m in models)
        self.e = np.zeros((len(self.cnt), self.M, self.K), complex if cx else np.float64)
        self.e[:, 0, :] = ek(mj.e_hat_)
        for m in range(1, self.M):
            mm = models[m]; Zm = get(mm._Zi).astype(np.int64); em = ek(mm.e_hat_); cm = get(mm.w_).astype(float)
            key = {tuple(z): i for i, z in enumerate(map(tuple, Zm))}
            # the marginal lattice has the joint's steps on its columns: joint bin -> marginal bin by its columns
            idx = np.array([key[tuple(z)] for z in map(tuple, self.Zi[:, self.masks[m]])])
            self.e[:, m, :] = em[idx] * (self.cnt / cm[idx])[:, None]
        self.key = {z: i for i, z in enumerate(map(tuple, self.Zi))}
        self.Q = np.full(self.M, self.n ** 2)
        self.pbin, self.pw = np.arange(len(self.cnt)), self.cnt.copy()
        return self

    def centres(self, I=None):
        return (self.Zi if I is None else self.Zi[I]) * self.h

    def bins(self, X, add=True):
        """Bin index per row of X (points, or bin centres with centres=True); add: new bins get e = 0."""
        Zi = self.pk.coords(X, self.h, centres=self.centres_in)
        idx = np.empty(len(Zi), np.int64); new = []
        for r, z in enumerate(map(tuple, Zi)):
            j = self.key.get(z)
            if j is None:                                   # (a revisit only meets zero-count pixels here)
                j = self.key[z] = len(self.key); new.append(z)
            idx[r] = j
        if new:
            grow(self, 'Zi', np.array(new, np.int64)); grow(self, 'e', np.zeros((len(new), self.M, self.K), self.e.dtype))
            grow(self, 'cnt', np.zeros(len(new)))
        return idx

    def add_counts(self, idx, wt):
        self.n += float(wt.sum()); np.add.at(self.cnt, idx, wt)
        grow(self, 'pbin', idx); grow(self, 'pw', wt)

    def sample_bins(self, m, rng):
        """Bins of m draws from the held data (bin b with probability cnt_b / n) and their draw counts."""
        p = self.pw / self.pw.sum()
        R = self.pbin[rng.choice(len(self.pbin), size=m, p=p)]
        R, inv = np.unique(R, return_inverse=True)
        return R, np.bincount(np.ravel(inv)).astype(float)

    # ---- held points (a trained input map moves them)
    def track_points(self, X, wt):
        """Track the bin of every held point (pbin) and its weight (pw), in the order they are held: the map's
        gradient pulls each bin's coordinate derivative back to its points, and move_points moves them."""
        self.pbin, self.pw = self.bins(X), np.asarray(wt, float).copy()

    def move_points(self, X_new, rng, m_rows=2000):
        """After a map step: each held point whose cell changed (its lattice columns now X_new) carries its share of
        its bin -- count w_i and coefficient mass w_i e_a / cnt_a (e_b is the bin's coefficient mass, the sum of its
        points' c_i; every mode) -- to its new bin (new bins start empty). O(n) with no kernel evaluation: the
        coefficients move with the points, as the map's gradient assumes (a bin's centre moves with its points). The
        normaliser is then re-estimated on m_rows bins drawn from the held data (O(m_rows B d), as batch_stats'
        estimate), the later touched updates refine the coefficients, and emptied bins (no data, no mass) are
        compacted. Needs track_points. Returns the indices of the points that moved."""
        Zn = self.pk.coords(X_new, self.h, centres=self.centres_in)
        ch = np.where(np.any(self.Zi[self.pbin] != Zn, axis=1))[0]
        if not len(ch):
            return ch
        io = self.pbin[ch]; inew = self.bins(X_new[ch]); w = self.pw[ch]
        share = (w / np.maximum(self.cnt[io], 1e-300))[:, None, None] * self.e[io]     # before the counts move
        np.add.at(self.e, io, -share); np.add.at(self.e, inew, share)
        np.add.at(self.cnt, io, -w); np.add.at(self.cnt, inew, w)
        self.pbin[ch] = inew
        empty = self.cnt <= 1e-12 * max(float(self.cnt.max()), 1.0)
        self.cnt[empty] = 0.0; self.e[empty] = 0.0
        R, cR = self.sample_bins(min(m_rows, len(self.pbin)), rng)
        SeR = self._rmK(self.Zi[R], self.Zi, self.e)[0]
        wR = (cR / self.cnt[R])[:, None, None] * np.conj(self.e[R])
        self.Q = np.maximum(self.n * np.real(np.einsum('rmk,rmk->m', wR, SeR)) / float(cR.sum()), 1e-300)
        self.renorm()
        self.compact()
        return ch

    def compact(self, frac=0.25):
        """Drop the bins with neither data nor coefficient (left behind by moved points) once they exceed frac of the
        bins: an O(B) relabelling, so B tracks the occupied cells and the per-step cost stays O(b B d)."""
        dead = (self.cnt <= 0) & np.all(np.abs(self.e) == 0, axis=(1, 2))
        if dead.sum() <= max(256, frac * len(dead)):
            return
        keep = ~dead; new = np.cumsum(keep) - 1
        self.Zi = np.ascontiguousarray(self.Zi[keep]); self.e = self.e[keep]; self.cnt = self.cnt[keep]
        ok = keep[self.pbin]; self.pbin = new[self.pbin[ok]]; self.pw = self.pw[ok]
        self.key = {z: i for i, z in enumerate(map(tuple, self.Zi))}

    def residual(self):
        """The one-shot solver's stopping test on this lattice: max over occupied bins and densities of
        ||(S_k e_mk)_b ||e_bm||^2 / (n w_b) - e_bmk|| / ||e_bm|| (rank 1 real: |(S e)_b e_b / (n w_b) - 1|), at e on the
        plane sum_k e_mk^H S_k e_mk = n^2. One full row pass (about one sweep); the state is not changed."""
        Se = self._rmK(self.Zi, self.Zi, self.e)[0]
        Q = np.real(np.einsum('bmk,bmk->m', np.conj(self.e), Se))
        sc = (self.n / np.sqrt(np.maximum(Q, 1e-300)))[None, :, None]
        e, Se = self.e * sc, Se * sc
        en2 = (np.abs(e) ** 2).sum(2)
        ok = (en2 > 0) & (self.cnt[:, None] > 0)
        dev = Se * np.where(ok, en2 / (self.n * np.maximum(self.cnt[:, None], 1e-300)), 0.0)[:, :, None] - e
        rho = np.sqrt((np.abs(dev) ** 2).sum(2) / np.where(ok, en2, 1.0))
        return float(np.max(np.where(ok, rho, 0.0))) if len(self.cnt) else 0.0

    def renorm(self):
        sc = self.n / np.sqrt(np.maximum(self.Q, 1e-300)); self.e *= sc[None, :, None]; self.Q = np.full(self.M, self.n ** 2)
        return sc

    # ---- density
    def g(self, X, want=(), grad=False, fcK=None):
        """g_mk at X (b, M, K) and, with want, its derivatives (b, M, K, len(want))."""
        return self._pmK(np.asarray(X, float), self.centres(), self.e / self.n, want, grad, fcK)

    def predict(self, X, m=0):
        return (np.abs(self.g(X)[0][:, m, :]) ** 2).sum(1)

    def _diag_m(self, want, fcK=None, periodsK=None):
        """S_bb per density and mode (M, K) and d S_bb / d theta (M, K, len(want)) for the wanted entries (inside each
        mask, of that mode)."""
        fcK = self.fcK if fcK is None else fcK; periodsK = self.periodsK if periodsK is None else periodsK
        sbb = np.zeros((self.M, self.K)); ds = np.zeros((self.M, self.K, len(want)))
        for k in range(self.K):
            units, umask = self.unitsK[k], self.umaskK[k]
            k0 = _unit_k0(self.kindsK[k], fcK[k], periodsK[k])
            sbb[:, k] = [np.prod([k0[u] for u in range(len(units)) if umask[m, u]]) for m in range(self.M)]
            pos, wk = self._mode_want(want, k)
            for q, (ui, p) in zip(pos, wk):
                kind, dims = units[ui]
                dk = float(np.ravel(kernel_class(kind).eval_dlog(unit_theta(kind, fcK[k], periodsK[k], dims), p,
                                                                  [np.zeros(1) for _ in dims]))[0])
                for m in range(self.M):
                    if umask[m, ui]:
                        ds[m, k, q] = sbb[m, k] / k0[ui] * dk
        return sbb, ds

    def _share(self, I, bc):
        return (np.where(self.cnt[I] > 0, bc / np.maximum(self.cnt[I], 1e-300), 0.0)[:, None, None] * self.e[I]) / self.n

    def logf(self, X, I, bc, revisit, eps_rel=1e-3):
        g = self.g(X)[0]
        if revisit:
            g = g - self._pmK(np.asarray(X, float), self.centres(I), self._share(I, bc))[0]
        f = (np.abs(g) ** 2).sum(2)
        return np.log(f + eps_rel * np.median(f, axis=0)[None, :] + 1e-300)

    def batch_stats(self, X, idx, bc, I, wt, want, revisit, eps_rel, train, rng, weight_fn=None, wsh=None):
        """lp (b, M) -- the prequential log densities (leave-batch-out on a revisit) -- and per-density gradient parts
        (lists over m), one point pass and one row pass per mode for all densities: cont (sum over points of d log(f +
        eps) / d log theta), resp (the one-sweep response of e_I, implicit_grad), dQ (~ e'dSe) and Qs (~ e'Se) on bins
        of draws from the held data. want = [(mode, dims, name)]. weight_fn(lp) -> per-point weights (b,) for the
        gradient (the score); default wt. The one-sweep response is rank-1 real only.
        wsh (trained input maps): dict(cols=[j], the mapped lattice columns). The score's derivative in those
        coordinates, summed over the modes, with dK the kernels' derivative in the coordinate difference,
        coef_bk = 2 w_b g_bk / (n (f_b + eps)), and each bin's centre moving with its points:
          vb (b, k)  batch points:      sum_k coef_bk sum_c e_ck dK(z_b - z_c)
          Vc (B, k)  bins (the model):  sum_k e_ck sum_b coef_bk dK(z_c - z_b)
          VR (r, k)  normaliser rows:   2 sum_k wR_ak sum_c e_ck dK(z_a - z_c) at the sampled bins R (dQ, as the fc's)
        -- the touched rows' and the sampled rows' derivative passes, one pass over the bins against the touched bins:
        O((|I| + |R|) B k K). Returned per density in st[m]['map'] = dict(vb, Vc, R, VR); the estimator pulls them
        back through each map (InputMap.vjp; held points take their bin's Vc / VR in proportion to their weight). The
        batch points sit at their bins' centres (an O(h) approximation, as the lattice everywhere). Also returns the
        rows (Se)_I for the coordinate update."""
        X = np.asarray(X, float); want = list(want); nW = len(want)
        shift = [(None, j, 'shift') for j in (wsh['cols'] if (wsh and train) else [])]; nS = len(shift)
        wall = want + shift
        g, dg = self.g(X, want, grad=train and nW > 0)                         # (b, M, K), (b, M, K, nW)
        if revisit:
            share = self._share(I, bc)
            g0, dg0 = self._pmK(X, self.centres(I), share, want, train and nW > 0)
            g, dg = g - g0, dg - dg0
        f = (np.abs(g) ** 2).sum(2)
        lp = np.log(np.maximum(f, 1e-300))
        dgrad = train and len(wall) > 0
        SeI, dSeI = self._rmK(self.Zi[I], self.Zi, self.e, wall if train else (), dgrad)
        R, cR = self.sample_bins(len(X), rng)
        SeR, dSeR = self._rmK(self.Zi[R], self.Zi, self.e, wall if train else (), dgrad)
        if nS:                                             # the batch's own share out of D_b on a revisit
            DI = dSeI[..., nW:nW + nS].copy()
            if revisit:
                DI -= self._rmK(self.Zi[I], self.Zi[I], share, wall, True)[1][..., nW:nW + nS]
        wR = (cR / self.cnt[R])[:, None, None] * np.conj(self.e[R])          # (r, M, K)
        Qhat = self.n * np.real(np.einsum('rmk,rmk->m', wR, SeR)) / float(cR.sum())
        self.Q = self.Q * np.exp(self.q_rate * (np.log(np.maximum(Qhat, 1e-300)) - np.log(self.Q)))
        sc = self.renorm()
        SeI, SeR, wR = SeI * sc[None, :, None], SeR * sc[None, :, None], wR * sc[None, :, None]
        dSeI, dSeR = dSeI * sc[None, :, None, None], dSeR * sc[None, :, None, None]
        st = None
        if train and (want or nS):
            w = wt if weight_fn is None else weight_fn(lp)
            eps = eps_rel * np.median(f, axis=0)                               # (M,)
            W = float(w.sum())
            sbb, dsbb = self._diag_m(want)
            st = []
            for m in range(self.M):
                cont = sum(np.real(((2 * w * np.conj(g[:, m, k]) / (f[:, m] + eps[m]))[:, None] * dg[:, m, k, :nW]).sum(0))
                           for k in range(self.K))
                resp = np.zeros(nW)
                if self.implicit_grad and not self.cx and self.K == 1:
                    a = 2 * w * g[:, m, 0].real / (self.n * (f[:, m] + eps[m]))
                    Wa = np.zeros((len(X), self.M)); Wa[:, m] = a
                    vB = np.real(self._pm(self.centres(I), X, Wa, 0)[0][:, m])
                    eB, SeB, dSeB = self.e[I, m, 0].real, SeI[:, m, 0].real, dSeI[:, m, 0, :nW].real
                    Rr = SeB - sbb[m, 0] * eB; den = 2 * sbb[m, 0] * eB + Rr; den = np.where(np.abs(den) > 1e-300, den, 1e-300)
                    dTr = -(eB[:, None] * (dSeB - dsbb[m, 0][None, :] * eB[:, None]) + (eB ** 2)[:, None] * dsbb[m, 0][None, :]) / den[:, None]
                    dTr *= (eB > 0)[:, None]
                    resp = dTr.T @ (vB - W * 2 * SeB / self.Q[m])
                st.append(dict(cont=cont, resp=resp, W=W, Wn=float(wt.sum()),
                               dQ=sum(np.real(wR[:, m, k] @ dSeR[:, m, k, :nW]) for k in range(self.K)),
                               Qs=float(sum(np.real(wR[:, m, k] @ SeR[:, m, k]) for k in range(self.K)))))
            if nS:                                         # maps: see the docstring (g, D_b at the pre-renorm scale)
                inv = np.searchsorted(I, idx)
                coef = np.stack([np.stack([2 * w * np.real(g[:, m, k]) / (self.n * (f[:, m] + eps[m])) for k in range(self.K)], 1)
                                 for m in range(self.M)], 1)                    # (b, M, K)
                CI = np.zeros((len(I), self.M, self.K)); np.add.at(CI, inv, coef)
                E = self._rmK(self.Zi, self.Zi[I], CI, wall, True)[1][..., nW:nW + nS]   # (B, M, K, nS)
                for m in range(self.M):
                    e_old = np.real(self.e[:, m, :]) / sc[m]
                    st[m]['map'] = dict(vb=sum(coef[:, m, k:k + 1] * np.real(DI[inv, m, k, :]) for k in range(self.K)),
                                        Vc=sum(e_old[:, k:k + 1] * np.real(E[:, m, k, :]) for k in range(self.K)),
                                        R=R, VR=sum(2 * np.real(wR[:, m, k])[:, None] * np.real(dSeR[:, m, k, nW:nW + nS])
                                                    for k in range(self.K)))
        return lp, SeI, st

    @staticmethod
    def _mode_rows(R, cc, s, e_now):
        """The rank-K row update of a bin (blml_trivial._solve_modes_cd): e_k = R_k / (lam - s_k), lam > max s_k the root
        of lam sum_k |R_k|^2 / (lam - s_k)^2 = cc (bisection; closed form when every s_k is equal). R (J, K) the
        off-diagonal sums, cc (J,) = n w_b, s (K,) S_bb per mode. A row without neighbours (R = 0) keeps its direction
        e_now (or mode 0) at the magnitude cc / (sum_k s_k d_k^2)."""
        a2 = np.abs(R) ** 2; ar2 = a2.sum(1); smax = float(s.max())
        out = np.zeros_like(R)
        live = ar2 > 0
        if np.allclose(s, s[0]):                          # closed form (the rank-1 root on |R|)
            ar = np.sqrt(ar2[live]); t = 2.0 * cc[live] / (ar + np.sqrt(ar * ar + 4.0 * s[0] * cc[live]))
            out[live] = t[:, None] * R[live] / ar[:, None]
        elif live.any():
            A, C = a2[live], cc[live]
            phi = lambda x: (smax + x) * (A / (smax + x[:, None] - s[None, :]) ** 2).sum(1)
            lo = np.zeros(len(C)); hi = np.ones(len(C))
            for _ in range(200):
                big = phi(hi) >= C
                if not big.any():
                    break
                hi = np.where(big, hi * 2, hi)
            for _ in range(80):
                mid = 0.5 * (lo + hi); up = phi(mid) > C
                lo = np.where(up, mid, lo); hi = np.where(up, hi, mid)
            lam = smax + 0.5 * (lo + hi)
            out[live] = R[live] / (lam[:, None] - s[None, :])
        dead = ~live
        if dead.any():
            d = e_now[dead]; nd = np.sqrt((np.abs(d) ** 2).sum(1))
            d = np.where(nd[:, None] > 0, d / np.where(nd > 0, nd, 1.0)[:, None], np.eye(R.shape[1])[0][None, :])
            out[dead] = np.sqrt(cc[dead] / np.maximum((s[None, :] * np.abs(d) ** 2).sum(1), 1e-300))[:, None] * d
        return out

    def cd(self, I, SeI, block=512, full_rows=None):
        """One coordinate update of the touched bins I for every density at once (blocks of <= block, later blocks'
        rows corrected; one fused row pass per block and mode); Q tracked exactly; then renormalised. Rank 1: the
        positive root with the solver's line search (complex: the solver's row update); rank K: the stationarity row
        update (_mode_rows) in blocks of 32 without a search, as the complex path. Bins without data go to e = 0.
        full_rows: S e for every bin (a full sweep, I a permutation of all bins). rho_run (when not None) keeps the running
        maximum of each updated bin's stationarity residual, from its fresh row just before its update (the stopping
        test of residual(), at no extra pass; FusedFit.residual for the definition)."""
        sbb = self._diag_m(())[0]
        if self.cx or self.K > 1:
            block = min(block, 32)
        SeI = (full_rows[I] if full_rows is not None else SeI).copy()
        z = np.all(self.e[I] == 0, axis=(1, 2))
        order = np.r_[np.where(z)[0], np.where(~z)[0]]; I, SeI = I[order], SeI[order]
        nz = int(z.sum())
        starts = list(range(0, nz, block)) + list(range(nz, len(I), block))
        tab = self.tables_k(0) if self.K == 1 else None
        for i_, s0 in enumerate(starts):
            s1 = starts[i_ + 1] if i_ + 1 < len(starts) else len(I)
            Jb = I[s0:s1]; SeJ = SeI[s0:s1]
            cc = self.n * self.cnt[Jb]
            if getattr(self, 'rho_run', None) is not None:
                eJ = self.e[Jb]; en2 = (np.abs(eJ) ** 2).sum(2); ok = (en2 > 0) & (cc[:, None] > 0)
                s2 = self.n ** 2 / np.maximum(self.Q, 1e-300)            # e on the plane e^H S e = n^2
                dev = SeJ * np.where(ok, s2[None, :] * en2 / np.maximum(cc, 1e-300)[:, None], 0.0)[:, :, None] - eJ
                rho = np.sqrt((np.abs(dev) ** 2).sum(2) / np.where(ok, en2, 1.0))
                self.rho_run = max(self.rho_run, float(np.max(np.where(ok, rho, 0.0)))) if len(Jb) else self.rho_run
            dl = np.zeros((len(Jb), self.M, self.K), self.e.dtype)
            for m in range(self.M):
                if self.K > 1:
                    new = self._mode_rows(SeJ[:, m, :] - sbb[m][None, :] * self.e[Jb, m, :], cc, sbb[m], self.e[Jb, m, :])
                    dl[:, m, :] = np.where(cc[:, None] > 0, new, 0.0) - self.e[Jb, m, :]
                elif self.cx:
                    Rb = SeJ[:, m, 0] - sbb[m, 0] * self.e[Jb, m, 0]; r2 = np.abs(Rb) ** 2
                    u = (r2 + np.sqrt(r2 * r2 + 4 * cc * sbb[m, 0] * r2)) / (2 * np.maximum(cc, 1e-300))
                    en = np.where(r2 > 0, Rb / np.maximum(u, 1e-300), np.sqrt(cc / sbb[m, 0]))
                    dl[:, m, 0] = np.where(cc > 0, en, 0.0) - self.e[Jb, m, 0]
                else:
                    dl[:, m, 0] = np.where(cc > 0, cd_root(SeJ[:, m, 0] - sbb[m, 0] * self.e[Jb, m, 0], cc, sbb[m, 0]), 0.0) - self.e[Jb, m, 0]
            if self.K == 1:
                SJd = self._rm(self.Zi[Jb], self.Zi[Jb], dl[:, :, 0], tab, 0)[0][:, :, None]
            else:
                SJd = self._rmK(self.Zi[Jb], self.Zi[Jb], dl)[0]
            for m in range(self.M):
                lin = sum(float(np.real(np.vdot(dl[:, m, k], SeJ[:, m, k]))) for k in range(self.K))
                quad = sum(float(np.real(np.vdot(dl[:, m, k], SJd[:, m, k]))) for k in range(self.K))
                t = 1.0 if (self.cx or self.K > 1) else cd_line_search(self.e[Jb, m, 0], dl[:, m, 0], lin, quad, cc, STEPS)
                dl[:, m] *= t
                self.Q[m] += 2 * t * lin + t * t * quad
            self.e[Jb] = self.e[Jb] + dl
            if s1 < len(I):
                if self.K == 1:
                    SeI[s1:] += self._rm(self.Zi[I[s1:]], self.Zi[Jb], dl[:, :, 0], tab, 0)[0][:, :, None]
                else:
                    SeI[s1:] += self._rmK(self.Zi[I[s1:]], self.Zi[Jb], dl)[0]
        self.renorm()

    def set_params(self, fc=None, periods=None, rng=None):
        """Move the cut-offs (fc: (d,) or (K, d)) and the periods (one list per mode, or one for every mode) at fixed
        coefficients; Q(new) / Q(old) estimated on bins of draws from the held data; rng=None: no estimate (a re-bin
        follows with the exact Q)."""
        fcK = self.fcK if fc is None else np.broadcast_to(np.atleast_2d(np.asarray(fc, float)), self.fcK.shape).copy()
        periodsK = self.periodsK if periods is None else _per_mode(periods, self.K, self.d)
        if rng is not None:
            I, bc = self.sample_bins(min(len(self.pbin), 2000), rng)
            wI = (bc / np.maximum(self.cnt[I], 1e-300))[:, None, None] * np.conj(self.e[I])
            on = self._rmK(self.Zi[I], self.Zi, self.e, fcK=fcK, periodsK=periodsK)[0]
            oo = self._rmK(self.Zi[I], self.Zi, self.e)[0]
            self.Q = self.Q * np.real(np.einsum('rmk,rmk->m', wI, on)) / np.maximum(np.real(np.einsum('rmk,rmk->m', wI, oo)), 1e-300)
        self.fcK, self.periodsK = fcK.copy(), [list(p) for p in periodsK]
        for k, pk in enumerate(self.pkK):
            pk.periods = list(self.periodsK[k]); pk.clear()
        self.renorm()

    def rebin(self, X, wt, h, sweeps=2):
        """A new lattice (steps h) for the points X (weights wt): each new bin starts at the current fit's stationary
        value -- e_bk = w_b g_k(z_b) / f(z_b) (rank 1: cnt_b / g(z_b), real only where g(z_b) > 0); 0 where f <= 1e-6
        max f -- then `sweeps` full
        coordinate sweeps."""
        Zi = self.pk.coords(X, h, centres=self.centres_in)
        u, inv = np.unique(Zi, axis=0, return_inverse=True)
        cnt = np.bincount(np.ravel(inv), weights=wt, minlength=len(u)).astype(float)
        gz = self._pmK(u * h, self.centres(), self.e / self.n)[0]               # (B_new, M, K)
        if self.K == 1:
            g1 = gz[:, :, 0]
            # real: e_b > 0 at the optimum ((S e)_b = n g(z_b)), so start from cnt / g only where g > 0; a bin
            # with g <= 0 starts at 0 and cd takes it to the positive root (a negative start sits outside the log
            # barrier: the block safeguard then rejects every step toward the root)
            ok = (np.abs(g1) if self.cx else g1) > 1e-3 * np.abs(g1).max(0)[None, :]
            den = np.where(np.abs(g1) > 0, np.conj(g1) if self.cx else g1, 1.0)
            e = np.where(ok & (cnt[:, None] > 0), cnt[:, None] / den, 0.0)[:, :, None]
        else:
            fz = (np.abs(gz) ** 2).sum(2)
            ok = fz > 1e-6 * fz.max(0)[None, :]
            e = np.where((ok & (cnt[:, None] > 0))[:, :, None], cnt[:, None, None] * gz / np.where(ok, fz, 1.0)[:, :, None], 0.0)
        self.h = np.asarray(h, float)
        self.Zi = np.ascontiguousarray(u); self.cnt = cnt
        self.e = e.astype(self.e.dtype)
        self.key = {z: i for i, z in enumerate(map(tuple, self.Zi))}
        self.pbin, self.pw = np.ravel(inv).astype(np.int64), np.asarray(wt, float).copy()
        for pk in self.pkK:
            pk.clear()
        allI = np.arange(len(self.cnt))
        Se = self._rmK(self.Zi, self.Zi, self.e)[0]
        self.Q = np.real(np.einsum('bmk,bmk->m', np.conj(self.e), Se)); sc = self.renorm(); Se = Se * sc[None, :, None]
        for k in range(sweeps):
            if k:
                Se = self._rmK(self.Zi, self.Zi, self.e)[0]
            self.cd(allI[np.random.default_rng(k).permutation(len(allI))], None, full_rows=Se)
