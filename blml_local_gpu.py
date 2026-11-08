"""GPU (CuPy) versions of blml_local's pair-accumulation kernels, same arguments and results:
  rows(ZA, ZB, w, tab, grad)                       lattice rows (S w)_r and d/dlog theta
  rows_multi(ZA, ZB, Wt, tab, umask, grad)         the same for several densities (unit masks), one pass
  cols(ZA, ZB, tab, out, out_m, depmask)           the explicit block S(ZA, ZB) on the device (one-shot solvers)
  point_rows(A, Bp, w, kinds, fc, periods, want, grad)   continuous kernel at points
One thread per output row and a slice of the inner index (grid.y), partial sums combined with atomicAdd (double;
compute capability >= 6.0). f32=True: kernel values, tables and products in float32 (consumer GPUs such as the T4
run float64 at 1/32 rate), with coordinate differences, the reduction of sin / cos arguments and every sum in
float64; f32=False: float64 throughout (A100 / H100, or exact parity). Inputs may be NumPy (copied up) or CuPy arrays; results are NumPy. The tables are the
host-built ones of blml_local (one flat row per column; building them is not the cost)."""
import numpy as np
import cupy as cp
from blml_local import _point_args
from blml_kernels import rank_coords

MAXU = 32                    # units (columns + radial groups) per kernel
MAXW = 64                    # derivative entries per kernel (a unit may have several: its parameters, coordinate shifts)
SLICE = 2048                 # inner-index slice per block row
MAXM = 4                     # densities per fused pass (joint + marginals)

_DVAL = r'''
/* derivative entry q of the pair (r, j) (blml_local._dval): a column's row (raw offsets; rank columns: their table),
   or a radial group's row on the r grid, times the offset of column wcol[q] for a shift row; a periodic ball's group
   (tor): its torus table at the group index */
__device__ __forceinline__ REAL dval(int q, const long long* ZA, const long long* ZB, long long r, long long j, int d,
                                     const long long* kinds, const long long* Jk, const long long* mk, const REAL* dT,
                                     const long long* dbase, const REAL* dR, int LR, const long long* wcol,
                                     const long long* wgrp, const long long* r2, const int* tor) {
    long long g = wgrp[q], k = wcol[q];
    if (g >= 0 && tor[g]) return dR[(long long)q * LR + r2[g]];   /* a periodic ball's torus row, read as is */
    if (g < 0) {
        long long za = ZA[r * d + k], zb = ZB[j * d + k];
        return dT[dbase[q] + (kinds[k] == 3 ? za * mk[k] + zb : za - zb + Jk[k])];
    }
    double t = sqrt((double)r2[g]) * @RQ@; int i0 = (int)t; REAL f = (REAL)(t - i0);
    REAL v = dR[(long long)q * LR + i0] * (1.0 - f) + dR[(long long)q * LR + i0 + 1] * f;
    if (k >= 0) v *= (REAL)(ZA[r * d + k] - ZB[j * d + k]);
    return v;
}
'''

_ROWS_SRC = r'''
#define MAXU @MAXU@
#define MAXW @MAXW@
#define USE_FLOAT @F32@
#if USE_FLOAT
typedef float REAL;
#else
typedef double REAL;
#endif
@LOOKUP@
@DVAL@
extern "C" __global__ void rows_kernel(
    const long long* ZA, const long long* ZB, const double* w, int R, int M, int d,
    const REAL* T, const long long* base, const long long* Jk, const long long* mk, const long long* kinds,
    const long long* nbins, const long long* grp, const REAL* RT, int LR, int G, const long long* dim_unit,
    const long long* grp_unit, int U, const REAL* dT, const long long* dbase, const REAL* dR,
    const long long* wunit, const long long* wcol, const long long* wgrp, int nW, int grad, int slice, double* s,
    double* ds)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;
    int j0 = blockIdx.y * slice, j1 = min(M, j0 + slice);
    REAL fac[MAXU], pre[MAXU + 1], suf[MAXU + 1];
    double dacc[MAXW];
    long long r2[MAXU]; int tor[MAXU];
    double acc = 0.0;
    for (int q = 0; q < nW; q++) dacc[q] = 0.0;
    for (int u = 0; u < U; u++) fac[u] = 1.0;
    for (int j = j0; j < j1; j++) {
        double wj = w[j];
        if (wj == 0.0) continue;
        for (int g = 0; g < G; g++) { r2[g] = 0; tor[g] = 0; }
        for (int k = 0; k < d; k++) {
            long long za = ZA[(long long)r * d + k], zb = ZB[(long long)j * d + k], off = za - zb;
            long long kd = kinds[k];
            if (kd == 2) { r2[grp[k]] += off * off; continue; }
            if (kd == 4) {                       /* torus: row-major circular offset over a periodic ball's group */
                long long nb = nbins[k], m = off < 0 ? off + nb : off; int g_ = (int)grp[k];
                r2[g_] = r2[g_] * nb + m; tor[g_] = 1; continue;
            }
            fac[(int)dim_unit[k]] = T[col_index(kd, za, zb, k, base, Jk, mk, nbins)];
        }
        for (int g = 0; g < G; g++) {
            if (tor[g]) { fac[(int)grp_unit[g]] = RT[g * LR + r2[g]]; continue; }
            double t = sqrt((double)r2[g]) * @RQ@; int i0 = (int)t; REAL f = (REAL)(t - i0);
            fac[(int)grp_unit[g]] = RT[g * LR + i0] * (1.0 - f) + RT[g * LR + i0 + 1] * f;
        }
        pre[0] = 1.0;
        for (int u = 0; u < U; u++) pre[u + 1] = pre[u] * fac[u];
        acc += (double)pre[U] * wj;
        if (grad) {
            suf[U] = 1.0;
            for (int u = U - 1; u >= 0; u--) suf[u] = suf[u + 1] * fac[u];
            for (int q = 0; q < nW; q++) {
                int u = (int)wunit[q];
                REAL dv = dval(q, ZA, ZB, r, j, d, kinds, Jk, mk, dT, dbase, dR, LR, wcol, wgrp, r2, tor);
                dacc[q] += (double)(dv * pre[u] * suf[u + 1]) * wj;
            }
        }
    }
    atomicAdd(&s[r], acc);
    if (grad) for (int q = 0; q < nW; q++) atomicAdd(&ds[(long long)r * nW + q], dacc[q]);
}
'''

_ROWS_MULTI_SRC = r'''
#define MAXU @MAXU@
#define MAXW @MAXW@
#define MAXM @MAXM@
#define USE_FLOAT @F32@
#if USE_FLOAT
typedef float REAL;
#else
typedef double REAL;
#endif
@LOOKUP@
@DVAL@
extern "C" __global__ void rows_multi_kernel(
    const long long* ZA, const long long* ZB, const double* Wt, const long long* umask, int Mm, int R, int M, int d,
    const REAL* T, const long long* base, const long long* Jk, const long long* mk, const long long* kinds,
    const long long* nbins, const long long* grp, const REAL* RT, int LR, int G, const long long* dim_unit,
    const long long* grp_unit, int U, const REAL* dT, const long long* dbase, const REAL* dR,
    const long long* wunit, const long long* wcol, const long long* wgrp, int nW, int grad, int slice, double* s,
    double* ds)
{
    /* rows_kernel for Mm densities at once (blml_local.rows_multi): the unit factors once per pair; density m
       multiplies the units with umask[m, u] = 1, weights Wt[:, m]. s (R, Mm), ds (R, Mm, nW). */
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;
    int j0 = blockIdx.y * slice, j1 = min(M, j0 + slice);
    REAL fac[MAXU], pre[MAXU + 1], suf[MAXU + 1], dv[MAXW];
    double acc[MAXM], dacc[MAXM * MAXW];
    long long r2[MAXU]; int tor[MAXU];
    for (int m = 0; m < Mm; m++) { acc[m] = 0.0; for (int q = 0; q < nW; q++) dacc[m * MAXW + q] = 0.0; }
    for (int u = 0; u < U; u++) fac[u] = 1.0;
    for (int j = j0; j < j1; j++) {
        int any = 0;
        for (int m = 0; m < Mm; m++) if (Wt[(long long)j * Mm + m] != 0.0) any = 1;
        if (!any) continue;
        for (int g = 0; g < G; g++) { r2[g] = 0; tor[g] = 0; }
        for (int k = 0; k < d; k++) {
            long long za = ZA[(long long)r * d + k], zb = ZB[(long long)j * d + k], off = za - zb;
            long long kd = kinds[k];
            if (kd == 2) { r2[grp[k]] += off * off; continue; }
            if (kd == 4) {                       /* torus: row-major circular offset over a periodic ball's group */
                long long nb = nbins[k], m = off < 0 ? off + nb : off; int g_ = (int)grp[k];
                r2[g_] = r2[g_] * nb + m; tor[g_] = 1; continue;
            }
            fac[(int)dim_unit[k]] = T[col_index(kd, za, zb, k, base, Jk, mk, nbins)];
        }
        for (int g = 0; g < G; g++) {
            if (tor[g]) { fac[(int)grp_unit[g]] = RT[g * LR + r2[g]]; continue; }
            double t = sqrt((double)r2[g]) * @RQ@; int i0 = (int)t; REAL f = (REAL)(t - i0);
            fac[(int)grp_unit[g]] = RT[g * LR + i0] * (1.0 - f) + RT[g * LR + i0 + 1] * f;
        }
        if (grad) for (int q = 0; q < nW; q++) dv[q] = dval(q, ZA, ZB, r, j, d, kinds, Jk, mk, dT, dbase, dR, LR, wcol, wgrp, r2, tor);
        for (int m = 0; m < Mm; m++) {
            double wj = Wt[(long long)j * Mm + m];
            if (wj == 0.0) continue;
            const long long* um = umask + (long long)m * U;
            pre[0] = 1.0;
            for (int u = 0; u < U; u++) pre[u + 1] = pre[u] * (um[u] ? fac[u] : (REAL)1.0);
            acc[m] += (double)pre[U] * wj;
            if (grad) {
                suf[U] = 1.0;
                for (int u = U - 1; u >= 0; u--) suf[u] = suf[u + 1] * (um[u] ? fac[u] : (REAL)1.0);
                for (int q = 0; q < nW; q++) {
                    int u = (int)wunit[q];
                    if (um[u]) dacc[m * MAXW + q] += (double)(dv[q] * pre[u] * suf[u + 1]) * wj;
                }
            }
        }
    }
    for (int m = 0; m < Mm; m++) {
        atomicAdd(&s[(long long)r * Mm + m], acc[m]);
        if (grad) for (int q = 0; q < nW; q++) atomicAdd(&ds[((long long)r * Mm + m) * nW + q], dacc[m * MAXW + q]);
    }
}
'''

_POINTS_SRC = r'''
#define MAXU @MAXU@
#define MAXW @MAXW@
#define USE_FLOAT @F32@
#if USE_FLOAT
typedef float REAL;
#define RSIN(t) sinf((float)remainder((double)(t), 6.283185307179586))
#define RCOS(t) cosf((float)remainder((double)(t), 6.283185307179586))
#else
typedef double REAL;
#define RSIN(t) sin(t)
#define RCOS(t) cos(t)
#endif
extern "C" __global__ void points_kernel(
    const double* A, const double* Bp, const double* w, int R, int M, int d,
    const long long* kinds, const long long* dim_unit, const long long* grp, const long long* grp_unit, int U,
    const double* fc, const double* FI, const double* P, const double* Mh, const double* scale,
    const double* prof, const double* dprof, const double* dsprof, int LP, int G, double dsg, const double* prodfc,
    const long long* wunit, const long long* wcode, const long long* wcol, const long long* wgrp, int nW,
    const long long* slot, const double* Hh, const long long* nh, const long long* gtor, const double* gsc, int HM, int DM,
    int grad, int slice, double* s, double* ds)
{
    /* periodic ball groups (gtor[g]; kinds 4): gsc sum_h cos(sum_c Hh[g, h, c] xg[c]) over its nh[g] harmonics, xg the
       group's differences by slot; derivative code 8: d/dt of the slot of column wcol. */
    /* derivative codes as blml_local.PD_*: 0 sinc d/dlog fc, 1 sinc d/dt, 2 dirichlet d/dlog P, 3 dirichlet d/dt,
       4 radial d/dlog fc, 5 radial d/dt of column wcol, 6 slits d/dlog fc, 7 slits d/dt; kinds: 0 sinc, 1 dirichlet,
       2 radial, 3 slits (fc sinc(fc x) - FI sinc(FI x)); on a dirichlet column FI is the inner harmonic count M_i
       of a periodic two-slit band, scale (D_M - D_{M_i}), 0 for the plain band */
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;
    int j0 = blockIdx.y * slice, j1 = min(M, j0 + slice);
    const double PI = 3.141592653589793;
    REAL fac[MAXU], pre[MAXU + 1], suf[MAXU + 1];
    double dacc[MAXW], s2[MAXU], xg[MAXU * 4];
    double acc = 0.0;
    for (int q = 0; q < nW; q++) dacc[q] = 0.0;
    for (int u = 0; u < U; u++) fac[u] = 1.0;
    for (int j = j0; j < j1; j++) {
        double wj = w[j];
        if (wj == 0.0) continue;
        for (int g = 0; g < G; g++) s2[g] = 0.0;
        for (int k = 0; k < d; k++) {
            double x = A[(long long)r * d + k] - Bp[(long long)j * d + k];
            long long kd = kinds[k];
            if (kd == 2) { s2[grp[k]] += (fc[k] * x) * (fc[k] * x); continue; }
            if (kd == 4) { xg[grp[k] * 4 + slot[k]] = x; continue; }
            int u = (int)dim_unit[k];
            if (kd == 0 || kd == 3) {
                double t = PI * fc[k] * x, v = fabs(t) < 1e-12 ? fc[k] : RSIN(t) / (PI * x);
                if (kd == 3 && FI[k] > 0.0) { double ti = PI * FI[k] * x; v -= fabs(ti) < 1e-12 ? FI[k] : RSIN(ti) / (PI * x); }
                fac[u] = (REAL)v;
            } else {
                double Pk = P[k], tt = fmod(fabs(x), Pk);
                if (tt > Pk - tt) tt = Pk - tt;
                fac[u] = (REAL)(scale[k] * (tt < 1e-10 ? Mh[k] - FI[k]
                                            : (RSIN(PI * Mh[k] * tt / Pk) - RSIN(PI * FI[k] * tt / Pk)) / RSIN(PI * tt / Pk)));
            }
        }
        for (int g = 0; g < G; g++) {
            int u = (int)grp_unit[g];
            if (gtor[g]) {
                double v = 0.0;
                for (int h = 0; h < nh[g]; h++) {
                    double ph = 0.0;
                    for (int c = 0; c < DM; c++) ph += Hh[((long long)g * HM + h) * DM + c] * xg[g * 4 + c];
                    v += cos(ph);
                }
                fac[u] = (REAL)(gsc[g] * v); continue;
            }
            double tq = sqrt(s2[g]) / dsg; int i0 = (int)tq; double f = tq - i0;
            fac[u] = (REAL)(prodfc[g] * (prof[g * LP + i0] * (1.0 - f) + prof[g * LP + i0 + 1] * f));
        }
        pre[0] = 1.0;
        for (int u = 0; u < U; u++) pre[u + 1] = pre[u] * fac[u];
        acc += (double)pre[U] * wj;
        if (grad) {
            suf[U] = 1.0;
            for (int u = U - 1; u >= 0; u--) suf[u] = suf[u + 1] * fac[u];
            for (int q = 0; q < nW; q++) {
                long long code = wcode[q], k = wcol[q], g = wgrp[q]; int u = (int)wunit[q];
                double x = k >= 0 ? A[(long long)r * d + k] - Bp[(long long)j * d + k] : 0.0, dv;
                if (code == 0) dv = fc[k] * RCOS(PI * fc[k] * x);
                else if (code == 6) dv = fc[k] * RCOS(PI * fc[k] * x) - FI[k] * RCOS(PI * FI[k] * x);
                else if (code == 1 || code == 7) {
                    dv = 0.0;
                    for (int e = 0; e < (code == 7 && FI[k] > 0.0 ? 2 : 1); e++) {
                        double f = e == 0 ? fc[k] : FI[k], pt = PI * f * x, de;
                        if (fabs(pt) < (USE_FLOAT ? 0.5 : 0.05)) {   // series: pt cos pt - sin pt cancels to -pt^3/3
                            double p2 = pt * pt;
                            de = PI * f * f * pt * (-1.0 / 3.0 + p2 / 30.0 - p2 * p2 / 840.0 + p2 * p2 * p2 / 45360.0);
                        } else de = (pt * RCOS(pt) - RSIN(pt)) / (PI * x * x);
                        dv += e == 0 ? de : -de;
                    }
                } else if (code == 2 || code == 3) {
                    double Pk = P[k], uu = 2 * PI * x / Pk, sn = RSIN(uu / 2), Mi = FI[k], Dv, dD;
                    if (fabs(sn) < 1e-6) { Dv = Mh[k] - Mi; dD = 0.0; }
                    else {
                        Dv = (RSIN(Mh[k] * uu / 2) - RSIN(Mi * uu / 2)) / sn;
                        dD = (0.5 * (Mh[k] * RCOS(Mh[k] * uu / 2) - Mi * RCOS(Mi * uu / 2)) * sn
                              - 0.5 * (RSIN(Mh[k] * uu / 2) - RSIN(Mi * uu / 2)) * RCOS(uu / 2)) / (sn * sn);
                    }
                    dv = code == 2 ? -(Dv + uu * dD) / Pk : scale[k] * dD * (2 * PI / Pk);
                } else if (code == 8) {
                    int i = (int)slot[k]; dv = 0.0;
                    for (int h = 0; h < nh[g]; h++) {
                        double ph = 0.0;
                        for (int c = 0; c < DM; c++) ph += Hh[((long long)g * HM + h) * DM + c] * xg[g * 4 + c];
                        dv -= Hh[((long long)g * HM + h) * DM + i] * sin(ph);
                    }
                    dv *= gsc[g];
                } else {
                    double tq = sqrt(s2[g]) / dsg; int i0 = (int)tq; double f = tq - i0;
                    if (code == 4) dv = prodfc[g] * (dprof[g * LP + i0] * (1.0 - f) + dprof[g * LP + i0 + 1] * f);
                    else dv = prodfc[g] * fc[k] * fc[k] * x * (dsprof[g * LP + i0] * (1.0 - f) + dsprof[g * LP + i0 + 1] * f);
                }
                dacc[q] += (double)((REAL)dv * pre[u] * suf[u + 1]) * wj;
            }
        }
    }
    atomicAdd(&s[r], acc);
    if (grad) for (int q = 0; q < nW; q++) atomicAdd(&ds[(long long)r * nW + q], dacc[q]);
}
'''


_LOOKUP = r"""
/* the index of column k's factor (blml_local._col_index): offset rows, circular rows, rank tables */
__device__ __forceinline__ long long col_index(long long kd, long long za, long long zb, int k, const long long* base,
                                               const long long* Jk, const long long* mk, const long long* nbins) {
    if (kd == 0) return base[k] + za - zb + Jk[k];
    if (kd == 3) return base[k] + za * mk[k] + zb;
    long long nb = nbins[k], m = za - zb;               /* |za - zb| <= nb: fold with branches, not 64-bit % */
    if (m < 0) m += nb; else if (m >= nb) m -= nb;
    if (m > nb - m) m = nb - m;
    return base[k] + m;
}
"""

_COLS_SRC = r"""
#define USE_FLOAT @F32@
#if USE_FLOAT
typedef float REAL;
#else
typedef double REAL;
#endif
@LOOKUP@
extern "C" __global__ void cols_kernel(
    const long long* ZA, const long long* ZB, long long R, long long M, int d,
    const REAL* T, const long long* base, const long long* Jk, const long long* mk, const long long* kinds,
    const long long* nbins, const long long* grp, const REAL* RT, long long LR, int G,
    long long depmask, REAL* out, REAL* out_m)
{
    /* out[r, j] = S(ZA_r, ZB_j); with depmask, out_m = the product over the columns not in depmask (the marginal
       of the others) from the same lookups; a radial group is on the dep side when its columns are (a group never
       straddles: blml_gpu checks). One thread per element. */
    long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= R * M) return;
    long long r = idx / M, j = idx % M;
    REAL v = 1, vdep = 1;
    long long r2[8]; int gdep[8], tor[8];
    for (int g = 0; g < G; g++) { r2[g] = 0; gdep[g] = 0; tor[g] = 0; }
    for (int k = 0; k < d; k++) {
        long long za = ZA[r * d + k], zb = ZB[j * d + k], kd = kinds[k];
        if (kd == 2) { long long off = za - zb; r2[grp[k]] += off * off; if ((depmask >> k) & 1LL) gdep[grp[k]] = 1; continue; }
        if (kd == 4) {                           /* torus: a periodic ball's group, row-major circular offset */
            long long nb = nbins[k], off = za - zb, m = off < 0 ? off + nb : off; int g_ = (int)grp[k];
            r2[g_] = r2[g_] * nb + m; tor[g_] = 1; if ((depmask >> k) & 1LL) gdep[g_] = 1; continue;
        }
        REAL tv = T[col_index(kd, za, zb, k, base, Jk, mk, nbins)];
        if ((depmask >> k) & 1LL) vdep *= tv; else v *= tv;
    }
    for (int g = 0; g < G; g++) {
        double t = sqrt((double)r2[g]) * @RQ@; long long i0 = (long long)t; double f = t - (double)i0;
        REAL tv = tor[g] ? RT[g * LR + r2[g]] : (REAL)(RT[g * LR + i0] * (1.0 - f) + RT[g * LR + i0 + 1] * f);
        if (gdep[g]) vdep *= tv; else v *= tv;
    }
    if (depmask != 0) out_m[idx] = v;
    out[idx] = v * vdep;
}
"""

_K = {}


def _kernel(name, f32):
    if (name, f32) not in _K:
        from blml_trivial import RADIAL_Q
        src = {'rows_kernel': _ROWS_SRC, 'rows_multi_kernel': _ROWS_MULTI_SRC, 'cols_kernel': _COLS_SRC}.get(name, _POINTS_SRC)
        src = src.replace('@LOOKUP@', _LOOKUP).replace('@DVAL@', _DVAL).replace('@MAXU@', str(MAXU)).replace(
            '@MAXW@', str(MAXW)).replace('@MAXM@', str(MAXM)).replace(
            '@RQ@', str(RADIAL_Q)).replace('@F32@', '1' if f32 else '0')
        _K[(name, f32)] = cp.RawKernel(src, name)
    return _K[(name, f32)]


_d = lambda a, dt: cp.ascontiguousarray(cp.asarray(a, dtype=dt))
_LAST = [None, None, None]    # device copy of the last bin-coordinate array (LocalFit replaces, never mutates, it)


def _bins_dev(Z, tab):
    if _LAST[0] is not Z or _LAST[2] is not tab["vals"]:          # ranks depend only on the rank values
        _LAST[0], _LAST[1], _LAST[2] = Z, _d(rank_coords(Z, tab), cp.int64), tab["vals"]
    return _LAST[1]


def dev_tab(tab, f32):
    """The device arrays of a ProductKernel table dict, made once and kept in it (the host table is cached by
    ProductKernel, so the device copy lives exactly as long; GPU and CPU share one cache)."""
    key = '_dev32' if f32 else '_dev64'
    if key not in tab:
        rt = cp.float32 if f32 else cp.float64
        G = int(tab['R'].shape[0]) if (np.asarray(tab['grp']) >= 0).any() else 0
        tab[key] = dict(T=_d(tab['T'], rt), dT=_d(tab['dT'], rt), R=_d(tab['R'][:max(G, 1)], rt),
                        dR=_d(tab['dR'], rt), G=G,
                        **{k: _d(tab[k], cp.int64) for k in ('base', 'Jk', 'mk', 'kinds', 'nbins', 'grp', 'dim_unit',
                                                              'grp_unit', 'dbase', 'wunit', 'wcol', 'wgrp')})
    return tab[key]


def _table_args(tab, f32):
    t = dev_tab(tab, f32)
    return (t['T'], t['base'], t['Jk'], t['mk'], t['kinds'], t['nbins'], t['grp'], t['R'], np.int32(t['R'].shape[1]),
            np.int32(t['G']), t['dim_unit'], t['grp_unit'], np.int32(int(tab['U'])), t['dT'], t['dbase'], t['dR'],
            t['wunit'], t['wcol'], t['wgrp'], np.int32(int(tab['nW'])))


def _launch(name, R, M, args, nW, grad, f32, Mm=None):
    shp = (R,) if Mm is None else (R, Mm)
    s = cp.zeros(shp); ds = cp.zeros(shp + (max(nW, 1),))
    if R and M:
        grid = ((R + 127) // 128, (M + SLICE - 1) // SLICE)
        _kernel(name, f32)(grid, (128,), args + (np.int32(SLICE), s, ds))
    return cp.asnumpy(s), cp.asnumpy(ds)


def _limits(who, U, nW, Mm=1):
    if U > MAXU or nW > MAXW or Mm > MAXM:
        raise ValueError(f"GPU {who}: {U} kernel units (max {MAXU}), {nW} derivative entries (max {MAXW}), "
                         f"{Mm} densities (max {MAXM})")


def rows(ZA, ZB, w, tab, grad=False, f32=False):
    U = int(tab['U'])
    _limits('rows', U, int(tab['nW']))
    R, d = ZA.shape; M = ZB.shape[0]; nW = int(tab['nW'])
    args = ((_d(rank_coords(ZA, tab), cp.int64), _bins_dev(ZB, tab), _d(w, cp.float64), np.int32(R), np.int32(M),
             np.int32(d)) + _table_args(tab, f32) + (np.int32(bool(grad)),))
    return _launch('rows_kernel', R, M, args, nW, grad, f32)


def rows_multi(ZA, ZB, Wt, tab, umask, grad=False, f32=False):
    """blml_local.rows_multi on the GPU: s (R, Mm), ds (R, Mm, nW), one pass for every density of umask (Mm, U)."""
    U = int(tab['U']); Mm = int(np.shape(umask)[0])
    _limits('rows_multi', U, int(tab['nW']), Mm)
    R, d = ZA.shape; M = ZB.shape[0]; nW = int(tab['nW'])
    args = ((_d(rank_coords(ZA, tab), cp.int64), _bins_dev(ZB, tab), _d(Wt, cp.float64), _d(umask, cp.int64),
             np.int32(Mm), np.int32(R), np.int32(M), np.int32(d)) + _table_args(tab, f32) + (np.int32(bool(grad)),))
    return _launch('rows_multi_kernel', R, M, args, nW, grad, f32, Mm)


def cols(ZA_dev, ZB_dev, tab, out, out_m=None, depmask=0, f32=False):
    """The explicit block S(ZA_r, ZB_j) into out (R, M) on the device (ZA_dev, ZB_dev: rank_coords, on the device);
    with depmask, out_m = the marginal of the columns not in depmask (blml_gpu's joint + marginal pass)."""
    t = dev_tab(tab, f32)
    R, d = ZA_dev.shape; M = ZB_dev.shape[0]
    if t['G'] > 8:
        raise ValueError("at most 8 radial groups on the GPU")
    if R and M:
        nth = 256; blocks = (R * M + nth - 1) // nth
        _kernel('cols_kernel', f32)((blocks,), (nth,), (ZA_dev, ZB_dev, np.int64(R), np.int64(M), np.int32(d), t['T'],
                                                    t['base'], t['Jk'], t['mk'], t['kinds'], t['nbins'], t['grp'], t['R'],
                                                    np.int64(t['R'].shape[1]), np.int32(t['G']),
                                                    np.int64(depmask), out, out if out_m is None else out_m))
    return out


def point_rows(A, Bp, w, kinds, fc, periods, want=(), grad=False, f32=False):
    A = np.ascontiguousarray(A, float); Bp = np.ascontiguousarray(Bp, float)
    (kd, dim_unit, grp, grp_unit, U, fcv, FIv, Pv, Mh, sc, prof, dprof, dsprof, dsg, prodfc, wunit, wcode, wcol, wgrp,
     nW, slot, Hh, nh, gtor, gsc) = _point_args(A, Bp, kinds, fc, periods, want)
    if Hh.shape[2] > 4:
        raise ValueError("GPU point_rows: a periodic ball over at most 4 columns")
    _limits('point_rows', U, nW)
    G = prof.shape[0]; LP = prof.shape[1] if G else 2
    pr, dpr, dspr = (prof, dprof, dsprof) if G else (np.zeros((1, 2)),) * 3
    R, d = A.shape; M = Bp.shape[0]
    args = (_d(A, cp.float64), _d(Bp, cp.float64), _d(w, cp.float64), np.int32(R), np.int32(M), np.int32(d),
            _d(kd, cp.int64), _d(dim_unit, cp.int64), _d(grp, cp.int64), _d(grp_unit, cp.int64), np.int32(U),
            _d(fcv, cp.float64), _d(FIv, cp.float64), _d(Pv, cp.float64), _d(Mh, cp.float64), _d(sc, cp.float64),
            _d(pr, cp.float64), _d(dpr, cp.float64), _d(dspr, cp.float64), np.int32(LP), np.int32(G), np.float64(dsg),
            _d(prodfc, cp.float64), _d(wunit, cp.int64), _d(wcode, cp.int64), _d(wcol, cp.int64), _d(wgrp, cp.int64),
            np.int32(nW), _d(slot, cp.int64), _d(Hh, cp.float64), _d(nh, cp.int64), _d(gtor, cp.int64), _d(gsc, cp.float64),
            np.int32(Hh.shape[1]), np.int32(Hh.shape[2]), np.int32(bool(grad)))
    return _launch('points_kernel', R, M, args, nW, grad, f32)


_CROSS_SRC = r'''
#define MAXD 32
#define USE_FLOAT @F32@
#if USE_FLOAT
typedef float REAL;
#define RSIN(t) sinf((float)remainder((double)(t), 6.283185307179586))
#define RCOS(t) cosf((float)remainder((double)(t), 6.283185307179586))
#else
typedef double REAL;
#define RSIN(t) sin(t)
#define RCOS(t) cos(t)
#endif
extern "C" __global__ void cross_kernel(const double* A, const double* B, const double* w, const double* fc,
                                        int R, int M, int d, int Q, int slice, double* S, double* D)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;
    int j0 = blockIdx.y * slice, j1 = min(M, j0 + slice);
    const double PI = 3.141592653589793;
    REAL kv[MAXD], dk[MAXD], pre[MAXD + 1], suf[MAXD + 1];
    double acc[8], dacc[8 * MAXD];
    for (int q = 0; q < Q; q++) { acc[q] = 0.0; for (int m = 0; m < d; m++) dacc[q * MAXD + m] = 0.0; }
    for (int j = j0; j < j1; j++) {
        pre[0] = 1.0;
        for (int m = 0; m < d; m++) {
            double t = A[(long long)r * d + m] - B[(long long)j * d + m], pt = PI * fc[m] * t;
            if (fabs(pt) < 1e-12) { kv[m] = (REAL)fc[m]; dk[m] = 0; }
            else if (fabs(pt) < (USE_FLOAT ? 0.5 : 0.05)) {   // series: pt cos pt - sin pt cancels to -pt^3/3
                double p2 = pt * pt;
                kv[m] = (REAL)(fc[m] * (1.0 - p2 / 6.0 + p2 * p2 / 120.0 - p2 * p2 * p2 / 5040.0));
                dk[m] = (REAL)(PI * fc[m] * fc[m] * pt * (-1.0 / 3.0 + p2 / 30.0 - p2 * p2 / 840.0 + p2 * p2 * p2 / 45360.0));
            } else {
                double s = RSIN(pt), c = RCOS(pt);
                kv[m] = (REAL)(s / (PI * t)); dk[m] = (REAL)((pt * c - s) / (PI * t * t));
            }
            pre[m + 1] = pre[m] * kv[m];
        }
        suf[d] = 1.0;
        for (int m = d - 1; m >= 0; m--) suf[m] = suf[m + 1] * kv[m];
        for (int q = 0; q < Q; q++) {
            double wj = w[(long long)j * Q + q];
            if (wj == 0.0) continue;
            acc[q] += (double)pre[d] * wj;
            for (int m = 0; m < d; m++) dacc[q * MAXD + m] += (double)(dk[m] * pre[m] * suf[m + 1]) * wj;
        }
    }
    for (int q = 0; q < Q; q++) {
        atomicAdd(&S[(long long)r * Q + q], acc[q]);
        for (int m = 0; m < d; m++) atomicAdd(&D[((long long)r * Q + q) * d + m], dacc[q * MAXD + m]);
    }
}
'''


def cross_sums(UA, UB, Wt, fc, f32=False):
    """blml_local.cross_sums on the GPU: S (nA, Q) = sum_b Wt[b, q] K(uA - uB_b), D (nA, Q, d) its gradient in uA, sinc
    product kernel; Q <= 8, d <= 32. Results are NumPy."""
    UA = np.ascontiguousarray(UA, float); UB = np.ascontiguousarray(UB, float); Wt = np.ascontiguousarray(Wt, float)
    R, d = UA.shape; M = UB.shape[0]; Q = Wt.shape[1]
    if Q > 8 or d > 32:
        raise ValueError("GPU cross_sums: Q <= 8 weight columns and d <= 32")
    key = ('cross_kernel', f32)
    if key not in _K:
        _K[key] = cp.RawKernel(_CROSS_SRC.replace('@F32@', '1' if f32 else '0'), 'cross_kernel')
    S = cp.zeros((R, Q)); D = cp.zeros((R, Q, d))
    if R and M:
        grid = ((R + 127) // 128, (M + SLICE - 1) // SLICE)
        _K[key](grid, (128,), (_d(UA, cp.float64), _d(UB, cp.float64), _d(Wt, cp.float64), _d(np.asarray(fc, float), cp.float64),
                               np.int32(R), np.int32(M), np.int32(d), np.int32(Q), np.int32(SLICE), S, D))
    return cp.asnumpy(S), cp.asnumpy(D)
