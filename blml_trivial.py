"""n-dimensional probability density estimation with the BLML-Trivial algorithm.

Reference
---------
Rahul Agarwal, Sridevi V. Sarma, and Zhe Chen,
"Nonparametric Estimation of Bandlimited Probability Density Functions",
arXiv:1503.06236, 2015.

BLML = Band-Limited Maximum Likelihood.

Overview
--------
Given n i.i.d. samples x_1, ..., x_n in R^d drawn from an unknown density
f, BLML assumes f is *band-limited* (its Fourier transform has finite
support, i.e. a cut-off frequency f_c).  Under this smoothness constraint a
nonparametric maximum likelihood estimator exists in closed form:

    f_hat(x) = ( (1/n) * sum_i c_hat_i * K(x, x_i) )^2                (1)

where K is the kernel and the coefficient vector c_hat solves the
constraint system

    rho_ni(c) = (1/n) * sum_j c_j * s_ij - 1 / c_i = 0,  i = 1..n    (2)

with s_ij = K(x_i, x_j) the kernel Gram matrix.  Squaring in (1) makes the
estimate non-negative by construction.

BLML-Trivial
------------
The system (2) is monotone with discontinuities at c_i = 0, so it has 2^n
solutions, one per orthant, each a local maximum of the likelihood in its
orthant.  When the true density is strictly positive and band-limited, the
global maximum asymptotically lies in the positive orthant
c0 = +[1, 1, ..., 1]^T (Theorem 6 of the paper).  BLML-Trivial exploits
this: it fixes the positive orthant and solves rho_n(c) = 0 there, using
the fixed-point iteration

    c_i <- n / sum_j s_ij * c_j,     c initialised to ones             (3)

which stays in the positive orthant.  If fixed-point iteration does not
reach the requested tolerance, a Newton solve of (2) is attempted as a
fallback.  The paper proves consistency of the resulting estimate when
f(x) > 0 everywhere and f is band-limited, with a convergence rate of
order 1/n for the MISE - faster than the O(n^{-4/5}) / O(n^{-12/13})
rates of standard higher-order kernel density estimators.

This module generalises the 1-D formulation to R^d with a *swappable*
kernel: only the default factory uses the sinc kernel; everything else
goes through the user-supplied kernel callable.
"""

import warnings
import numpy as np

_TINY = 1e-300


# ---------------------------------------------------------------------------
# Periodic (Dirichlet) kernel convention
# ---------------------------------------------------------------------------
# The sinc kernel fc*sinc(fc*t) = sin(pi fc t)/(pi t) is the reproducing kernel
# of functions band-limited to |nu| <= fc/2 (so g = sqrt(f) has band fc/2 and
# f = g^2 has band fc). Its P-periodic analogue keeps the harmonics m/P with
# |m| <= N, N = floor(fc*P/2), normalized by 1/P:
#
#     K(t) = (1/P) * sin(pi (2N+1) t / P) / sin(pi t / P),   K(0) = (2N+1)/P ~ fc.
#
# N must be an integer: with a fractional mode count the "kernel" is neither
# P-periodic nor positive semidefinite, and the BLML objective stops being
# convex. The 1/P normalization makes K reproducing over one period, so the
# fitted density integrates to 1.

import os
# BLML_DIRICHLET_LEGACY=1 restores the pre-fix kernel (M = 2*fc*P + 1 modes,
# possibly fractional, no 1/P) so older results can be reproduced exactly.
DIRICHLET_LEGACY = os.environ.get('BLML_DIRICHLET_LEGACY', '0') == '1'


def dirichlet_harmonics(fc, P):
    """Number of harmonics N kept on a P-periodic dim with cutoff fc."""
    return max(0, int(np.floor(fc * P / 2.0 + 1e-9)))


def cd_root(b_term, cc, sbb):
    """The coordinate update of BLML's block coordinate descent: the positive root of sbb t^2 + b t - cc = 0
    (b = (S e)_b - S_bb e_b, cc = n w_b), in the cancellation-free form. Shared by the solver's sweeps and the
    streaming touched-bin update (blml_local.LocalFit.cd)."""
    disc = np.sqrt(b_term * b_term + 4.0 * sbb * cc)
    return np.where(b_term >= 0, 2.0 * cc / np.maximum(b_term + disc, 1e-300), (-b_term + disc) / (2.0 * sbb))


def cd_line_search(e_blk, d, lin, quad, cc, steps):
    """Block safeguard: the largest t in steps with no increase of Phi = e'Se/2 - sum cc log e along e + t d
    (lin = d'(Se)_blk, quad = d'S_blk d); 0 if none. A block of coordinates all at 0 (new bins) takes the full
    step (any step toward the positive root decreases Phi); do not mix them with positive ones (their log
    gain is unbounded, so the test would be meaningless)."""
    pos = e_blk > 0
    if not pos.any():
        return 1.0
    r = np.where(pos, d / np.where(pos, e_blk, 1.0), 0.0).astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        logs = np.log1p(steps[:, None] * r[None, :]) @ np.where(pos, cc, 0.0).astype(np.float64)
    ok = (steps * lin + 0.5 * steps ** 2 * quad - logs <= 0) & np.all(1 + steps[:, None] * r[None, :] > 0, axis=1)
    return float(steps[np.argmax(ok)]) if ok.any() else 0.0


def dirichlet_inner_M(fc, P, fill):
    """M_i = 2 N_i + 1 of the harmonics |k| <= N_i a periodic two-slit band removes (fill < 1): the band keeps
    ceil(fi P / 2) <= |k| <= N, fi = (1 - fill) fc; 0 (D_0 = 0: nothing removed) for fill None or 1."""
    if fill is None or fill >= 1.0:
        return 0
    if DIRICHLET_LEGACY:
        raise NotImplementedError("dirichlet fill: not with DIRICHLET_LEGACY")
    Ni = int(np.ceil((1.0 - fill) * fc * P / 2.0 - 1e-9)) - 1
    return max(2 * Ni + 1, 0)


def dirichlet_modes_scale(fc, P):
    """(M, scale) with K(t) = scale * sin(pi M t/P) / sin(pi t/P)."""
    if DIRICHLET_LEGACY:
        return 2.0 * fc * P + 1.0, 1.0
    return 2 * dirichlet_harmonics(fc, P) + 1, 1.0 / P


# ---- Radial (non-separable) band kernels --------------------------------------------------
# A radial kind couples a group of dims through s = ||(fc_k * x_k)_k|| (fc_k * x_k is unitless):
#     K(x) = prod_k fc_k * profile(s, d_g),
# where profile is the FT of the indicator of the band set at unit fc (so the set has radius 1/2)
# and d_g the group's dimension. K is then the reproducing kernel of functions band-limited to that
# set, so the BLML system is unchanged. kinds tokens: '<name>' or '<name>:<tag>'; dims sharing a
# tag form one group (untagged '<name>' dims form one group). To add a band shape, register its
# profile in RADIAL_KERNELS -- nothing else changes. On a lattice with equal fc_k * h_k = c across
# the group, s^2 = c^2 * (integer r^2), so the Gram reads one cached 1-D table over integer r^2.

def _ball_profile(s, dg):
    """FT of the ball |nu| <= 1/2 in dg dims: (1/2)^(dg/2) J_{dg/2}(pi s) / s^(dg/2);
    dg = 1 gives sinc(s), dg = 2 the jinc. Value at s = 0: volume of that ball."""
    from scipy.special import jv, gamma
    s = np.asarray(s, dtype=float)
    out = np.full(s.shape, np.pi ** (dg / 2) * 0.5 ** dg / gamma(dg / 2 + 1))
    nz = s > 0
    out[nz] = 0.5 ** (dg / 2) * jv(dg / 2, np.pi * s[nz]) / s[nz] ** (dg / 2)
    return out


def _ball_profile_dlog(s, dg):
    """s p'(s) for the ball profile p: -pi (1/2)^(dg/2) s^(1 - dg/2) J_{dg/2 + 1}(pi s) (from
    d/dx [x^-v J_v(x)] = -x^-v J_{v+1}(x)); 0 at s = 0. dg = 1: cos(pi s) - sinc(s)."""
    from scipy.special import jv
    s = np.asarray(s, dtype=float)
    out = np.zeros(s.shape)
    nz = s > 0
    out[nz] = -np.pi * 0.5 ** (dg / 2) * s[nz] ** (1 - dg / 2) * jv(dg / 2 + 1, np.pi * s[nz])
    return out


def bessel_j(nu, x, xp=np, j01=None):
    """J_nu(x) for x >= 0 and nu an integer or half-integer (the ball profiles need nu = D/2 and D/2 + 1),
    from elementwise primitives only, so it runs on the GPU too (xp = cupy, j01 = cupyx's (j0, j1)):
    the power series for x < nu + 12 (upward recurrence loses accuracy below the order) and, above,
    upward recurrence J_{m+1} = (2m/x) J_m - J_{m-1} from J_1/2, J_3/2 (closed forms in sin, cos) or
    from J_0, J_1. Accurate to ~1e-12 relative where |J| is not tiny."""
    import math
    x = xp.asarray(x, dtype=float)
    out = xp.zeros(x.shape)
    small = x < nu + 12.0
    if bool(small.any()):
        xs = x[small] / 2.0
        term = xs ** nu / math.gamma(nu + 1.0); acc = term.copy()
        for k in range(1, 80):
            term = -term * xs * xs / (k * (k + nu))
            acc = acc + term
        out[small] = acc
    big = ~small
    if bool(big.any()):
        xb = x[big]
        if abs(nu - round(nu)) > 1e-9:             # half-integer: J_1/2, J_3/2 closed forms
            c = xp.sqrt(2.0 / (np.pi * xb))
            jm, jc, m = c * xp.sin(xb), c * (xp.sin(xb) / xb - xp.cos(xb)), 0.5
        else:
            j0, j1 = j01 if j01 is not None else (__import__('scipy.special', fromlist=['j0']).j0,
                                                   __import__('scipy.special', fromlist=['j1']).j1)
            jm, jc, m = j0(xb), j1(xb), 0.0
        if abs(nu - m) < 1e-9:
            out[big] = jm
        else:
            while abs((m + 1) - nu) > 1e-9:
                jm, jc, m = jc, (2 * (m + 1) / xb) * jc - jm, m + 1
            out[big] = jc
    return out


def ball_profile_xp(s, dg, xp=np, j01=None):
    """_ball_profile from bessel_j (elementwise only, so on the GPU with xp = cupy)."""
    import math
    s = xp.asarray(s, dtype=float)
    out = xp.full(s.shape, np.pi ** (dg / 2) * 0.5 ** dg / math.gamma(dg / 2 + 1))
    nz = s > 0
    out[nz] = 0.5 ** (dg / 2) * bessel_j(dg / 2, np.pi * s[nz], xp, j01) / s[nz] ** (dg / 2)
    return out


def ball_profile_dlog_xp(s, dg, xp=np, j01=None):
    """_ball_profile_dlog from bessel_j (elementwise only)."""
    s = xp.asarray(s, dtype=float)
    out = xp.zeros(s.shape)
    nz = s > 0
    out[nz] = -np.pi * 0.5 ** (dg / 2) * s[nz] ** (1 - dg / 2) * bessel_j(dg / 2 + 1, np.pi * s[nz], xp, j01)
    return out


RADIAL_KERNELS = {'ball': _ball_profile}
# the same profiles from elementwise primitives (bessel_j), for building tables on the GPU
RADIAL_KERNELS_XP = {'ball': (ball_profile_xp, ball_profile_dlog_xp)}
# Radial tables are on a grid in r (not r^2): RADIAL_Q entries per lattice step, linear interpolation in
# between (relative error ~ (pi c / RADIAL_Q)^2 / 8 with c = fc h ~ 0.1 n^-1/4, under 1e-5), so a table has
# ~ RADIAL_Q sqrt(r2max) entries -- linear in the span, where one entry per integer r^2 grows as its square.
RADIAL_Q = 4
# s p'(s) of each profile: every kernel unit has the scaling form K(t) = prod_k fc_k p(||fc * t||), so
# with the unit's fc scaled together, dK/dlog fc = prod_k fc_k (D p(s) + s p'(s)), D = the unit's dims.
RADIAL_DLOG = {'ball': _ball_profile_dlog}


def _split_kind(kind):
    name, _, tag = str(kind).partition(':')
    return name, tag


PERIODIC_GROUP_KINDS = ('pball',)   # multi-column units on periodic columns (not functions of r: no profile table)


def is_radial(kind):
    """Whether a kind is a multi-column unit (a radial band, or a periodic ball over periodic columns)."""
    name = _split_kind(kind)[0]
    if name in ('sinc', 'dirichlet', 'slits'):
        return False
    if name not in RADIAL_KERNELS and name not in PERIODIC_GROUP_KINDS:
        raise ValueError(f"unknown kernel kind {kind!r}; multi-column kinds: {sorted(RADIAL_KERNELS) + list(PERIODIC_GROUP_KINDS)}")
    return True


def is_periodic_group(kind):
    return _split_kind(kind)[0] in PERIODIC_GROUP_KINDS


def is_periodic_kind(kind):
    """A kind whose columns have a period (one-column dirichlet, or a periodic ball's columns)."""
    return _split_kind(kind)[0] in ('dirichlet',) + PERIODIC_GROUP_KINDS


def radial_groups(kinds):
    """[(profile name, [dims]), ...] in order of first appearance; [] if no radial dims."""
    groups = {}
    for k, kind in enumerate(kinds or []):
        if is_radial(kind):
            name, tag = _split_kind(kind)
            g = groups.setdefault(tag, (name, []))
            if g[0] != name:
                raise ValueError(f"radial group {tag!r} mixes kinds {g[0]!r} and {name!r}")
            g[1].append(k)
    return list(groups.values())


_RADIAL_INFO_SHOWN = set()


def check_radial(kinds, fc_vec):
    """A radial group has one fc (the band is round in the data's units). Raise if a direct fit
    passes unequal fc within a group; print a one-time note on input scaling per group layout."""
    groups = radial_groups(kinds)
    for name, dims in groups:
        f = np.asarray(fc_vec, dtype=float)[dims]
        if not np.allclose(f, f[0], rtol=1e-9, atol=0):
            raise ValueError(f"radial group {name} over dims {dims} uses one fc for every direction; "
                             f"got {f}. Pass equal fc (and rescale the data if the dims differ in scale).")
    key = tuple((name, tuple(dims)) for name, dims in groups)
    if groups and key not in _RADIAL_INFO_SHOWN:
        _RADIAL_INFO_SHOWN.add(key)
        print("info: radial kernel groups " + ", ".join(f"{name} over dims {dims}" for name, dims in groups)
              + " assume the same fc in every direction of the group; make sure those dims are on a common "
              "scale in the input data (same units, comparable spread).", flush=True)


def _radial_scale(dims, fc_vec, h_vec):
    """c = fc_k * h_k, which must be equal across a radial group so r^2 stays an integer."""
    c = fc_vec[dims] * h_vec[dims]
    if not np.allclose(c, c[0], rtol=1e-9, atol=0):
        raise ValueError(f"radial group dims {dims}: fc*h must be equal across the group, got {c}")
    return float(c[0])


def radial_grid_len(r2max):
    """Entries of a radial table covering lattice r^2 <= r2max (grid r = j / RADIAL_Q, +2 for interpolation)."""
    return int(np.floor(np.sqrt(r2max) * RADIAL_Q)) + 2


def radial_table(name, dims, fc_vec, h_vec, r2max, xp=np, j01=None):
    """K for one radial group on the grid r = j / RADIAL_Q lattice steps, j = 0..radial_grid_len(r2max) - 1
    (the kernels interpolate linearly in r). xp = cupy (with j01): built on the GPU."""
    c = _radial_scale(dims, fc_vec, h_vec)
    s = c * xp.arange(radial_grid_len(r2max), dtype=float) / RADIAL_Q
    prof = RADIAL_KERNELS[name](s, len(dims)) if xp is np else RADIAL_KERNELS_XP[name][0](s, len(dims), xp, j01)
    return np.prod(fc_vec[dims]) * prof


def radial_dlog_table(name, dims, fc_vec, h_vec, r2max, xp=np, j01=None):
    """dK/dlog fc for one radial group (its fc scaled together) on the radial_table grid."""
    c = _radial_scale(dims, fc_vec, h_vec)
    s = c * xp.arange(radial_grid_len(r2max), dtype=float) / RADIAL_Q
    D = len(dims)
    if xp is np:
        return np.prod(fc_vec[dims]) * (D * RADIAL_KERNELS[name](s, D) + RADIAL_DLOG[name](s, D))
    p, dl = RADIAL_KERNELS_XP[name]
    return np.prod(fc_vec[dims]) * (D * p(s, D, xp, j01) + dl(s, D, xp, j01))


def radial_dshift_table(name, dims, fc_vec, h_vec, r2max, i, xp=np, j01=None):
    """dK/dt_i of one radial group divided by the lattice offset of its column i, on the radial_table grid r (lattice
    steps): prod fc fc_i p'(c r) / r = prod fc fc_i s p'(s) / (c r^2), s = c r. At r = 0 the offset is 0, so that
    entry only bounds the interpolation (it takes its neighbour's)."""
    c = _radial_scale(dims, fc_vec, h_vec)
    r = xp.arange(radial_grid_len(r2max), dtype=float) / RADIAL_Q
    s = c * r; D = len(dims)
    dl = RADIAL_DLOG[name](s, D) if xp is np else RADIAL_KERNELS_XP[name][1](s, D, xp, j01)
    out = np.prod(fc_vec[dims]) * fc_vec[dims][i] * dl / (c * xp.where(r > 0, r * r, 1.0))
    out[0] = out[1]
    return out


def kernel_units(kinds, d):
    """The product kernel's factors: [(kind, [dims])], one per sinc / dirichlet dim and one per radial
    group (each unit's fc moves together, so each has one d/dlog fc)."""
    kinds = list(kinds) if kinds is not None else ['sinc'] * d
    units = [(kinds[k], [k]) for k in range(d) if not is_radial(kinds[k])]
    return units + [(name, list(dims)) for name, dims in radial_groups(kinds)]


# ---- kernel classes: one per factor kind of the product kernel ------------------------------------------
# A unit is one factor: a sinc or dirichlet column, or a radial group of columns. Its parameters th are a dict
# {'fc': (D,) cut-offs of its columns, 'period': P or None}. Each class provides, natively,
#   eval(th, diffs) / eval_dlog(th, p, diffs): K and dK/dlog p at coordinate differences (one array per column),
#   lattice(th, h, span) / lattice_dlog(th, p, h, span): the solver's table row and its derivative row,
# and declares its parameters (KERNEL_PARAMS): {name: (type, trainable by default)}. Every kernel formula lives
# here; the solver tables (CPU and GPU), kernel_unit and the gradients (param_grad) call these.
# A unit also owns its lattice (blml_kernels.ProductKernel assembles them):
#   step(th, h_target): its lattice step for a target width (a folded period is tiled by whole bins),
#   quantize(th, x, h): the bin centres of its coordinates,
#   block(th, h, J, vals) / dblock(th, p, h, J, vals): its value block for the row kernels -- (code, values, m):
#   code 0 a row over offsets -J..J, 1 a row over circular distance (m = bins per period), 3 a rank table over
#   the m occupied values `vals` (m x m, used when m^2 < 2J + 1: few values over a wide span).
# Besides its declared parameters every class takes p = 'shift' (a one-column unit) or ('shift', i) (column i of
# the unit) in eval_dlog / dblock: dK/dt_i, the derivative in the coordinate difference (an input map moves the
# coordinates; shift_index). A radial unit's lattice shift row is dK/dt_i divided by the offset of column i
# (the row kernels multiply by it).


def shift_index(p):
    """The column (within its unit) of a coordinate derivative p = 'shift' (0) or ('shift', i); None for a parameter."""
    if isinstance(p, str):
        return 0 if p == 'shift' else None
    return int(p[1]) if isinstance(p, tuple) and len(p) == 2 and p[0] == 'shift' else None


class SincKernel:
    """K(t) = fc sinc(fc t): the band |nu| <= fc/2 in one column. dK/dlog fc = fc cos(pi fc t)."""
    params = {'fc': ('scale', True)}

    def eval(self, th, diffs, xp=np):
        f = float(th['fc'][0])
        return f * xp.sinc(f * diffs[0])

    def eval_dlog(self, th, p, diffs, xp=np):
        if shift_index(p) is not None:
            return self.eval_dshift(th, diffs, xp)
        _check_param(self, p)
        f = float(th['fc'][0])
        return f * xp.cos(np.pi * f * diffs[0])

    def eval_dshift(self, th, diffs, xp=np):
        """dK/dt, the derivative in the coordinate difference t (a mapped unit's parameters move t):
        (pi f t cos(pi f t) - sin(pi f t)) / (pi t^2); its series -f (pi f)^2 t / 3 near t = 0 (odd: 0 at 0)."""
        f = float(th['fc'][0]); t = diffs[0]; pt = np.pi * f * t
        small = xp.abs(pt) < 1e-4
        ts = xp.where(small, 1.0, t)
        with np.errstate(divide='ignore', invalid='ignore'):
            full = (pt * xp.cos(pt) - xp.sin(pt)) / (np.pi * ts * ts)
        return xp.where(small, -f * (np.pi * f) ** 2 * t / 3.0, full)

    def lattice(self, th, h, span, xp=np):
        """Row over lattice offsets j = -span..span (indexed off + span by the solver kernels)."""
        f = float(th['fc'][0]); js = xp.arange(-span, span + 1, dtype=float)
        return f * xp.sinc(f * js * float(h[0]))

    def lattice_dlog(self, th, p, h, span, xp=np):
        _check_param(self, p)
        f = float(th['fc'][0]); js = xp.arange(-span, span + 1, dtype=float)
        return f * xp.cos(np.pi * f * js * float(h[0]))

    def step(self, th, h_target):
        return float(h_target)

    def quantize(self, th, x, h):
        return np.round(x / h) * h

    @staticmethod
    def _rank(vals, J):
        return vals is not None and len(vals) ** 2 < 2 * J + 1

    def block(self, th, h, J, vals=None, xp=np):
        if self._rank(vals, J):                            # the occupied values' pairs only
            v = np.asarray(vals, np.int64)
            return 3, self.eval(th, [xp.asarray((v[:, None] - v[None, :]).ravel() * float(h))], xp), len(v)
        return 0, xp.asarray(self.lattice(th, [h], J, xp=xp)), 0

    def dblock(self, th, p, h, J, vals=None, xp=np):
        """p: a parameter (dK/dlog p), or 'shift' (dK/dt: the coordinate difference, for a projection's W)."""
        if self._rank(vals, J):
            v = np.asarray(vals, np.int64)
            return 3, self.eval_dlog(th, p, [xp.asarray((v[:, None] - v[None, :]).ravel() * float(h))], xp), len(v)
        if shift_index(p) is not None:
            return 0, self.eval_dshift(th, [xp.arange(-J, J + 1, dtype=float) * float(h)], xp), 0
        return 0, self.lattice_dlog(th, p, [h], J, xp=xp), 0



class SlitsKernel:
    """Two bands symmetric about zero in one column: the far field of two equal slits, whose spectrum is the aperture
    |nu| in [(1 - w) fc / 2, fc / 2], w = fill = 2a / (d + a) in (0, 1] for slits of width a at separation d
    (w = 1: one interval, the sinc kernel). K(t) = fc sinc(fc t) - fi sinc(fi t), fi = (1 - w) fc. fill is a shape: fixed, part of the
    unit's kind token internally ('slits:<w>', as parse_modes writes it); fc moves both edges, dK/dlog fc =
    fc cos(pi fc t) - fi cos(pi fi t)."""
    params = {'fc': ('scale', True), 'fill': ('shape', False)}

    def __init__(self, fill=None):
        self.fill = None if fill is None else float(fill)

    def _f(self, th):
        if self.fill is None:
            raise ValueError("slits kernel: fill missing (give the column fill=, 0 < fill <= 1)")
        f = float(th['fc'][0]); return f, (1.0 - self.fill) * f

    def eval(self, th, diffs, xp=np):
        f, fi = self._f(th)
        return f * xp.sinc(f * diffs[0]) - fi * xp.sinc(fi * diffs[0])

    def eval_dlog(self, th, p, diffs, xp=np):
        if shift_index(p) is not None:
            return self.eval_dshift(th, diffs, xp)
        _check_param(self, p)
        if p != 'fc':
            raise ValueError("slits fill is a fixed shape: no gradient (select it by an outer search)")
        f, fi = self._f(th)
        return f * xp.cos(np.pi * f * diffs[0]) - fi * xp.cos(np.pi * fi * diffs[0])

    def eval_dshift(self, th, diffs, xp=np):
        f, fi = self._f(th); S = SincKernel()
        out = S.eval_dshift(dict(fc=[f]), diffs, xp)
        return out - S.eval_dshift(dict(fc=[fi]), diffs, xp) if fi > 0 else out

    def lattice(self, th, h, span, xp=np):
        return self.eval(th, [xp.arange(-span, span + 1, dtype=float) * float(h[0])], xp)

    def lattice_dlog(self, th, p, h, span, xp=np):
        return self.eval_dlog(th, p, [xp.arange(-span, span + 1, dtype=float) * float(h[0])], xp)

    def step(self, th, h_target):
        return float(h_target)

    def quantize(self, th, x, h):
        return np.round(x / h) * h

    def block(self, th, h, J, vals=None, xp=np):
        if SincKernel._rank(vals, J):
            v = np.asarray(vals, np.int64)
            return 3, self.eval(th, [xp.asarray((v[:, None] - v[None, :]).ravel() * float(h))], xp), len(v)
        return 0, xp.asarray(self.lattice(th, [h], J, xp=xp)), 0

    def dblock(self, th, p, h, J, vals=None, xp=np):
        if SincKernel._rank(vals, J):
            v = np.asarray(vals, np.int64)
            return 3, self.eval_dlog(th, p, [xp.asarray((v[:, None] - v[None, :]).ravel() * float(h))], xp), len(v)
        return 0, self.lattice_dlog(th, p, [h], J, xp=xp), 0


class DirichletKernel:
    """Periodic band on a circle of length P: K(t) = (1/P) D(u), u = 2 pi t / P, D(u) = sin(M u/2) / sin(u/2),
    M = 2N + 1 harmonics, N = floor(fc P / 2) (fc moves in whole harmonics: no gradient). At fixed N,
    dK/dlog P = -(1/P) (D(u) + u D'(u)) at a fixed raw offset t (not symmetric in t <-> P - t, so its lattice
    row is over raw offsets, unlike K's circular row). With fill < 1 (kind token 'dirichlet:<fill>', a fixed shape
    as for 'slits') the periodic two-slit band: D_M - D_{M_i}, the harmonics ceil(fi P/2) <= |k| <= N kept, fi =
    (1 - fill) fc; every expression below is linear in D, so the inner term enters each the same way."""
    params = {'fc': ('integer', True), 'period': ('scale', False), 'fill': ('shape', False)}

    def __init__(self, fill=None):
        self.fill = None if fill is None else float(fill)

    def _Mi(self, th):
        return dirichlet_inner_M(float(th['fc'][0]), float(th['period']), self.fill)

    @staticmethod
    def _D(M, u, xp):
        s2 = xp.sin(u / 2)
        small = xp.abs(s2) < 1e-9
        ss = xp.where(small, 1.0, s2)
        Dv = xp.where(small, float(M), xp.sin(M * u / 2) / ss)
        dD = xp.where(small, 0.0, (0.5 * M * xp.cos(M * u / 2) * ss - 0.5 * xp.sin(M * u / 2) * xp.cos(u / 2)) / ss ** 2)
        return Dv, dD

    def eval(self, th, diffs, xp=np):
        P, f = float(th['period']), float(th['fc'][0])
        t = xp.mod(xp.abs(diffs[0]), P); t = xp.minimum(t, P - t)
        M, scale = dirichlet_modes_scale(f, P); Mi = self._Mi(th)
        with np.errstate(divide='ignore', invalid='ignore'):
            val = (xp.sin(np.pi * M * t / P) - xp.sin(np.pi * Mi * t / P)) / xp.sin(np.pi * t / P)
        return scale * xp.where(t < 1e-10, M - Mi, val)

    def eval_dlog(self, th, p, diffs, xp=np):
        if shift_index(p) is not None:
            return self.eval_dshift(th, diffs, xp)
        _check_param(self, p)
        if p == 'fc':
            raise ValueError("dirichlet fc moves in whole harmonics: no gradient (use the harmonic test)")
        if p == 'fill':
            raise ValueError("dirichlet fill is a fixed shape: no gradient (select it by an outer search)")
        if DIRICHLET_LEGACY:
            raise NotImplementedError("period gradient: not with DIRICHLET_LEGACY")
        P, f = float(th['period']), float(th['fc'][0])
        M, _ = dirichlet_modes_scale(f, P); Mi = self._Mi(th)
        u = 2 * np.pi * diffs[0] / P
        Dv, dD = self._D(M, u, xp)
        if Mi:
            Di, dDi = self._D(Mi, u, xp); Dv, dD = Dv - Di, dD - dDi
        return -(Dv + u * dD) / P

    def eval_dshift(self, th, diffs, xp=np):
        """dK/dt = scale D'(u) 2 pi / P at u = 2 pi t / P (P-periodic, odd in t)."""
        if DIRICHLET_LEGACY:
            raise NotImplementedError("dirichlet shift derivative: not with DIRICHLET_LEGACY")
        P, f = float(th['period']), float(th['fc'][0])
        M, scale = dirichlet_modes_scale(f, P); Mi = self._Mi(th)
        _, dD = self._D(M, 2 * np.pi * diffs[0] / P, xp)
        if Mi:
            dD = dD - self._D(Mi, 2 * np.pi * diffs[0] / P, xp)[1]
        return scale * dD * (2 * np.pi / P)

    def lattice(self, th, h, span, xp=np):
        """Row over circular lattice distance 0..n_bins-1 (n_bins = P / h)."""
        P = float(th['period']); nb = int(round(P / float(h[0])))
        return xp.asarray(dirichlet_table(float(th['fc'][0]), P, nb, self.fill))

    def lattice_dlog(self, th, p, h, span, xp=np):
        """Derivative row over RAW lattice offsets j = -span..span (index off + span, as a sinc row)."""
        js = xp.arange(-span, span + 1, dtype=float)
        return self.eval_dlog(th, p, [js * float(h[0])], xp)

    def step(self, th, h_target, fold=True):
        """A folded column's step tiles the period with whole bins (>= 8); an unfolded one is binned as sinc."""
        if not fold or DIRICHLET_LEGACY:
            return float(h_target)
        P = float(th['period'])
        return P / max(8, int(np.ceil(P / h_target - 1e-9)))

    def quantize(self, th, x, h, fold=True):
        if not fold:
            return np.round(x / h) * h
        P = float(th['period'])
        return np.mod(np.round(np.mod(x, P) / h) * h, P)

    def block(self, th, h, J, vals=None, xp=np, fold=True):
        if not fold:                                       # raw offsets: the kernel is P-periodic by itself
            return 0, self.eval(th, [xp.arange(-J, J + 1) * float(h)], xp), 0
        row = xp.asarray(self.lattice(th, [h], J, xp=xp))
        return 1, row, len(row)

    def dblock(self, th, p, h, J, vals=None, xp=np, fold=True):
        return 0, self.lattice_dlog(th, p, [h], J, xp=xp), 0     # raw offsets either way



class RadialKernel:
    """A radial band over a group of D columns: K(t) = prod fc p(||fc * t||), p the profile of the band set at
    unit fc (RADIAL_KERNELS[name]); with the group's fc scaled together, dK/dlog fc = prod fc (D p + s p')."""
    params = {'fc': ('scale', True)}

    def __init__(self, name):
        self.name = name

    def eval(self, th, diffs, xp=np):
        fc = np.asarray(th['fc'], float)
        s = np.sqrt(sum((f * dk) ** 2 for f, dk in zip(fc, diffs)))
        return float(np.prod(fc)) * RADIAL_KERNELS[self.name](s, len(fc))

    def eval_dlog(self, th, p, diffs, xp=np):
        i = shift_index(p)
        fc = np.asarray(th['fc'], float); D = len(fc)
        s = np.sqrt(sum((f * dk) ** 2 for f, dk in zip(fc, diffs)))
        if i is not None:                                  # dK/dt_i = prod fc p'(s) fc_i^2 t_i / s; s p'(s) = RADIAL_DLOG
            return float(np.prod(fc)) * RADIAL_DLOG[self.name](s, D) * fc[i] ** 2 * diffs[i] / np.where(s > 0, s * s, 1.0)
        _check_param(self, p)
        return float(np.prod(fc)) * (D * RADIAL_KERNELS[self.name](s, D) + RADIAL_DLOG[self.name](s, D))

    def lattice(self, th, h, span, xp=np, j01=None):
        """Row on the r grid (radial_table), span = the group's per-column spans in lattice steps."""
        D = len(th['fc'])
        return radial_table(self.name, list(range(D)), np.asarray(th['fc'], float), np.asarray(h, float),
                            int((np.asarray(span) ** 2).sum()), xp=xp, j01=j01)

    def lattice_dlog(self, th, p, h, span, xp=np, j01=None):
        D = len(th['fc']); i = shift_index(p)
        args = (self.name, list(range(D)), np.asarray(th['fc'], float), np.asarray(h, float),
                int((np.asarray(span) ** 2).sum()))
        if i is not None:                                  # / the offset of column i (the row kernels multiply by it)
            return radial_dshift_table(*args, i, xp=xp, j01=j01)
        _check_param(self, p)
        return radial_dlog_table(*args, xp=xp, j01=j01)

    def step(self, th, h_target):
        return float(h_target)

    def quantize(self, th, x, h):
        return np.round(x / h) * h

    def block(self, th, h, span, vals=None, xp=np, j01=None):
        """The group's row on the r^2 grid (code 2); h, span: the group's per-column steps and spans."""
        return 2, self.lattice(th, h, span, xp=xp, j01=j01), 0

    def dblock(self, th, p, h, span, vals=None, xp=np, j01=None):
        return 2, self.lattice_dlog(th, p, h, span, xp=xp, j01=j01), 0



def pball_harmonics(fc, periods):
    """The harmonics of a periodic ball: integer vectors k with sum_i (k_i / P_i)^2 <= (fc / 2)^2 (on one column the
    dirichlet rule |k| <= floor(fc P / 2)), as an (H, D) int array, and the scale 1 / prod P."""
    P = np.atleast_1d(np.asarray(periods, float)); R = float(fc) / 2.0
    N = [max(0, int(np.floor(R * p + 1e-9))) for p in P]
    Kv = np.stack([g.ravel() for g in np.meshgrid(*[np.arange(-n, n + 1) for n in N], indexing='ij')], 1)
    keep = ((Kv / P) ** 2).sum(1) <= R * R * (1 + 1e-9) + 1e-12
    return Kv[keep], 1.0 / float(np.prod(P))


class PeriodicBallKernel:
    """A ball band over a group of D periodic columns (periods P_i): the harmonics k / P with ||k / P|| <= fc / 2,
    K(t) = (1 / prod P) sum_k cos(2 pi sum_i k_i t_i / P_i) -- the ball kernel made periodic (its Poisson sum), the
    round analogue of 'dirichlet' (on one column it is dirichlet). fc moves in whole harmonics (no gradient); the
    periods are fixed. On a folded lattice (whole bins tile each period) its block is the table over the group's
    circular offsets (code 4, 'torus': index sum over its columns of off_k mod m_k, row-major in column order)."""
    params = {'fc': ('integer', True), 'period': ('scale', False)}

    @staticmethod
    def _parts(th):
        P = np.atleast_1d(np.asarray(th['period'], float))
        Kv, sc = pball_harmonics(float(np.atleast_1d(th['fc'])[0]), P)
        return P, Kv, sc

    def eval(self, th, diffs, xp=np):
        P, Kv, sc = self._parts(th)
        out = 0.0
        for k in Kv:
            ph = 0.0
            for i in range(len(P)):
                if k[i]:
                    ph = ph + (2 * np.pi * k[i] / P[i]) * diffs[i]
            out = out + xp.cos(ph)
        return sc * (out + 0.0 * sum(diffs))             # the broadcast shape of diffs even for one harmonic

    def eval_dlog(self, th, p, diffs, xp=np):
        i = shift_index(p)
        if i is None:
            _check_param(self, p)
            raise ValueError(f"pball {p}: " + ("fc moves in whole harmonics: no gradient" if p == 'fc'
                                               else "the periods are fixed"))
        P, Kv, sc = self._parts(th)
        out = 0.0 * sum(diffs)
        for k in Kv:
            if not k[i]:
                continue
            ph = 0.0
            for j in range(len(P)):
                if k[j]:
                    ph = ph + (2 * np.pi * k[j] / P[j]) * diffs[j]
            out = out - (2 * np.pi * k[i] / P[i]) * xp.sin(ph)
        return sc * out

    def eval_dshift(self, th, diffs, xp=np):
        return self.eval_dlog(th, ('shift', 0), diffs, xp)

    @staticmethod
    def _bins(th, h):
        P = np.atleast_1d(np.asarray(th['period'], float)); h = np.atleast_1d(np.asarray(h, float))
        ms = np.round(P / h).astype(np.int64)
        if np.any(np.abs(ms * h - P) > 1e-9 * P):
            raise ValueError(f"pball: the lattice steps {h} must tile the periods {P} with whole bins (a folded lattice)")
        return ms

    def _torus(self, th, h, f, xp):
        ms = self._bins(th, h); h = np.atleast_1d(np.asarray(h, float))
        grids = np.meshgrid(*[np.arange(m) * hh for m, hh in zip(ms, h)], indexing='ij')
        return xp.asarray(np.ravel(f([g for g in grids]))), ms

    def lattice(self, th, h, span, xp=np, j01=None):
        """The table over the group's circular lattice offsets 0..m_k - 1 per column (row-major in column order)."""
        return self._torus(th, h, lambda d: self.eval(th, d), xp)[0]

    def lattice_dlog(self, th, p, h, span, xp=np, j01=None):
        """d/dt_i on the same table (the value itself: not divided by an offset, unlike a radial shift row)."""
        return self._torus(th, h, lambda d: self.eval_dlog(th, p, d), xp)[0]

    def block(self, th, h, span, vals=None, xp=np, j01=None):
        t, ms = self._torus(th, h, lambda d: self.eval(th, d), xp)
        return 4, t, ms

    def dblock(self, th, p, h, span, vals=None, xp=np, j01=None):
        t, ms = self._torus(th, h, lambda d: self.eval_dlog(th, p, d), xp)
        return 4, t, ms


def _check_param(kern, p):
    if p not in kern.params:
        raise ValueError(f"{type(kern).__name__} has no parameter {p!r} (has {sorted(kern.params)})")


def is_mapped(key, c):
    """A mapped unit: a tuple key whose kernel acts on u = map(x[key]) -- given as map= (a blml_maps.InputMap) or
    W= / k= (a LinearMap, u = x W with orthonormal W). Its k outputs are lattice columns of the unit's kind: sinc or
    dirichlet (one column each, own fc; dirichlet with period=), or a radial kind (one round band over the k outputs,
    one fc). The map's parameters are its own (projected_modes)."""
    return isinstance(key, tuple) and any(a in c for a in ('map', 'W', 'k'))


class ProjectedInputs:
    """The input -> lattice map of modes with mapped units (projected_modes): the lattice columns are each unit's k
    outputs u = map(x[cols]) (blml_maps; W= / k=: a LinearMap), in key order, then the other input columns."""

    def __init__(self, units, others, d_in):
        self.units, self.others, self.d_in = units, others, d_in      # units: [dict(cols, map, trainable)]
        self.k = sum(u['map'].k for u in units)

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[1] != self.d_in:
            raise ValueError(f"mapped units: {X.shape[1]} input columns, the modes have {self.d_in}")
        return np.concatenate([u['map'].forward(X[:, u['cols']]) for u in self.units] + [X[:, self.others]], axis=1)

    def spans(self):
        """(unit, first lattice column, k) of each mapped unit."""
        out, L = [], 0
        for u in self.units:
            out.append((u, L, u['map'].k)); L += u['map'].k
        return out

    def lattice_col(self, j):
        """The lattice column of input column j (not a projected input)."""
        return self.k + self.others.index(j)

    def rest(self, Xrest, dep_in):
        """Lattice rest columns of points given in the input's non-dep columns."""
        Xrest = np.atleast_2d(np.asarray(Xrest, float))
        rest_in = [j for j in range(self.d_in) if j not in dep_in]
        Xf = np.zeros((len(Xrest), self.d_in)); Xf[:, rest_in] = Xrest
        dep_lat = {self.lattice_col(j) for j in dep_in}
        return self(Xf)[:, [j for j in range(self.k + len(self.others)) if j not in dep_lat]]


def projected_modes(modes, seed=0):
    """(ProjectedInputs, lattice modes) for modes with mapped units (is_mapped: a tuple key with map= or W= / k=),
    else (None, modes); the outputs take the unit's kind (is_mapped). A unit gives W (D, k) or k (u = x W, a LinearMap; without W a random orthonormal start from seed), or
    map= any blml_maps.InputMap (its own parameters). trainable= on the unit freezes (False) or trains its map, any kind
    (W given as dict(value=, trainable=) may say the same; a contradiction raises). The units must be the same in every
    mode; their fc may differ. A mapped unit cannot be dep."""
    isp = lambda m, c: is_mapped(c, m[c])
    if not isinstance(modes, (list, tuple)) or not any(isp(m, c) for m in modes for c in m):
        return None, modes
    keys = [c for c in modes[0] if isp(modes[0], c)]
    units, used = [], []
    for key in keys:
        c = modes[0][key]
        if c.get('dep'):
            raise ValueError(f"columns {key}: a mapped unit cannot be dep")
        if 'h' in c:
            lattice_fixed([{key: c}], 'a fixed lattice (h in the modes)')
        if c.get('map') is not None:                   # any input map (blml_maps.InputMap): its own parameters
            mp = c['map']; tr = bool(c.get('trainable', any(t for _, t in mp.params.values())))
        else:                                           # W / k: u = x W
            from blml_maps import LinearMap
            Wv, Wt = param_value(c.get('W'))
            if Wv is None:
                if c.get('k') is None:
                    raise ValueError(f"columns {key}: give W (D, k) or k (random start)")
                Wv = np.random.default_rng([seed, len(units)]).normal(size=(len(key), int(c['k'])))
            Wv = np.asarray(Wv, float)
            if c.get('k') is not None and Wv.shape[1] != int(c['k']):
                raise ValueError(f"columns {key}: k = {c['k']} but W has {Wv.shape[1]} columns")
            if Wv.ndim != 2 or Wv.shape[0] != len(key):
                raise ValueError(f"columns {key}: W must be ({len(key)}, k), got {Wv.shape}")
            if 'trainable' in c and Wt is not None and bool(c['trainable']) != bool(Wt):
                raise ValueError(f"columns {key}: trainable={c['trainable']} contradicts W's trainable={Wt}")
            tr = bool(c['trainable']) if 'trainable' in c else (True if Wt is None else bool(Wt))   # unit or W flag
            mp = LinearMap(Wv)
        units.append(dict(cols=list(key), map=mp, trainable=tr))
        used += list(key)
    if len(set(used)) != len(used):
        raise ValueError("mapped units overlap")
    d_in = max(used + [j for c in modes[0] if c not in keys for j in (c if isinstance(c, tuple) else (c,))]) + 1
    others = [j for j in range(d_in) if j not in used]
    units_k = sum(u['map'].k for u in units)
    out = []
    for m in modes:
        if [c for c in m if isp(m, c)] != keys:
            raise ValueError("mapped units must be the same in every mode")
        lm, L = {}, 0
        for key, u in zip(keys, units):
            c, k = m[key], u['map'].k
            if set(c) - _MAPPED_KEYS:
                raise ValueError(f"columns {key}: unknown keys {sorted(set(c) - _MAPPED_KEYS)}")
            kind = c.get('kind', 'sinc')
            f, ft = param_value(c.get('fc'))
            if is_radial(kind):                          # one round band over the outputs: one fc
                fv = None if f is None else np.asarray(f, float).ravel()
                if fv is not None and not np.allclose(fv, fv[0]):
                    raise ValueError(f"columns {key}: a radial kind has one fc, got {f}")
                cj = dict(kind=kind)
                if ft is not None:
                    cj['fc'] = dict(trainable=ft, **({} if fv is None else dict(value=float(fv[0]))))
                elif fv is not None:
                    cj['fc'] = float(fv[0])
                lm[tuple(range(L, L + k))] = cj; L += k
                continue
            fv = None if f is None else np.broadcast_to(np.asarray(f, float), (k,))
            pv = None if c.get('period') is None else c['period']
            for j in range(k):
                cj = dict(kind=kind)
                if kind == 'slits':                      # slits outputs: the unit's fill (a fixed shape) each
                    if 'fill' not in c:
                        raise ValueError(f"columns {key}: kind 'slits' needs fill=")
                    cj['fill'] = c['fill']
                elif kind == 'dirichlet' and 'fill' in c:  # periodic two-slit outputs
                    cj['fill'] = c['fill']
                if ft is not None:
                    cj['fc'] = dict(trainable=ft, **({} if fv is None else dict(value=float(fv[j]))))
                elif fv is not None:
                    cj['fc'] = float(fv[j])
                if pv is not None:                       # dirichlet outputs: one period each (or one for all)
                    pval, ptr_ = param_value(pv)
                    pj = float(np.broadcast_to(np.asarray(pval, float), (k,))[j])
                    cj['period'] = pj if ptr_ is None else dict(value=pj, trainable=ptr_)
                lm[L] = cj; L += 1
        pos = {j: units_k + i for i, j in enumerate(others)}
        for key, c in m.items():                       # the other columns and units (e.g. ball), renumbered
            if key not in keys:
                lm[tuple(pos[j] for j in key) if isinstance(key, tuple) else pos[key]] = c
        out.append(lm)
    return ProjectedInputs(units, others, d_in), out


def unprojected_modes(proj, lmodes, modes):
    """Modes in the input's form (tuple keys) from lattice modes: each unit's outputs' fc and its current W."""
    out = []
    for m, lm in zip(modes, lmodes):
        mm, L = {}, 0
        for u in proj.units:
            key, k = tuple(u['cols']), u['map'].k; c = dict(m[key])
            if is_radial(c.get('kind', 'sinc')):
                fcs = param_value(lm[tuple(range(L, L + k))].get('fc'))[0]
            else:
                fcs = [param_value(lm[L + j].get('fc'))[0] for j in range(k)]
                if c.get('period') is not None:
                    pts = [param_value(lm[L + j].get('period'))[0] for j in range(k)]
                    ptr_ = param_value(c['period'])[1]
                    c['period'] = pts if ptr_ is None else dict(value=pts, trainable=ptr_)
            ft = param_value(c.get('fc'))[1]
            c['fc'] = fcs if ft is None else dict(value=fcs, trainable=ft)
            if c.get('map') is None:                      # a LinearMap given by W / k: its current W
                c['W'] = dict(value=u['map'].W.copy(), trainable=u['trainable'])
            mm[key] = c; L += k
        back = {proj.k + i: j for i, j in enumerate(proj.others)}
        for key, c in lm.items():
            if (key[0] if isinstance(key, tuple) else key) >= proj.k:   # (a mapped unit's keys are below proj.k)
                mm[tuple(back[j] for j in key) if isinstance(key, tuple) else back[key]] = c
        out.append(mm)
    return out


_MAPPED_KEYS = {'kind', 'k', 'W', 'fc', 'period', 'fill', 'map', 'trainable'}


def lattice_fixed(modes, who):
    """Estimators on counts keep the data's lattice (the instrument's resolution: a data constraint, never re-binned).
    A mapped unit needs a regular lattice in its outputs u = map(x) (Nyquist holds on a lattice regular in the
    kernel's coordinates; projected bin centres are not one), i.e. re-binning: fit the bin centres as points with
    their counts as weights (BLMLQuick / BLMLQuickStreaming, sample_weight) instead."""
    if any(is_mapped(key, c) for m in modes for key, c in m.items()):
        raise ValueError(f"{who}: a mapped unit needs a lattice regular in its outputs, but {who} keeps the "
                         "data's lattice; fit the bin centres as points weighted by their counts (BLMLQuick or "
                         "BLMLQuickStreaming with sample_weight)")


def projected_front(cls):
    """Class decorator for point estimators: modes may hold mapped units. fit maps the modes and the data
    to the lattice columns (projected_modes; W as given) and fits there; predict, sqrt_density, predict_ratio (raw
    inputs) and predict_marginal (raw non-dep inputs) project the queries first, once (nested calls see lattice
    columns). self.proj_ (ProjectedInputs or None), self.spec_in_ (the modes as given)."""
    import functools
    fit0 = cls.fit

    @functools.wraps(fit0)
    def fit(self, X, modes, *a, **kw):
        proj, lmodes = projected_modes(modes, getattr(self, 'seed', 0) or 0)
        self.proj_ = None
        if proj is None:
            return fit0(self, X, modes, *a, **kw)
        init = kw.get('init')
        guard = init is not None and getattr(init, 'proj_', None) is not None
        if guard:
            init._in_lat = True                        # a warm start reads init on the lattice columns
        try:
            fit0(self, proj(X), lmodes, *a, **kw)
        finally:
            if guard:
                init._in_lat = False
        self.proj_, self.spec_in_ = proj, modes
        return self
    cls.fit = fit

    def query(name, rest=False):
        f0 = getattr(cls, name)

        @functools.wraps(f0)
        def f(self, Xq, *a, **kw):
            p = getattr(self, 'proj_', None)
            if p is None or getattr(self, '_in_lat', False):
                return f0(self, Xq, *a, **kw)
            Xq = p.rest(Xq, [p.others[j - p.k] for j in self.dep_]) if rest else p(Xq)
            self._in_lat = True
            try:
                return f0(self, Xq, *a, **kw)
            finally:
                self._in_lat = False
        setattr(cls, name, f)
    for name in ('predict', 'sqrt_density', 'predict_ratio'):
        query(name)
    query('predict_marginal', rest=True)
    query('sample_conditional', rest=True)
    if hasattr(cls, 'param_grad'):                   # the kernel parameters of the lattice units, at raw queries
        query('param_grad')
    return cls


def kernel_class(kind):
    """The kernel class of a kind token ('sinc', 'dirichlet', or a radial name / 'name:tag')."""
    name = _split_kind(kind)[0]
    if name == 'sinc':
        return SincKernel()
    if name == 'dirichlet':
        tag = _split_kind(kind)[1]
        return DirichletKernel(float(tag) if tag else None)
    if name == 'slits':
        tag = _split_kind(kind)[1]
        return SlitsKernel(float(tag) if tag else None)
    if name == 'pball':
        return PeriodicBallKernel()
    return RadialKernel(name)


def unit_theta(kind, fc_vec, periods, dims):
    """A unit's parameters {'fc': (D,), 'period': P or None} from the per-column fc and periods."""
    name = _split_kind(kind)[0]
    if name == 'pball':                                   # one period per column of the group
        per = None if periods is None else np.array([periods[j] for j in dims], float)
    else:
        per = (None if periods is None else periods[dims[0]]) if name == 'dirichlet' else None
    return dict(fc=np.asarray(fc_vec, float)[list(dims)], period=per)


def kernel_unit(kind, fc_vec, dims, diffs, periods, need_grad=True, xp=np):
    """(K, dK/dlog fc) of one kernel unit at coordinate differences diffs (one array per column of the
    unit), from its kernel class; dK/dlog fc is None for dirichlet (whole harmonics) or need_grad=False."""
    cls, th = kernel_class(kind), unit_theta(kind, fc_vec, periods, dims)
    K = cls.eval(th, diffs, xp)
    grad = need_grad and cls.params['fc'][0] != 'integer'
    return K, (cls.eval_dlog(th, 'fc', diffs, xp) if grad else None)


def fc_grad_from_parts(g, dg, Zn, dZ, eps=0.0):
    """d/dlog fc_u [log(sum_k |g_k|^2 + eps) - log Z] from g (m, K), dg (U, m, K), Z and dZ (U,)."""
    F = np.maximum((np.abs(g) ** 2).sum(1), 1e-300) + eps
    out = np.stack([2 * np.real((np.conj(g) * dg[u]).sum(1)) / F for u in range(len(dZ))], 1)
    return out - np.asarray(dZ)[None, :] / Zn


def sweep_response(e, Se, dSe, w, n, sbb, dsbb):
    """d e_b / d log fc_u of one coordinate-descent sweep at the current coefficients (real rank 1),
    linearised there: each e_b is the positive root of S_bb t^2 + R_b t - n w_b = 0 with
    R_b = (S e)_b - S_bb e_b, so dt = -(t dR + t^2 dS_bb) / (2 S_bb t + R) at t = e_b, with
    dR_b = (dS e)_b - dS_bb e_b. e, Se (B,); dSe (U, B); dsbb (U,). Returns (U, B)."""
    R = Se - sbb * e
    den = np.where(np.abs(2 * sbb * e + R) > 1e-300, 2 * sbb * e + R, 1e-300)
    return np.stack([-(e * (dSe[u] - dsbb[u] * e) + e ** 2 * dsbb[u]) / den for u in range(len(dsbb))])


def pooled_fc_grad(model, acc, want, allu, fc, periods, E64, SE, Zn, dZ, dSE, n, implicit_grad):
    """Pooled weighted-mean gradient (U,) from per-batch accumulators acc (fc_grad mode='points') and the
    current fit's lattice terms: mean of the per-point parts - dZ/Z, plus the one-sweep response with
    the pooled response weights mapped onto the current bins (bins a batch's fit lacked contribute 0)."""
    W = sum(a['W'] for a in acc)
    out = sum(a['cont'] for a in acc) / W - np.asarray(dZ) / Zn
    if implicit_grad:
        Zi = model._Zi
        key = lambda A: np.ascontiguousarray(A, dtype=np.int64).view(np.dtype((np.void, 8 * A.shape[1]))).ravel()
        v = np.zeros(len(Zi))
        for a in acc:
            _, ia, ib = np.intersect1d(key(Zi), key(a['Zi']), return_indices=True)
            v[ia] += np.real(np.asarray(a['v']))[ib]
        e = np.real(E64[:, 0]); Se = np.real(SE[:, 0]); w = np.asarray(model.w_, float)
        sbb, dsbb = _diag_and_dlog(allu, want, fc, periods)
        dT = sweep_response(e, Se, np.stack([np.real(np.asarray(dd)[:, 0]) for dd in dSE]), w, n, sbb, dsbb)
        out = out + dT @ (v / W - 2.0 * Se / (n ** 2 * Zn))
    return out


def param_grad_modes(Zi, h, Zc, E, n, mspecs, Xq, units, eps_rel=0.0, weights=None, rows_fn=None, pts_fn=None):
    """d log f(x) / d theta at the queries Xq (m, U) for any fit on a lattice, each mode with its own kernel --
    f = sum_k |g_k|^2 / Z, g_k = (1/n) sum_b E_bk K_k(x - z_b), Z = sum_k e_k^H S_k e_k / n^2 (coefficients fixed,
    renormalised). mspecs: one (fc, kinds, periods, fold) per mode. units: dims (fc), (dims, name), or (mode, dims,
    name) for one mode's parameter (without a mode: every mode's, the modes sharing the unit). Zi, h: the bins'
    lattice coordinates and steps; Zc the bin centres; rows_fn / pts_fn: the engine's row and point kernels
    (blml_local or blml_local_gpu). The general path behind param_grad (per-mode kernels, conditionals)."""
    from blml_kernels import ProductKernel
    from blml_local import rows as rows_cpu, point_rows as pts_cpu
    rows_fn = rows_fn or rows_cpu; pts_fn = pts_fn or pts_cpu
    Xq = np.atleast_2d(np.asarray(Xq, float)); K = len(mspecs); d = Zi.shape[1]
    E = np.asarray(E); E = E[:, None] if E.ndim == 1 else E
    spec = []
    for u in units:
        if isinstance(u, tuple) and len(u) == 3:
            spec.append((u[0], list(u[1]), u[2]))
        elif isinstance(u, tuple) and len(u) == 2 and isinstance(u[1], str):
            spec.append((None, list(u[0]), u[1]))
        else:
            spec.append((None, list(u), 'fc'))
    U, m = len(spec), len(Xq)
    cplx = lambda fn, A, B, w, *a: (lambda r, i: (r[0] + 1j * i[0], r[1] + 1j * i[1]))(
        fn(A, B, np.ascontiguousarray(w.real), *a), fn(A, B, np.ascontiguousarray(w.imag), *a)) \
        if np.iscomplexobj(w) else fn(A, B, np.ascontiguousarray(w, float), *a)
    g = np.zeros((m, K), complex if np.iscomplexobj(E) else float); dg = np.zeros((U, m, K), g.dtype)
    Zq, dZq, fb = 0.0, np.zeros(U), 0.0
    for k, (fc, kinds, periods, fold) in enumerate(mspecs):
        kinds = list(kinds); periods = list(periods) if periods is not None else [None] * d
        fc = np.broadcast_to(np.asarray(fc, float), (d,)).copy(); allu = kernel_units(kinds, d)
        qk = [q for q, (mk, _, _) in enumerate(spec) if mk is None or mk == k]
        want = _want_params(allu, [(spec[q][1], spec[q][2]) for q in qk])
        tab = ProductKernel(kinds, periods, fold).tables(fc, periods, h, Zi, want)
        Se, dSe = cplx(rows_fn, Zi, Zi, E[:, k], tab, bool(want))
        Zq += float(np.real(np.vdot(E[:, k], Se))) / n ** 2
        fb = fb + np.abs(Se / n) ** 2
        for j, q in enumerate(qk):
            dZq[q] += float(np.real(np.vdot(E[:, k], dSe[:, j]))) / n ** 2
        gk, dgk = cplx(pts_fn, Xq, Zc, E[:, k] / n, kinds, fc, periods, want, bool(want))
        g[:, k] = gk
        for j, q in enumerate(qk):
            dg[q, :, k] = dgk[:, j]
    eps = float(eps_rel * np.median(fb / Zq)) if eps_rel else 0.0
    return fc_grad_from_parts(g, dg, Zq, dZq, eps)


def conditional_param_grad(joint_fn, marg_fn, Xq, units, dep, d):
    """d log p(x_dep | x_rest) / d theta = the joint's gradient minus the marginal's (a unit of the conditioning
    columns is in both; one of the dep columns only in the joint). joint_fn(Xq, units), marg_fn(Xrest, units in the
    marginal's columns) -> (m, U)."""
    rest = [j for j in range(d) if j not in dep]; pos = {j: i for i, j in enumerate(rest)}
    Xq = np.atleast_2d(np.asarray(Xq, float))
    out = np.asarray(joint_fn(Xq, units), float).copy()
    mq, mu = [], []
    for q, u in enumerate(units):
        mode, dims, name = (u[0], u[1], u[2]) if (isinstance(u, tuple) and len(u) == 3) else \
            ((None, u[0], u[1]) if (isinstance(u, tuple) and len(u) == 2 and isinstance(u[1], str)) else (None, u, 'fc'))
        if all(j in pos for j in dims):
            mq.append(q); mapped = [pos[j] for j in dims]
            mu.append((mode, mapped, name) if mode is not None else (mapped, name))
    if mq:
        out[:, mq] -= np.asarray(marg_fn(Xq[:, rest], mu), float)
    return out


def _want_params(allu, units):
    """[(unit index, parameter name)] for units given as dims (their fc) or (dims, name); checks each."""
    out = []
    for u in units:
        dims, p = (u[0], u[1]) if (isinstance(u, tuple) and len(u) == 2 and isinstance(u[1], str)) else (u, 'fc')
        i = _unit_index(allu, list(dims))
        cls = kernel_class(allu[i][0])
        _check_param(cls, p)
        if cls.params[p][0] == 'integer':
            raise ValueError(f"param_grad: {p} of {allu[i][0]} columns {allu[i][1]} is discrete (no gradient)")
        out.append((i, p))
    return out


def _diag_and_dlog(allu, want, fc, periods):
    """S_bb = prod of the units' K(0), and its d/dlog theta for each wanted (unit, parameter)."""
    z = lambda dims: [np.zeros(1) for _ in dims]
    K0 = [float(kernel_class(kind).eval(unit_theta(kind, fc, periods, dims), z(dims))[0]) for kind, dims in allu]
    dK0 = [float(kernel_class(allu[i][0]).eval_dlog(unit_theta(allu[i][0], fc, periods, allu[i][1]), p,
                                                    z(allu[i][1]))[0]) for i, p in want]
    return float(np.prod(K0)), [dK0[u] * float(np.prod([K0[j] for j in range(len(allu)) if j != i]))
                                for u, (i, p) in enumerate(want)]


def _unit_index(units, dims):
    for i, (_, ud) in enumerate(units):
        if sorted(ud) == sorted(dims):
            return i
    raise ValueError(f"fc_grad: {dims} is not a kernel unit (a single sinc dim or a whole radial group)")


def _radial_factor(Xi, Yi, kinds, fc_vec, h_vec):
    """Product of radial-group factors for integer lattice coords (1 if no radial dims)."""
    G = 1.0
    for name, dims in radial_groups(kinds):
        r2 = sum((Xi[:, k, None] - Yi[None, :, k]) ** 2 for k in dims)
        c = _radial_scale(dims, fc_vec, h_vec)        # exact (no table): K at s = c sqrt(r^2)
        G = G * np.prod(fc_vec[dims]) * RADIAL_KERNELS[name](c * np.sqrt(r2.astype(float)), len(dims))
    return G



# ---------------------------------------------------------------------------
# Fixed sign patterns for g = sqrt(f)
# ---------------------------------------------------------------------------
# BLML's g need not be positive: f = g^2 only fixes |g|. For a fixed sign pattern s (one +-1 per
# sample or bin), writing c = s * t with t > 0 turns the objective into
#     0.5 t'(D S D) t - n sum_b w_b log t_b,   D = diag(s),
# the same convex problem with kernel s_i s_j S_ij, so every solver runs unchanged on D S D and
# returns c = s * t. Needed when the true g changes sign (e.g. a real far-field amplitude h, whose
# zeros |h| would turn into corners that no band-limited g can follow).

def resolve_signs(signs, Zc, inverse=None, n_samples=None):
    """Return a +-1 float array, one per centre in Zc, or None.

    signs: None; a callable s(Z) -> values whose sign is used, evaluated at the centres Zc
    (same coordinates as the data passed to fit); an array with one entry per centre; or, when
    `inverse` (sample -> centre index) is given, an array with one entry per sample, reduced to
    one sign per centre by majority (a warning reports centres whose samples disagree).
    """
    if signs is None:
        return None
    B = len(Zc)
    if callable(signs):
        v = np.asarray(signs(Zc), dtype=float).reshape(-1)
        if len(v) != B:
            raise ValueError("signs(Z) must return one value per centre")
    else:
        v = np.asarray(signs, dtype=float).reshape(-1)
        if inverse is not None and n_samples is not None and len(v) == n_samples and len(v) != B:
            sv = np.where(v < 0, -1.0, 1.0)
            tot = np.bincount(inverse, weights=sv, minlength=B)
            cnt = np.bincount(inverse, minlength=B)
            mixed = int((np.abs(tot) < cnt).sum())
            if mixed:
                print(f"warning: signs disagree within {mixed} of {B} bins; using the majority sign", flush=True)
            v = tot
        elif len(v) != B:
            raise ValueError(f"signs has length {len(v)}; expected {B} (centres)"
                             + (f" or {n_samples} (samples)" if n_samples is not None else ""))
    return np.where(v < 0, -1.0, 1.0)

def _band_grid(shape, h_vec, fc_arr, kinds):
    """FFT frequencies of a lattice box (spacing h_vec), scaled so the band is the unit box (sinc dims) or
    unit ball (each radial group), and the band indicator. Bands: |nu_j| <= fc_j/2, ||nu_G|| <= fc/2."""
    U = np.stack(np.meshgrid(*[np.fft.fftfreq(m, hj) / (0.5 * f) for m, hj, f in zip(shape, h_vec, fc_arr)],
                             indexing='ij'), -1)
    band = np.ones(shape, bool); rad = set()
    for _, dims in radial_groups(kinds):
        band &= (U[..., dims] ** 2).sum(-1) <= 1; rad |= set(dims)
    for j in range(len(shape)):
        if j not in rad:
            band &= np.abs(U[..., j]) <= 1
    return U, band


def _even_kicks(U):
    """Even low-order phase directions on the unit band: 2r^2-1, u_0^2-u_j^2, 2u_iu_j, 6r^4-6r^2+1 (on a 2-D
    ball: Zernike defocus, the two astigmatisms, spherical)."""
    d = U.shape[-1]; r2 = (U ** 2).sum(-1)
    return ([2 * r2 - 1] + [U[..., 0] ** 2 - U[..., j] ** 2 for j in range(1, d)]
            + [2 * U[..., i] * U[..., j] for i in range(d) for j in range(i + 1, d)] + [6 * r2 ** 2 - 6 * r2 + 1])


_KINDS = ('sinc', 'dirichlet', 'ball', 'slits', 'pball')
_STRUCT_COLS = {'kind', 'dep', 'h', '_unit'}           # structure: the model, never parameters
# Kernel parameters per kind: name -> (type, trainable by default). type: 'scale' (log steps, IQR-scaled by
# the selector), 'integer' (discrete: dirichlet harmonics move one at a time), 'shape' (unitless). A
# parameter is given as a bare value (default trainability) or as {'value': v, 'trainable': bool}; a
# trainable parameter's value, when given, is where the selector / streaming start; a fixed one is used
# as is. Structure keys (kind, dep) and the keys themselves (a column, or a tuple of columns: one multi-column
# unit) define the model and are never parameters.
KERNEL_PARAMS = {kind: kernel_class(kind).params for kind in _KINDS}     # as each kernel class declares them
LATTICE_PARAMS = ('fc',)        # parameters that set the bin width (h ~ n^-1/4 / fc): the ones a selector searches
_STRUCT_KEYS = ('kind', 'dep', 'h')


def _percol(m):
    """One mode as per-column views {j: column dict}, j = 0..d-1: an int key is one column; a tuple key one
    multi-column unit (a radial band: kind 'ball'), each of its columns seeing the unit's dict (its one fc) with
    the private '_unit' = the key. Mapped units (is_mapped) are mapped to lattice columns before (projected_modes)."""
    out = {}
    for key, c in m.items():
        kind = c.get('kind', 'sinc')
        if isinstance(key, tuple):
            if is_mapped(key, c):
                raise ValueError(f"columns {key}: mapped units are mapped by projected_modes (estimators do it)")
            if not is_radial(kind):
                raise ValueError(f"columns {key}: a tuple key is a multi-column unit (a radial kind such as 'ball', or "
                                 "an input map: map= / W= / k=)")
            for i, j in enumerate(key):
                cj = dict(c, _unit=key)
                per = param_value(c.get('period'))[0]
                if isinstance(per, (list, tuple, np.ndarray)):   # a periodic ball: one period per column
                    if len(per) != len(key):
                        raise ValueError(f"columns {key}: period needs one value per column, got {per}")
                    cj['period'] = float(per[i]) if not isinstance(c.get('period'), dict) else dict(c['period'], value=float(per[i]))
                out[j] = cj
        else:
            if 'group' in c:
                raise ValueError(f"column {key}: radial units are tuple keys, e.g. {{(0, 1): dict(kind='ball', fc=...)}}")
            if is_radial(kind):
                raise ValueError(f"column {key}: kind {kind!r} spans columns: give its unit as a tuple key")
            out[key] = c
    return dict(sorted(out.items()))


def _retuple(m, percol):
    """Per-column views (as from _percol) back under m's keys (a unit from its first column)."""
    return {key: ({a: v for a, v in percol[key[0]].items() if a != '_unit'} if isinstance(key, tuple) else percol[key])
            for key in m}


def param_value(v):
    """(value or None, trainable or None) of a parameter given as a bare value or {'value', 'trainable'}."""
    if isinstance(v, dict):
        if set(v) - {'value', 'trainable'}:
            raise ValueError(f"parameter {v}: keys must be 'value' and/or 'trainable'")
        return v.get('value'), v.get('trainable')
    return v, None


def parse_modes(modes, d, need_fc=True):
    """The model spec. modes: list with one entry per mode (rank K = len(modes)); each entry maps every
    column 0..d-1, or a tuple of columns (one multi-column unit: a round band, kind 'ball'), to {'fc': cut-off,
    'kind': 'sinc' | 'dirichlet' | 'ball' | 'slits' (default 'sinc'), 'period': period (dirichlet only), 'fill': the slits' fill (fixed, 0 < fill <= 1; on dirichlet optional: the periodic two-slit band), 'dep': True for
    a dependent column or unit (conditional fits; the same in every mode), 'h': the column's lattice step when the
    data are counts on a fixed lattice (bin centres h * integers, e.g. detector pixels: given for every column or
    none, the same in every mode; never re-binned)}. f = sum_k |g_k|^2 with g_k band-limited under mode k's kernel.
    Kernel parameters (fc, period; see KERNEL_PARAMS) are bare values or {'value': v, 'trainable': bool}:
    fc is trainable by default (a given value is the start), period fixed. kind, dep and the keys are
    structure, never parameters.
    Returns (specs, dep): specs, K dicts {fc (d,), kinds (d,), periods (d,), fc_trainable (d,)} in the
    internal form (radial kinds 'ball:<group>'); dep, the sorted dependent columns, or None.
    need_fc=False allows missing fc (NaN), for selector templates."""
    if isinstance(modes, dict) or not isinstance(modes, (list, tuple)) or not modes:
        raise TypeError("modes: a non-empty list of mode dicts {column: {...}}; wrap one mode as [mode]")
    specs, deps = [], []
    for k, m in enumerate(modes):
        m = _percol(m)
        if set(m) != set(range(d)):
            raise ValueError(f"mode {k}: columns {sorted(m)}, expected every column 0..{d - 1}")
        kinds, dep, vals, trs = [], [], {}, {}
        for j in range(d):
            c = m[j]
            for sk in _STRUCT_KEYS:
                if isinstance(c.get(sk), dict):
                    raise ValueError(f"mode {k} column {j}: {sk} is structure (it defines the model), not a "
                                     "parameter; it cannot be given as {'value', 'trainable'}")
            kind, g = c.get('kind', 'sinc'), c.get('_unit')
            if kind not in _KINDS:
                raise ValueError(f"mode {k} column {j}: kind {kind!r}, expected one of {_KINDS}")
            declared = KERNEL_PARAMS[kind]                 # the parameters its kernel class declares
            if set(c) - _STRUCT_COLS - set(declared):
                raise ValueError(f"mode {k} column {j}: unknown keys {sorted(set(c) - _STRUCT_COLS - set(declared))} "
                                 f"(kind {kind!r} declares {sorted(declared)})")
            for name, (ptype, default) in declared.items():
                v, t = param_value(c.get(name))
                t = default if t is None else bool(t)
                if v is None and kind == 'dirichlet' and name == 'fill':
                    continue                               # optional: without it the plain periodic band
                if v is None:                              # only a selector's template may leave a searched one out
                    if name not in LATTICE_PARAMS:
                        raise ValueError(f"mode {k} column {j}: {name} missing (kind {kind!r} needs it)")
                    if not t:
                        raise ValueError(f"mode {k} column {j}: a fixed (trainable=False) {name} needs a value")
                    if need_fc:
                        raise ValueError(f"mode {k} column {j}: {name} missing")
                if v is not None and ptype in ('scale', 'integer') and not float(v) > 0:
                    raise ValueError(f"mode {k} column {j}: {name} must be positive, got {v}")
                if kind == 'pball' and name == 'period' and t:
                    raise ValueError(f"mode {k} column {j}: a pball period is fixed (trainable=False)")
                if name == 'fill':                         # a shape: fixed, in (0, 1]; carried in the kind token
                    if t:
                        raise ValueError(f"mode {k} column {j}: {kind} fill is a fixed shape (trainable=False); "
                                         "select it by an outer search")
                    if not 0 < float(v) <= 1:
                        raise ValueError(f"mode {k} column {j}: {kind} fill must be in (0, 1], got {v}")
                vals.setdefault(name, [None] * d)[j] = v; trs.setdefault(name, [False] * d)[j] = t
            kinds.append(f"{kind}:{'-'.join(map(str, g))}" if g is not None else
                         f"{kind}:{float(param_value(c['fill'])[0])!r}" if kind in ('slits', 'dirichlet') and
                         param_value(c.get('fill'))[0] is not None else kind)
            dep.append(bool(c.get('dep', False)))
        fc = [np.nan if v is None else float(v) for v in vals.get('fc', [None] * d)]
        periods = list(vals.get('period', [None] * d))
        tr, ptr = list(trs.get('fc', [False] * d)), list(trs.get('period', [False] * d))
        for name, dims in radial_groups(kinds):        # one round band: one fc, so one trainability
            if len({tr[j] for j in dims}) > 1:
                raise ValueError(f"mode {k}: radial group over columns {dims} mixes trainable and fixed fc")
        specs.append(dict(fc=np.array(fc), kinds=kinds, periods=periods, fc_trainable=np.array(tr),
                          period_trainable=np.array(ptr), fold=lattice_fold(kinds, ptr))); deps.append(dep)
    if any(dp != deps[0] for dp in deps):
        raise ValueError("dep must mark the same columns in every mode")
    modes_lattice(modes, d)                           # h: every column or none, the same in every mode
    if need_fc:
        for sp in specs:
            check_radial(sp['kinds'], sp['fc'])
    dep = [j for j in range(d) if deps[0][j]]
    return specs, (dep or None)


def lattice_fold(kinds, period_trainable):
    """Per column: whether its lattice is folded onto its period -- a dirichlet column with a fixed period is (whole
    bins tile it); one with a trainable period is not (raw coordinate: its row over raw offsets is P-periodic by
    itself, so a period step only rebuilds that row). The same rule in every fit."""
    return [not (_split_kind(k)[0] == 'dirichlet' and bool(t)) for k, t in zip(kinds, period_trainable)]


def lattice_target(h_t, kinds_list):
    """Target bin widths (d,) for a lattice shared by modes with these kinds (one list per mode): a radial unit has
    one step over its columns (fc h equal across the group), so every radial group of any mode takes its columns'
    finest width."""
    h = np.array(h_t, float)
    for kinds in kinds_list:
        for _, dims in radial_groups(kinds):
            h[dims] = h[dims].min()
    return h


def lattice_fold_specs(specs):
    """The shared lattice's fold per column for modes that share it (parse_modes specs): folded only where every
    mode folds -- a dirichlet column with one fixed period in every mode; a column some mode reads raw (another
    period, a trainable one, or another kind) is raw for all (each kernel row is then over raw offsets)."""
    d = len(specs[0]['kinds'])
    out = []
    for j in range(d):
        dirs = [is_periodic_kind(sp['kinds'][j]) for sp in specs]
        if not any(dirs):
            out.append(True)
            continue
        out.append(all(dirs) and all(sp['fold'][j] for sp in specs)
                   and len({float(sp['periods'][j]) for sp in specs}) == 1
                   and len({_split_kind(sp['kinds'][j])[0] for sp in specs}) == 1)
    return out


def modes_lattice(modes, d):
    """The fixed lattice steps (d,) the modes declare (column key 'h'), or None (the fit picks and re-bins its lattice)."""
    hs = []
    for k, m in enumerate(modes):
        m = _percol(m)
        hk = [m[j].get('h') for j in range(d)]
        if any(v is None for v in hk) and any(v is not None for v in hk):
            raise ValueError(f"mode {k}: h (a fixed lattice) must be given for every column or none")
        hs.append(None if hk[0] is None else np.array(hk, float))
    if any((a is None) != (hs[0] is None) or (a is not None and not np.allclose(a, hs[0])) for a in hs):
        raise ValueError("h (the lattice) must be the same in every mode")
    if hs[0] is not None and not np.all(hs[0] > 0):
        raise ValueError("h must be positive")
    return hs[0]


def with_lattice(modes, h):
    """modes on the fixed lattice h (scalar or one per column): counts on bin centres h * integers."""
    out = []
    for m in modes:
        mm = {}
        for key, c in m.items():
            cols = key if isinstance(key, tuple) else (key,)
            hv = np.broadcast_to(np.asarray(h, float), (max(max(cols) + 1, np.size(h)),))
            if isinstance(key, tuple) and len({float(hv[j]) for j in cols}) > 1:
                raise ValueError(f"columns {key}: one unit, one lattice step")
            mm[key] = dict(c, h=float(hv[cols[0]]))
        out.append(mm)
    return out


def marginal_counts(Z, w, cols):
    """Counts on a fixed lattice summed over the columns not in cols: (unique bin centres in cols, their counts)."""
    Z = np.atleast_2d(np.asarray(Z, float))[:, list(cols)]
    u, inv = np.unique(np.ascontiguousarray(Z), axis=0, return_inverse=True)
    return u, np.bincount(np.ravel(inv), weights=np.asarray(w, float), minlength=len(u))


def strip_lattice(modes):
    return [{key: {a: v for a, v in c.items() if a != 'h'} for key, c in m.items()} for m in modes]


def keep_lattice(init, h_target, rebin_tol):
    """Whether a refit may stay on init's lattice: every dim's target width h = bin_scale n^{-1/4} / fc
    is within a factor (1 + rebin_tol)^{1/4} of the target the lattice was built for (rebin_tol = 1:
    n may double, or fc move by ~19%, before re-binning). Small fc steps keep the lattice."""
    h0 = getattr(init, 'h_target_', None)
    if h0 is None or np.shape(h0) != np.shape(h_target):
        return False
    return bool(np.all(np.abs(np.log(np.asarray(h_target) / h0)) <= 0.25 * np.log1p(rebin_tol) + 1e-12))


def warm_start_e(init, Z, Zi, w, same_lattice):
    """Starting coefficients (B,) or (B, K) from a previous fit `init` (warm start), or None.
    same_lattice: bins of the two fits share the lattice, so bins present in both keep init's
    coefficients and new bins start at 0 (the solvers then begin them from their neighbours).
    Otherwise e_b = w_b g(z_b) / |g(z_b)|^2 from init's g (exact at its optimum); bins where init's g
    is <= 1e-3 of its largest value (or, real rank 1, negative) start at 0 too: w/g would blow up there."""
    if init is None or getattr(init, 'e_hat_', None) is None:
        return None
    E = np.asarray(init.e_hat_)
    if same_lattice and getattr(init, '_Zi', None) is not None and init._Zi.shape[1] == Zi.shape[1]:
        key = lambda A: np.ascontiguousarray(A, dtype=np.int64).view(np.dtype((np.void, 8 * A.shape[1]))).ravel()
        common, ia, ib = np.intersect1d(key(Zi), key(init._Zi), return_indices=True)
        e0 = np.zeros((len(Zi),) + E.shape[1:], dtype=E.dtype)
        e0[ia] = E[ib]
        return e0
    if not hasattr(init, 'sqrt_density'):
        return None
    G = np.asarray(init.sqrt_density(Z)); G = G[:, None] if G.ndim == 1 else G
    F = np.maximum((np.abs(G) ** 2).sum(1), 1e-300)
    e0 = np.asarray(w, float)[:, None] * G / F[:, None]
    bad = np.sqrt(F) <= 1e-3 * np.sqrt(F.max())
    if G.shape[1] == 1 and not np.iscomplexobj(G) and getattr(init, 'signs_', None) is None:
        bad |= G[:, 0] <= 0     # unsigned real g is positive at the optimum
    e0[bad] = 0
    return e0[:, 0] if e0.shape[1] == 1 else e0


def warm_start_Se(init, Zi, fc_arr, kinds, periods, h_vec):
    """The previous fit's exact S e (init.Se_) moved onto the bins Zi, when the kernel and lattice are
    unchanged (same fc, kinds, periods and h, one shared kernel, no fixed signs): (Se0, new, Zi_rem, e_rem)
    with Se0 init's S e at the bins both share (0 at new bins), new the mask of bins init lacks, and the
    removed bins' lattice coordinates and coefficients; the solver then needs kernel columns only for
    new and removed bins instead of a full S e pass. None otherwise."""
    Se = getattr(init, 'Se_', None)
    if (Se is None or getattr(init, 'modes_', None) is not None or getattr(init, 'signs_', None) is not None
            or getattr(init, '_Zi', None) is None or init._Zi.shape[1] != Zi.shape[1]
            or not np.allclose(np.asarray(init._fc_arr, float), np.asarray(fc_arr, float))
            or not np.allclose(init._h_vec, h_vec) or list(init.kinds_ or []) != list(kinds or [])
            or list(init.periods_ or []) != list(periods or [])):
        return None
    key = lambda A: np.ascontiguousarray(A, dtype=np.int64).view(np.dtype((np.void, 8 * A.shape[1]))).ravel()
    _, ia, ib = np.intersect1d(key(Zi), key(init._Zi), return_indices=True)
    Se = np.asarray(Se); E = np.asarray(init.e_hat_)
    Se0 = np.zeros((len(Zi),) + Se.shape[1:], dtype=np.result_type(Se.dtype, np.float64)); Se0[ia] = Se[ib]
    new = np.ones(len(Zi), bool); new[ia] = False
    rem = np.ones(len(init._Zi), bool); rem[ib] = False
    return Se0, new, init._Zi[rem], E[rem]


def _shared_kernel(specs):
    key = lambda sp: (tuple(sp['fc']), tuple(sp['kinds']), tuple(sp['periods']))
    return all(key(sp) == key(specs[0]) for sp in specs)


def make_modes(fc, kinds=None, periods=None, rank=1, dep=None, d=None):
    """Build a modes list from per-dim settings. fc: scalar, (d,), or (K, d) with one row per mode;
    kinds: per dim 'sinc' | 'dirichlet' | 'dirichlet:<fill>' | 'slits:<fill>' | 'ball' | 'ball:<tag>' (columns sharing a tag, or all untagged ball
    columns, form one unit: a tuple key in the modes); periods: per dim (used on dirichlet
    dims); rank: K when fc has a single row; dep: dependent column(s); d: needed only when fc is a
    scalar and kinds is None. NaN fc leaves fc out (selector templates)."""
    F = np.asarray(fc, dtype=float)
    if d is None:
        d = len(kinds) if kinds is not None else (F.shape[-1] if F.ndim else None)
    if d is None:
        raise ValueError("make_modes: give kinds or d with a scalar fc")
    F = F if F.ndim == 2 else np.broadcast_to(F, (d,))[None]
    F = np.broadcast_to(F, (int(rank) if len(F) == 1 else len(F), d))
    kinds = list(kinds) if kinds is not None else ['sinc'] * d
    deps = set() if dep is None else {int(j) for j in np.atleast_1d(dep)}
    units = {tuple(dims): dims[0] for _, dims in radial_groups(kinds)}
    def col(k, j):
        kind, _, tag = str(kinds[j]).partition(':')
        c = dict(kind=kind)
        if kind in ('slits', 'dirichlet') and tag:         # 'slits:<fill>', 'dirichlet:<fill>'
            c['fill'] = float(tag)
        if np.isfinite(F[k, j]):
            c['fc'] = float(F[k, j])
        if kind == 'dirichlet':
            c['period'] = periods[j]
        if kind == 'pball':                                # the unit's periods, one per column of its group
            key = next(u for u in units if j in u)
            c['period'] = [periods[i] for i in key]
        if j in deps:
            c['dep'] = True
        return c
    def mode(k):
        out = {}
        for j in range(d):
            key = next((u for u in units if j in u), j)
            if key == j or j == key[0]:
                out[key] = col(k, j)
        return out
    return [mode(k) for k in range(len(F))]


def _lattice_form(modes):
    """Modes with mapped units in their lattice form (projected_modes; the helpers below read lattice columns)."""
    proj, lm = projected_modes(modes)
    return lm


def _on_lattice(fn):
    """A modes -> modes helper applied on the lattice form and mapped back (mapped units keep their map)."""
    import functools

    @functools.wraps(fn)
    def f(modes, *a, **k):
        proj, lm = projected_modes(modes)
        return fn(modes, *a, **k) if proj is None else unprojected_modes(proj, fn(lm, *a, **k), modes)
    return f


def modes_fc(modes):
    """(K, d) cut-off values of a modes list (NaN where fc is missing; bare or {'value', 'trainable'} form); mapped
    units: their lattice columns (projected_modes)."""
    modes = _lattice_form(modes)
    val = lambda c: param_value(c.get('fc'))[0]
    return np.array([[np.nan if val(m[j]) is None else val(m[j]) for j in range(len(m))] for m in map(_percol, modes)], dtype=float)


def modes_trainable(modes):
    """(K, d) bool: whether each column's fc is trainable (default per kind, see KERNEL_PARAMS)."""
    modes = _lattice_form(modes)

    def tr(c):
        t = param_value(c.get('fc'))[1]
        return KERNEL_PARAMS[c.get('kind', 'sinc')]['fc'][1] if t is None else bool(t)
    return np.array([[tr(m[j]) for j in range(len(m))] for m in map(_percol, modes)], dtype=bool)


def modes_period(modes):
    """(K, d) period values (NaN for non-dirichlet columns)."""
    modes = _lattice_form(modes)
    val = lambda c: param_value(c.get('period'))[0]
    return np.array([[np.nan if val(m[j]) is None else val(m[j]) for j in range(len(m))] for m in map(_percol, modes)], dtype=float)


def modes_period_trainable(modes):
    """(K, d) bool: whether each dirichlet column's period is trainable (default fixed)."""
    modes = _lattice_form(modes)

    def tr(c):
        if c.get('kind', 'sinc') != 'dirichlet':
            return False
        t = param_value(c.get('period'))[1]
        return KERNEL_PARAMS['dirichlet']['period'][1] if t is None else bool(t)
    return np.array([[tr(m[j]) for j in range(len(m))] for m in map(_percol, modes)], dtype=bool)


@_on_lattice
def with_period(modes, P):
    """Copy of modes with the dirichlet columns' period values replaced ((d,) or (K, d), NaN elsewhere);
    trainability kept."""
    Pm = np.broadcast_to(np.asarray(P, dtype=float), modes_period(modes).shape)
    def put(c, v):
        if c.get('kind', 'sinc') != 'dirichlet':
            return dict(c)
        t = param_value(c.get('period'))[1]
        return dict(c, period=float(v) if t is None else dict(value=float(v), trainable=t))
    return [_retuple(m, {j: put(c, Pm[k, j]) for j, c in _percol(m).items()}) for k, m in enumerate(modes)]


@_on_lattice
def with_fc(modes, fc):
    """Copy of modes with fc values replaced (scalar, (d,) for every mode, or (K, d)); trainability kept."""
    F = np.broadcast_to(np.asarray(fc, dtype=float), modes_fc(modes).shape)
    def put(c, v):
        t = param_value(c.get('fc'))[1]
        return dict(c, fc=float(v) if t is None else dict(value=float(v), trainable=t))
    return [_retuple(m, {j: put(c, F[k, j]) for j, c in _percol(m).items()}) for k, m in enumerate(modes)]


def marginal_modes(modes, keep):
    """Modes restricted to the columns in keep (renumbered 0..), dep flags dropped; a unit is kept whole or not."""
    pos = {j: i for i, j in enumerate(keep)}
    def one(m):
        out = {}
        for key, c in m.items():
            cols = key if isinstance(key, tuple) else (key,)
            inside = [j in pos for j in cols]
            if all(inside):
                out[tuple(pos[j] for j in cols) if isinstance(key, tuple) else pos[key]] = {a: v for a, v in c.items() if a != 'dep'}
            elif any(inside):
                raise ValueError(f"columns {key}: a unit straddles kept and dropped columns")
        return dict(sorted(out.items(), key=lambda t: t[0][0] if isinstance(t[0], tuple) else t[0]))
    return [one(m) for m in modes]


def strip_dep(modes):
    return [{j: {a: v for a, v in c.items() if a != 'dep'} for j, c in m.items()} for m in modes]


def complex_starts(Zi, w, h_vec, fc_arr, kinds, n_hio=3, n_sb=2, hio_pool=30, sb_delta=1.0,
                   n_hio_iter=400, n_er_iter=100, beta=0.9, seed=0, max_cells=1 << 22):
    """Starting coefficients for a complex BLML fit (f = |g|^2, g band-limited). The real positive root is a
    saddle / wrong basin whenever the true phase is even, so candidates come from the FFT lattice of the bin
    box (zero-padded 2x per dim; O(M log M) per iteration, O(M) memory, M ~ 2^d * box):
    (a) HIO from random band phases (hio_pool runs, n_hio_iter HIO + n_er_iter error-reduction iterations);
    (b) symmetry breaking: the positive root times exp(+-i sb_delta p) for each even direction p
        (_even_kicks), then n_er_iter error-reduction iterations.
    Data constraint: |g| = sqrt(w) * scale on the box (0 on empty lattice points); band constraint as in
    _band_grid. Candidates are scored by the Poisson log-likelihood of |g|^2 normalised on the box.
    The search grid is the bin lattice coarsened by an integer factor q_j to spacing <= 0.5/fc_j (2x
    Nyquist; q = 1 for pixel data), counts summed per coarse cell; g at the original bins is then the
    exact band-limited sum over the band's frequencies.
    Returns [(family, score, e0), ...] with the top n_hio HIO and top n_sb symmetry-breaking starts,
    e0_b = w_b / conj(g_b) (the warm-start convention of _solve_modes_cd), or None if the grid has more than
    max_cells cells."""
    Zi0 = np.asarray(Zi); w0 = np.asarray(w, dtype=np.float64); h0 = np.asarray(h_vec, float)
    q = np.maximum(1, np.floor(0.5 / (np.asarray(fc_arr, float) * h0))).astype(np.int64)
    Zc = np.floor_divide(Zi0 - Zi0.min(0), q)
    if np.all(q == 1):
        Zi, w = Zc, w0
    else:
        Zi, inv = np.unique(Zc, axis=0, return_inverse=True); w = np.bincount(np.ravel(inv), weights=w0)
    h_vec = h0 * q
    lo = Zi.min(0); L = Zi.max(0) - lo + 1
    shape = tuple(1 << int(np.ceil(np.log2(2 * int(l)))) for l in L)
    if np.prod(shape, dtype=float) > max_cells:
        return None
    idx = tuple((Zi - lo).T)
    U, band = _band_grid(shape, h_vec, fc_arr, kinds)
    box = tuple(slice(0, int(l)) for l in L)
    A = np.zeros(shape); A[idx] = np.sqrt(w); Ab = A[box]; Ab2 = Ab.ravel() @ Ab.ravel()

    def project(g):                                 # data: keep the phase, set |g| = scale * sqrt(w) on the box
        gb = g[box]; s = np.sqrt((np.abs(gb) ** 2).sum() / Ab2)
        g = g.copy(); g[box] = Ab * s * np.exp(1j * np.angle(gb)); return g

    def run(P, n_h, n_e):
        for it in range(n_h + n_e):
            Pn = np.fft.fftn(project(np.fft.ifftn(P)))
            P = np.where(band, Pn, (P - beta * Pn) if it < n_h else 0)
        return P * band

    def score(P):
        f = np.abs(np.fft.ifftn(P)) ** 2
        return float(w @ np.log(np.maximum(f[idx] / f[box].sum(), 1e-300)))

    def top(family, gen, k):                        # keep only the k best grids (O(k M) memory)
        best = []
        for P in gen:
            best.append((score(P), P)); best.sort(key=lambda t: -t[0]); best = best[:k]
        out = []
        for sc, P in best:
            if np.all(q == 1):
                g = np.fft.ifftn(P)[idx]
            else:                                   # exact band-limited values at the original bins
                nz = np.nonzero(band); nu = np.stack([np.fft.fftfreq(m, hj)[k] for m, hj, k in zip(shape, h_vec, nz)], 1)
                c = P[nz] / P.size; x = (Zi0 - Zi0.min(0)) * h0; g = np.empty(len(x), complex)
                for s0 in range(0, len(x), 4096):
                    g[s0:s0 + 4096] = np.exp(2j * np.pi * x[s0:s0 + 4096] @ nu.T) @ c
            out.append((family, sc, w0 / np.conj(np.where(np.abs(g) == 0, 1e-12, g))))
        return out

    rng = np.random.default_rng(seed); starts = []
    if n_hio > 0:
        starts += top('hio', (run(band * np.exp(2j * np.pi * rng.random(shape)), n_hio_iter, n_er_iter)
                              for _ in range(hio_pool)), n_hio)
    if n_sb > 0:
        P0 = np.fft.fftn(A) * band
        starts += top('symbreak', (run(P0 * np.exp(1j * sg * sb_delta * p), 0, n_er_iter)
                                   for p in _even_kicks(U) for sg in (1, -1)), n_sb)
    return starts


def dirichlet_kernel(dist, fc, P, fill=None):
    """Dirichlet kernel at circular distance ``dist`` in [0, P/2]; with fill < 1 the periodic two-slit band, the
    inner harmonics' D_{M_i} subtracted (dirichlet_inner_M)."""
    M, scale = dirichlet_modes_scale(fc, P)
    Mi = dirichlet_inner_M(fc, P, fill)
    dist = np.asarray(dist, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        val = (np.sin(np.pi * M * dist / P) - np.sin(np.pi * Mi * dist / P)) / np.sin(np.pi * dist / P)
    return scale * np.where(dist < 1e-10, M - Mi, val)


def dirichlet_table(fc, P, n_bins, fill=None):
    """Kernel values at lattice distances j*P/n_bins, j = 0..n_bins-1.

    Index with the min circular distance min(j, n_bins - j).
    """
    return dirichlet_kernel(np.arange(n_bins) * (P / n_bins), fc, P, fill)


def solve_rho_positive_orthant(matvec, n, max_iter=2000, tol=1e-10,
                               verbose=False, tag="rho", dense_S=None):
    """Solve rho(c) = (1/n) S c - 1/c = 0 over the positive orthant.

    ``matvec`` is the Gram operator c -> S c (a callable, so the Gram
    matrix itself never needs to be formed).  ``n`` is the system size.

    Strategy, mirroring the paper's BLML-Trivial algorithm:
      1. Fixed-point iteration ``c <- n / S c`` while it stays in the
         positive orthant (a contraction for positive kernels; one
         Gram matvec per iteration).
      2. If the iterate leaves the orthant or stalls (typical for
         sign-indefinite kernels such as sinc, or ill-conditioned
         Grams), fall back to a constrained Newton-type solve of
         rho(c) = 0 over c > 0 via nonlinear least squares
         (minimising ||rho||^2, whose global minimum is the root):
         with the analytic dense Jacobian (1/n) S + diag(1/c^2) when
         ``dense_S`` is supplied, otherwise matrix-free with a
         LinearOperator Jacobian built from ``matvec``.

    Returns
    -------
    (c, max_rho, n_iter, solver_name, converged)
    """
    def rho(c):
        return matvec(c) / n - 1.0 / np.maximum(c, _TINY)

    c = np.ones(n)
    converged = False
    solver = "fixed-point"
    it_done = 0
    Sc = matvec(c)
    for it in range(max_iter):
        if not np.all(np.isfinite(Sc)) or np.any(Sc <= 0):
            if verbose:
                print(f"  [{tag}] fixed point left the positive orthant "
                      f"at iter {it + 1}; switching to fallback.")
            break
        c_new = n / Sc
        Sc_new = matvec(c_new)
        res = float(np.max(np.abs(Sc_new / n - 1.0 / c_new)))
        it_done = it + 1
        if verbose and (it % 100 == 0 or res < tol):
            print(f"  [{tag}] iter {it_done:4d}  max|rho| = {res:.3e}")
        if res < tol:
            c, Sc = c_new, Sc_new
            converged = True
            break
        c, Sc = c_new, Sc_new

    if not converged:
        from scipy.optimize import least_squares
        from scipy.sparse.linalg import LinearOperator

        c0 = (np.clip(c, _TINY, None)
              if np.all(np.isfinite(c)) and np.max(np.abs(c)) < 1e8
              else np.ones(n))

        def F(cc):
            return matvec(cc) / n - 1.0 / np.maximum(cc, _TINY)

        if dense_S is not None:
            def J(cc):  # analytic dense Jacobian
                return dense_S / n + np.diag(1.0 / np.maximum(cc, _TINY) ** 2)
            jac = J
            solver = "newton-fallback"
        else:
            def jac(cc):  # matrix-free Jacobian operator (symmetric)
                d = 1.0 / np.maximum(cc, _TINY) ** 2

                def mv(v):
                    v1 = np.ravel(v)
                    return (matvec(v1) / n + d * v1).reshape(np.shape(v))

                return LinearOperator((n, n), matvec=mv, rmatvec=mv)
            solver = "newton-krylov-fallback"

        sol = least_squares(
            F, c0, jac=jac, bounds=(_TINY, np.inf), method="trf",
            # x_scale='jac' incompatible with LinearOperator jac (scipy>=1.14);
            # use 1.0 for the matrix-free path
            x_scale="jac" if solver.startswith("newton-fallback") else 1.0,
            xtol=1e-12, ftol=1e-12, gtol=1e-12,
            max_nfev=max(2000, 20 * n), verbose=2 if verbose else 0,
        )
        c = sol.x
        solver = f"{solver} ({sol.nfev} evals)"
        res = float(np.max(np.abs(rho(c))))
        converged = res < tol
        if verbose:
            print(f"  [{tag}] {solver}: max|rho| = {res:.3e}")
    else:
        res = float(np.max(np.abs(rho(c))))
    return c, res, it_done, solver, converged


# ---------------------------------------------------------------------------
# Kernel factories
# ---------------------------------------------------------------------------

try:
    import numba                                       # the row kernels themselves: blml_local (rows, cols)
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def lattice_kernel(fc, h, kinds, periods, d, fold=None):
    """The product kernel at fc on the lattice h (blml_kernels.BinnedGram): the solvers' binned Gram, quantizer and S_bb.
    fold: per column (lattice_fold; default all folded)."""
    from blml_kernels import ProductKernel
    kinds = list(kinds) if kinds is not None else ['sinc'] * d
    periods = list(periods) if periods is not None else [None] * d
    return ProductKernel(kinds, periods, fold).binned(np.broadcast_to(np.asarray(fc, float), (d,)), h)


def _local_rows(*a, **k):
    """blml_local.rows (imported lazily: blml_local imports this module)."""
    from blml_local import rows
    return rows(*a, **k)


def _local_cols(*a, **k):
    from blml_local import cols
    return cols(*a, **k)


def rank_coords(Z, tab):
    from blml_kernels import rank_coords as rc
    return rc(Z, tab)


def default_sinc_kernel(fc):
    """Band-limited sinc kernel (the BLML default), generalised to R^d.

    K(x, y) = prod_k sin(pi * fc_k * (x_k - y_k)) / (pi * (x_k - y_k)),
    with the continuous limit fc_k when x_k == y_k.

    Parameters
    ----------
    fc : float or array-like of shape (d,)
        Cut-off frequency (bandwidth parameter of the smoothness
        constraint). May be a scalar or one value per dimension.

    Returns
    -------
    kernel : callable
        kernel(X, Y) -> Gram matrix of shape (len(X), len(Y)).
    """
    def kernel(X, Y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        d = X.shape[1]
        if Y.shape[1] != d:
            raise ValueError("X and Y must have the same number of dimensions")
        fc_vec = np.broadcast_to(np.asarray(fc, dtype=float), (d,)).copy()
        G = np.ones((X.shape[0], Y.shape[0]))
        for k in range(d):
            D = X[:, k, None] - Y[None, :, k]
            T = np.ones_like(D)
            nz = D != 0
            T[nz] = np.sin(np.pi * fc_vec[k] * D[nz]) / (np.pi * D[nz])
            T[~nz] = fc_vec[k]          # limit of sin(a t)/(pi t) as t -> 0
            G *= T
        return G

    kernel.__name__ = "sinc_kernel"
    # Metadata consumed by blml_inducing.py: for the sinc kernel the
    # overlap matrix Q_{ml} = int K(x,z_m) K(x,z_l) dx equals the Gram
    # matrix kernel(Z, Z) by the reproducing property of band-limited
    # functions.
    kernel._overlap = "gram"
    return kernel



if _HAS_NUMBA:
    @numba.njit(cache=True)
    def _psd_gs_block(I, Sblk, e_re, e_im, Se_re, Se_im, w, n, sbb):
        """Sequential exact updates of rank-K (complex) BLML over the bins I (Sblk = S[:, I]).
        e, Se have shape (B, K). For bin b: R = (S e)_b - sbb e_b, e_b = t R/||R||,
        sbb t^2 + ||R|| t - n w_b = 0."""
        B, K = Se_re.shape
        r_re = np.empty(K)
        r_im = np.empty(K)
        for jj in range(I.shape[0]):
            b = I[jj]
            nr2 = 0.0
            for k in range(K):
                r_re[k] = Se_re[b, k] - sbb * e_re[b, k]
                r_im[k] = Se_im[b, k] - sbb * e_im[b, k]
                nr2 += r_re[k] * r_re[k] + r_im[k] * r_im[k]
            ar = np.sqrt(nr2)
            cc = n * w[b]
            t = 2.0 * cc / (ar + np.sqrt(ar * ar + 4.0 * sbb * cc))
            if ar > 0.0:
                for k in range(K):
                    r_re[k] = t * r_re[k] / ar
                    r_im[k] = t * r_im[k] / ar
            else:
                a0 = 0.0
                for k in range(K):
                    a0 += e_re[b, k] * e_re[b, k] + e_im[b, k] * e_im[b, k]
                a0 = np.sqrt(a0)
                for k in range(K):
                    if a0 > 0.0:
                        r_re[k] = t * e_re[b, k] / a0
                        r_im[k] = t * e_im[b, k] / a0
                    else:
                        r_re[k] = t if k == 0 else 0.0
                        r_im[k] = 0.0
            for k in range(K):
                dr = r_re[k] - e_re[b, k]
                di = r_im[k] - e_im[b, k]
                e_re[b, k] = r_re[k]
                e_im[b, k] = r_im[k]
                if dr != 0.0 or di != 0.0:
                    for q in range(B):
                        Se_re[q, k] += Sblk[q, jj] * dr
                        Se_im[q, k] += Sblk[q, jj] * di


if _HAS_NUMBA:
    @numba.njit(cache=True)
    def _modes_gs_block(I, Sblks, sbbs, e_re, e_im, Se_re, Se_im, w, n):
        """Sequential exact updates when mode k has its own Gram S_k (Sblks[k] = S_k[:, I]).
        For bin b: R_k = (S_k e_k)_b - s_k e_kb; e_kb = R_k/(lam - s_k), with lam the root of
        lam * sum_k |R_k|^2/(lam - s_k)^2 = n w_b (decreasing in lam > max s_k; bisection)."""
        K = Sblks.shape[0]
        B = Se_re.shape[0]
        r_re = np.empty(K)
        r_im = np.empty(K)
        for jj in range(I.shape[0]):
            b = I[jj]
            nr2 = 0.0
            smax = 0.0
            for k in range(K):
                r_re[k] = Se_re[b, k] - sbbs[k] * e_re[b, k]
                r_im[k] = Se_im[b, k] - sbbs[k] * e_im[b, k]
                nr2 += r_re[k] * r_re[k] + r_im[k] * r_im[k]
                if sbbs[k] > smax:
                    smax = sbbs[k]
            cc = n * w[b]
            if nr2 <= 0.0:
                continue
            lo = 0.0
            hi = 1.0
            # phi(smax + x) = (smax + x) sum |R_k|^2 / (smax - s_k + x)^2, decreasing in x > 0
            while True:
                lam = smax + hi
                ph = 0.0
                for k in range(K):
                    dd = lam - sbbs[k]
                    ph += (r_re[k] * r_re[k] + r_im[k] * r_im[k]) / (dd * dd)
                if lam * ph < cc:
                    break
                hi *= 2.0
            for it in range(80):
                mid = 0.5 * (lo + hi)
                lam = smax + mid
                ph = 0.0
                for k in range(K):
                    dd = lam - sbbs[k]
                    ph += (r_re[k] * r_re[k] + r_im[k] * r_im[k]) / (dd * dd)
                if lam * ph > cc:
                    lo = mid
                else:
                    hi = mid
            lam = smax + 0.5 * (lo + hi)
            for k in range(K):
                nr = r_re[k] / (lam - sbbs[k])
                ni = r_im[k] / (lam - sbbs[k])
                dr = nr - e_re[b, k]
                di = ni - e_im[b, k]
                e_re[b, k] = nr
                e_im[b, k] = ni
                if dr != 0.0 or di != 0.0:
                    for q in range(B):
                        Se_re[q, k] += Sblks[k, q, jj] * dr
                        Se_im[q, k] += Sblks[k, q, jj] * di



def default_gaussian_kernel(h):
    """Isotropic Gaussian (RBF) kernel, provided for comparison.

    K(x, y) = exp(-||x - y||^2 / (2 h^2)).

    Note: a Gaussian kernel is *not* band-limited, so using it steps
    outside the assumptions of BLML theory (the estimate is still
    non-negative, but consistency guarantees no longer apply).

    Parameters
    ----------
    h : float
        Kernel bandwidth (standard deviation).

    Returns
    -------
    kernel : callable
        kernel(X, Y) -> Gram matrix of shape (len(X), len(Y)).
    """
    h = float(h)
    if h <= 0:
        raise ValueError("bandwidth h must be positive")

    def kernel(X, Y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        D2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=-1)
        return np.exp(-D2 / (2.0 * h * h))

    kernel.__name__ = "gaussian_kernel"
    # Metadata consumed by blml_inducing.py: closed-form Gaussian overlap
    # Q_{ml} = (pi h^2)^{d/2} exp(-||z_m - z_l||^2 / (4 h^2)).
    kernel._overlap = ("gaussian", h)
    return kernel


def default_ball_kernel(fc):
    """Isotropic band-limited kernel: reproducing kernel of the Paley-Wiener
    space of functions band-limited to a Euclidean *ball* in frequency.

    The frequency response is the indicator of the d-ball of radius fc/2
    (0 or 1 - the condition that makes 1_Omega^2 = 1_Omega, i.e. the
    reproducing property int K(x-z) K(z-y) dz = K(x-y), hold).  Its
    inverse Fourier transform has the closed form

        K(x, y) = (fc/2)^{d/2} * r^{-d/2} * J_{d/2}(pi * fc * r),
        r = ||x - y||,

    with K(x, x) = pi^{d/2} (fc/2)^d / Gamma(d/2 + 1) (the ball volume).
    In 1-D this reduces exactly to the paper's sinc kernel
    sin(pi*fc*(x-y)) / (pi*(x-y)).

    Why this shape in high dimensions: the entire BLML derivation
    (constraint system, convex Phi, convergence rate) is shape-agnostic,
    so we are free to pick Omega.  The ball gives one Bessel evaluation
    per pair instead of d sinc evaluations, is isotropic (no preferred
    axes, unlike the cuboid/product-sinc), and decays uniformly as
    r^{-(d+1)/2} in every direction (product-sinc decays fast on the
    diagonal but only as 1/|x_k| along axes).

    Parameters
    ----------
    fc : float
        Cut-off frequency (diameter of the frequency ball); matches the
        paper's sinc convention in 1-D.

    Returns
    -------
    kernel : callable
        kernel(X, Y) -> Gram matrix of shape (len(X), len(Y)).
    """
    from scipy.special import gamma, jv
    fc = float(fc)
    if fc <= 0:
        raise ValueError("fc must be positive")

    def kernel(X, Y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        d = X.shape[1]
        if Y.shape[1] != d:
            raise ValueError("X and Y must have the same number of dimensions")
        nu = d / 2.0
        a = fc / 2.0
        R = np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=-1))
        K = np.empty_like(R)
        # Dimensionless radius; use the closed-form limit when tiny.
        tiny = R * fc <= 1e-8
        K[tiny] = (np.pi ** nu) * (a ** d) / gamma(nu + 1.0)
        rnz = R[~tiny]
        z = np.pi * fc * rnz
        if d % 2 == 1:
            # Odd d: J_{n+1/2} has an exact closed form via the spherical
            # Bessel recurrence j_0 = sin z / z,
            # j_k = (2k-1)/z * j_{k-1} - j_{k-2},
            # J_{n+1/2}(z) = sqrt(2z/pi) * j_n(z).  Much cheaper than jv.
            n_sph = (d - 1) // 2
            j0 = np.sin(z) / z
            if n_sph == 0:
                jn = j0
            else:
                j1 = np.sin(z) / z ** 2 - np.cos(z) / z
                jm2, jm1 = j0, j1
                for k in range(2, n_sph + 1):
                    jk = (2 * k - 1) / z * jm1 - jm2
                    jm2, jm1 = jm1, jk
                jn = jm1
            Jnu = np.sqrt(2.0 * z / np.pi) * jn
        else:
            Jnu = jv(nu, z)
        K[~tiny] = (a ** nu) * (rnz ** (-nu)) * Jnu
        return K

    kernel.__name__ = "ball_kernel"
    # Reproducing property => overlap matrix Q equals the Gram matrix.
    kernel._overlap = "gram"
    return kernel


# ---------------------------------------------------------------------------
# BLML-Trivial estimator
# ---------------------------------------------------------------------------

class _Rescaler:
    """Per-dimension data standardization for unit-cutoff fits.

    Fitting BLML with cutoff ``fc_k`` on raw data ``X`` is exactly
    equivalent to fitting with cutoff 1 on ``X / s`` with ``s_k = 1/fc_k``
    (a change of variables: if p_X is band-limited to ``fc``, then
    p_{X/s} is band-limited to 1), up to the Jacobian ``prod(s)`` on the
    density.  Standardizing first keeps every Gram entry in [-1, 1] with
    an exact unit diagonal, which removes the ``prod(fc)`` overflow
    regime entirely.

    ``mode``: "std" (default, True maps here), "iqr" (raw IQR = q75-q25),
    a positive scalar, or per-dimension scales.  None/False disables.
    """

    def __init__(self, mode):
        if mode is True:
            mode = "std"
        self.mode = mode
        self.enabled = mode is not None and mode is not False
        self.s_ = None

    def _compute_scales(self, X):
        mode = self.mode
        if isinstance(mode, str):
            if mode == "std":
                return X.std(axis=0)
            elif mode == "iqr":
                q75, q25 = np.percentile(X, [75, 25], axis=0)
                return q75 - q25  # raw IQR, not /1.349 (user: "if we call it iqr")
            elif mode == "auto":
                raise ValueError('rescale="auto" needs a kernel; it is '
                                 "resolved in fit(), not here")
            else:
                raise ValueError('rescale must be "std", "iqr", "auto", True, '
                                 "a positive scalar, or per-dimension scales")
        return np.broadcast_to(np.asarray(mode, dtype=float),
                               (X.shape[1],)).copy()

    def fit_transform(self, X, scales=None):
        if scales is None:
            scales = self._compute_scales(X)
        s = np.broadcast_to(np.asarray(scales, dtype=float),
                            (X.shape[1],)).copy()
        if not np.all(np.isfinite(s)) or np.any(s <= 0):
            raise ValueError("rescale scales must be positive and finite; "
                             "is an input dimension constant?")
        self.s_ = s
        return X / self.s_

    def transform(self, X):
        return np.asarray(X, dtype=float) / self.s_

    @property
    def jacobian(self):
        """prod(s): predict() divides the rescaled-space density by this."""
        return float(np.prod(self.s_))


def auto_scale(X, gram_fn=None, target_neighbors=32, seed=0, sub_n=256):
    """Choose per-dimension rescaling scales data-adaptively.

    Sets ``s = m * std(X)`` with a dimension-aware multiplier

        m = 0.6 * sqrt(d).

    Why this form: for the d-dimensional *product* sinc kernel the
    Gram collapses unless the per-dimension rescaled spread shrinks
    with d -- the d-fold product of per-dim sinc factors kills
    off-diagonal entries unless typical per-dim offsets stay O(1),
    which needs m growing like sqrt(d).  Calibrated on Gaussian-mixture
    MSE sweeps (d = 1, 2, 10), where the MSE-optimal multipliers were
    m* = 0.5, 1.0, 2.0, i.e. m*/sqrt(d) = 0.50, 0.71, 0.63:

        d=1 : m=0.5 -> relMSE 0.005 (m=1.0 -> 0.009)
        d=2 : m=1.0 -> relMSE 0.025 (m=0.5 -> 0.066)
        d=10: m=2.0 -> relMSE 0.158 (m=1.5 -> 0.537, m=2.5 -> 0.331)

    The previous neighbor-count (perplexity-style) tuner is retired:
    the count statistic is nearly discontinuous in m at high d (it
    jumps from ~5 to ~146 neighbors across a tiny m range), so any
    fixed neighbor target is miscalibrated somewhere.  The closed-form
    rule is continuous, costs no Gram evaluations, and matches the
    observed optima.

    ``gram_fn`` is kept for backward compatibility and ignored.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Training data.
    gram_fn : callable, optional
        Ignored (kept for compatibility).
    target_neighbors, seed, sub_n : ignored
        Kept for compatibility.

    Returns
    -------
    scales : ndarray, shape (d,)
        Per-dimension scales; pass as ``rescale=scales``.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, d = X.shape
    base = X.std(axis=0)
    base[~np.isfinite(base) | (base <= 0)] = 1.0
    m_star = 0.6 * np.sqrt(d)
    return m_star * base


class BLMLTrivial:
    """DEPRECATED: Use BLMLQuick instead. This dense O(n^2) version is kept
    only for backward compatibility with demo.py. Do not use for new code,
    especially not for 2-D (will OOM)."""
    """Band-Limited Maximum Likelihood density estimator (trivial variant).

    Solves the BLML constraint system in the positive orthant with a
    fixed-point iteration (Newton fallback), then evaluates the squared
    weighted kernel sum as the density estimate.

    Parameters
    ----------
    kernel : callable or None
        Swappable kernel.  Called as ``kernel(X, Y, **kernel_params)``
        returning the (len(X), len(Y)) Gram matrix; a plain scalar
        ``kernel(x, y)`` is also accepted and evaluated elementwise.
        ``None`` selects the default band-limited sinc kernel.
    kernel_params : dict or None
        Extra keyword arguments forwarded to ``kernel``.
    fc : float or array-like
        Cut-off frequency used only when ``kernel`` is None (sinc default).
    max_iter : int
        Maximum fixed-point iterations for solving rho_n(c) = 0.
    tol : float
        Convergence tolerance: stop when max |rho_n(c)| < tol.
    rescale : {None, True, "std", "iqr"} or array-like, optional
        Standardize each dimension by a data-driven scale before fitting
        (see :class:`_Rescaler`), so the kernel's ``fc`` acts in
        standardized units -- e.g. ``rescale="std"`` with a unit-cutoff
        sinc kernel fits ``fc = 1`` per standardized dimension, with the
        density mapped back by the Jacobian.  Keeps Gram entries in
        [-1, 1] with unit diagonal.  Default None (no rescaling).
    predict_kernel : callable or None, optional
        Kernel used for the query-vs-train Gram at prediction time
        (``predict`` / ``score_samples``).  The recommended pairing is a
        binned kernel (e.g. ``lattice_kernel(fc, h, kinds, periods, d)``) as ``kernel`` for
        the fit with the exact kernel (e.g. ``default_sinc_kernel`` at
        the same ``fc``) as ``predict_kernel``: cheap cached fit, exact
        lattice-free inference.  When set and the fit kernel is a binned
        kernel, training points are snapped to the bin centers at fit
        time so the exact prediction is the true binned-model expansion.
        Default None (predict with the fit kernel).
    verbose : bool
        Print iteration diagnostics.

    Attributes
    ----------
    c_hat_ : ndarray, shape (n,)
        Estimated coefficients after :meth:`fit`.
    S_ : ndarray, shape (n, n)
        Kernel Gram matrix on the training data.
    residual_ : float
        max |rho_n(c_hat)| at the fitted solution.
    scales_ : ndarray or None
        Per-dimension rescaling factors (None when rescale is off).
    """

    def __init__(self, kernel=None, kernel_params=None, fc=1.0,
                 max_iter=1000, tol=1e-8, rescale=None, verbose=False,
                 predict_kernel=None):
        if kernel is None:
            kernel = default_sinc_kernel(fc)
        self.kernel = kernel
        self.kernel_params = dict(kernel_params) if kernel_params else {}
        self.fc = fc
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.rescale = rescale
        self.predict_kernel = predict_kernel
        self._rescaler = _Rescaler(rescale)
        self.scales_ = None
        self.verbose = bool(verbose)

    # -- internal helpers ---------------------------------------------------

    def _maybe_rescale_fit(self, X, seed=0):
        """Apply rescaling at fit time; returns the rescaled X.

        Handles rescale="auto" via auto_scale(); otherwise delegates to
        the _Rescaler.  Sets self.scales_.
        """
        if self._rescaler.enabled:
            scales = None
            if isinstance(self._rescaler.mode, str) and \
                    self._rescaler.mode == "auto":
                scales = auto_scale(X, self._gram, seed=seed)
            X = self._rescaler.fit_transform(X, scales=scales)
        self.scales_ = (self._rescaler.s_.copy()
                        if self._rescaler.enabled else None)
        return X

    def _maybe_quantize_fit(self, X):
        """Snap training points to the binned lattice for cached-fit /
        exact-predict inference.

        When ``predict_kernel`` is set and the fit kernel is a binned
        (lattice) kernel, the fitted model is the BLML solution of the
        *binned* problem, whose expansion is naturally centered on the
        bin centers.  Snapping ``X`` here makes ``self.X_`` those
        centers, so exact-sinc prediction evaluates the true binned-model
        expansion instead of mixing binned coefficients with unbinned
        centers.  The snap is idempotent, so the fit itself is unchanged.
        """
        if self.predict_kernel is not None:
            quantize = getattr(self.kernel, "quantize", None)
            if callable(quantize):
                X = quantize(X)
        return X

    # -- internal helpers ---------------------------------------------------

    def _gram(self, A, B, kernel=None):
        """Gram matrix between rows of A and rows of B via the kernel.

        ``kernel`` overrides the fit kernel (used for exact-sinc
        prediction when ``predict_kernel`` is set); defaults to the fit
        kernel.
        """
        kernel = self.kernel if kernel is None else kernel
        A = np.atleast_2d(np.asarray(A, dtype=float))
        B = np.atleast_2d(np.asarray(B, dtype=float))
        try:
            G = kernel(A, B, **self.kernel_params)
            G = np.asarray(G, dtype=float)
            if G.shape == (A.shape[0], B.shape[0]):
                return G
        except TypeError:
            pass  # not a vectorised (X, Y) kernel; try scalar form below
        # Elementwise fallback for scalar kernel(x, y) callables.
        G = np.empty((A.shape[0], B.shape[0]))
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                G[i, j] = kernel(a, b, **self.kernel_params)
        return G

    def _rho(self, c, S=None):
        """Constraint residuals rho_n(c) = (1/n) S c - 1/c."""
        S = self.S_ if S is None else S
        return (S @ c) / self.n_samples_ - 1.0 / c

    # -- public API ----------------------------------------------------------

    def fit(self, X, signs=None):
        """Fit the BLML-Trivial estimator to samples X.

        Parameters
        ----------
        X : array-like, shape (n,) or (n, d)
            Training samples.
        signs : None, array of length n, or callable s(X) -> +-values
            Optional fixed sign of g at each sample (see resolve_signs).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError("X must be 1-D or 2-D")
        n, d = X.shape
        if n == 0:
            raise ValueError("X must contain at least one sample")

        sg = resolve_signs(signs, X)          # callables see the samples in the caller's units
        X = self._maybe_rescale_fit(X)
        X = self._maybe_quantize_fit(X)

        S = self._gram(X, X)
        S = 0.5 * (S + S.T)  # symmetrise against numerical asymmetry
        if sg is not None:
            S = S * sg[:, None] * sg[None, :]     # fixed sign pattern: solve on D S D

        self.X_ = X
        self.S_ = S
        self.n_samples_ = n
        self.dim_ = d

        # Positive-orthant solve of rho_n(c) = 0 via the shared
        # fixed-point/Newton routine (see solve_rho_positive_orthant).
        c, res, it_done, solver, converged = solve_rho_positive_orthant(
            lambda cc: S @ cc, n,
            max_iter=self.max_iter, tol=self.tol,
            verbose=self.verbose, tag="BLMLTrivial", dense_S=S,
        )

        if sg is not None:
            c = c * sg
            self.S_ = S * sg[:, None] * sg[None, :]   # store the plain Gram
        self.signs_ = sg
        self.c_hat_ = c
        self.residual_ = res
        self.n_iter_ = it_done
        self.solver_ = solver
        self.converged_ = converged
        if self.verbose:
            print(f"  [BLMLTrivial] done in {it_done} iters, "
                  f"max|rho| = {self.residual_:.3e}, converged = {converged}")
        return self

    def predict(self, Xq):
        """Evaluate the density estimate at query points.

        f_hat(x) = ((1/n) * sum_i c_hat_i * K(x, x_i))^2.

        Parameters
        ----------
        Xq : array-like, shape (m,) or (m, d)

        Returns
        -------
        ndarray, shape (m,)
            Estimated density values (non-negative by construction).
        """
        if not hasattr(self, "c_hat_"):
            raise RuntimeError("Call fit() before predict().")
        Xq = np.asarray(Xq, dtype=float)
        if Xq.ndim == 1:
            Xq = Xq[:, None]
        if Xq.shape[1] != self.dim_:
            raise ValueError(
                f"query dimension {Xq.shape[1]} != fitted dimension {self.dim_}")
        if self._rescaler.enabled:
            Xq = self._rescaler.transform(Xq)
        # Exact-sinc prediction when predict_kernel is set: inference is
        # the continuous expansion, not bound to the bin lattice.
        G = self._gram(Xq, self.X_, kernel=self.predict_kernel)
        v = (G @ self.c_hat_) / self.n_samples_
        f = v ** 2
        if self._rescaler.enabled:
            # Map the rescaled-space density back (change of variables).
            f = f / self._rescaler.jacobian
        return f

    def score_samples(self, X):
        """Density values of the fitted model at X (like sklearn scorers)."""
        return self.predict(X)

    def constraint_residual(self, c=None):
        """max |rho_n(c)| - the constraint residual at the solution.

        Parameters
        ----------
        c : array-like or None
            Coefficients to check; defaults to the fitted ``c_hat_``.
        """
        if not hasattr(self, "c_hat_"):
            raise RuntimeError("Call fit() before constraint_residual().")
        c = self.c_hat_ if c is None else np.asarray(c, dtype=float)
        return float(np.max(np.abs(self._rho(c))))

    def __repr__(self):
        kname = getattr(self.kernel, "__name__", type(self.kernel).__name__)
        return (f"BLMLTrivial(kernel={kname}, max_iter={self.max_iter}, "
                f"tol={self.tol}, rescale={self.rescale})")


def unit_eval(kind, th, diffs, xp=np, j01=None):
    """K of one kernel unit at coordinate differences, from its kernel class (xp = cupy: radial profiles from the
    elementwise RADIAL_KERNELS_XP, the classes' own formulas otherwise)."""
    if xp is not np and is_radial(kind) and _split_kind(kind)[0] in RADIAL_KERNELS_XP:
        name = _split_kind(kind)[0]; fc = np.asarray(th['fc'], float)
        s = xp.sqrt(sum((f * dk) ** 2 for f, dk in zip(fc, diffs)))
        return float(np.prod(fc)) * RADIAL_KERNELS_XP[name][0](s, len(fc), xp, j01)
    return kernel_class(kind).eval(th, diffs, xp)


def sample_conditional(model, Xrest, n_samples=100, seed=0, n_prop=400, xp=np, j01=None, dt=float, max_elems=1 << 23):
    """Samples of a conditional fit's dep columns given the rest (any kernel kinds, rank K, real or complex g, per-mode
    kernels), by sampling-importance-resampling. On the query's slice f(y | x) is proportional to
    sum_k |g_k(y)|^2, g_k(y) = sum_b A_bk K_dep,k(y - y_b) with A_bk = e_bk K_rest,k(x - x_b) / n. Proposals: a bin b
    with probability proportional to sum_k |A_bk|, then each dep column from a bump around y_b -- y_bj + u / fc_j
    with u ~ sinc^2 (sinc and radial columns; the bump fc sinc^2(fc t) a proper density) or uniform on the circle
    (dirichlet columns). Each of n_prop proposals per query is weighted by target / proposal density (both
    unnormalised), and n_samples draws are resampled with those weights (consistent as n_prop grows; plain rejection
    with a Cauchy-Schwarz envelope accepts only ~4% on CAISO and can stall in sparse covariate regions). No grid, no
    normalising constant; one proposal costs one pass over the B bins. xp = cupy (with j01, dt the kernel dtype): on
    the GPU. Returns (samples (m, n_samples, D) in the order of dep_, mean effective sample size / n_prop)."""
    dep = list(getattr(model, 'dep_', None) or [])
    if not dep:
        raise RuntimeError("fit with dep columns to sample the conditional")
    d = model.dim_; rest = [j for j in range(d) if j not in dep]; D = len(dep)
    Xrest = np.atleast_2d(np.asarray(Xrest, dtype=float))
    E = np.asarray(model.e_hat_.get() if hasattr(model.e_hat_, 'get') else model.e_hat_)
    E = E[:, None] if E.ndim == 1 else E; K = E.shape[1]
    if getattr(model, 'modes_', None) is not None:                 # per-mode kernels (CPU specs)
        specs = [(sp['fc'], sp['kinds'], sp['periods']) for sp in model.modes_]
    elif getattr(model, 'mode_specs_', None) is not None:          # (GPU)
        specs = [tuple(ms) for ms in model.mode_specs_]
    else:
        specs = [(model._fc_arr, model.kinds_, model.periods_)] * K
    specs = [(np.broadcast_to(np.asarray(f, float), (d,)), list(k) if k is not None else ['sinc'] * d,
              list(p) if p is not None else [None] * d) for f, k, p in specs]
    ZZ = xp.asarray(model.Z_, dtype=float); Ex = xp.asarray(E); n = float(model.n_samples_); B = ZZ.shape[0]
    rests = [_continuous_product_kernel(f[rest], [k[j] for j in rest], [p[j] for j in rest], len(rest), xp, j01)
             for f, k, p in specs]
    deps = []
    for f, k, p in specs:                                  # each mode's dep units, as (kind, theta, dep positions)
        us = []
        for kind, dims in kernel_units(k, d):
            if dims[0] in dep:
                if not all(j in dep for j in dims):
                    raise ValueError(f"a kernel unit over {dims} straddles dep and conditioning columns")
                us.append((kind, unit_theta(kind, f, p, dims), [dep.index(j) for j in dims]))
        deps.append(us)
    f0, k0, p0 = specs[0]
    prop = [('circ', float(p0[j])) if is_periodic_kind(k0[j]) else ('sinc2', float(f0[j])) for j in dep]
    rs = xp.random.RandomState(seed)

    def sinc2(shape):                                      # u ~ sinc^2(u) = (sin(pi u) / (pi u))^2
        out = xp.empty(shape, dtype=dt).ravel(); filled = 0
        while filled < out.size:
            k_ = int(1.4 * (out.size - filled)) + 64
            tail = rs.random_sample(k_) < 0.5
            v = rs.random_sample(k_)
            u = xp.where(tail, 1.0 / (np.pi * xp.maximum(v, 1e-12)), v / np.pi)   # |u| from min(1, 1/(pi u)^2)
            hh = xp.where(tail, 1.0 / (np.pi * u) ** 2, 1.0)
            acc = rs.random_sample(k_) * hh < xp.sinc(u) ** 2
            u = (u * xp.where(rs.random_sample(k_) < 0.5, -1.0, 1.0))[acc][:out.size - filled]
            out[filled:filled + len(u)] = u; filled += len(u)
        return out.reshape(shape)

    P = n_prop
    Qc = max(1, max_elems // max(B * P, 1))
    out = np.empty((len(Xrest), n_samples, D)); ess = []
    for q in range(0, len(Xrest), Qc):
        Xq = xp.asarray(Xrest[q:q + Qc]); m_ = Xq.shape[0]
        A = [rests[k](Xq, ZZ[:, rest]).astype(dt) * Ex[None, :, k] / n for k in range(K)]      # (m, B) per mode
        absA = sum(xp.abs(a) for a in A)
        cdf = xp.cumsum(absA.astype(xp.float64), 1) / absA.sum(1, dtype=xp.float64)[:, None]   # float64: row offsets below
        off = xp.arange(m_, dtype=xp.float64)[:, None] * 2
        b = xp.searchsorted((cdf + off).ravel(), (rs.random_sample((m_, P)) + off).ravel())
        b = xp.minimum(b.reshape(m_, P) - xp.arange(m_)[:, None] * B, B - 1)              # (m, P) bins
        y = xp.stack([rs.random_sample((m_, P)) * pv if kd == 'circ' else ZZ[:, j][b] + sinc2((m_, P)) / pv
                      for (kd, pv), j in zip(prop, dep)], -1).astype(dt)                    # (m, P, D)
        rho = xp.ones((m_, P, B), dtype=dt)               # the proposal's bump at each bin (circle: constant)
        for i, ((kd, pv), j) in enumerate(zip(prop, dep)):
            if kd == 'sinc2':
                rho *= (pv * xp.sinc(pv * (y[:, :, i, None] - ZZ[None, None, :, j])) ** 2).astype(dt)
        env = (absA[:, None, :] * rho).sum(-1)
        tgt = 0.0
        for k in range(K):
            Kd = xp.ones((m_, P, B), dtype=dt)
            for kind, th, pos in deps[k]:
                Kd *= unit_eval(kind, th, [y[:, :, i, None] - ZZ[None, None, :, dep[i]] for i in pos], xp, j01).astype(dt)
            gk = (A[k][:, None, :] * Kd).sum(-1)
            tgt = tgt + xp.abs(gk) ** 2
        w = (tgt / xp.maximum(env, 1e-300)).astype(xp.float64)
        w = w / xp.maximum(w.sum(1, keepdims=True), 1e-300)
        ess.append((1.0 / (w * w).sum(1)) / P)
        wc = xp.cumsum(w, 1); wc[:, -1] = 1.0
        rr = xp.searchsorted((wc + off).ravel(), (rs.random_sample((m_, n_samples)) + off).ravel())
        rr = xp.minimum(rr.reshape(m_, n_samples) - xp.arange(m_)[:, None] * P, P - 1)
        smp = y[xp.arange(m_)[:, None], rr]
        out[q:q + m_] = smp.get() if hasattr(smp, 'get') else smp
    ess = [e.get() if hasattr(e, 'get') else e for e in ess]
    return out, float(np.mean(np.concatenate(ess)))


def _continuous_product_kernel(fc, kinds, periods, d, xp=np, j01=None):
    """Continuous (non-binned) product kernel for exact prediction: the kernel classes' units (unit_eval)."""
    fc_vec = np.broadcast_to(np.asarray(fc, dtype=float), (d,)).copy()
    kinds = ['sinc'] * d if kinds is None else list(kinds)
    periods = [None] * d if periods is None else list(periods)
    units = kernel_units(kinds, d)

    def kern(X, Y):
        X = xp.atleast_2d(xp.asarray(X, dtype=float)); Y = xp.atleast_2d(xp.asarray(Y, dtype=float))
        G = xp.ones((X.shape[0], Y.shape[0]))
        for kind, dims in units:
            G *= unit_eval(kind, unit_theta(kind, fc_vec, periods, dims), [X[:, k, None] - Y[None, :, k] for k in dims],
                           xp, j01)
        return G
    return kern


def lbfgs_psd(e, Se, w, n, SE_of, loglik, residual, tol, max_iter, mem, verbose=False, xp=np):
    """solver='lbfgs' of the psd (complex / rank K) fits of BLMLQuick (CPU) and its CUDA class: L-BFGS ascent of
    L = sum_b w_b log F_b - n log Q (Q = sum_k e_k^H S_k e_k) on the coefficients e (B, K), gradient
    2 (S_k (w u_k / F) - n u_k / Q) with S applied by SE_of (kernel rows streamed in blocks, as the sweeps: memory
    O(B x block + mem B K)); xp = numpy or cupy. Memory mem, Armijo backtracking from the unit step (the first step
    1e-3 of |e|), a pair stored only with positive curvature, one memory reset on a failed line search before
    stopping. Stops at max|rho| < tol, when L / n gains < 1e-11 over 10 iterations (converged in the likelihood: the
    residual of the near-null directions of S, which L does not see, can stay above tol), or after max_iter
    iterations; then the coefficients take their canonical form e = (Q / n) (w / F) S e (the stationarity condition:
    a function of the field alone; L-BFGS leaves components in the near-null space of S where the start put them),
    kept unless it loses likelihood. Returns e, S e, iterations, converged, max|rho|, rho and -L/n histories."""
    dot = lambda a, b: float(xp.real(xp.vdot(a.astype(xp.complex128 if xp.iscomplexobj(a) else xp.float64),
                                              b.astype(xp.complex128 if xp.iscomplexobj(b) else xp.float64))))
    w = w.astype(xp.float64)
    def grad(e, Se):
        F = (xp.abs(Se) ** 2).sum(1); Q = dot(e, Se)
        return (2 * (SE_of(((w / xp.maximum(F, 1e-300))[:, None] * Se).astype(e.dtype)) - (n / Q) * Se)).astype(e.dtype)
    def nrm(e, Se):
        sc = n / np.sqrt(max(dot(e, Se), 1e-300)); return e * sc, Se * sc
    L = loglik(e, Se) * n; g = grad(e, Se); S_, Y_, R_ = [], [], []
    it = 0; cap = np.inf if max_iter is None else max_iter
    max_rho = residual(*nrm(e, Se)); rho_hist, obj_hist = [max_rho], [-L / n]; reset = llstop = False
    while max_rho >= tol and it < cap:
        if len(obj_hist) > 10 and obj_hist[-11] - obj_hist[-1] < 1e-11:
            llstop = True; break
        it += 1
        q = g.copy(); al = []
        for s_, y_, r_ in zip(reversed(S_), reversed(Y_), reversed(R_)):
            a = r_ * dot(s_, q); al.append(a); q = q - a * y_
        q = q * (dot(S_[-1], Y_[-1]) / dot(Y_[-1], Y_[-1]) if S_ else 1e-3 * np.sqrt(dot(e, e) / max(dot(g, g), 1e-300)))
        for (s_, y_, r_), a in zip(zip(S_, Y_, R_), reversed(al)):
            q = q + (a - r_ * dot(y_, q)) * s_
        slope = dot(g, q)
        if not slope > 0:
            S_, Y_, R_ = [], [], []; q = g * (1e-3 * np.sqrt(dot(e, e) / max(dot(g, g), 1e-300))); slope = dot(g, q)
        t = 1.0; ok = False
        for _ in range(40):
            en = (e + t * q).astype(e.dtype); Sen = SE_of(en); Ln = loglik(en, Sen) * n
            if Ln >= L + 1e-4 * t * slope:
                ok = True; break
            t *= 0.5
        if not ok:
            if reset or not S_:
                break
            S_, Y_, R_ = [], [], []; reset = True; continue
        reset = False
        gn = grad(en, Sen); s_, y_ = en - e, g - gn; sy = dot(s_, y_)
        if sy > 1e-12 * np.sqrt(dot(s_, s_) * dot(y_, y_)):
            S_.append(s_); Y_.append(y_); R_.append(1.0 / sy)
            if len(S_) > mem:
                S_.pop(0); Y_.pop(0); R_.pop(0)
        e, Se, L, g = en, Sen, Ln, gn
        max_rho = residual(*nrm(e, Se)); rho_hist.append(max_rho); obj_hist.append(-L / n)
        if verbose and it % 100 == 0:
            print(f"  lbfgs {it}: max|rho|={max_rho:.2e} -L/n={-L / n:.9f}", flush=True)
    F = (xp.abs(Se) ** 2).sum(1); Q = dot(e, Se)
    ec = ((Q / n) * (w / xp.maximum(F, 1e-300))[:, None] * Se).astype(e.dtype); Sec = SE_of(ec)
    if loglik(ec, Sec) * n >= L - 1e-9 * n:
        e, Se = ec, Sec
    e, Se = nrm(e, Se); max_rho = residual(e, Se); converged = max_rho < tol or llstop
    rho_hist.append(max_rho); obj_hist.append(-loglik(e, Se))
    return e, Se, it, converged, max_rho, rho_hist, obj_hist


@projected_front
class BLMLQuick:
    """BLML with collapsed duplicates (the full BLMLQuick idea).

    Quantizes training points to the kernel's lattice (lattice_kernel), collapses
    duplicate bins (keeping counts w_b), and solves the weighted BLML
    system on the B unique bins instead of n points via block-CD with
    cached sums (O(B) memory, never materializing the BxB Gram).
    When B << n (large n, low d, moderate fc) this is orders of
    magnitude faster than the full n-point solve.

    Math: after quantization, points in the same bin are identical, so the
    optimal coefficients are equal within bins by symmetry. Writing e_b for
    the total coefficient mass in bin b (sum of c_i over the bin), the
    constraint system becomes
        S^bin_{b,b} e_b^2 + (sum_{b'!=b} S^bin_{b,b'} e_{b'}) e_b - n w_b = 0,
    solved by block coordinate descent with the positive root. The density
    is f(x) = ((1/n) sum_b e_b K(x, z_b))^2, with the usual Jacobian
    correction when rescaling is used.
    """

    def __new__(cls, *a, engine='cpu', **kw):
        """engine='gpu': the CUDA implementation (blml_gpu), same API (its own keywords, e.g. block_size)."""
        if engine == 'gpu' and cls is BLMLQuick:
            from blml_gpu import _BLMLQuickCUDA
            return _BLMLQuickCUDA(*a, **kw)
        if engine not in ('cpu', 'gpu'):
            raise ValueError(f"engine {engine!r}: 'cpu' or 'gpu'")
        return super().__new__(cls)

    def __init__(self, tol=1e-4, batch_size=2000, seed=0, verbose=False, dtype=np.float32, bin_scale=0.1,
                 max_sweeps=None, engine='cpu', solver='cd', lbfgs_m=20):
        if solver not in ('cd', 'lbfgs'):
            raise ValueError("solver: 'cd' (block coordinate descent with Anderson mixing) or 'lbfgs'")
        self.solver, self.lbfgs_m = solver, int(lbfgs_m)   # complex / rank K fits; real rank 1 always cd
        self.tol = tol
        self.batch_size = batch_size
        self.seed = seed
        self.bin_scale = bin_scale  # bin width h = bin_scale * n^{-1/4} / fc
        self.verbose = verbose
        self.dtype = np.dtype(dtype)
        # None: sweep until max|rho| < tol. k: stop after k sweeps (0: keep the start, e.g. a warm start
        # at a new fc), converged_ False unless tol was met; e is then rescaled to e'Se = n^2, the
        # optimum's normalisation (int f = e'Se / n^2 = 1), so the density stays normalised.
        self.max_sweeps = max_sweeps
        self.phase_noise = 0.1      # complex_g: size of the random phases added to the real start
        self.anderson = 5           # complex_g / rank > 1: Anderson history (0 = off); fixed visiting order
        self.n_starts = (3, 2)      # complex_g: polished starts (HIO, symmetry breaking), best LL kept; (0, 0) = one start
        self._fit_kernel = None  # built in fit() with paper h
        self._predict_kernel = None  # built in fit() from kinds/periods

    def fit(self, X, modes, init=None, rebin_tol=1.0, signs=None, complex_g=False, sample_weight=None):
        """X: (n, d) samples. modes: the model, one dict per mode {column: {fc, kind, period, group, dep}}
        (see parse_modes); rank K = len(modes); f = sum_k |g_k|^2, g_k band-limited, real or complex.
        Real fits use one start (the positive root). Complex fits polish self.n_starts = (HIO,
        symmetry-breaking) starts and keep the best log-likelihood (see _solve_psd); self.starts_
        lists them. Columns marked dep: the joint is fitted on all columns and the marginal on the
        others (same spec); predict_marginal and predict_ratio (the conditional) become available.
        signs: real rank-1 only; fixed sign of g, one entry per sample (majority per bin) or a callable
        s(Z) at the bin centres; see resolve_signs. The lattice is fine enough for the largest fc over
        the modes.
        init: a previous fit with the same modes (warm start for streaming / sliding windows; complex
        fits then skip the start search). While the target bin width bin_scale n^{-1/4} / fc stays
        within (1 + rebin_tol)^{1/4} of the one the lattice was built for (keep_lattice: n may
        double, or fc move ~19%), the lattice is kept and bins present in both fits keep their coefficients; otherwise the data are re-binned for the current
        n and every bin starts at w_b g_prev(z_b) / |g_prev(z_b)|^2. Same rule on the GPU (engine='gpu').
        sample_weight: (n,) positive weights (default 1): point i counts w_i times (a bin's count is its points'
        total weight, n their sum); integer weights equal repeating the points.
        Counts on a fixed lattice (modes with h, see parse_modes / with_lattice): X are the bin centres, sample_weight
        their counts (zeros allowed: such bins are dropped); see _fit_lattice."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))   # quantize in float64: float32 misplaces
        n_rows, d = X.shape                                    # points near bin edges once |X|/h > ~1e5
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        h_fixed = modes_lattice(modes, d)
        if h_fixed is not None:                              # counts on a fixed lattice: never re-binned
            w = np.ones(n_rows) if sw is None else sw
            if w.shape != (n_rows,) or not np.all(w >= 0):
                raise ValueError("sample_weight: (n,) non-negative counts")
            return self._fit_lattice(X, w, h_fixed, modes, signs=signs, complex_g=complex_g, init=init)
        if sw is not None and (sw.shape != (n_rows,) or not np.all(sw > 0)):
            raise ValueError("sample_weight: (n,) positive weights")
        n = n_rows if sw is None else float(sw.sum())
        specs, dep = parse_modes(modes, d)
        if dep is not None:
            return self._fit_conditional(X, modes, dep, signs, complex_g, init, rebin_tol, sw)
        K, shared = len(specs), _shared_kernel(specs)
        self.spec_ = modes
        psd = complex_g or K > 1
        if psd and signs is not None:
            raise ValueError("complex_g / rank > 1 estimate the phases; do not pass signs as well")
        kinds, periods, fc_arr = specs[0]['kinds'], specs[0]['periods'], specs[0]['fc']
        self.kinds_, self.periods_, self._fc_arr = kinds, periods, fc_arr
        # Paper (arXiv:1503.06236): bins of size < 0.5/fc (Nyquist), with h ~ n^{-0.25} so the
        # discretization ISE is O(1/n); h = bin_scale * n^{-0.25} / fc satisfies both.
        fc_lat = np.max([sp['fc'] for sp in specs], axis=0)
        h_paper = lattice_target(self.bin_scale * (n ** -0.25) / fc_lat, [sp['kinds'] for sp in specs])
        rebin = init is not None and not keep_lattice(init, h_paper, rebin_tol)
        self.h_target_ = h_paper if (init is None or rebin) else init.h_target_
        if init is not None and not rebin:     # keep init's lattice (see keep_lattice)
            h_paper = np.asarray(init._h_vec, dtype=float)
        self.n_lattice_ = n if (init is None or rebin) else init.n_lattice_
        self.fc_lattice_, self.rebinned_ = fc_lat, rebin
        self._fit_kernel = lattice_kernel(fc_lat, h_paper, kinds, periods, d, lattice_fold_specs(specs))
        self._predict_kernel = _continuous_product_kernel(fc_arr, kinds, periods, d)
        # Snap to the lattice (round to nearest center; periodic dims wrap to [0, P)) and collapse
        # duplicate bins.
        Xq = self._fit_kernel.quantize(X)
        Xq_c = np.ascontiguousarray(Xq)
        vdt = np.dtype((np.void, Xq_c.dtype.itemsize * Xq_c.shape[1]))
        _, idx, inv, counts = np.unique(Xq_c.view(vdt), return_index=True,
                                        return_inverse=True, return_counts=True)
        Z = Xq[idx]                      # (B, d) unique bin centers
        w = (counts if sw is None else np.bincount(np.ravel(inv), weights=sw, minlength=len(idx))).astype(self.dtype)
        B = Z.shape[0]
        h_vec = self._fit_kernel.h
        Zi = np.round(Z / h_vec).astype(np.int64)
        self._h_vec = h_vec
        self._Zi = Zi
        sg = resolve_signs(signs, Z, inverse=np.ravel(inv), n_samples=n_rows)
        self._signs_cd = None if sg is None else sg.astype(self.dtype)
        self.complex_g_, self.rank_ = bool(complex_g), K
        self.modes_ = None if shared else specs
        self._init_e = warm_start_e(init, Z, Zi, w, same_lattice=not rebin)
        self._init_Se = warm_start_Se(init, Zi, fc_arr, kinds, periods, h_vec) if (init is not None and not rebin) else None
        if psd:
            out = self._solve_psd(Zi, w, n, B, Z, h_vec, d, specs, bool(complex_g))
        else:
            out = self._solve_weighted_cd(Zi, w, n, B, Z, h_vec, d)
        self._init_e = self._init_Se = None
        e, n_sweeps, converged, max_rho, rho_hist, obj_hist = out
        if sg is not None:
            e = (e * self._signs_cd).astype(self.dtype)
        self.Se_ = self._final_Se(e, sg)
        self.signs_ = sg
        self.e_hat_ = e
        self.Z_ = Z
        self.w_ = w
        self.n_samples_ = n
        self.n_bins_ = B
        self.dim_ = d
        self.n_sweeps_ = n_sweeps
        self.converged_ = converged
        self.residual_ = max_rho
        self.rho_history_ = rho_hist
        self.obj_history_ = obj_hist
        return self

    def _solve_psd(self, Zi, w, n, B, Z, h_vec, d, specs, complex_g):
        """f = sum_k |g_k|^2 (rank K = len(specs), real or complex g_k). Real: one solve from the real
        rank-1 start. Complex: self.n_starts = (HIO, symmetry-breaking) starts from complex_starts on
        mode 0's band, each polished by _solve_modes_cd (start in mode 0, small noise in the others);
        the highest log-likelihood wins and self.starts_ records every start. One start when a warm
        start (init=) is given, n_starts = (0, 0), a dim is periodic, or the FFT grid is too large."""
        def solve():
            out = list(self._solve_modes_cd(Zi, w, n, B, Z, h_vec, d, specs, complex_g=complex_g))
            if len(specs) == 1:
                out[0] = out[0][:, 0]
            return out
        self.starts_ = None
        nh, ns = self.n_starts
        if not complex_g or getattr(self, '_init_e', None) is not None or nh + ns == 0:
            return solve()
        if any(is_periodic_kind(k) for k in specs[0]['kinds']):
            warnings.warn("complex_g: start search not implemented for periodic dims; using one start")
            return solve()
        starts = complex_starts(Zi, w, h_vec, specs[0]['fc'], specs[0]['kinds'], nh, ns, seed=self.seed)
        if starts is None:
            warnings.warn("complex_g: FFT grid of the bin box too large for the start search; using one start")
            return solve()
        best, self.starts_ = None, []
        self._init_Se = None
        for family, sc, e0 in starts:
            self._init_e = e0
            out = solve()
            ll = -n * out[5][-1] if out[5] and np.isfinite(out[5][-1]) else -np.inf    # sum_b w_b log f(z_b)
            self.starts_.append(dict(family=family, start_score=sc, loglik=float(ll), converged=bool(out[2]),
                                     n_sweeps=int(out[1])))
            if best is None or ll > best[0]:
                best = (ll, out, getattr(self, '_Se_end', None))
        self._init_e = None
        self._Se_end = best[2]
        return best[1]

    def _final_Se(self, e, sg):
        """The solver's final exact S e in the coefficients' space, shaped like e (None if not exact).
        Fixed signs: the solver works in t = |e| with kernel D S D, so S e = D (D S D t)."""
        Se = getattr(self, '_Se_end', None)
        self._Se_end = None
        if Se is None:
            return None
        Se = np.asarray(Se)
        if e.ndim == 1 and Se.ndim == 2:
            Se = Se[:, 0]
        return Se * sg if sg is not None else Se

    def _kernel_tab(self, Zspan, h_vec, fc_arr, d, want=()):
        """The value tables of the current lattice kernel's kinds and periods at cut-offs fc_arr for the bins Zspan
        (blml_kernels.ProductKernel, one per kinds / periods, cached there)."""
        from blml_kernels import ProductKernel
        kinds, periods, fold = self._fit_kernel.kinds, self._fit_kernel.periods, self._fit_kernel.pk.fold
        key = (tuple(kinds), tuple(periods), tuple(fold))
        pks = self.__dict__.setdefault('_pks', {})
        if key not in pks:
            pks[key] = ProductKernel(kinds, periods, fold)
        return pks[key].tables(np.broadcast_to(np.asarray(fc_arr, float), (d,)), periods, h_vec, Zspan, want)

    def _solve_weighted_cd(self, Zi, w, n, B, Z, h_vec, d):
        """Block-Jacobi CD for the weighted bin system (S e) * e = n w.

        This is the stationarity condition of the convex objective
            Phi(e) = 0.5 e'Se - n sum_b w_b log e_b,
        whose exact per-coordinate minimizer is the positive root of
        sbb*e_b^2 + b_term*e_b - n*w_b = 0. Se = S @ e is kept as a cached
        sum, so memory is O(B*Bs) and the B x B Gram is never formed.
        Se drifts slightly under incremental float32 updates, so it is
        recomputed exactly before convergence is declared.

        Returns (e, n_sweeps, converged, max_rho, rho_history, obj_history).
        """
        if not _HAS_NUMBA:
            raise RuntimeError("numba required (no Python fallback for mixed kernels)")
        pre0 = getattr(self, '_init_Se', None)       # tables must span removed bins too (their columns)
        Zspan = Zi if pre0 is None or not len(pre0[2]) else np.vstack([Zi, pre0[2]])
        tab = self._kernel_tab(Zspan, h_vec, self._fc_arr, d)
        RZ = rank_coords(Zi, tab)                     # the bins as the kernels read them, once
        sbb_const = float(self._fit_kernel.k0())
        Bs = min(32, B)  # fixed block size, O(B) memory
        block_buf = np.empty((B, Bs), dtype=self.dtype)

        def kernel_block(I, out):
            _local_cols(RZ, RZ[I], tab, out, ranked=True)
        sg = getattr(self, '_signs_cd', None)
        if sg is not None:   # fixed sign pattern: kernel DSD (see _fit_lattice)
            kb0 = kernel_block
            def kernel_block(I, out):
                kb0(I, out)
                out *= sg[:, None] * sg[I][None, :]

        def exact_Se(e):
            acc = np.zeros(B)
            buf = np.empty((B, min(512, B)), dtype=self.dtype)
            for s in range(0, B, buf.shape[1]):
                I = np.arange(s, min(s + buf.shape[1], B))
                kernel_block(I, buf[:, :len(I)])
                acc += buf[:, :len(I)].astype(np.float64) @ e[I].astype(np.float64)
            return acc.astype(self.dtype)

        def residual(e, Se):
            with np.errstate(divide='ignore', invalid='ignore'):
                rho = np.abs(Se * e / (n * w) - 1.0)
            rho = np.where((e > 0) & (w > 0), rho, 0.0)
            return float(np.max(rho)) if B > 0 else 0.0

        def objective(e, Se):
            e64 = e.astype(np.float64)
            return float(0.5 * e64 @ Se - n * (w * np.log(np.maximum(e64, _TINY))).sum()) / n

        rng = np.random.default_rng(self.seed)
        steps = 0.5 ** np.arange(0, 31)
        self._Se_end = None
        def kernel_cols(Zj, out):                      # S between the bins and other lattice points Zj
            _local_cols(RZ, rank_coords(Zj, tab), tab, out, ranked=True)
        e0w = getattr(self, '_init_e', None)          # warm start (init=): magnitudes
        new = None
        if e0w is not None and np.shape(e0w)[0] == B:
            a0 = np.abs(e0w) if np.ndim(e0w) == 1 else np.sqrt((np.abs(e0w) ** 2).sum(1))
            new = ~(a0 > 0)
            e = np.where(new, 0.0, a0).astype(self.dtype)
        else:
            e = np.ones(B, dtype=self.dtype)
        pre = getattr(self, '_init_Se', None) if (new is not None and sg is None) else None
        if pre is not None and not new.all():
            # previous fit's S e on the shared bins; kernel columns only for removed and new bins
            Se0, newp, Zi_rem, e_rem = pre
            Se = np.real(np.asarray(Se0)).astype(np.float64)
            cb = np.empty((B, min(512, max(len(Zi_rem), int(new.sum()), 1))), dtype=self.dtype)
            for s0 in range(0, len(Zi_rem), cb.shape[1]):
                Zr = np.ascontiguousarray(Zi_rem[s0:s0 + cb.shape[1]]); m = len(Zr)
                kernel_cols(Zr, cb[:, :m]); Se -= cb[:, :m].astype(np.float64) @ np.real(e_rem[s0:s0 + m]).astype(np.float64)
            newI = np.where(new)[0]
            for s0 in range(0, len(newI), cb.shape[1]):   # rows of the new bins (their e is 0 now)
                I = newI[s0:s0 + cb.shape[1]]; kernel_block(I, cb[:, :len(I)])
                Se[I] = cb[:, :len(I)].astype(np.float64).T @ e.astype(np.float64)
            bt_, cc = Se[newI], n * w[newI].astype(np.float64)
            disc = np.sqrt(bt_ * bt_ + 4.0 * sbb_const * cc)
            e[newI] = np.where(bt_ >= 0, 2.0 * cc / (bt_ + disc), (-bt_ + disc) / (2.0 * sbb_const))
            for s0 in range(0, len(newI), cb.shape[1]):   # their columns, now that they have e
                I = newI[s0:s0 + cb.shape[1]]; kernel_block(I, cb[:, :len(I)])
                Se += cb[:, :len(I)].astype(np.float64) @ e[I].astype(np.float64)
            Se = Se.astype(self.dtype)
        else:
            Se = exact_Se(e)
            if new is not None and new.any():
                # bins without a start (new bins, or g_prev ~ 0 there): exact 1-D minimizer given the others
                if new.all():
                    e = np.ones(B, dtype=self.dtype); Se = exact_Se(e)
                else:
                    bt_, cc = Se[new].astype(np.float64), n * w[new].astype(np.float64)
                    disc = np.sqrt(bt_ * bt_ + 4.0 * sbb_const * cc)
                    e_new = np.where(bt_ >= 0, 2.0 * cc / (bt_ + disc), (-bt_ + disc) / (2.0 * sbb_const))
                    newI = np.where(new)[0]; e[newI] = e_new
                    cb = np.empty((B, min(512, len(newI))), dtype=self.dtype)
                    Se = Se.astype(np.float64)
                    for s0 in range(0, len(newI), cb.shape[1]):   # add the new bins' columns, not a full pass
                        I = newI[s0:s0 + cb.shape[1]]; kernel_block(I, cb[:, :len(I)])
                        Se += cb[:, :len(I)].astype(np.float64) @ e[I].astype(np.float64)
                    Se = Se.astype(self.dtype)
        rho_hist, obj_hist = [], []
        n_sweeps = 0
        converged = False

        def sweep(e, Se, safeguard):
            perm = rng.permutation(B)
            for start in range(0, B, Bs):
                I = perm[start:start + Bs]
                m = len(I)
                kernel_block(I, block_buf[:, :m])
                Sblk = block_buf[:, :m]
                cc = n * w[I]
                e_star = cd_root(Se[I] - sbb_const * e[I], cc, sbb_const)
                d = (e_star - e[I]).astype(self.dtype)
                v = Sblk @ d
                if safeguard:   # see the GPU fit (blml_gpu)
                    d64 = d.astype(np.float64)
                    t = cd_line_search(e[I], d64, d64 @ Se[I].astype(np.float64), d64 @ v[I].astype(np.float64), cc, steps)
                    e[I] = e[I] + t * d
                    Se += (t * v).astype(self.dtype)
                else:
                    e[I] = e_star
                    Se += v

        # Plain parallel sweeps until one increases Phi; that sweep is redone
        # with the block-level safeguard, which then stays on (see the GPU fit).
        safeguard = False
        obj_prev = objective(e, Se)
        cap = getattr(self, 'max_sweeps', None)
        max_rho = residual(e, Se)
        while True:
            if cap is not None and n_sweeps >= cap:
                break
            n_sweeps += 1
            e_save, Se_save = e.copy(), Se.copy()
            sweep(e, Se, safeguard)
            obj_now = objective(e, Se)
            if not safeguard and obj_now > obj_prev + 1e-10 * abs(obj_prev):
                e, Se = e_save, Se_save
                safeguard = True
                sweep(e, Se, safeguard)
                obj_now = objective(e, Se)
            obj_prev = obj_now
            # Divergence guard: NaN e would mask to rho=0 and false-converge.
            if not np.all(np.isfinite(e)):
                max_rho = np.inf
                rho_hist.append(max_rho)
                break
            max_rho = residual(e, Se)
            if max_rho < self.tol:
                Se = exact_Se(e)
                max_rho = residual(e, Se)
            rho_hist.append(max_rho)
            obj_hist.append(objective(e, Se) if max_rho < self.tol else obj_now)
            if self.verbose:
                print(f"  sweep {n_sweeps}: max|rho|={max_rho:.2e} "
                      f"Phi/n={obj_hist[-1]:.6f}", flush=True)
            if max_rho < self.tol:
                converged = True
                break
            # Stall guard (not an iteration cap): stop only if the residual has
            # improved by < 1% over the last 50 sweeps, e.g. when tol is below
            # what the working precision can reach. converged_ stays False.
            if len(rho_hist) > 50 and rho_hist[-1] > 0.99 * min(rho_hist[:-50]):
                break
        if not converged and cap is not None and np.all(np.isfinite(e)):     # capped: normalise exactly
            Se = exact_Se(e)
            sc = n / np.sqrt(max(float(e.astype(np.float64) @ Se.astype(np.float64)), 1e-300))
            e, Se = (e * sc).astype(self.dtype), (Se * sc).astype(self.dtype)
            max_rho = residual(e, Se); converged = max_rho < self.tol
            obj_hist.append(objective(e, Se)); rho_hist.append(max_rho)
            self._Se_end = Se                          # exact (recomputed above)
        elif converged:
            self._Se_end = Se                          # exact: recomputed when the residual met tol
        return e, n_sweeps, converged, max_rho, rho_hist, obj_hist

    def _mode_kernels(self, modes, Zi, h_vec, d, Zspan=None):
        """Per-mode kernel machinery: [(kernel_block, sbb, predict_kernel)], one per mode spec
        {'fc', 'kinds', 'periods'}; every mode lives on the same bin lattice."""
        save = (self._fit_kernel, self._fc_arr, getattr(self, '_radial', None))
        out = []
        for md in modes:
            kinds = list(md.get('kinds') or ['sinc'] * d)
            periods = list(md.get('periods') or [None] * d)
            fc_arr = np.broadcast_to(np.asarray(md['fc'], dtype=float), (d,)).copy()
            check_radial(kinds, fc_arr)
            self._fit_kernel = lattice_kernel(fc_arr, h_vec, kinds, periods, d, self._fit_kernel.pk.fold)
            self._fc_arr = fc_arr
            tab = self._kernel_tab(Zi if Zspan is None else Zspan, h_vec, fc_arr, d)
            RZ = rank_coords(Zi, tab)

            def kz(Zj, buf, tab=tab, RZ=RZ):
                _local_cols(RZ, rank_coords(Zj, tab), tab, buf, ranked=True)
            kb = lambda I, buf, tab=tab, RZ=RZ: _local_cols(RZ, RZ[I], tab, buf, ranked=True)
            sbb = float(self._fit_kernel.k0())
            # (columns at bins I, sbb, exact kernel, columns at arbitrary lattice points Zj)
            out.append((kb, sbb, _continuous_product_kernel(fc_arr, kinds, periods, d), kz))
        self._fit_kernel, self._fc_arr, self._radial = save
        return out

    def _solve_modes_cd(self, Zi, w, n, B, Z, h_vec, d, modes, complex_g=False):
        """Rank-K BLML where mode k has its own kernel S_k (own kinds, periods, fc per dim):
        f = sum_k |g_k|^2, g_k = (1/n) sum_b e_bk K_k(., z_b), plane normalisation sum_k e_k^H S_k e_k = n^2.
        Stationarity: (S_k e_k)_b = n w_b e_bk / ||e_b||^2 for every k and b. Row update: e_bk =
        R_bk/(lam - S_k,bb) with lam the root of lam sum_k |R_bk|^2/(lam - S_k,bb)^2 = n w_b
        (closed form when all S_k,bb are equal). Block-Jacobi + Anderson (fixed order), switching
        to sequential updates if a parallel sweep lowers L. Start: real rank-1 solution of mode 0
        in column 0, small noise in the other columns (and phases if complex_g)."""
        if not _HAS_NUMBA:
            raise RuntimeError("numba required")
        K = len(modes)
        spec = lambda md: (tuple(np.broadcast_to(np.asarray(md['fc'], float), (d,))),
                           tuple(md.get('kinds') or ['sinc'] * d), tuple(md.get('periods') or [None] * d))
        shared = all(spec(md) == spec(modes[0]) for md in modes)   # one kernel for every mode
        pre0 = getattr(self, '_init_Se', None)       # tables must span removed bins too (their columns)
        Zspan = None if pre0 is None or not len(pre0[2]) else np.vstack([Zi, pre0[2]])
        mk = self._mode_kernels(modes[:1] if shared else modes, Zi, h_vec, d, Zspan)
        if shared:
            mk = mk * K
        self._mode_predict_kernels = [m[2] for m in mk]
        sbbs = np.array([m[1] for m in mk])
        Bs = min(32, B)
        bufs = np.empty((1 if shared else K, B, Bs), dtype=np.float64)
        cdt = complex if complex_g else np.float64
        w64 = w.astype(np.float64)

        def SE_of(e):
            acc = np.zeros((B, K), dtype=cdt)
            buf = np.empty((B, min(512, B)), dtype=np.float64)
            for s0 in range(0, B, buf.shape[1]):
                I = np.arange(s0, min(s0 + buf.shape[1], B))
                if shared:
                    mk[0][0](I, buf[:, :len(I)])
                    acc += buf[:, :len(I)] @ e[I]
                    continue
                for k in range(K):
                    mk[k][0](I, buf[:, :len(I)])
                    acc[:, k] += buf[:, :len(I)] @ e[I, k]
            return acc

        def loglik(e, Se):
            F = (np.abs(Se) ** 2).sum(1)
            return float((w64 @ np.log(np.maximum(F, 1e-300)) - n * np.log(max(np.real(np.vdot(e, Se)), 1e-300))) / n)

        def residual(e, Se):
            en2 = (np.abs(e) ** 2).sum(1)
            dev = Se * (en2 / (n * w64))[:, None] - e
            return float(np.max(np.sqrt((np.abs(dev) ** 2).sum(1) / np.maximum(en2, 1e-300)))) if B else 0.0

        def row_solve(R, cc):
            """Vectorised root of lam sum_k |R_k|^2/(lam - s_k)^2 = cc over rows of R (m, K)."""
            a2 = np.abs(R) ** 2
            smax = sbbs.max()
            lo = np.zeros(len(cc)); hi = np.ones(len(cc))
            phi = lambda x: (smax + x) * (a2 / (smax + x[:, None] - sbbs[None]) ** 2).sum(1)
            for _ in range(200):
                big = phi(hi) >= cc
                if not big.any():
                    break
                hi = np.where(big, hi * 2, hi)
            for _ in range(80):
                mid = 0.5 * (lo + hi); up = phi(mid) > cc
                lo = np.where(up, mid, lo); hi = np.where(up, hi, mid)
            lam = smax + 0.5 * (lo + hi)
            return R / (lam[:, None] - sbbs[None])

        rng = np.random.default_rng(self.seed)
        pn = getattr(self, 'phase_noise', 0.1)
        # real rank-1 start under mode 0's kernel
        save = (self._fit_kernel, self._fc_arr, getattr(self, '_radial', None), getattr(self, '_signs_cd', None))
        md0 = modes[0]
        self._fit_kernel = lattice_kernel(md0['fc'], h_vec, md0.get('kinds'), md0.get('periods'), d, self._fit_kernel.pk.fold)
        self._fc_arr = np.broadcast_to(np.asarray(md0['fc'], float), (d,)).copy(); self._signs_cd = None
        e0w = getattr(self, '_init_e', None)          # warm start (init=): previous coefficients
        if e0w is not None and np.shape(e0w)[0] == B:
            self._fit_kernel, self._fc_arr, self._radial, self._signs_cd = save
            e0w = np.asarray(e0w); e0w = e0w[:, None] if e0w.ndim == 1 else e0w
            e = np.zeros((B, K), dtype=cdt)
            kk = min(K, e0w.shape[1])
            e[:, :kk] = e0w[:, :kk] if complex_g else np.real(e0w[:, :kk])
            for k in range(kk, K):                    # modes the warm start lacks: small noise (zero would stay zero)
                noise = ((rng.standard_normal(B) + 1j * rng.standard_normal(B)) / np.sqrt(2) if complex_g
                         else rng.standard_normal(B))
                e[:, k] = pn * np.abs(e[:, 0]) * noise
        else:
            e_real = self._solve_weighted_cd(Zi, w, n, B, Z, h_vec, d)[0].astype(np.float64)
            self._fit_kernel, self._fc_arr, self._radial, self._signs_cd = save
            e = np.zeros((B, K), dtype=cdt)
            e[:, 0] = e_real * (np.exp(1j * pn * rng.standard_normal(B)) if complex_g else 1.0)
            for k in range(1, K):
                noise = ((rng.standard_normal(B) + 1j * rng.standard_normal(B)) / np.sqrt(2) if complex_g
                         else rng.standard_normal(B))
                e[:, k] = pn * e_real * noise
        self._Se_end = None
        pre = getattr(self, '_init_Se', None)
        if pre is not None and shared and e0w is not None and np.shape(e0w)[0] == B:
            # previous fit's S e on the shared bins (warm coefficients are zero at new bins)
            Se0, newp, Zi_rem, e_rem = pre
            Se = np.asarray(Se0, dtype=cdt).reshape(B, -1)[:, :K].copy()
            if len(Zi_rem):
                e_rem = np.asarray(e_rem).reshape(len(Zi_rem), -1)[:, :K]
            T0 = mk[0][3]
            cb = np.empty((B, 512))
            for s0 in range(0, len(Zi_rem), 512):
                Zr = np.ascontiguousarray(Zi_rem[s0:s0 + 512]); m = len(Zr)
                T0(Zr, cb[:, :m]); Se -= cb[:, :m] @ (e_rem[s0:s0 + m] if complex_g else np.real(e_rem[s0:s0 + m]))
            newI = np.where(newp)[0]
            for s0 in range(0, len(newI), 512):
                I = newI[s0:s0 + 512]; mk[0][0](I, cb[:, :len(I)]); Se[I] = cb[:, :len(I)].T @ e
        else:
            Se = SE_of(e)
        sc = n / np.sqrt(max(np.real(np.vdot(e, Se)), 1e-300))
        e *= sc; Se *= sc                              # S (c e) = c S e: no second pass
        if getattr(self, 'solver', 'cd') == 'lbfgs':
            e, Se, *out = lbfgs_psd(e, Se, w64, n, SE_of, loglik, residual, self.tol, getattr(self, 'max_sweeps', None),
                                    self.lbfgs_m, self.verbose, np)
            self._Se_end = Se
            return (e, *out)
        aa_m = int(getattr(self, 'anderson', 5))
        perm_fixed = rng.permutation(B) if aa_m else None

        def sweep_jacobi(e, Se):
            perm = perm_fixed if perm_fixed is not None else rng.permutation(B)
            for start in range(0, B, Bs):
                I = perm[start:start + Bs]; m = len(I)
                if shared:          # one kernel block, closed-form row update
                    mk[0][0](I, bufs[0, :, :m])
                    R = Se[I] - sbbs[0] * e[I]
                    ar = np.sqrt((np.abs(R) ** 2).sum(1)); cc = n * w64[I]
                    t = 2.0 * cc / (ar + np.sqrt(ar * ar + 4.0 * sbbs[0] * cc))
                    en = np.sqrt((np.abs(e[I]) ** 2).sum(1))
                    dirn = np.where((ar > 0)[:, None], R / np.where(ar > 0, ar, 1.0)[:, None],
                                    e[I] / np.maximum(en, 1e-300)[:, None])
                    dlt = t[:, None] * dirn - e[I]
                    e[I] += dlt
                    Se += bufs[0, :, :m] @ dlt
                    continue
                for k in range(K):
                    mk[k][0](I, bufs[k, :, :m])
                R = Se[I] - sbbs[None] * e[I]
                new = row_solve(R, n * w64[I])
                dlt = new - e[I]
                e[I] = new
                for k in range(K):
                    Se[:, k] += bufs[k, :, :m] @ dlt[:, k]

        def sweep_gs(e, Se):
            perm = perm_fixed if perm_fixed is not None else rng.permutation(B)
            er = np.ascontiguousarray(e.real, dtype=np.float64); sr = np.ascontiguousarray(Se.real, dtype=np.float64)
            ei = np.ascontiguousarray(e.imag, dtype=np.float64) if complex_g else np.zeros((B, K))
            si = np.ascontiguousarray(Se.imag, dtype=np.float64) if complex_g else np.zeros((B, K))
            for start in range(0, B, Bs):
                I = perm[start:start + Bs]; m = len(I)
                if shared:
                    mk[0][0](I, bufs[0, :, :m])
                    _psd_gs_block(I.astype(np.int64), bufs[0, :, :m], er, ei, sr, si, w64, float(n), sbbs[0])
                    continue
                for k in range(K):
                    mk[k][0](I, bufs[k, :, :m])
                _modes_gs_block(I.astype(np.int64), np.ascontiguousarray(bufs[:, :, :m]), sbbs, er, ei, sr, si, w64, float(n))
            e[:] = er + 1j * ei if complex_g else er
            Se[:] = sr + 1j * si if complex_g else sr

        def flat(a):
            return np.concatenate([a.real.ravel(), a.imag.ravel()]) if complex_g else a.ravel()
        use_gs = getattr(self, 'complex_update', 'jacobi') == 'gs'
        hG, hSG, hR = [], [], []
        self.aa_accepted_ = 0
        rho_hist, obj_hist = [], []
        n_sweeps, converged = 0, False
        L_prev = loglik(e, Se)
        cap = getattr(self, 'max_sweeps', None)
        max_rho = residual(e, Se)
        while True:
            if cap is not None and n_sweeps >= cap:
                break
            n_sweeps += 1
            x_prev = e.copy()
            if use_gs:
                sweep_gs(e, Se); L_now = loglik(e, Se)
            else:
                Se_save = Se.copy()
                sweep_jacobi(e, Se); L_now = loglik(e, Se)
                if L_now < L_prev - 1e-12 * abs(L_prev):
                    e[:], Se[:] = x_prev, Se_save
                    use_gs = True
                    sweep_gs(e, Se); L_now = loglik(e, Se)
            if aa_m:
                hG.append(e.copy()); hSG.append(Se.copy()); hR.append(flat(e - x_prev))
                if len(hG) > aa_m + 1:
                    hG.pop(0); hSG.pop(0); hR.pop(0)
                if len(hG) >= 2:
                    dR = np.stack([hR[j + 1] - hR[j] for j in range(len(hR) - 1)], 1)
                    gam = np.linalg.lstsq(dR, hR[-1], rcond=None)[0]
                    e_aa, Se_aa = hG[-1].copy(), hSG[-1].copy()
                    for j, gj in enumerate(gam):
                        e_aa -= gj * (hG[j + 1] - hG[j]); Se_aa -= gj * (hSG[j + 1] - hSG[j])
                    L_aa = loglik(e_aa, Se_aa) if np.all(np.isfinite(e_aa)) else -np.inf
                    if L_aa >= L_now:
                        e[:], Se[:] = e_aa, Se_aa; L_now = L_aa; self.aa_accepted_ += 1
                    else:
                        hG, hSG, hR = hG[-1:], hSG[-1:], hR[-1:]
            L_prev = L_now
            if not np.all(np.isfinite(e)):
                max_rho = np.inf; rho_hist.append(max_rho); break
            max_rho = residual(e, Se)
            if max_rho < self.tol:
                Se = SE_of(e); max_rho = residual(e, Se)
            rho_hist.append(max_rho); obj_hist.append(-L_now)
            if self.verbose and n_sweeps % 100 == 0:
                print(f"  sweep {n_sweeps}: max|rho|={max_rho:.2e} -L/n={-L_now:.6f}", flush=True)
            if max_rho < self.tol:
                converged = True; break
            if (len(rho_hist) > 500 and rho_hist[-1] > 0.99 * min(rho_hist[:-500])
                    and obj_hist[-501] - obj_hist[-1] < 1e-9 * abs(obj_hist[-1])):
                break
        if not converged and cap is not None and np.all(np.isfinite(e)):     # capped: normalise exactly
            Se = SE_of(e)
            sc = n / np.sqrt(max(float(np.real(np.vdot(e, Se))), 1e-300))
            e, Se = e * sc, Se * sc
            max_rho = residual(e, Se); converged = max_rho < self.tol
            obj_hist.append(-loglik(e, Se)); rho_hist.append(max_rho)
            self._Se_end = Se
        elif converged:
            self._Se_end = Se
        return e, n_sweeps, converged, max_rho, rho_hist, obj_hist

    def predict(self, Xq):
        """Density f(x) = ((1/n) sum_b e_b K(x, z_b))^2 with the exact kernel."""
        if not hasattr(self, "e_hat_"):
            raise RuntimeError("Call fit() before predict().")
        Xq = np.asarray(Xq, dtype=float)
        if Xq.ndim == 1:
            Xq = Xq[:, None]
        m = Xq.shape[0]
        B = self.Z_.shape[0]
        batch = max(1, min(m, int(2e7 // max(B, 1))))  # ~20M Gram entries max
        E = self.e_hat_ if self.e_hat_.ndim == 2 else self.e_hat_[:, None]
        v = np.empty(m)
        if getattr(self, 'modes_', None):
            for i in range(0, m, batch):
                v[i:i+batch] = sum(np.abs(pk(Xq[i:i+batch], self.Z_) @ E[:, k] / self.n_samples_) ** 2
                                   for k, pk in enumerate(self._mode_predict_kernels))
            return v
        for i in range(0, m, batch):
            Gb = self._predict_kernel(Xq[i:i+batch], self.Z_)
            v[i:i+batch] = (np.abs(Gb @ E / self.n_samples_) ** 2).sum(1)
        return v

    def sqrt_density(self, Xq):
        """g(x) = (1/n) sum_b e_b K(x, z_b): real (signed) or complex, shape (m,) or (m, K) for rank K;
        f = sum_k |g_k|^2."""
        Xq = np.asarray(Xq, dtype=float)
        if Xq.ndim == 1:
            Xq = Xq[:, None]
        e = self.e_hat_
        out = np.empty((len(Xq),) + e.shape[1:], dtype=np.result_type(e.dtype, np.float64))
        step = max(1, (1 << 22) // max(1, len(self.Z_)))           # ~32 MB kernel block, never (m, B) at once
        for s in range(0, len(Xq), step):
            out[s:s + step] = self._predict_kernel(Xq[s:s + step], self.Z_) @ e
        return out / self.n_samples_

    def param_grad(self, Xq, units, implicit_grad=False, eps_rel=0.0, mode='full', weights=None, acc=None):
        """d log f(x) / d log theta_u at the query points Xq for each unit u in `units`: a list of dims (its fc:
        one sinc dim, or a whole radial group) or a pair (dims, parameter name), e.g. ([1], 'period') for a
        dirichlet column's period. Coefficients fixed and the density renormalised:
        d/dlog theta_u [log sum_k |g_k(x)|^2 - log Z], Z = sum_k e_k^H S e_k / n^2 (= int f). Analytic, from
        the kernel classes (eval_dlog at Xq; lattice_dlog rows for the normaliser, through the solver's
        kernels). Shared-kernel fits only; dirichlet fc moves in whole harmonics (no gradient). Returns
        (m, len(units)).
        implicit_grad=True (real rank 1, no fixed signs): also the coefficients' one-sweep response
        (sweep_response), linearised at the current fit: the score on new points is not the training
        objective, so c's response matters. Costs no extra lattice pass.
        eps_rel > 0: the gradient of log(f + eps), eps = eps_rel * the median density at the bins: log f is
        singular where g crosses zero, and the few points in such lobes would otherwise dominate the mean.
        mode (pooling over batches): 'points' returns this batch's accumulator (weighted sums at Xq; no
        lattice derivative pass); 'lattice' takes a list of accumulators (acc), does the derivative passes
        once on this fit and returns the pooled weighted-mean gradient (U,).
        Per-mode kernels (each mode its own fc / kinds): a unit may be (mode, dims, name), one mode's parameter;
        conditional fits (dep columns): the gradient of log p(x_dep | x_rest) (conditional_param_grad).
        Pooling (mode != 'full') is for shared-kernel joint fits."""
        if getattr(self, '_joint', None) is not None:      # conditional: joint minus marginal
            kw = dict(implicit_grad=implicit_grad, eps_rel=eps_rel, weights=weights)
            if mode != 'full':
                raise ValueError("param_grad: pooling (mode='points' / 'lattice') is for joint fits")
            return conditional_param_grad(lambda X, u: self._joint.param_grad(X, u, **kw),
                                          lambda X, u: self._marg.param_grad(X, u, **kw), Xq, units, self.dep_, self.dim_)
        if getattr(self, 'modes_', None) is not None:      # per-mode kernels
            if implicit_grad or mode != 'full':
                raise ValueError("param_grad: per-mode kernels (rank K > 1) have no one-sweep response and no pooling")
            return param_grad_modes(self._Zi, self._h_vec, self.Z_, self.e_hat_, float(self.n_samples_),
                                    [(sp['fc'], sp['kinds'], sp['periods'], sp.get('fold')) for sp in self.modes_],
                                    Xq, units, eps_rel, weights)
        if implicit_grad and (np.iscomplexobj(self.e_hat_) or self.e_hat_.ndim != 1
                         or getattr(self, 'signs_', None) is not None):
            raise ValueError("param_grad(implicit_grad=True): real rank-1 fits without fixed signs")
        d, n, Z = self.dim_, float(self.n_samples_), self.Z_
        fc = np.broadcast_to(np.asarray(self._fc_arr, float), (d,)).copy()
        kinds = list(self.kinds_) if self.kinds_ is not None else ['sinc'] * d
        periods = list(self.periods_) if self.periods_ is not None else [None] * d
        allu = kernel_units(kinds, d)
        want = _want_params(allu, units)
        E = self.e_hat_ if self.e_hat_.ndim == 2 else self.e_hat_[:, None]
        E64 = E.astype(np.complex128 if np.iscomplexobj(E) else np.float64)
        # ---- normaliser and its derivatives on the lattice
        Zi, h = self._Zi, self._h_vec; B = len(Zi)
        quad = lambda SE: float(np.real(np.vdot(E64, SE))) / n ** 2
        wl = list(want) if mode != 'points' else []
        tab = self._kernel_tab(Zi, h, fc, d, wl)

        def mv(grad):                                  # S E and, with grad, (dS_u E) for every wanted unit: one pass
            parts = [E64.real, E64.imag] if np.iscomplexobj(E64) else [E64]
            out = []
            for P in parts:
                cols_ = [_local_rows(Zi, Zi, np.ascontiguousarray(P[:, k]), tab, grad=grad) for k in range(P.shape[1])]
                out.append((np.stack([c[0] for c in cols_], 1), np.stack([c[1] for c in cols_], 1)))
            if len(out) == 2:
                return out[0][0] + 1j * out[1][0], out[0][1] + 1j * out[1][1]
            return out[0]
        Se_ = getattr(self, 'Se_', None)
        if wl:
            SE_full, dS = mv(True)                    # dS: (B, K, nW)
        SE = (np.asarray(Se_, dtype=E64.dtype).reshape(E64.shape) if Se_ is not None
              else (SE_full if wl else mv(False)[0]))
        Zn = quad(SE)
        eps = float(eps_rel * np.median((np.abs(SE / n) ** 2).sum(1) / Zn)) if eps_rel else 0.0
        dSE = [dS[:, :, q] for q in range(len(wl))]
        dZ = [quad(x) for x in dSE]
        if mode == 'lattice':
            return pooled_fc_grad(self, acc, want, allu, fc, periods, E64, SE, Zn, dZ, dSE, n, implicit_grad)
        # ---- g and dg at the queries (exact kernel), in row chunks
        Xq = np.atleast_2d(np.asarray(Xq, dtype=float))
        m = len(Xq); U = len(want)
        wq = np.ones(m) if weights is None else np.asarray(weights, float)
        g = np.zeros((m,) + E64.shape[1:], dtype=E64.dtype); dg = np.zeros((U,) + g.shape, dtype=E64.dtype)
        step = max(1, int(2e7 // max(1, B * (len(allu) + 1))))
        for i0 in range(0, m, step):
            Xc = Xq[i0:i0 + step]
            diffs = [Xc[:, k, None] - Z[None, :, k] for k in range(d)]
            facs = [kernel_unit(kind, fc, dims, [diffs[k] for k in dims], periods, need_grad=False)[0]
                    for kind, dims in allu]
            G = np.prod(facs, axis=0)
            g[i0:i0 + step] = G @ E64 / n
            for u, (i, p) in enumerate(want):
                kind, dims = allu[i]
                dF = kernel_class(kind).eval_dlog(unit_theta(kind, fc, periods, dims), p, [diffs[k] for k in dims])
                dG = dF * np.prod([F for j, F in enumerate(facs) if j != i], axis=0)
                dg[u, i0:i0 + step] = dG @ E64 / n
            if implicit_grad:                               # v = d mean score / d e at the queries (kernel part)
                gc = g[i0:i0 + step, 0]
                v_acc = (v_acc if i0 else 0) + G.T @ (wq[i0:i0 + step] * 2.0 * gc / (n * (gc ** 2 + eps)))
        if mode == 'points':
            F = np.maximum((np.abs(g) ** 2).sum(1), 1e-300) + eps
            cont = np.array([wq @ (2 * np.real((np.conj(g) * dg[u]).sum(1)) / F) for u in range(U)])
            return dict(cont=cont, W=float(wq.sum()), v=(v_acc if implicit_grad else None), Zi=self._Zi.copy())
        out = fc_grad_from_parts(g, dg, Zn, dZ, eps)
        if implicit_grad:
            e = E64[:, 0]; Se = SE[:, 0]; w = np.asarray(self.w_, float)
            sbb, dsbb = _diag_and_dlog(allu, want, fc, periods)
            dT = sweep_response(e, Se, np.stack([dd[:, 0] for dd in dSE]), w, n, sbb, dsbb)
            out = out + (dT @ (v_acc / wq.sum() - 2.0 * Se / (n ** 2 * Zn)))[None, :]
        return out

    def fc_grad(self, Xq, units, **kw):
        """param_grad for the fc of each unit (a list of dims per unit)."""
        return self.param_grad(Xq, units, **kw)

    def kernel_matvec(self, X):
        """S @ X with s_bb' = K(z_b - z_b') on the fitted bins, without forming S: FFT convolution on the
        bin lattice (O(B log B)) when the lattice box is small, else row blocks of 512 (O(512 B) memory).
        X: (B,) or (B, K), real or complex. Shared-kernel fits only (not per-mode kernels)."""
        if getattr(self, 'modes_', None) is not None:
            raise ValueError("kernel_matvec: per-mode kernels; use one mode's kernel explicitly")
        X = np.asarray(X); Zi, h, B = self._Zi, self._h_vec, len(self._Zi)
        lo = Zi.min(0); L = Zi.max(0) - lo + 1
        if np.prod(2 * L) <= 1 << 24:
            d = Zi.shape[1]; shp = tuple(int(x) for x in 2 * L)
            offs = np.stack(np.meshgrid(*[np.r_[0:l, -l:0] for l in L], indexing='ij'), -1).reshape(-1, d)
            Kf = np.fft.fftn(self._fit_kernel(offs * h, np.zeros((1, d)))[:, 0].reshape(shp))
            idx = tuple((Zi - lo).T)
            Xc = X.reshape(B, -1); out = np.empty(Xc.shape, dtype=np.result_type(X.dtype, np.float64))
            for k in range(Xc.shape[1]):
                G = np.zeros(shp, dtype=complex); G[idx] = Xc[:, k]
                r = np.fft.ifftn(np.fft.fftn(G) * Kf)[idx]
                out[:, k] = r if np.iscomplexobj(out) else r.real
            return out.reshape(X.shape)
        Z = self.Z_; out = np.zeros(X.shape, dtype=np.result_type(X.dtype, np.float64))
        for s in range(0, B, 512):
            J = slice(s, min(s + 512, B))
            out += self._fit_kernel(Z, Z[J]) @ X[J]
        return out

    def certificate(self, return_vector=False):
        """Global-optimality certificate lambda_max(D^1/2 S D^1/2)/n, D = diag(sum_k |e_bk|^2 / w_b); <= 1 means
        the fit solves the convex PSD lifting (no sign pattern, complex g or sum of squares does better).
        Lanczos on kernel_matvec, never forms S. return_vector: also beta = D^1/2 y, the uphill direction
        (S D S beta = lambda S beta)."""
        from scipy.sparse.linalg import LinearOperator, eigsh
        C = self.e_hat_ if self.e_hat_.ndim == 2 else self.e_hat_[:, None]
        sd = np.sqrt((np.abs(C) ** 2).sum(1) / self.w_); B = len(sd)
        op = LinearOperator((B, B), matvec=lambda y: sd * self.kernel_matvec(sd * np.ravel(y)), dtype=np.float64)
        lam, V = eigsh(op, k=1, which='LA', tol=1e-8)
        c = float(lam[0] / self.n_samples_)
        return (c, sd * V[:, 0]) if return_vector else c

    def constraint_residual(self):
        """max |rho| recomputed from scratch (no cached sums)."""
        if not hasattr(self, "e_hat_"):
            raise RuntimeError("Call fit() before constraint_residual().")
        e, w, n, Z, B = self.e_hat_, self.w_, self.n_samples_, self.Z_, self.Z_.shape[0]
        Se = np.zeros(B)
        for s in range(0, B, 512):
            J = slice(s, min(s + 512, B))
            Se += self._fit_kernel(Z, Z[J]) @ e[J].astype(np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            rho = np.abs(Se * e / (n * w) - 1.0)
        return float(np.max(rho))

    def _clone(self):
        c = type(self)(tol=self.tol, batch_size=self.batch_size, seed=self.seed, verbose=self.verbose,
                       dtype=self.dtype, bin_scale=self.bin_scale, max_sweeps=self.max_sweeps)
        c.phase_noise, c.anderson, c.n_starts = self.phase_noise, self.anderson, self.n_starts
        return c

    def _fit_conditional(self, X, modes, dep, signs, complex_g, init=None, rebin_tol=1.0, sw=None):
        """Joint fit on all columns and marginal fit on the non-dep columns (same spec there); a
        conditional init warm-starts both."""
        if signs is not None:
            raise ValueError("signs with dep columns (the marginal's g has its own signs)")
        rest = [j for j in range(X.shape[1]) if j not in dep]
        if not rest:
            raise ValueError("dep covers every column; need at least one conditioning column")
        kw = dict(complex_g=complex_g, rebin_tol=rebin_tol, sample_weight=sw)
        joint = self._clone().fit(X, strip_dep(modes), init=getattr(init, '_joint', None), **kw)
        marg = self._clone().fit(X[:, rest], marginal_modes(modes, rest), init=getattr(init, '_marg', None), **kw)
        self.__dict__.update(joint.__dict__)
        self._joint, self._marg, self.dep_, self.spec_ = joint, marg, dep, modes
        return self

    def predict_marginal(self, Xrest):
        """Marginal density of the non-dep columns (fit with dep columns), at points in those columns."""
        if getattr(self, '_marg', None) is None:
            raise RuntimeError("fit with dep columns to get the marginal")
        return self._marg.predict(Xrest)

    def sample_conditional(self, Xrest, n_samples=100, seed=0, n_prop=400):
        """Samples of the dep columns given the rest (blml_trivial.sample_conditional)."""
        return sample_conditional(self, Xrest, n_samples, seed, n_prop)

    def predict_ratio(self, Xq):
        """Conditional density of the dep column(s) given the others: joint / marginal."""
        Xq = np.atleast_2d(np.asarray(Xq, dtype=float))
        rest = [j for j in range(Xq.shape[1]) if j not in self.dep_]
        return self.predict(Xq) / np.maximum(self.predict_marginal(Xq[:, rest]), 1e-300)

    def _fit_lattice(self, Z, w, h, modes, signs=None, complex_g=False, init=None):
        """BLML on counts recorded on a fixed lattice: bin centres Z on the lattice h * integers (e.g. camera
        pixels and photon counts). With c_b the common coefficient of the w_b points in bin b, the BLML
        stationarity condition c_i (S c)_i = n becomes, for every point in bin b, c_b (S diag(w) c)_b = n; in terms
        of e_b = w_b c_b this is the system (S e)_b e_b = n w_b, solved by the same block CD. g(x) = (1/n) sum_b e_b
        K(x, z_b); at the solution e'Se = n^2. No jitter and no re-binning: the lattice is the data's own (the
        paper's Nyquist condition h < 0.5/fc applies).
        Z: (B, d) bin centres; w: (B,) counts; h: (d,) steps. modes, complex_g: as in BLMLQuick.fit (e.g. a PSF mode at 2NA/lambda and a background
        mode at a small fc); dep columns: the joint on the counts and the marginal on them summed over the dep columns
        (marginal_counts), as _fit_conditional.
        signs: real rank-1 only; sign of g per bin (array of length B) or callable s(Z). With
        e = signs * t (t > 0) the objective is 0.5 t'(DSD)t - n sum w log t, D = diag(signs): the
        same convex problem with kernel s_ij * signs_i * signs_j. Needed when g = sqrt(f) must change
        sign (e.g. a real far-field amplitude h).
        Bins with w = 0 are dropped before solving (they do not enter the likelihood); Z, w and an
        array of signs all have the caller's length B.
        init: a previous fit on counts (e.g. the previous EM iteration) starts the solver (warm
        start; complex fits then skip the start search): its coefficients on the same occupied bins,
        otherwise e_b = w_b g(z_b) / |g(z_b)|^2 from its g (exact at its optimum)."""
        lattice_fixed(modes, 'a fixed lattice (h in the modes)')
        modes = strip_lattice(modes)
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        w = np.asarray(w, dtype=float)
        keep = w > 0
        B0, d = Z.shape
        specs, dep = parse_modes(modes, d)
        if dep is not None:
            if signs is not None:
                raise ValueError("signs with dep columns (the marginal's g has its own signs)")
            rest = [j for j in range(d) if j not in dep]
            if not rest:
                raise ValueError("dep covers every column; need at least one conditioning column")
            h_vec = np.broadcast_to(np.asarray(h, dtype=float), (d,))
            Zm, wm = marginal_counts(Z, w, rest)
            joint = self._clone()._fit_lattice(Z, w, h_vec, strip_dep(modes), complex_g=complex_g,
                                               init=getattr(init, '_joint', None))
            marg = self._clone()._fit_lattice(Zm, wm, h_vec[rest], marginal_modes(modes, rest), complex_g=complex_g,
                                              init=getattr(init, '_marg', None))
            self.__dict__.update(joint.__dict__)
            self._joint, self._marg, self.dep_, self.spec_ = joint, marg, dep, modes
            return self
        K, shared = len(specs), _shared_kernel(specs)
        self.spec_ = modes
        psd = complex_g or K > 1
        if psd and signs is not None:
            raise ValueError("complex_g / rank > 1 estimate the phases; do not pass signs as well")
        sg = resolve_signs(signs, Z)
        sg = None if sg is None else sg[keep]
        Z, w = Z[keep], w[keep].astype(self.dtype)
        B = Z.shape[0]
        n = float(w.sum())
        kinds, periods, fc_arr = specs[0]['kinds'], specs[0]['periods'], specs[0]['fc']
        h_vec = np.broadcast_to(np.asarray(h, dtype=float), (d,)).copy()
        self.kinds_, self.periods_, self._fc_arr = kinds, periods, fc_arr
        self._fit_kernel = lattice_kernel(fc_arr, h_vec, kinds, periods, d, lattice_fold_specs(specs))
        self._predict_kernel = _continuous_product_kernel(fc_arr, kinds, periods, d)
        Zi = np.round(Z / h_vec).astype(np.int64)
        if np.abs(Zi * h_vec - Z).max() > 1e-6 * h_vec.max():
            raise ValueError("bin centers must lie on the lattice h * integers")
        self._h_vec, self._Zi = h_vec, Zi
        self._signs_cd = None if sg is None else sg.astype(self.dtype)
        same = init is not None and np.allclose(getattr(init, '_h_vec', np.nan), h_vec)
        self._init_e = warm_start_e(init, Z, Zi, w, same_lattice=same)
        self._init_Se = warm_start_Se(init, Zi, fc_arr, kinds, periods, h_vec) if (init is not None and same) else None
        if init is not None and self._init_e is None:
            warnings.warn("counts: init ignored (different bins and no sqrt_density to evaluate it on them)")
        self.complex_g_, self.rank_ = bool(complex_g), K
        self.modes_ = None if shared else specs
        if psd:
            out = self._solve_psd(Zi, w, n, B, Z, h_vec, d, specs, bool(complex_g))
        else:
            out = self._solve_weighted_cd(Zi, w, n, B, Z, h_vec, d)
        e, n_sweeps, converged, max_rho, rho_hist, obj_hist = out
        if sg is not None:
            e = (e * self._signs_cd).astype(self.dtype)
        self._init_e = self._init_Se = None
        self.Se_ = self._final_Se(e, sg)
        self.signs_ = sg
        self.e_hat_, self.Z_, self.w_ = e, Z, w
        self.n_samples_, self.n_bins_, self.dim_ = n, B, d
        self.n_sweeps_, self.converged_, self.residual_ = n_sweeps, converged, max_rho
        self.rho_history_, self.obj_history_ = rho_hist, obj_hist
        return self

    def __repr__(self):
        r = getattr(self, 'rank_', None)
        return f"{type(self).__name__}(tol={self.tol}" + (f", rank={r}, fc={self._fc_arr})" if r else ")")




def sampled_relmse(fh, ft):
    """Importance-sampled relative MSE.
    
    Assumes test points x_i were sampled i.i.d. from the TRUE density f.
    Estimates ∫(f̂-f)²dx / ∫f²dx via:
        MSE = mean((f̂_i - f_i)² / f_i)
        relMSE = MSE / mean(f_i)
    
    This is the correct Monte Carlo estimator of the L2 error. Sampling
    from f (not uniform) puts points where mass is, avoiding wasted
    samples in empty high-d regions. No trimming needed.
    """
    fh = np.asarray(fh, dtype=np.float64)
    ft = np.asarray(ft, dtype=np.float64)
    # Guard against tiny ft (shouldn't happen often since x_i ~ f)
    ft_safe = np.maximum(ft, 1e-300)
    mse = np.mean((fh - ft) ** 2 / ft_safe)
    denom = np.mean(ft)  # estimates ∫f²
    return mse / denom if denom > 0 else np.inf
