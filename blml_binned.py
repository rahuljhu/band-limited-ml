"""Full-lattice binned BLML: BLMLBinned (one-shot) and BLMLBinnedStreaming (counts arriving in batches).

The data are a histogram on a dense regular lattice (images, stacks, photon-counting frames): counts w on every
point z_b = origin + h * index of a box of shape (m_1, ..., m_d). Every lattice point is a bin; an empty bin has
w_b = 0 and so e_b = 0 at every stationary point (the coordinate update gives 0 there), the same model as
BLMLQuick on the occupied bins (with_lattice). On the full lattice S_ab = K(z_a - z_b) is a convolution, so
S e costs one FFT pair (zero-padded to >= 2m - 1 per dim; a dirichlet dim whose box is exactly one period is
circular). The solver is the coordinate update of BLMLQuick applied to every bin at once (Jacobi), accelerated
by Anderson mixing and safeguarded by a merit function; nothing of size B x B (or B x block) is ever formed.

    m = BLMLBinned().fit(counts, make_modes([fc, fc], ['ball', 'ball']), h=1.0, complex_g=True)
    m.density_         f at the bins (B,)-shaped like counts; m.predict(X) / m.sqrt_density(X) anywhere
    m.predict_grid(r)  f on the lattice refined r times per dim (r^d FFT pairs)
    Stops at max|rho| < tol, after max_sweeps, after `stall` sweeps without progress in the residual or the merit
    (the round-off floor), or, with ll_tol set, when the merit gained less than ll_tol per count over the last
    ll_window sweeps (large lattices: max|rho| is set by a few slowly relaxing bins long after the likelihood has
    converged); converged_ (max|rho| < tol) and stop_reason_ say which.
    m.certificate()    lambda_max(D^1/2 S D^1/2) / n (shared kernel; <= 1 + tol: global optimum of the convex lift)
    m.fc_grad()        d (mean log-likelihood) / d log fc per mode and unit, at the fitted e (envelope theorem)

    s = BLMLBinnedStreaming(decay=1.0); s.partial_fit(frame, modes) ...   (warm-started; same state as BLMLBinned)
    e = BLMLEmpirical(reg='ridge', lam=1e-2).fit(counts, modes)   (paper 7: normalised on the observed lattice,
        S c = n sqrt(h) regularised; CG / FISTA with the same FFT mat-vec; BLMLEmpirical.select for lam)

Update rules (c_b = n w_b, s_k = K_k(0), R_bk = (S_k e_k)_b - s_k e_bk):
  real rank 1 (positive root, or fixed signs sigma): e_b = sigma_b t, s t^2 + sigma_b R_b t - c_b = 0 (cd_root);
    merit Phi = e'Se/2 - sum c log|e| (convex for fixed signs).
  complex and/or rank K: e_bk = R_bk / (lam_b - s_k), lam_b the root above max_k s_k of
    lam sum_k |R_bk|^2 / (lam - s_k)^2 = c_b (closed form t = 2c/(|R| + sqrt(|R|^2 + 4 s c)) when the s_k are equal);
    merit: the scale-free log-likelihood L = sum_b c_b log F_b - n log Q, F_b = sum_k |(S_k e_k)_b|^2, Q = sum_k e_k^H S_k e_k.
  After every sweep e is rescaled to Q = n^2. Converged when max_b rho_b < tol,
    rho_b = || (S e)_b ||e_b||^2 / c_b - e_b || / ||e_b|| (rank 1: |e_b conj((Se)_b) / c_b - 1|).
Engines: 'cpu' (numpy + scipy.fft, threads) and 'gpu' (cupy + cuFFT); dtype float64 or float32 (complex64)."""
import numpy as np
from blml_trivial import parse_modes, kernel_units, kernel_class, unit_theta, cd_root

__all__ = ['BLMLBinned', 'BLMLBinnedStreaming', 'BLMLEmpirical', 'BLMLEmpiricalStreaming']


def _backend(engine):
    if engine == 'gpu':
        import cupy as cp
        import cupyx.scipy.fft as cfft
        return cp, cfft
    import scipy.fft as sfft
    return np, sfft


class _ModeKernel:
    """One mode's kernel on the padded lattice: the offset table, its FFT, K(0) and (on request) dK/dlog fc tables."""

    def __init__(self, spec, shape, h, pad, xp, fft, rdt, workers):
        self.spec, self.shape, self.h, self.pad = spec, tuple(shape), np.asarray(h, float), tuple(pad)
        self.xp, self.fft, self.rdt, self.workers = xp, fft, rdt, workers
        self.units = kernel_units(spec['kinds'], len(shape))
        diffs, mask = self._offsets()
        self._diffs, self._mask = diffs, mask
        T = self._table(diffs, mask)
        self.s = float(T[(0,) * len(shape)])
        self.Kf = self._ffts(T)
        self._dKf = None

    def _offsets(self):
        """Per dim: the offset (in coordinate units) of each padded index, broadcastable to the padded grid, and
        the indicator of the offsets the convolution reads (|j| <= m - 1 for linear dims; all for circular)."""
        d = len(self.shape); diffs, masks = [], []
        for j, (m, L) in enumerate(zip(self.shape, self.pad)):
            idx = np.arange(L); off = np.where(idx < (L + 1) // 2, idx, idx - L).astype(float)
            mk = np.ones(L) if L == m else (np.abs(off) <= m - 1).astype(float)
            sh = [1] * d; sh[j] = L
            diffs.append((off * self.h[j]).reshape(sh)); masks.append(mk.reshape(sh))
        return diffs, masks

    def _unit_vals(self, kind, dims, diffs, dlog=False):
        cls = kernel_class(kind); th = unit_theta(kind, self.spec['fc'], self.spec['periods'], dims)
        dd = [diffs[j] for j in dims]
        return cls.eval_dlog(th, 'fc', dd) if dlog else cls.eval(th, dd)

    def _table(self, diffs, mask, dlog_unit=None):
        T = 1.0
        for u, (kind, dims) in enumerate(self.units):
            T = T * self._unit_vals(kind, dims, diffs, dlog=(u == dlog_unit))
        for mk in mask:
            T = T * mk
        return np.broadcast_to(T, self.pad).astype(np.float64)

    def _ffts(self, T):
        xp = self.xp
        return self.fft.rfftn(xp.asarray(T, dtype=self.rdt), s=self.pad, workers=self.workers) if xp is np else \
            self.fft.rfftn(xp.asarray(T, dtype=self.rdt), s=self.pad)

    def dKf(self):
        """FFTs of dK/dlog fc_u, one per unit (dirichlet units: None, fc moves in whole harmonics)."""
        if self._dKf is None:
            out = []
            for u, (kind, dims) in enumerate(self.units):
                if kernel_class(kind).params['fc'][0] == 'integer':
                    out.append(None); continue
                out.append(self._ffts(self._table(self._diffs, self._mask, dlog_unit=u)))
            self._dKf = out
        return self._dKf


class BLMLBinned:
    """BLML on counts on a full regular lattice (see the module docstring)."""

    def __init__(self, tol=1e-5, max_sweeps=None, anderson=5, engine='cpu', dtype=np.float64, phase_noise=0.1,
                 seed=0, workers=-1, verbose=False, stall=1000, ll_tol=None, ll_window=100, solver='cd', lbfgs_m=20):
        if solver not in ('cd', 'lbfgs'):
            raise ValueError("solver: 'cd' (coordinate / Jacobi updates with Anderson mixing) or 'lbfgs'")
        self.solver, self.lbfgs_m = solver, int(lbfgs_m)
        self.tol, self.max_sweeps, self.anderson = tol, max_sweeps, anderson
        self.engine, self.dtype, self.phase_noise, self.seed = engine, np.dtype(dtype), phase_noise, seed
        self.workers, self.verbose, self.stall = workers, verbose, stall
        self.ll_tol, self.ll_window = ll_tol, ll_window
        self.xp, self.fft = _backend(engine)

    # ---------------------------------------------------------------- setup
    def _setup(self, counts, modes, h, origin):
        xp = self.xp
        counts = np.asarray(counts.get() if hasattr(counts, 'get') else counts)
        if counts.ndim < 1 or np.any(counts < 0) or not np.all(np.isfinite(counts)):
            raise ValueError("counts: a non-negative finite array on the lattice, one axis per dimension")
        d = counts.ndim; shape = counts.shape
        specs, dep = parse_modes(modes, d)
        if dep is not None:
            raise ValueError("BLMLBinned: conditional (dep) modes are not supported; fit the joint")
        self.h_ = np.broadcast_to(np.asarray(h, float), (d,)).copy()
        self.origin_ = np.broadcast_to(np.asarray(origin, float), (d,)).copy()
        pad = []
        for j in range(d):                       # circular only if every mode is periodic in dim j (dirichlet, or a
            circ = all(sp['kinds'][j].split(':')[0] in ('dirichlet', 'pball') and sp['periods'][j] is not None and  # pball column) with period m h
                       abs(shape[j] * self.h_[j] - sp['periods'][j]) <= 1e-9 * sp['periods'][j] for sp in specs)
            pad.append(shape[j] if circ else int(self.fft.next_fast_len(2 * shape[j] - 1, real=True)
                                                 if xp is np else _next_fast_len(2 * shape[j] - 1)))
        self.pad_ = tuple(pad); self.shape_ = shape; self.d_ = d; self.specs_ = specs; self.modes_ = modes
        self.rdt_ = np.float32 if self.dtype == np.float32 else np.float64
        self.cdt_ = np.complex64 if self.dtype == np.float32 else np.complex128
        self._build_kernels()
        self.K_ = len(specs)

    def _build_kernels(self):
        """Kernel tables of every mode from self.specs_ (modes with the same kernel share one table)."""
        key = lambda sp: (tuple(sp['fc']), tuple(sp['kinds']), tuple(sp['periods']))
        cache, old = {}, {key(kk.spec): kk for kk in getattr(self, 'kern_', [])}
        self.kern_ = []
        for sp in self.specs_:
            k = key(sp)
            if k not in cache:
                cache[k] = old[k] if k in old else _ModeKernel(dict(sp, fc=np.array(sp['fc'])), self.shape_, self.h_,
                                                               self.pad_, self.xp, self.fft, self.rdt_, self.workers)
            self.kern_.append(cache[k])
        self.shared_ = len(cache) == 1
        self.s_ = np.array([kk.s for kk in self.kern_])

    def _S(self, k, e):
        """S_k e for one mode: e has the lattice shape (real or complex)."""
        xp, fft, kk = self.xp, self.fft, self.kern_[k]
        sl = tuple(slice(0, m) for m in self.shape_)
        kw = dict(workers=self.workers) if xp is np else {}
        if xp.iscomplexobj(e):
            re = fft.irfftn(fft.rfftn(e.real, s=self.pad_, **kw) * kk.Kf, s=self.pad_, **kw)[sl]
            im = fft.irfftn(fft.rfftn(e.imag, s=self.pad_, **kw) * kk.Kf, s=self.pad_, **kw)[sl]
            return re + 1j * im
        return fft.irfftn(fft.rfftn(e, s=self.pad_, **kw) * kk.Kf, s=self.pad_, **kw)[sl]

    def _Sall(self, E):
        return self.xp.stack([self._S(k, E[k]) for k in range(self.K_)])

    # ---------------------------------------------------------------- merit, update, residual
    def _merit(self, E, U):
        """Larger is better. psd: scale-free log-likelihood / n; real rank 1: -Phi / n."""
        xp, c, occ, n = self.xp, self.c_, self.occ_, self.n_
        if self.psd_:
            F = (xp.abs(U) ** 2).sum(0); Q = float(xp.real(xp.sum(xp.conj(E) * U, dtype=self._acc(E))))
            if not Q > 0:
                return -np.inf
            return float((c * xp.log(xp.maximum(F, 1e-30))).sum(dtype=np.float64)) / n - n * np.log(Q)   # c = 0 off occ
        e, u = E[0], U[0]
        if bool(xp.any(occ & (self.sg_ * e <= 0))):
            return -np.inf
        return -(0.5 * float(xp.sum(e * u, dtype=np.float64)) - float((c * xp.log(xp.where(occ, xp.abs(e), 1.0))).sum(dtype=np.float64))) / n

    def _update(self, E, U):
        """The coordinate update of every bin from the current U = S E (Jacobi)."""
        xp, c = self.xp, self.c_
        s = xp.asarray(self.s_, dtype=self.rdt_).reshape((-1,) + (1,) * self.d_)
        R = U - s * E
        if not self.psd_:
            sg = self.sg_
            t = xp.where(self.occ_, _cd_root(xp, sg * R[0], c, float(self.s_[0])), 0.0)   # empty bins stay 0
            return (sg * t)[None].astype(E.dtype)
        a2 = (xp.abs(R) ** 2).sum(0); a = xp.sqrt(a2)
        if np.allclose(self.s_, self.s_[0]):
            s0 = float(self.s_[0]); t = 2 * c / (a + xp.sqrt(a2 + 4 * s0 * c))
            out = R * (t / xp.where(a > 0, a, 1.0))[None]
            lone = (a == 0) & (c > 0)                         # no field from the others: mode 0 at the root
            if bool(xp.any(lone)):
                out[0] = xp.where(lone, xp.sqrt(c / s0), out[0])
            return out.astype(E.dtype)
        lone = (a2 == 0) & (c > 0)
        lam = _secular(xp, xp.abs(R) ** 2, s, c)
        out = xp.where((c > 0)[None] & ~lone[None], R / (lam[None] - s), 0)
        if bool(xp.any(lone)):
            out[0] = xp.where(lone, xp.sqrt(c / float(self.s_[0])), out[0])
        return out.astype(E.dtype)

    def _acc(self, E):
        return np.complex128 if self.xp.iscomplexobj(E) else np.float64

    def _rescale(self, E, U):
        Q = float(self.xp.real(self.xp.sum(self.xp.conj(E) * U, dtype=self._acc(E))))
        sc = float(self.n_ / np.sqrt(Q)) if Q > 0 else 1.0
        return E * sc, U * sc

    def _resid(self, E, U):
        xp, c, occ = self.xp, self.c_, self.occ_
        nrm2 = (xp.abs(E) ** 2).sum(0)
        r = xp.sqrt((xp.abs(U * (nrm2 / xp.where(occ, c, 1.0))[None] - E) ** 2).sum(0)) / xp.sqrt(xp.maximum(nrm2, 1e-300))
        return float(xp.max(xp.where(occ, r, 0.0)))

    # ---------------------------------------------------------------- solver
    def _solve(self, E, max_sweeps):
        if self.solver == 'lbfgs' and self.psd_:
            return self._solve_lbfgs(E, max_sweeps)
        xp = self.xp
        U = self._Sall(E); E, U = self._rescale(E, U)
        cur = self._merit(E, U); hX, hF = [], []; mA = self.anderson
        n_and = n_damp = n_grad = 0; it = 0; res = self._resid(E, U); hist = [res]
        cap = np.inf if max_sweeps is None else max_sweeps
        best_res, best_it, best_m = res, 0, cur
        mhist = [cur]; self.stop_reason_ = 'tol'
        while res >= self.tol and it < cap:
            if it - best_it >= self.stall:           # no progress in residual or merit: round-off floor, stop
                self.stop_reason_ = 'stall'; break
            if self.ll_tol is not None and len(mhist) > self.ll_window and \
                    (mhist[-1] - mhist[-1 - self.ll_window]) / self.n_ < self.ll_tol:
                self.stop_reason_ = 'll_tol'; break
            it += 1
            EJ = self._update(E, U); UJ = self._Sall(EJ); mJ = self._merit(EJ, UJ)
            if mA:
                f = (EJ - E).ravel()
                hX.append(E.ravel()); hF.append(f); hX, hF = hX[-(mA + 1):], hF[-(mA + 1):]
                if len(hX) >= 3:
                    try:
                        EA = E + f.reshape(E.shape) - _anderson_mix(xp, hX, hF, f).reshape(E.shape)
                        UA = self._Sall(EA); mAc = self._merit(EA, UA)
                        if mAc > mJ:
                            EJ, UJ, mJ = EA, UA, mAc; n_and += 1
                    except Exception:                     # singular history: plain step
                        hX, hF = [], []
            if mJ < cur - 1e-12 * abs(cur):                # no ascent: damp the Jacobi step
                hX, hF = [], []; n_damp += 1
                EP = self._update(E, U); UP = self._Sall(EP); a = 0.5
                while a > 1e-8:
                    EJ, UJ = E + a * (EP - E), U + a * (UP - U); mJ = self._merit(EJ, UJ)
                    if mJ >= cur:
                        break
                    a *= 0.5
                else:
                    EJ, UJ, mJ = E, U, cur
                    if self.psd_:                          # the aligned update is no ascent direction here:
                        EJ, UJ, mJ = self._grad_steps(E, U, cur)   # gradient ascent on L (always one, unless stationary)
                        n_grad += 1
            E, U = self._rescale(EJ, UJ); cur = mJ if self.psd_ else self._merit(E, U); mhist.append(cur)   # psd: scale-free
            res = self._resid(E, U); hist.append(res)
            if res < 0.9 * best_res or cur > best_m + 1e-13 * abs(best_m):
                best_res, best_it, best_m = min(res, best_res), it, max(cur, best_m)
            if self.verbose and it % 100 == 0:
                print(f'  sweep {it}: max|rho|={res:.2e} merit={cur:.9f} anderson={n_and} damped={n_damp} gradient={n_grad}', flush=True)
        if res >= self.tol and self.stop_reason_ == 'tol':
            self.stop_reason_ = 'max_sweeps'
        self.n_sweeps_, self.residual_, self.converged_ = it, res, res < self.tol
        self.merit_history_ = mhist
        self.rho_history_, self.n_anderson_, self.n_damped_, self.n_gradient_ = hist, n_and, n_damp, n_grad
        return E, U

    def _lgrad(self, E, U):
        """L = sum_b w_b log F_b - n log Q (the psd merit) and its gradient in the real inner product Re <A, B>,
        2 (S_k (w u_k / F) - n u_k / Q), zero on empty bins (their e stays 0)."""
        xp = self.xp
        F = (xp.abs(U) ** 2).sum(0); Q = float(xp.real(xp.sum(xp.conj(E) * U, dtype=self._acc(E))))
        if not Q > 0:
            return -np.inf, None
        L = float((self.w_ * xp.log(xp.where(self.occ_, F, 1.0))).sum(dtype=np.float64)) - self.n_ * np.log(Q)
        V = xp.where(self.occ_, self.w_ / xp.where(self.occ_, F, 1.0), 0.0)[None] * U
        G = self._Sall(V) - (self.n_ / Q) * U
        return L, 2 * xp.where(self.occ_[None], G, 0)

    def _solve_lbfgs(self, E, max_iter):
        """psd fits (complex / rank K): L-BFGS ascent of L on the coefficients (real and imaginary parts), memory
        lbfgs_m, Armijo backtracking from the unit step (the first step scaled to 1e-3 of |E|); a pair enters the memory
        only with positive curvature, and a failed line search clears the memory once (a gradient step) before
        stopping as 'stall'. Stops at max|rho| < tol, when L / n gains less than ll_tol (default 1e-11) over 10
        iterations ('ll_tol': the residual of the flat directions, which the likelihood does not see, can stay above
        tol), or after max_iter iterations ('max_sweeps'). Each iteration costs one gradient (two FFT pairs per mode) per trial step."""
        xp = self.xp; n = self.n_; m_ = self.lbfgs_m
        U = self._Sall(E); E, U = self._rescale(E, U)
        L, g = self._lgrad(E, U)
        S_, Y_, R_ = [], [], []; it = 0; res = self._resid(*self._rescale(E, U)); hist = [res]; mhist = [L / n]   # rho at Q = n^2
        cap = np.inf if max_iter is None else max_iter; self.stop_reason_ = 'tol'; reset = False
        ftol, fwin = (self.ll_tol if self.ll_tol is not None else 1e-11), 10   # L / n gain over 10 iterations
        while res >= self.tol and it < cap:
            if len(mhist) > fwin and mhist[-1] - mhist[-1 - fwin] < ftol:
                self.stop_reason_ = 'll_tol'; break
            it += 1
            q = g.copy(); al = []
            for s_, y_, r_ in zip(reversed(S_), reversed(Y_), reversed(R_)):
                a = r_ * _rdot(xp, s_, q); al.append(a); q = q - a * y_
            if S_:
                q = q * (_rdot(xp, S_[-1], Y_[-1]) / _rdot(xp, Y_[-1], Y_[-1]))
            else:
                q = q * (1e-3 * np.sqrt(_rdot(xp, E, E) / max(_rdot(xp, g, g), 1e-300)))
            for (s_, y_, r_), a in zip(zip(S_, Y_, R_), reversed(al)):
                b = r_ * _rdot(xp, y_, q); q = q + (a - b) * s_
            slope = _rdot(xp, g, q)                                   # ascent: q is the direction
            if not slope > 0:
                S_, Y_, R_ = [], [], []; q = g * (1e-3 * np.sqrt(_rdot(xp, E, E) / max(_rdot(xp, g, g), 1e-300))); slope = _rdot(xp, g, q)
            t = 1.0; ok = False
            for _ in range(40):
                En = E + t * q; Un = self._Sall(En); Ln, gn = self._lgrad(En, Un)
                if Ln >= L + 1e-4 * t * slope:
                    ok = True; break
                t *= 0.5
            if not ok:
                if reset or not S_:
                    self.stop_reason_ = 'stall'; break
                S_, Y_, R_ = [], [], []; reset = True; continue
            reset = False
            s_, y_ = En - E, g - gn                                   # y = -(change of the ascent gradient)
            sy = _rdot(xp, s_, y_)
            if sy > 1e-12 * np.sqrt(_rdot(xp, s_, s_) * _rdot(xp, y_, y_)):
                S_.append(s_); Y_.append(y_); R_.append(1.0 / sy)
                if len(S_) > m_:
                    S_.pop(0); Y_.pop(0); R_.pop(0)
            E, U, L, g = En, Un, Ln, gn
            res = self._resid(*self._rescale(E, U)); hist.append(res); mhist.append(L / n)
            if self.verbose and it % 100 == 0:
                print(f'  lbfgs {it}: max|rho|={res:.2e} L/n={L / n:.9f}', flush=True)
        if res >= self.tol and self.stop_reason_ == 'tol':
            self.stop_reason_ = 'max_sweeps'
        # the coefficients in their canonical form: at a stationary point e_k = (Q / n) (w / F) u_k on the occupied
        # bins (Theorem 1), a function of the field alone; L-BFGS leaves e's components in the near-null space of S
        # wherever the start put them, and the fc gradients (d(S e)/dfc) and the certificate read e itself
        F = (xp.abs(U) ** 2).sum(0); Q = float(xp.real(xp.sum(xp.conj(E) * U, dtype=self._acc(E))))
        Ec = xp.where(self.occ_, (Q / n) * self.w_ / xp.where(self.occ_, F, 1.0), 0.0)[None] * U
        Uc = self._Sall(Ec); Lc, _ = self._lgrad(Ec, Uc)
        if Lc >= L - 1e-9 * n:                                        # kept unless it loses likelihood (not stationary)
            E, U = Ec, Uc
        E, U = self._rescale(E, U); res = self._resid(E, U)
        self.n_sweeps_, self.residual_ = it, res
        self.converged_ = res < self.tol or self.stop_reason_ == 'll_tol'   # ll_tol: converged in the likelihood
        self.merit_history_ = mhist; self.rho_history_ = hist
        self.n_anderson_ = self.n_damped_ = self.n_gradient_ = 0
        return E, U

    def _grad_steps(self, E, U, cur, steps=5):
        """Fallback for complex / rank K when the damped coordinate (Jacobi) step does not raise L: steepest ascent
        on L = sum_b c_b log F_b / n - n log Q, dL/d conj(e_k) = S_k (w u_k / F) - n u_k / Q (one FFT pair per mode),
        with Armijo backtracking and a step that doubles after each success (kept between calls)."""
        xp = self.xp
        for _ in range(steps):
            F = (xp.abs(U) ** 2).sum(0); Q = float(xp.real(xp.sum(xp.conj(E) * U, dtype=self._acc(E))))
            V = xp.where(self.occ_, self.w_ / xp.where(self.occ_, F, 1.0), 0.0)[None] * U
            G = self._Sall(V) - (self.n_ / Q) * U
            G = xp.where(self.occ_[None], G, 0)             # empty bins keep e = 0
            SG = self._Sall(G); g2 = float(xp.real(xp.sum(xp.conj(G) * G, dtype=self._acc(G))))
            if not g2 > 0:
                break
            a = getattr(self, '_gstep', None) or 1e-3 * np.sqrt(float(xp.sum(xp.abs(E) ** 2, dtype=np.float64)) / g2)
            ok = False
            for _ in range(40):
                En, Un = E + a * G, U + a * SG; mn = self._merit(En, Un)
                if mn > cur:                                # backtrack to a strict gain
                    ok = True; break
                a *= 0.5
            if not ok:
                break
            E, U, cur = En, Un, mn; self._gstep = 2 * a
        return E, U, cur

    # ---------------------------------------------------------------- fit
    def fit(self, counts, modes, h=1.0, origin=0.0, complex_g=False, signs=None, init=None):
        """counts: (m_1, ..., m_d) histogram on the lattice origin + h * index. modes: as in BLMLQuick.fit (make_modes);
        rank K = len(modes). complex_g: complex g_k. signs: real rank 1 only, sign of g per bin (array shaped like
        counts, or a callable of the (B, d) bin centres). init: a previous fit on the same lattice (warm start)."""
        xp = self.xp
        self._setup(counts, modes, h, origin)
        w = xp.asarray(np.asarray(counts, float), dtype=self.rdt_)
        self.w_ = w; self.n_ = float(w.sum(dtype=np.float64))
        if not self.n_ > 0:
            raise ValueError("counts: no observations")
        self.c_ = self.n_ * w; self.occ_ = w > 0
        self.complex_g_ = bool(complex_g); self.psd_ = bool(complex_g) or self.K_ > 1
        if signs is not None and self.psd_:
            raise ValueError("complex_g / rank > 1 estimate the phases; do not pass signs as well")
        sg = np.ones(self.shape_) if signs is None else (
            np.sign(signs(self.centres())).reshape(self.shape_) if callable(signs) else np.sign(np.asarray(signs, float)))
        if np.any(sg[np.asarray(counts) > 0] == 0):
            raise ValueError("signs: nonzero on every occupied bin")
        self.sg_ = xp.asarray(sg, dtype=self.rdt_); self.signs_ = None if signs is None else sg
        dt = self.cdt_ if self.complex_g_ else self.rdt_
        if init is not None and getattr(init, 'e_hat_', None) is not None and init.shape_ == self.shape_:
            E0 = xp.asarray(init.e_hat_)
            E = xp.zeros((self.K_,) + self.shape_, dtype=dt)
            kk = min(self.K_, E0.shape[0]); E[:kk] = E0[:kk] if self.complex_g_ else xp.real(E0[:kk])
            E[:, ~self.occ_] = 0
            E = self._fill_modes(E, kk, phases=False)
        else:
            E = self._start(dt)
        E, U = self._solve(E, self.max_sweeps)
        self._finish(E, U)
        return self

    def _start(self, dt):
        """Real positive root of mode 0 (one convex solve), then small phases / small other modes."""
        xp = self.xp
        c, s0 = self.c_, float(self.s_[0])
        e0 = xp.where(self.occ_, xp.sqrt(c / s0), 0.0).astype(self.rdt_) * self.sg_
        if not self.psd_:
            return e0[None]
        save = (self.psd_, self.K_, self.kern_, self.s_)
        self.psd_, self.K_, self.kern_, self.s_ = False, 1, self.kern_[:1], self.s_[:1]
        er, _ = self._solve(e0[None], self.max_sweeps)       # the real rank-1 solution under mode 0's kernel
        self.psd_, self.K_, self.kern_, self.s_ = save
        E = xp.zeros((self.K_,) + self.shape_, dtype=dt); E[0] = er[0]
        return self._fill_modes(E, 1)

    def _fill_modes(self, E, kk, phases=True):
        xp = self.xp; rng = np.random.default_rng(self.seed); pn = self.phase_noise; sh = self.shape_
        if self.complex_g_ and phases:
            E[0] = E[0] * xp.asarray(np.exp(1j * pn * rng.standard_normal(sh)), dtype=E.dtype)
        for k in range(kk, self.K_):                          # modes the start lacks: small noise (zero stays zero)
            z = rng.standard_normal(sh) + (1j * rng.standard_normal(sh) if self.complex_g_ else 0)
            E[k] = pn * xp.abs(E[0]) * xp.asarray(z / (np.sqrt(2) if self.complex_g_ else 1), dtype=E.dtype)
        return E

    def _finish(self, E, U):
        xp = self.xp
        self.e_hat_, self.Se_ = E, U
        F = (xp.abs(U) ** 2).sum(0)
        self.density_ = F / self.n_ ** 2                       # f at the bins (f = sum_k |g_k|^2, g_k = (S_k e_k)/n)
        occ = self.occ_
        self.loglik_ = float((self.w_[occ] * xp.log(self.density_[occ])).sum(dtype=np.float64)) / self.n_
        self.n_samples_ = self.n_

    # ---------------------------------------------------------------- outputs
    def centres(self):
        g = np.meshgrid(*[self.origin_[j] + self.h_[j] * np.arange(m) for j, m in enumerate(self.shape_)], indexing='ij')
        return np.stack([x.ravel() for x in g], 1)

    def sqrt_density(self, X, chunk=4096):
        """g_k(x) = (1/n) sum_b e_bk K_k(x - z_b) at arbitrary points X (Nq, d): (Nq,) for rank 1, else (Nq, K)."""
        X = np.atleast_2d(np.asarray(X, float)); Zb = self.centres()
        E = _np(self.e_hat_).reshape(self.K_, -1); occ = _np(self.occ_).ravel()
        Zb, E = Zb[occ], E[:, occ]
        out = np.zeros((len(X), self.K_), dtype=complex if self.complex_g_ else float)
        for i in range(0, len(X), chunk):
            diffs = X[i:i + chunk, None, :] - Zb[None, :, :]
            for k, sp in enumerate(self.specs_):
                Kv = 1.0
                for kind, dims in kernel_units(sp['kinds'], self.d_):
                    th = unit_theta(kind, sp['fc'], sp['periods'], dims)
                    Kv = Kv * kernel_class(kind).eval(th, [diffs[..., j] for j in dims])
                out[i:i + chunk, k] = Kv @ E[k] / self.n_
        return out[:, 0] if self.K_ == 1 else out

    def predict(self, X):
        G = self.sqrt_density(X)
        return np.abs(G) ** 2 if G.ndim == 1 else (np.abs(G) ** 2).sum(1)

    def predict_grid(self, r):
        """f on the lattice refined r times per dim (origin, steps h / r; the last fine point is the last bin):
        shape ((m_j - 1) r + 1, ...). One FFT pair per sub-lattice shift and mode."""
        import itertools
        xp, fft = self.xp, self.fft; d = self.d_
        sl = tuple(slice(0, m) for m in self.shape_)
        fine = tuple((m - 1) * r + 1 for m in self.shape_)
        out = np.zeros(fine)
        kw = dict(workers=self.workers) if xp is np else {}
        for sh in itertools.product(range(r), repeat=d):
            dens = 0
            for k, kk in enumerate(self.kern_):
                diffs = [kk._diffs[j] + self.h_[j] * sh[j] / r for j in range(d)]
                T = 1.0
                for kind, dims in kk.units:
                    T = T * kk._unit_vals(kind, dims, diffs)
                for mk in kk._mask:
                    T = T * mk
                Kf = fft.rfftn(xp.asarray(np.broadcast_to(T, self.pad_), dtype=self.rdt_), s=self.pad_, **kw)
                e = self.e_hat_[k]
                parts = [e.real, e.imag] if xp.iscomplexobj(e) else [e]
                g = [fft.irfftn(fft.rfftn(p, s=self.pad_, **kw) * Kf, s=self.pad_, **kw)[sl] for p in parts]
                gk = g[0] + 1j * g[1] if len(g) == 2 else g[0]
                dens = dens + xp.abs(gk / self.n_) ** 2
            dens = _np(dens)
            idx = tuple(slice(s, None, r) for s in sh)
            tgt = out[idx]; out[idx] = dens[tuple(slice(0, n_) for n_ in tgt.shape)]
        return out

    def certificate(self, k=1, tol=1e-6):
        """lambda_max(D^1/2 S D^1/2) / n, D_b = ||e_b||^2 / w_b (shared kernel). <= 1 (+ solver tolerance): the fit
        solves the convex lift, so no fit of any rank or sign pattern has a larger likelihood."""
        if not self.shared_:
            raise ValueError("certificate: shared kernel only (per-mode kernels: no convex lift)")
        xp = self.xp
        D = xp.where(self.occ_, (xp.abs(self.e_hat_) ** 2).sum(0) / xp.where(self.occ_, self.w_, 1.0), 0.0)
        dh = xp.sqrt(D).astype(np.float64); B = int(np.prod(self.shape_))      # Lanczos in float64 at any fit dtype
        op = lambda x: (dh * self._S(0, dh * x.reshape(self.shape_).astype(np.float64))).ravel()
        if xp is np:
            from scipy.sparse.linalg import LinearOperator, eigsh
            lam = eigsh(LinearOperator((B, B), matvec=op, dtype=np.float64), k=k, which='LA', tol=tol,
                        return_eigenvectors=False)
        else:
            from cupyx.scipy.sparse.linalg import LinearOperator, eigsh
            lam = _np(eigsh(LinearOperator((B, B), matvec=op, dtype=np.float64), k=k, which='LA', tol=tol,
                            return_eigenvectors=False))
        return float(np.max(lam)) / self.n_

    def fc_grad(self):
        """d loglik_/d log fc of every (mode, unit) at the fitted coefficients (the envelope theorem: e is a
        stationary point of the scale-free likelihood). Returns a list per mode of arrays (one entry per unit;
        nan for dirichlet units). With modes sharing one kernel table the gradient is per mode (each mode's own fc).
        Training gradient: the band-limited spaces are nested in fc, so it never says 'smaller'; the streaming
        classes ascend the prequential one (param_grad on the incoming batch) instead."""
        return self.param_grad(self.w_)

    def _apply(self, Kf, e):
        """The convolution with a kernel spectrum Kf of one mode's coefficients e (real or complex)."""
        xp, fft = self.xp, self.fft
        sl = tuple(slice(0, m) for m in self.shape_)
        kw = dict(workers=self.workers) if xp is np else {}
        parts = [e.real, e.imag] if xp.iscomplexobj(e) else [e]
        g = [fft.irfftn(fft.rfftn(p, s=self.pad_, **kw) * Kf, s=self.pad_, **kw)[sl] for p in parts]
        return g[0] + 1j * g[1] if len(g) == 2 else g[0]

    def _norm(self, E, U):
        """The normaliser Z of f = F / Z (the plane: Z = sum_k e_k^H S_k e_k) and dZ from (e_k, dU_k)."""
        xp = self.xp
        Z = float(xp.real(xp.sum(xp.conj(E) * U, dtype=self._acc(E))))
        return Z, lambda k, dU: float(xp.real(xp.sum(xp.conj(E[k]) * dU, dtype=self._acc(dU))))

    def param_grad(self, counts, eps=0.0):
        """d/dlog fc_u of sum_b m_b log(F_b / Z + eps) / sum m at the current coefficients, for counts m on the
        lattice (the training counts: fc_grad; a new batch: the prequential gradient). eps (in F units) smooths the
        log where g crosses zero; eps is in density units (f = F / Z, as streaming BLMLQuick's grad_eps * median f), so
        the score is scale-free in the coefficients. A list per mode (nan: integer parameters)."""
        xp = self.xp
        E, U = self.e_hat_, self.Se_
        m = xp.asarray(counts, dtype=self.rdt_); M = float(m.sum(dtype=np.float64))
        F = (xp.abs(U) ** 2).sum(0); Z, dZ = self._norm(E, U)
        wn = xp.where(m > 0, m / xp.maximum(F + eps * Z, 1e-300), 0.0)
        fz = float(xp.sum(wn * F, dtype=np.float64)) / M           # sum m F / (F + eps Z) / M (1 at eps = 0)
        out = []
        for k, kk in enumerate(self.kern_):
            gk = []
            for dKf in kk.dKf():
                if dKf is None:
                    gk.append(np.nan); continue
                dU = self._apply(dKf, E[k])
                gk.append(2 * float(xp.sum(wn * xp.real(xp.conj(U[k]) * dU), dtype=np.float64)) / M - fz * dZ(k, dU) / Z)
            out.append(np.array(gk))
        return out

    # ---------------------------------------------------------------- implicit (refit-response) gradient
    def implicit_grad(self, counts, eps=0.0, tol=1e-6, maxit=300):
        """d/dlog fc_u of the same score as param_grad with the coefficients' response to the move included:
        dP/dtheta = dP/dtheta|_e - <v, d grad/dtheta>, H v = dP/de, H the Hessian of the fit's objective at its
        stationary point (one CG solve per call, Hessian-vector products by FFT; frozen coordinates -- empty bins,
        l1's off-support coefficients -- excluded; gauge directions are null for H and orthogonal to dP/de). On a
        fixed lattice the coefficients carry the counts, so the fixed-coefficient part alone is close to the
        training gradient (always 'larger'); this is the held-out derivative (results/binned_learn_checks.py)."""
        G = self.param_grad(counts, eps)
        st = self._ift_state()
        b = self._dP_de(st, counts, eps); mask = self._free(st)
        v, self.ift_iters_, self.ift_res_ = _icg(self.xp, lambda x: self._hvp(st, x) * mask, b * mask, tol, maxit)
        for k, kk in enumerate(self.kern_):
            for u, dKf in enumerate(kk.dKf()):
                if dKf is not None:
                    G[k][u] -= _rdot(self.xp, v, self._mixed(st, k, dKf) * mask)
        return G

    def _ift_state(self):
        xp = self.xp
        E, U = self.e_hat_, self.Se_
        F = (xp.abs(U) ** 2).sum(0); Q = float(xp.real(xp.sum(xp.conj(E) * U, dtype=self._acc(E))))
        return dict(E=E, U=U, F=F, Q=Q, Fs=xp.where(self.occ_, F, 1.0))

    def _free(self, st):
        return self.occ_[None].astype(self.rdt_)

    def _dP_de(self, st, counts, eps):
        """dP/de (real inner product Re<., .>) of P = sum m log(F / Q + eps) / M."""
        xp = self.xp; m = xp.asarray(counts, dtype=self.rdt_); M = float(m.sum(dtype=np.float64))
        wn = m / xp.maximum(st['F'] + eps * st['Q'], 1e-300); fz = float(xp.sum(wn * st['F'], dtype=np.float64)) / M
        return 2 * self._Sall((wn / M)[None] * st['U']) - fz * 2 * st['U'] / st['Q']

    def _hvp(self, st, D):
        """Hessian of the minimised objective (psd: -L, real rank 1: Phi) times D."""
        xp, w, n = self.xp, self.w_, self.n_
        E, U, F, Q, Fs = st['E'], st['U'], st['F'], st['Q'], st['Fs']
        dU = self._Sall(D)
        if not self.psd_:
            e = E[0]; c = self.c_
            return dU + xp.where(self.occ_, c / xp.where(self.occ_, e * e, 1.0), 0.0)[None] * D
        dF = 2 * xp.real(xp.conj(U) * dU).sum(0); dQ = 2 * _rdot(xp, U, D)
        inner = (w / Fs)[None] * dU - (w * dF / Fs ** 2)[None] * U
        return -2 * self._Sall(xp.where(self.occ_[None], inner, 0)) + 2 * n * dU / Q - 2 * n * U * dQ / Q ** 2

    def _mixed(self, st, k, dKf):
        """d(grad of the minimised objective)/dlog fc for one unit of mode k (dKf), at fixed coefficients."""
        xp, w, n = self.xp, self.w_, self.n_
        E, U, F, Q, Fs = st['E'], st['U'], st['F'], st['Q'], st['Fs']
        dUk = self._apply(dKf, E[k])
        if not self.psd_:
            return dUk[None]
        dF = 2 * xp.real(xp.conj(U[k]) * dUk); dQ = float(xp.real(xp.sum(xp.conj(E[k]) * dUk, dtype=self._acc(dUk))))
        occ = self.occ_; ws = xp.where(occ, w / Fs, 0.0); wd = xp.where(occ, w * dF / Fs ** 2, 0.0)
        out = []
        for j in range(self.K_):
            t = -wd * U[j]
            if j == k:
                t = t + ws * dUk
            g = -2 * self._S(j, t) - 2 * n * U[j] * dQ / Q ** 2
            if j == k:
                g = g - 2 * self._apply(dKf, ws * U[k]) + 2 * n * dUk / Q
            out.append(g)
        return xp.stack(out)

    def log_density(self, counts, eps=0.0):
        """Mean over counts m on the lattice of log(f_b + eps) under the current fit (f = F / Z, unrenormalised:
        loglik_ for the training counts; a new batch: its prequential score)."""
        xp = self.xp
        U = self.Se_; F = (xp.abs(U) ** 2).sum(0); Z, _ = self._norm(self.e_hat_, U)
        m = xp.asarray(counts, dtype=self.rdt_); occ = m > 0
        return float((m[occ] * xp.log(xp.maximum(F[occ] / Z + eps, 1e-300))).sum(dtype=np.float64)) / float(m.sum(dtype=np.float64)) \
            + self._log_scale()

    def _log_scale(self):
        """log of the factor from F / Z to the density (BLMLBinned: f = F / Q exactly, the plane normalisation)."""
        return 0.0

    def _val_score(self, counts):
        """The held-out score the streaming burn-in selection maximises: mean log f at counts on the lattice."""
        return self.log_density(counts)


class _KernelLearner:
    """Kernel-parameter learning for the streaming classes, with the functionality of blml_rolling._StreamingBase /
    BLMLQuickStreaming on counts with the lattice fixed by the input (no re-binning: a parameter move rebuilds that
    mode's kernel spectrum only; conditional (dep) modes and input maps need moving points and are not available):
      0. Burn-in: counts are held until burn_in (0: the first batch) and get an exact fit; fc entries left out of the
         modes (NaN: selector templates) are selected first by the selector's search on the held counts -- an 80/20
         binomial split of the counts, or with split_groups 20% of whole groups for validation (groups are never
         separated) -- search='shared' (one doubling bracket of every free fc, no refinement) or 'refine'.
      1. Prequential score: each batch after the burn-in is scored by the current model before it is used (preq_).
         A revisited batch (fit's later epochs) is scored leave-batch-out (_left_out: its share of the held counts
         removed from the fit), so the score and its gradient stay out of sample however many epochs run.
      2. Its gradient for every trainable (mode, unit) parameter, d/dlog fc of the batch's mean log(F + eps) - log Z,
         eps = grad_eps * the median f on the batch's bins, pooled over param_every batches. implicit_grad=True
         (default here, unlike BLMLQuickStreaming): with the coefficients' refit response (implicit_grad, one CG solve
         per batch). On a fixed lattice the fixed-coefficient part alone (param_grad) is close to the training
         gradient, which never says 'smaller' (nested band-limited spaces): on a sign-changing 1-d truth it stays
         >= 0 up to fc = 3 x the truth while the held-out slope is negative above 1.2 x.
      3. Every param_every batches: 'scale' parameters (sinc / ball fc) by Adam on log fc (blml_maps.adam_step);
         'integer' parameters (dirichlet harmonics), and with fc_update='hillclimb' every parameter, by one step
         (one harmonic, N = floor(fc P / 2) +- 1, as selector._move; or fc * step^+-1) once its pooled paired gain on
         the batches (each candidate scored at the fixed coefficients) exceeds z_se standard errors. The coefficients
         are not refitted when a parameter moves (S e is recomputed); the next batches' sweeps catch up. Periods stay
         fixed (the box sets them).
    fit(batches, modes, epochs, ..., split_groups, on_epoch, tol, tol_check): epoch 1 adds every batch (random
    group order, as BLMLQuickStreaming.fit); later epochs revisit them -- counts not added again, scored
    leave-batch-out, the coefficients swept, the parameters moving; tol stops after the first epoch whose fit meets
    the solver's residual test (running: the maximum over the epoch's revisits before their sweeps; exact: the
    residual after the epoch) with the parameters stationary (_param_stationary: the epoch's leave-batch-out
    gradients within z_se standard errors of 0, or pushing into the Nyquist cap; no pending hill-climb move);
    converged_, epochs_, residual_. bcd_frac < 1: a fixed binomial share 1 - bcd_frac of
    each batch's counts (integer counts) is scored and drives the parameters without being fitted (out of sample at
    every epoch); freeze_frac > 0: the last freeze_frac of the batch steps fit everything at frozen parameters (the
    learn counts join then). trace_trainable: history_ entries carry the trainable values ({mode: {'fc': [...]}}).
    fc_update=None: no learning. Each mode's units own their parameters (modes sharing a kernel can drift apart;
    the certificate then needs a shared kernel again). history_: per batch n, preq, fc per mode, what moved,
    revisit; per epoch (with tol) epoch, residual."""

    def _learner_init(self, burn_in, fc_update, lr, betas, step, z_se, param_every, grad_eps, implicit_grad=True,
                      search='shared', bcd_frac=1.0, freeze_frac=0.0, trace_trainable=False):
        if fc_update not in (None, 'adam', 'hillclimb'):
            raise ValueError("fc_update: 'adam', 'hillclimb' or None")
        if search not in ('shared', 'refine'):
            raise ValueError("search: 'shared' or 'refine'")
        if not (0.0 < bcd_frac <= 1.0 and 0.0 <= freeze_frac < 1.0):
            raise ValueError("bcd_frac in (0, 1], freeze_frac in [0, 1)")
        self.burn_in, self.fc_update, self.lr, self.betas, self.step, self.z_se = burn_in, fc_update, lr, betas, step, z_se
        self.param_every, self.grad_eps, self._implicit = max(1, int(param_every)), grad_eps, bool(implicit_grad)
        self.search, self.bcd_frac, self.freeze_frac, self.trace_trainable = search, float(bcd_frac), float(freeze_frac), trace_trainable
        self.preq_, self.history_ = [], []
        self._n_learn, self._pool, self._hc, self._adam, self._epoch_g = 0, [], {}, {}, []
        self._held, self._held_groups, self.n_batches_ = None, [], 0
        self.converged_ = self.epochs_ = self.residual_ = None

    @property
    def implicit(self):                                       # the flag (the constructor's implicit_grad; earlier name)
        return self._implicit

    def _registry(self):
        reg = []
        if self.fc_update is None:
            return reg
        for k, kk in enumerate(self.kern_):
            for u, (kind, dims) in enumerate(kk.units):
                if not bool(self.specs_[k]['fc_trainable'][dims[0]]):
                    continue
                ptype = kernel_class(kind).params['fc'][0]
                rule = 'adam' if ptype == 'scale' and self.fc_update == 'adam' else 'hill'
                reg.append(dict(mode=k, unit=u, dims=tuple(dims), ptype=ptype, rule=rule))
        return reg

    def _fc_now(self):
        return [np.array(sp['fc'], float) for sp in self.specs_]

    def _candidate(self, e, value):
        """fc of e's (mode, unit) moved to value: (spec, kernel)."""
        k, dims = e['mode'], list(e['dims'])
        fc = np.array(self.specs_[k]['fc'], float); fc[dims] = value; sp = dict(self.specs_[k], fc=fc)
        return sp, _ModeKernel(sp, self.shape_, self.h_, self.pad_, self.xp, self.fft, self.rdt_, self.workers)

    def _learn(self, counts):
        """Steps 1-2 for one batch (before it is added): the prequential score, the gradient and hill-climb gains."""
        xp = self.xp
        m = xp.asarray(counts, dtype=self.rdt_)
        U = self.Se_; F = (xp.abs(U) ** 2).sum(0); occ = m > 0
        Z0, _ = self._norm(self.e_hat_, U)
        eps = self.grad_eps * float(xp.median(F[occ])) / Z0 if bool(xp.any(occ)) else 0.0   # density units
        reg = self._registry()
        if any(e['rule'] == 'adam' for e in reg):
            G = self.implicit_grad(m, eps) if self._implicit else self.param_grad(m, eps)
            self._pool.append((float(m.sum(dtype=np.float64)), [G[e['mode']][e['unit']] for e in reg if e['rule'] == 'adam']))
            self._epoch_g.append(self._pool[-1])
        hill = [e for e in reg if e['rule'] == 'hill']
        if hill:
            base = xp.log(F / Z0 + eps)
            for e in hill:
                k, dims = e['mode'], list(e['dims']); v = float(self.specs_[k]['fc'][dims[0]])
                for sgn in (+1, -1):
                    if e['ptype'] == 'integer':                 # dirichlet: one harmonic, N = floor(fc P / 2) +- 1, at
                        P = float(self.specs_[k]['periods'][dims[0]])   # the middle of its fc interval (selector._move)
                        N = int(np.floor(v * P / 2.0)) + sgn
                        if N < 0:
                            continue
                        cand = (N + 0.5) * 2.0 / P
                    else:
                        cand = v * self.step ** sgn
                    if not self._in_range(e, cand):
                        continue
                    Fc, Zc = self._candidate_fit(e, cand)
                    dl = (xp.log(Fc / Zc + eps) - base)[occ]; mw = m[occ]
                    acc = self._hc.setdefault((k, e['unit'], sgn), dict(s=0.0, ss=0.0, W=0.0, cand=cand))
                    acc['s'] += float((mw * dl).sum(dtype=np.float64)); acc['ss'] += float((mw * dl * dl).sum(dtype=np.float64))
                    acc['W'] += float(mw.sum(dtype=np.float64)); acc['cand'] = cand

    def _nyq(self, e):
        """Largest identifiable value of e's parameter: fc <= 0.5 / h (the lattice Nyquist limit: a band beyond it is
        aliased and the fit only interpolates the counts); dirichlet harmonics N < period / (2 h)."""
        return 0.5 / float(self.h_[e['dims'][0]])

    def _in_range(self, e, value):
        return 0.0 < value <= self._nyq(e) + 1e-12

    def _candidate_fit(self, e, cand):
        """(F, Z) of the fit with e's parameter at cand: the kernel swapped and a warm refit of the held counts
        (to the solver's tol, as the leave-batch-out refit) -- the held counts exclude the batch being scored (a new
        batch is not added yet; a revisit is scored leave-batch-out), so candidates compare out of sample. At fixed
        coefficients the comparison is biased towards the richer kernel (as the fixed-coefficient fc gradient)."""
        saved = dict(self.__dict__); xp = self.xp
        try:
            self._set_fc(e['mode'], e['dims'], cand); self._build_kernels()
            self._candidate_refit()
            U = self.Se_; F = (xp.abs(U) ** 2).sum(0); Z, _ = self._norm(self.e_hat_, U)
            return F, Z
        finally:
            pools = (self._pool, self._hc, self._adam, self._epoch_g)
            self.__dict__.clear(); self.__dict__.update(saved); self._pool, self._hc, self._adam, self._epoch_g = pools

    def _set_fc(self, k, dims, value):
        fc = np.array(self.specs_[k]['fc'], float); fc[list(dims)] = value
        self.specs_ = [dict(sp, fc=fc) if j == k else sp for j, sp in enumerate(self.specs_)]

    def _param_step(self):
        """Step 3: returns what moved [(mode, unit, change)]."""
        from blml_maps import adam_step
        reg = self._registry(); moved = []
        adam = [e for e in reg if e['rule'] == 'adam']
        if adam and self._pool:
            Wt = sum(p[0] for p in self._pool)
            G = sum(p[0] * np.asarray(p[1]) for p in self._pool) / Wt
            if not np.all(np.isfinite(G)):
                raise FloatingPointError(f"streaming: non-finite gradient {G}")
            for q, e in enumerate(adam):
                k, dims = e['mode'], e['dims']; old = float(self.specs_[k]['fc'][dims[0]])
                st = self._adam.setdefault((k, e['unit']), [0.0, 0.0, 0])
                new = min(float(adam_step(old, G[q], st, 'scale', self.lr, self.betas, dlog=True)), self._nyq(e))
                self._set_fc(k, dims, new); moved.append((k, e['unit'], round(float(np.log(new / old)), 5)))
        self._pool = []
        for e in (e for e in reg if e['rule'] == 'hill'):
            for sgn in (+1, -1):
                acc = self._hc.get((e['mode'], e['unit'], sgn))
                if not acc or acc['W'] <= 1:
                    continue
                mu = acc['s'] / acc['W']; se = np.sqrt(max(acc['ss'] / acc['W'] - mu * mu, 0.0) / (acc['W'] - 1))
                if mu > 0 and mu > self.z_se * se:
                    self._set_fc(e['mode'], e['dims'], acc['cand']); moved.append((e['mode'], e['unit'], sgn))
                    self._hc = {k_: v for k_, v in self._hc.items() if k_[:2] != (e['mode'], e['unit'])}
                    break
        if moved:
            self._build_kernels(); self._refresh()
        return moved

    def _param_stationary(self):
        """Whether the epoch's parameters are stationary: for each Adam parameter, the mean of the epoch's per-batch
        gradients (count-weighted) within z_se standard errors of 0, or > 0 at the Nyquist cap; no pending hill-climb
        move (a gain beyond z_se standard errors). Returns (bool, max |mean| / se over the Adam parameters)."""
        reg = self._registry(); adam = [e for e in reg if e['rule'] == 'adam']; zmax = 0.0
        if adam and self._epoch_g:
            W = np.array([p[0] for p in self._epoch_g]); G = np.array([np.ravel(np.asarray(p[1], float)) for p in self._epoch_g])
            if len(W) < 2:
                return False, np.inf
            mu = (W[:, None] * G).sum(0) / W.sum(); var = (W[:, None] * (G - mu) ** 2).sum(0) / W.sum()
            se = np.sqrt(var / (len(W) - 1)); ok = True
            for q, e in enumerate(adam):
                at_cap = float(self.specs_[e['mode']]['fc'][e['dims'][0]]) >= self._nyq(e) - 1e-12
                z = abs(mu[q]) / se[q] if se[q] > 0 else (0.0 if mu[q] == 0 else np.inf)
                if not (z <= self.z_se or (at_cap and mu[q] > 0)):
                    ok = False; zmax = max(zmax, z)
        else:
            ok = not adam
        for acc in self._hc.values():
            if acc['W'] > 1:
                mu = acc['s'] / acc['W']; se = np.sqrt(max(acc['ss'] / acc['W'] - mu * mu, 0.0) / (acc['W'] - 1))
                if mu > 0 and mu > self.z_se * se:
                    ok = False
        return ok, zmax

    def _trainable_now(self):
        return {k: {'fc': [float(v) for v, t in zip(sp['fc'], sp['fc_trainable']) if t]} for k, sp in enumerate(self.specs_)}

    def _stream_step(self, counts, absorb, revisit=False, learn=True):
        """One batch: score (leave-batch-out on a revisit) and learn (after the burn-in), absorb (add and refit; on a
        revisit refit only; None: a learn-only batch), then a parameter step every param_every learning batches."""
        burned = self.n_ >= self.burn_in
        do_learn = learn and burned and self.fc_update is not None
        with self._left_out(counts if revisit else None):
            preq = self.log_density(counts)
            if do_learn:
                self._learn(counts)
        if absorb is not None:
            absorb(burned)
        moved = []
        if do_learn:
            self._n_learn += 1
            if self._n_learn % self.param_every == 0:
                moved = self._param_step()
        self.preq_.append(preq)
        h = dict(n=self.n_, preq=preq, fc=[f.tolist() for f in self._fc_now()], moved=moved, revisit=bool(revisit))
        if self.trace_trainable:
            h['trainable'] = self._trainable_now()
        self.history_.append(h)

    # ---------------------------------------------------------------- leave-batch-out
    def _left_out(self, counts):
        """Context: the fit with counts' share of the held counts removed (counts None: unchanged), restored after."""
        import contextlib
        @contextlib.contextmanager
        def ctx():
            if counts is None:
                yield; return
            saved = dict(self.__dict__)
            try:
                self._loo_enter(np.asarray(counts, float))
                yield
            finally:
                pools = (self._pool, self._hc, self._adam, self._epoch_g)   # gradients and gains gathered inside are kept
                self.__dict__.clear(); self.__dict__.update(saved)
                self._pool, self._hc, self._adam, self._epoch_g = pools
        return ctx()

    # ---------------------------------------------------------------- burn-in (held counts, selection)
    def _hold(self, counts, modes, h, origin, complex_g, group):
        if self._held is None:
            if modes is None:
                raise ValueError("partial_fit: modes on the first call")
            self._modes_args = (modes, h, origin, complex_g); self._held = counts.copy(); self._held_list = [counts]
        else:
            if counts.shape != self._held.shape:
                raise ValueError("partial_fit: the batch must be on the lattice of the first one")
            self._held = self._held + counts; self._held_list.append(counts)
        self._held_groups.append(group)
        return float(self._held.sum()) >= self.burn_in

    def _select(self):
        """Selector search for the fc entries the modes leave out (NaN), on the held counts: train / validation as
        an 80/20 binomial split of the counts, or 20% of whole groups (split_groups) for validation."""
        from blml_trivial import modes_fc
        import selector as sel
        modes, h, origin, cg = self._modes_args
        F = modes_fc(modes); free = ~np.isfinite(F)
        if not free.any():
            return modes
        specs = parse_modes(modes, self._held.ndim, need_fc=False)[0]
        rng = np.random.default_rng(self.seed)
        groups = [g for g in self._held_groups]
        if all(g is not None for g in groups):
            u = np.unique(groups); V = rng.choice(u, max(1, int(round(0.2 * len(u)))), replace=False)
            tr = sum((b for b, g in zip(self._held_list, groups) if g not in V), np.zeros(self._held.shape))
            va = sum((b for b, g in zip(self._held_list, groups) if g in V), np.zeros(self._held.shape))
        else:
            if not np.allclose(self._held, np.round(self._held)):
                raise ValueError("burn-in selection: integer counts for the binomial split (or pass split_groups)")
            tr = rng.binomial(np.round(self._held).astype(np.int64), 0.8).astype(float); va = self._held - tr
        kinds, periods, idx = [], [], np.argwhere(free)
        for k, j in idx:
            kinds.append(specs[k]['kinds'][j].split(':')[0]); periods.append(specs[k]['periods'][j])
        hh = np.broadcast_to(np.asarray(h, float), (self._held.ndim,))
        periods = [None if p is None else p / hh[j] for p, (_, j) in zip(periods, idx)]   # in lattice units

        def obj(v):
            if np.any(np.asarray(v) > 0.5 / hh[idx[:, 1]] + 1e-12):          # beyond the lattice Nyquist limit
                return -np.inf
            try:
                mm = self._base().fit(tr, _modes_with_fc(modes, _fill(F, free, v)), h=h, origin=origin, complex_g=cg)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                return -np.inf
            return mm._val_score(va)
        fc0 = 2.0 / (max(self._held.shape) * float(np.max(hh)))   # one cycle across the box: below any band the data resolve
        v, _ = sel._search(obj, len(idx), kinds, periods, fc0=fc0, max_sweeps=0 if self.search == 'shared' else 6,
                           tag='burn-in', factors=(np.sqrt(2.0), 2.0 ** 0.25, 2.0 ** 0.125))
        return _modes_with_fc(modes, _fill(F, free, v))

    # ---------------------------------------------------------------- epochs
    def fit(self, batches, modes=None, epochs=2, h=1.0, origin=0.0, complex_g=False, split_groups=None, on_epoch=None,
            tol=None, tol_check='running'):
        """batches: a sequence of count arrays on one lattice (each a batch; e.g. readouts or frames). Epoch 1 adds
        them in random group order (split_groups: a label per batch; batches sharing a label are never separated,
        here or between the burn-in selection's sides); later epochs revisit them (see the class docstring)."""
        batches = [np.asarray(b, float) for b in batches]; nb = len(batches)
        grp = None if split_groups is None else list(np.asarray(split_groups).tolist())
        if grp is not None and len(grp) != nb:
            raise ValueError("split_groups: one label per batch")
        if tol is not None and not tol > 0:
            raise ValueError("tol: None (fixed epochs) or > 0")
        if tol is not None and (self.bcd_frac < 1.0 or self.freeze_frac > 0.0):
            raise ValueError("tol needs bcd_frac = 1 and freeze_frac = 0")
        if tol_check not in ('running', 'exact'):
            raise ValueError("tol_check: 'running' or 'exact'")
        split = self.bcd_frac < 1.0
        if split:
            if not all(np.allclose(b, np.round(b)) for b in batches):
                raise ValueError("bcd_frac < 1: integer counts (the learn share is a binomial thinning)")
            r0 = np.random.default_rng([self.seed, 11])
            L = [r0.binomial(np.round(b).astype(np.int64), 1.0 - self.bcd_frac).astype(float) for b in batches]
        held, joined = [False] * nb, [False] * nb
        T_learn = int(round((1.0 - self.freeze_frac) * epochs * nb)); t = 0
        self.converged_, self.epochs_, self.residual_ = None, 0, None
        first = dict(modes=modes, h=h, origin=origin, complex_g=complex_g)
        for ep in range(epochs):
            run_max = 0.0; self._epoch_g = []
            for i in _batch_order(nb, grp, np.random.default_rng(self.seed + ep)):
                frozen = t >= T_learn; t += 1
                both = not split and not frozen
                Fi = batches[i] if (not split or frozen) else batches[i] - L[i]
                if not held[i]:
                    self.partial_fit(Fi, split_groups=None if grp is None else grp[i], _learn=both and not frozen, **first)
                    held[i] = True; first = {}
                elif self.n_batches_ > 0:
                    run_max = max(run_max, self._residual())
                    self._again(Fi, learn=both)
                if split and not frozen and self.n_batches_ > 0:
                    self._stream_step(L[i], None, learn=True)  # scored, gradient pooled, not added
                if split and frozen and not joined[i] and self.n_batches_ > 0:
                    self.partial_fit(L[i], _learn=False); joined[i] = True
            if self.n_batches_ == 0:
                raise ValueError(f"fit: burn_in = {self.burn_in} exceeds the counts given ({sum(float(b.sum()) for b in batches):g})")
            self.epochs_ = ep + 1
            if tol is not None and (tol_check == 'exact' or ep > 0):
                self.residual_ = self._residual() if tol_check == 'exact' else run_max
                still, z = self._param_stationary() if self.fc_update is not None else (True, 0.0)
                self.converged_ = bool(self.residual_ < tol and still)
                self.history_.append(dict(epoch=ep + 1, residual=self.residual_, params_stationary=still, grad_z=z))
            if on_epoch is not None:
                on_epoch(ep, self)
            if tol is not None and self.converged_:
                break
        return self


def _fill(F, free, v):
    F = F.copy(); F[free] = np.asarray(v, float); return F


def _modes_with_fc(modes, F):
    """modes with the fc of mode k, column j set to F[k, j] (template entries filled; trainability kept)."""
    out = []
    for k, m in enumerate(modes):
        mm = {}
        for key, c in m.items():
            c = dict(c); cols = key if isinstance(key, tuple) else (key,)
            val = float(F[k][cols[0]])
            if isinstance(c.get('fc'), dict):
                c['fc'] = dict(c['fc'], value=val)
            else:
                c['fc'] = val
            mm[key] = c
        out.append(mm)
    return out


def _batch_order(nb, groups, rng):
    """Batch indices in random group order (no groups: a random permutation), whole groups together."""
    if groups is None:
        return list(rng.permutation(nb))
    u, inv = np.unique(np.asarray(groups), return_inverse=True)
    members = [[] for _ in u]
    for i, k in enumerate(np.ravel(inv)):
        members[k].append(i)
    return [i for k in rng.permutation(len(u)) for i in members[k]]


class BLMLBinnedStreaming(_KernelLearner, BLMLBinned):
    """Counts arriving in batches on one fixed lattice, with the functionality of BLMLQuickStreaming (_KernelLearner;
    no re-binning: the lattice is the input's). partial_fit(counts) holds counts until burn_in, then adds each batch
    (after multiplying the held counts by decay: decay < 1 forgets old data exponentially) and refits warm-started
    from the current coefficients, to tol or for batch_sweeps sweeps (a fixed per-batch budget for real-time use; to
    tol during the burn-in). fit(batches, modes, epochs, ...) revisits the batches over epochs. The state between
    batches is the coefficients and S e only: the lattice never changes, so there is nothing to re-bin or re-cache.
    Kernel parameters: fc_update='adam' by default (pass fc_update=None, or trainable=False fc, to keep them fixed);
    an fc left out of the modes (NaN) is selected at the burn-in. Defaults kept from earlier versions: burn_in=0
    (the first batch), implicit_grad=True (alias: implicit)."""

    def __init__(self, decay=1.0, batch_sweeps=None, burn_in=0, fc_update='adam', lr=0.1, betas=(0.9, 0.999),
                 step=1.05, z_se=1.0, param_every=3, grad_eps=1e-3, implicit_grad=True, search='shared', bcd_frac=1.0,
                 freeze_frac=0.0, trace_trainable=False, loo_sweeps=None, **kw):
        if 'implicit' in kw:
            implicit_grad = kw.pop('implicit')
        super().__init__(**kw)
        self._base_kw = dict(kw)
        self.decay, self.batch_sweeps, self.loo_sweeps = decay, batch_sweeps, loo_sweeps
        self._learner_init(burn_in, fc_update, lr, betas, step, z_se, param_every, grad_eps, implicit_grad,
                           search, bcd_frac, freeze_frac, trace_trainable)

    def _base(self):
        return BLMLBinned(**self._base_kw)

    def _refresh(self):
        E, U = self._rescale(self.e_hat_, self._Sall(self.e_hat_))
        self._finish(E, U)

    def _residual(self):
        return self._resid(self.e_hat_, self.Se_)

    def _hill_sweeps(self):
        return self.loo_sweeps if self.loo_sweeps is not None else self.max_sweeps

    def _candidate_refit(self):
        E, U = self._solve(self.e_hat_, self._hill_sweeps()); self._finish(E, U)

    def _loo_enter(self, m):
        """The fit with m removed: each bin's coefficient scaled by its remaining share (w - m) / w (the start), then a
        warm refit on the held counts less m to the solver's tol (loo_sweeps: a cap on its sweeps). A scaling alone
        leaves the batch's influence in the other coefficients and the gradient in sample; the implicit gradient
        assumes a stationary fit, so a capped refit biases it (complex, 1-d, fc 1.5 x the truth: d/dlog fc of the
        leave-batch-out score -0.18 after 100 sweeps, -0.006 to tol, the finite difference -0.016)."""
        xp = self.xp; m = xp.asarray(m, dtype=self.rdt_)
        w = xp.maximum(self.w_ - m, 0.0); n = float(w.sum(dtype=np.float64))
        if not n > 0:
            return
        f = xp.where(self.w_ > 0, w / xp.where(self.w_ > 0, self.w_, 1.0), 0.0)
        self.w_, self.n_, self.c_, self.occ_ = w, n, n * w, w > 0
        E, U = self._solve(self.e_hat_ * f[None], self._hill_sweeps())
        self._finish(E, U)

    def partial_fit(self, counts, modes=None, h=1.0, origin=0.0, complex_g=False, split_groups=None, _learn=True):
        xp = self.xp
        counts = np.asarray(counts, float)
        if self.n_batches_ == 0:
            if not self._hold(counts, modes, h, origin, complex_g, split_groups):
                return self
            modes = self._select(); _, h, origin, complex_g = self._modes_args
            cap, self.max_sweeps = self.max_sweeps, (self.batch_sweeps if self.batch_sweeps is not None and not
                                                     self.burn_in else self.max_sweeps)
            BLMLBinned.fit(self, self._held, modes, h, origin, complex_g)
            self.max_sweeps = cap; self.n_batches_ = 1; self._held_list = None
            self.history_.append(dict(n=self.n_, preq=np.nan, fc=[f.tolist() for f in self._fc_now()], moved=[], burn_in=True))
            self.preq_.append(np.nan)
            return self
        if counts.shape != self.shape_:
            raise ValueError("partial_fit: the batch must be on the lattice of the first one")

        def absorb(burned):
            w = self.decay * self.w_ + xp.asarray(counts, dtype=self.rdt_)
            n_old = self.n_
            self.w_ = w; self.n_ = float(w.sum(dtype=np.float64)); self.c_ = self.n_ * w; self.occ_ = w > 0
            E = self.e_hat_ * (self.n_ / n_old)                    # same g; the rescale in _solve fixes Q = n^2
            E = xp.where(self.occ_[None], E, 0)
            new = self.occ_ & (xp.abs(E).sum(0) == 0)               # newly occupied bins: their own root from S e
            if bool(xp.any(new)):
                U = self._Sall(E); E = xp.where(new[None], self._update(E, U), E)
            sweeps = self.batch_sweeps if (self.batch_sweeps is not None and burned) else self.max_sweeps
            E, U = self._solve(E, sweeps)
            self._finish(E, U)

        self._stream_step(counts, absorb, learn=_learn); self.n_batches_ += 1
        return self

    def _again(self, counts, learn=True):
        """A held batch revisited: scored leave-batch-out, its gradient pooled, the coefficients swept (not added)."""
        def absorb(burned):
            sweeps = self.batch_sweeps if (self.batch_sweeps is not None and burned) else self.max_sweeps
            E, U = self._solve(self.e_hat_, sweeps); self._finish(E, U)
        self._stream_step(counts, absorb, revisit=True, learn=learn)


class BLMLEmpirical(BLMLBinned):
    """Paper 7: the normalisation on the observed lattice, sum_b g(z_b)^2 Delta = 1 in place of int g^2 = 1. Then
    g(z_b)^2 = h_b = w_b / (n Delta) and S c = n sqrt(h): a band-limited interpolant of the square-root histogram,
    regularised (S is ill-conditioned on a lattice fine for the band). Empty bins are part of the fit (h_b = 0).
      reg='ridge': min ||Sc/n - sqrt h||^2 + lam c'Sc/n^2  ->  (S + lam I) c = n sqrt h        (CG, FFT mat-vec)
      reg='l2'   : min ||Sc/n - sqrt h||^2 + lam ||c/n||^2   ->  (S^2 + lam I) c = n S sqrt h     (CG)
      reg='l1'   : min ||Sc/n - sqrt h||^2 + lam ||c/n||_1                                     (FISTA on a = c/n)
    Every penalty is on the natural coefficients a = c/n (g = sum_b a_b K(. - z_b)), so lam does not scale with n.
    anscombe: sqrt(w + 3/8) in place of sqrt(w) (variance-stabilised target), renormalised to the same n.
    normalise: 'window' (sum_b g_b^2 Delta = 1: f is a density on the observed box) or 'plane' (int g^2 = 1, the
    reproducing-property norm c'Sc/n^2). predict / sqrt_density / predict_grid as BLMLBinned.

    Hidden phase (retrieve): the histogram fixes only |g_b| = sqrt(h_b); the sign (real rank 1), phase (complex) or
    direction (rank K: g_b in R^K or C^K) in each bin is free. retrieve=True estimates it by alternating projections
    (Gerchberg-Saxton with the regularised band-limited fit as the projection): t_bk = sqrt(h_b) u_bk with
    u_b = g_b / |g_b| from the current fit, then the linear fit of every mode to its t_k (warm-started). Each half-step
    minimises J = sum_k ||S_k c_k / n - t_k||^2 + lam pen(c_k) over its own variables, so the reduced objective
    J_red = sum_b (|g_b| - sqrt h_b)^2 + lam pen (J at the best directions: a minimum Hellinger distance fit) never rises
    (J_history_); Anderson mixing (anderson) on the alternation map, kept only when it lowers J_red more than the plain
    step. Stops when |grad J_red| / its first value < outer_tol (l1: the relative change of c), or after max_outer;
    it is non-convex (no certificate; twins / local fixed points are possible), started from the rank-1 fit, modes
    beyond the first and phases from phase_noise. retrieve=None: on for rank K > 1 or complex_g, off for real rank 1
    (the +sqrt h fit); retrieve=True on real rank 1 lets the sign flip (zeros of g, e.g. dark rings)."""

    def __init__(self, reg='ridge', lam=1e-2, anscombe=False, normalise='window', tol=1e-8, max_iter=20000,
                 retrieve=None, outer_tol=1e-6, max_outer=500, anderson=5, phase_noise=0.1, seed=0,
                 engine='cpu', dtype=np.float64, workers=-1, verbose=False):
        super().__init__(tol=tol, engine=engine, dtype=dtype, workers=workers, verbose=verbose,
                         phase_noise=phase_noise, seed=seed, anderson=anderson)
        if reg not in ('ridge', 'l2', 'l1') or normalise not in ('window', 'plane'):
            raise ValueError("reg: 'ridge' | 'l2' | 'l1'; normalise: 'window' | 'plane'")
        self.reg, self.lam, self.anscombe, self.normalise, self.max_iter = reg, float(lam), anscombe, normalise, max_iter
        self.retrieve, self.outer_tol, self.max_outer = retrieve, outer_tol, max_outer

    def fit(self, counts, modes, h=1.0, origin=0.0, init=None, complex_g=False):
        self._setup(counts, modes, h, origin)
        w = np.asarray(counts, float)
        if not w.sum() > 0:
            raise ValueError("counts: no observations")
        self.complex_g_ = bool(complex_g)
        rt = self.retrieve if self.retrieve is not None else (self.K_ > 1 or self.complex_g_)
        if not rt and (self.K_ > 1 or self.complex_g_):
            raise ValueError("BLMLEmpirical: rank K > 1 or complex_g need retrieve=True (the hidden phase)")
        self.retrieve_ = bool(rt)
        return self._fit_counts(w, init)

    # ---------------------------------------------------------------- the linear fit of one mode
    def _lin(self, k, t, c0):
        """The regularised fit of mode k to the target t (real; complex: real and imaginary parts separately)."""
        xp = self.xp
        if xp.iscomplexobj(t):
            r = self._lin(k, t.real, None if c0 is None else c0.real); i = self._lin(k, t.imag, None if c0 is None else c0.imag)
            return r[0] + 1j * i[0], r[1] + i[1], max(r[2], i[2])
        n = self.n_; S = lambda x: self._S(k, x)
        if self.reg == 'ridge':
            return _cg(xp, lambda x: S(x) + self.lam * x, n * t, c0, self.tol, self.max_iter)
        if self.reg == 'l2':
            return _cg(xp, lambda x: S(S(x)) + self.lam * x, n * S(t), c0, self.tol, self.max_iter)
        Lmax = float(xp.max(xp.abs(self.kern_[k].Kf)))          # >= lambda_max(S) (a section of the circulant)
        a, it, res = _fista(xp, S, t, self.lam, 2 * Lmax ** 2, None if c0 is None else c0 / n, self.tol, self.max_iter)
        return n * a, it, res

    def _targets(self, U, t):
        """t_bk = sqrt(h_b) u_bk, u_b = g_b / |g_b| (mode 0 where g_b = 0)."""
        xp = self.xp
        a = xp.sqrt((xp.abs(U) ** 2).sum(0))
        Uu = U / xp.where(a > 0, a, 1.0)[None]
        Uu[0] = xp.where(a > 0, Uu[0], 1.0)
        return t[None] * Uu

    def _Jred(self, C, U, t):
        """The reduced objective sum_b (|g_b| - sqrt h_b)^2 + lam pen (Proposition: J minimised over the directions)."""
        return self._objective(C, U, self._targets(U, t))

    def _grad_norm(self, C, U, t):
        """|grad J_red| (ridge / l2; the stationarity measure of the alternation)."""
        xp, n = self.xp, self.n_
        a = xp.sqrt((xp.abs(U) ** 2).sum(0)); As = xp.maximum(a, 1e-30 * (float(xp.max(a)) or 1.0))
        g = (2 / n) * self._Sall(((a / n - t) / As)[None] * U)
        if self.reg == 'ridge':
            g = g + 2 * self.lam * U / n ** 2
        elif self.reg == 'l2':
            g = g + 2 * self.lam * C / n ** 2
        return float(xp.sqrt(xp.sum(xp.abs(g) ** 2)))

    def _objective(self, C, U, T):
        xp, n = self.xp, self.n_
        J = float(xp.sum(xp.abs(U / n - T) ** 2, dtype=np.float64))
        if self.reg == 'ridge':
            J += self.lam * float(xp.real(xp.sum(xp.conj(C) * U, dtype=self._acc(C)))) / n ** 2
        elif self.reg == 'l2':
            J += self.lam * float(xp.sum(xp.abs(C) ** 2, dtype=np.float64)) / n ** 2
        else:
            ab = (xp.abs(C.real) + xp.abs(C.imag)) if xp.iscomplexobj(C) else xp.abs(C)
            J += self.lam * float(xp.sum(ab, dtype=np.float64)) / n
        return J

    def _fit_counts(self, w, init=None):
        """The fit on the counts w at the current kernels (fit; the streaming refit)."""
        xp = self.xp
        n = float(w.sum())
        self.Delta_ = float(np.prod(self.h_))
        r = np.sqrt(w + 0.375) if self.anscombe else np.sqrt(w)
        r = r * np.sqrt(n / (r ** 2).sum())                    # same total as sqrt(w)
        t = xp.asarray(r / np.sqrt(n * self.Delta_), dtype=self.rdt_)   # sqrt(h)
        self.w_ = xp.asarray(w, dtype=self.rdt_); self.n_ = n; self.occ_ = self.w_ > 0
        self.psd_ = False
        if not self.retrieve_:
            c0 = None if init is None or getattr(init, 'c_raw_', None) is None or init.c_raw_.shape != self.shape_ \
                else xp.asarray(init.c_raw_, dtype=self.rdt_)
            c, it, res = self._lin(0, t, c0)
            C, U = c[None], self._S(0, c)[None]; self.c_raw_ = c
            self.n_outer_, self.J_history_ = 0, []
        else:
            dt = self.cdt_ if self.complex_g_ else self.rdt_
            if init is not None and getattr(init, 'retrieve_', False) and init.c_raw_.shape == (self.K_,) + self.shape_:
                C = xp.asarray(init.c_raw_).astype(dt) * (n / init.n_raw_)     # same g at the new n
                U = self._Sall(C)
            else:                                              # the rank-1 +sqrt h fit, then phases / small modes
                c, _, _ = self._lin(0, t, None); u0 = self._S(0, c)
                rng = np.random.default_rng(self.seed); pn = self.phase_noise; sh = self.shape_
                U = xp.zeros((self.K_,) + sh, dtype=dt); U[0] = u0
                if self.complex_g_:
                    U[0] = u0 * xp.asarray(np.exp(1j * pn * rng.standard_normal(sh)), dtype=dt)
                for k in range(1, self.K_):
                    z = rng.standard_normal(sh) + (1j * rng.standard_normal(sh) if self.complex_g_ else 0)
                    U[k] = pn * xp.abs(u0) * xp.asarray(z / (np.sqrt(2) if self.complex_g_ else 1), dtype=dt)
                C = None
            it = 0; res = 0.0
            def step(C, U):                                     # one alternation: directions from U, then the fits
                nonlocal it, res
                T = self._targets(U, t); Cn = []
                for k in range(self.K_):
                    ck, i_, r_ = self._lin(k, T[k], None if C is None else C[k]); Cn.append(ck); it += i_; res = max(res, r_)
                Cn = xp.stack(Cn); return Cn, self._Sall(Cn)
            C, U = step(C, U); Jc = self._Jred(C, U, t); Jh = [Jc]
            g0 = self._grad_norm(C, U, t) or 1.0; hX, hF = [], []; mA = self.anderson; n_and = 0
            for outer in range(1, self.max_outer + 1):
                CP, UP = step(C, U); JP = self._Jred(CP, UP, t)
                if mA:                                          # Anderson on the alternation map, safeguarded by J
                    f = (CP - C).ravel(); hX.append(C.ravel()); hF.append(f); hX, hF = hX[-(mA + 1):], hF[-(mA + 1):]
                    if len(hX) >= 3:
                        try:
                            CA = (C.ravel() + f - _anderson_mix(xp, hX, hF, f)).reshape(C.shape); UA = self._Sall(CA)
                            JA = self._Jred(CA, UA, t)
                            if JA < JP:
                                CP, UP, JP = CA, UA, JA; n_and += 1
                        except np.linalg.LinAlgError:
                            hX, hF = [], []
                dC = float(xp.sqrt(xp.sum(xp.abs(CP - C) ** 2))) / (float(xp.sqrt(xp.sum(xp.abs(CP) ** 2))) or 1.0)
                C, U, Jc = CP, UP, JP; Jh.append(Jc)
                crit = dC if self.reg == 'l1' else self._grad_norm(C, U, t) / g0
                if self.verbose and outer % 10 == 0:
                    print(f'  outer {outer}: J={Jc:.12g} crit={crit:.2e} anderson={n_and}', flush=True)
                if crit < self.outer_tol:
                    break
            self.n_anderson_ = n_and; self.outer_crit_ = crit
            self.c_raw_ = C; self.n_outer_, self.J_history_ = outer, Jh
        self.n_raw_ = n
        self.n_iter_, self.residual_, self.converged_ = it, res, res < self.tol
        F = (xp.abs(U) ** 2).sum(0)
        Zw = float(xp.sum(F, dtype=np.float64)) / n ** 2 * self.Delta_            # sum_b g_b^2 Delta
        Zp = float(xp.real(xp.sum(xp.conj(C) * U, dtype=self._acc(C)))) / n ** 2  # int g^2 (reproducing property)
        self.window_norm_, self.plane_norm_ = Zw, Zp
        if not (Zw > 0 and Zp > 0):
            raise ValueError(f"BLMLEmpirical: the fit is zero (reg={self.reg!r}, lam={self.lam:g} removes every coefficient); lower lam")
        sc = float(1 / np.sqrt(Zw if self.normalise == 'window' else Zp))
        self._set_state(C * sc, U * sc)
        self.sparsity_ = float(xp.mean(xp.abs(C) > 0)) if self.reg == 'l1' else 1.0
        self.n_samples_ = n; self.sweeps_ = it
        return self

    def _set_state(self, E, U):
        xp, n = self.xp, self.n_
        self.e_hat_, self.Se_ = E, U
        self.density_ = (xp.abs(U) ** 2).sum(0) / n ** 2           # f at the bins under the chosen normalisation
        occ = self.occ_
        self.loglik_ = float((self.w_[occ] * xp.log(xp.maximum(self.density_[occ], 1e-300))).sum(dtype=np.float64)) / n

    def _norm(self, E, U):
        """window: Z = sum_b F_b Delta (f a density on the box); plane: Z = sum_k e_k^H S_k e_k."""
        if self.normalise == 'plane':
            return super()._norm(E, U)
        xp, D = self.xp, self.Delta_
        Z = float(xp.sum((xp.abs(U) ** 2).sum(0), dtype=np.float64)) * D
        return Z, lambda k, dU: 2 * D * float(xp.real(xp.sum(xp.conj(U[k]) * dU, dtype=self._acc(dU))))

    def _refresh(self):
        """After a kernel move at fixed coefficients: S e and the normalisation again."""
        xp = self.xp
        U = self._Sall(self.e_hat_); Z, _ = self._norm(self.e_hat_, U); sc = float(self.n_ / np.sqrt(Z))
        self._set_state(self.e_hat_ * sc, U * sc)

    # ---------------------------------------------------------------- implicit gradient: J at the raw coefficients
    def _ift_state(self):
        xp = self.xp
        C = self.c_raw_ if self.retrieve_ else self.c_raw_[None]
        U = self._Sall(C); F = (xp.abs(U) ** 2).sum(0)
        r = np.sqrt(_np(self.w_) + 0.375) if self.anscombe else np.sqrt(_np(self.w_))
        r = r * np.sqrt(self.n_ / (r ** 2).sum())
        t = xp.asarray(r / np.sqrt(self.n_ * self.Delta_), dtype=self.rdt_)
        a = xp.sqrt(F); As = xp.maximum(a, 1e-30 * float(xp.max(a)))
        return dict(C=C, U=U, F=F, t=t, a=a, As=As)

    def _free(self, st):
        if self.reg == 'l1':
            return (self.xp.abs(st['C']) > 0).astype(self.rdt_)
        return self.xp.ones((1,) * (self.d_ + 1), dtype=self.rdt_)

    def _dP_de(self, st, counts, eps):
        xp = self.xp; m = xp.asarray(counts, dtype=self.rdt_); M = float(m.sum(dtype=np.float64))
        C, U, F = st['C'], st['U'], st['F']                          # P = sum m log(F / Z + eps) / M (scale-free)
        if self.normalise == 'plane':
            Z = _rdot(xp, C, U); dlZ = 2 * U / Z
        else:
            Z = float(xp.sum(F, dtype=np.float64)) * self.Delta_; dlZ = 2 * self._Sall(U) / float(xp.sum(F, dtype=np.float64))
        wn = m / xp.maximum(F + eps * Z, 1e-300); fz = float(xp.sum(wn * F, dtype=np.float64)) / M
        return 2 * self._Sall((wn / M)[None] * U) - fz * dlZ

    def _q(self, st, dU):
        """d of the per-bin residual field q (grad J = (2/n) S q + pen) for a change dU of U."""
        xp, n = self.xp, self.n_
        U, t, a, As = st['U'], st['t'], st['a'], st['As']
        if not self.retrieve_:
            return dU / n
        da = xp.real(xp.conj(U) * dU).sum(0) / As; rho = a / n - t
        return (da / n)[None] * U / As[None] + (rho / As)[None] * (dU - U * (da / As)[None])

    def _hvp(self, st, D):
        xp, n, lam = self.xp, self.n_, self.lam
        dU = self._Sall(D)
        H = (2 / n) * self._Sall(self._q(st, dU))
        if self.reg == 'ridge':
            H = H + 2 * lam * dU / n ** 2
        elif self.reg == 'l2':
            H = H + 2 * lam * D / n ** 2
        return H

    def _mixed(self, st, k, dKf):
        xp, n, lam = self.xp, self.n_, self.lam
        C, U, t, a, As = st['C'], st['U'], st['t'], st['a'], st['As']
        dUk = self._apply(dKf, C[k]); dU = xp.zeros_like(U); dU[k] = dUk
        q = (U / n - t[None]) if not self.retrieve_ else ((a / n - t) / As)[None] * U
        dq = self._q(st, dU)
        out = []
        for j in range(self.K_):
            g = (2 / n) * self._S(j, dq[j])
            if j == k:
                g = g + (2 / n) * self._apply(dKf, q[k])
                if self.reg == 'ridge':
                    g = g + 2 * lam * dUk / n ** 2
            out.append(g)
        return xp.stack(out)

    def score(self, counts):
        """Mean log-probability per count of held-out counts on the same lattice, with f normalised over the box."""
        p = _np(self.density_).astype(np.float64); p = p / p.sum()
        w = np.asarray(counts, float); m = w > 0
        return float((w[m] * np.log(np.maximum(p[m], 1e-300))).sum() / w.sum())

    @classmethod
    def select(cls, train, val, modes, lams, fit_kw=None, **kw):
        """Fit for each lam (warm-started along the grid, largest first) and keep the best held-out score on val."""
        best, prev, rows = None, None, []
        for lam in sorted(lams, reverse=True):
            m = cls(lam=lam, **kw).fit(train, modes, init=prev, **(fit_kw or {}))
            s = m.score(val); rows.append((lam, s)); prev = m
            if best is None or s > best[0]:
                best = (s, m)
        best[1].path_ = sorted(rows)
        return best[1]


class BLMLEmpiricalStreaming(_KernelLearner, BLMLEmpirical):
    """Paper 7 on counts arriving in batches on one fixed lattice, with the streaming functionality of _KernelLearner
    (burn-in with selection, fit over epochs with leave-batch-out revisits, tol, split_groups, bcd_frac / freeze_frac,
    trace_trainable): partial_fit adds the batch (held counts times decay first) and refits warm-started (CG / FISTA
    from the current coefficients; with retrieve, the alternation from the current phases, for at most batch_outer
    outer steps after the burn-in). Leave-batch-out and hill-climb candidates: a warm refit on the held counts less
    the batch to convergence (max_outer; the implicit gradient needs a stationary fit). The residual for tol: the
    outer criterion of the last refit (outer_crit_; the CG / FISTA residual without retrieve). Kernel parameters: on
    the prequential score of the empirical density (fc_update='adam' by default); implicit_grad=True (alias: implicit)."""

    def __init__(self, decay=1.0, batch_outer=None, burn_in=0, fc_update='adam', lr=0.1, betas=(0.9, 0.999),
                 step=1.05, z_se=1.0, param_every=3, grad_eps=1e-3, implicit_grad=True, search='shared', bcd_frac=1.0,
                 freeze_frac=0.0, trace_trainable=False, **kw):
        if 'implicit' in kw:
            implicit_grad = kw.pop('implicit')
        super().__init__(**kw)
        self._base_kw = dict(kw)
        self.decay, self.batch_outer = decay, batch_outer
        self._learner_init(burn_in, fc_update, lr, betas, step, z_se, param_every, grad_eps, implicit_grad,
                           search, bcd_frac, freeze_frac, trace_trainable)

    def _base(self):
        return BLMLEmpirical(**self._base_kw)

    def _val_score(self, counts):
        return self.score(counts)

    def _residual(self):
        return float(getattr(self, 'outer_crit_', None) if self.retrieve_ else self.residual_)

    def _refit(self, w, burned):
        cap = self.max_outer
        if burned and self.batch_outer is not None:
            self.max_outer = self.batch_outer
        try:
            self._fit_counts(w, init=self)
        finally:
            self.max_outer = cap

    def _candidate_refit(self):                               # to convergence (max_outer), as _loo_enter
        self._refit(_np(self.w_).astype(float), False)

    def _loo_enter(self, m):                                  # to convergence: the implicit gradient needs a stationary fit
        w = np.maximum(_np(self.w_).astype(float) - m, 0.0)
        if w.sum() > 0:
            self._refit(w, False)

    def partial_fit(self, counts, modes=None, h=1.0, origin=0.0, complex_g=False, split_groups=None, _learn=True):
        counts = np.asarray(counts, float)
        if self.n_batches_ == 0:
            if not self._hold(counts, modes, h, origin, complex_g, split_groups):
                return self
            modes = self._select(); _, h, origin, complex_g = self._modes_args
            BLMLEmpirical.fit(self, self._held, modes, h=h, origin=origin, complex_g=complex_g)
            self.n_batches_ = 1; self._held_list = None
            self.history_.append(dict(n=self.n_, preq=np.nan, fc=[f.tolist() for f in self._fc_now()], moved=[], burn_in=True))
            self.preq_.append(np.nan)
            return self
        if counts.shape != self.shape_:
            raise ValueError("partial_fit: the batch must be on the lattice of the first one")
        self._stream_step(counts, lambda burned: self._refit(self.decay * _np(self.w_).astype(float) + counts, burned),
                          learn=_learn)
        self.n_batches_ += 1
        return self

    def _again(self, counts, learn=True):
        self._stream_step(counts, lambda burned: self._refit(_np(self.w_).astype(float), burned), revisit=True, learn=learn)


def _anderson_mix(xp, hX, hF, f):
    """(dX + dF) gamma of Anderson type II from the histories: the small normal equations from inner products
    accumulated in float64 (no float64 copies of the history, which a GPU without fast float64 cannot afford),
    solved on the host."""
    dX = [hX[i + 1] - hX[i] for i in range(len(hX) - 1)]; dF = [hF[i + 1] - hF[i] for i in range(len(hF) - 1)]
    m = len(dF); G = np.empty((m, m)); rhs = np.array([_rdot(xp, a, f) for a in dF])
    for i in range(m):
        for j in range(i, m):
            G[i, j] = G[j, i] = _rdot(xp, dF[i], dF[j])
    g = np.linalg.solve(G + 1e-12 * np.trace(G) * np.eye(m), rhs)
    if not np.all(np.isfinite(g)):
        raise np.linalg.LinAlgError
    out = float(g[0]) * (dX[0] + dF[0])
    for i in range(1, m):
        out += float(g[i]) * (dX[i] + dF[i])
    return out


def _rdot(xp, a, b):
    return float(xp.real(xp.sum(xp.conj(a) * b, dtype=np.complex128 if xp.iscomplexobj(a) or xp.iscomplexobj(b) else np.float64)))


def _icg(xp, A, b, tol, maxit):
    """CG for a symmetric positive semidefinite operator A on (complex as real) arrays, consistent b."""
    x = xp.zeros_like(b); r = b.copy(); p = r.copy(); rs = _rdot(xp, r, r); bn = np.sqrt(rs) or 1.0
    for it in range(1, maxit + 1):
        Ap = A(p); pAp = _rdot(xp, p, Ap)
        if not pAp > 0:
            return x, it, np.sqrt(rs) / bn
        a = rs / pAp; x = x + a * p; r = r - a * Ap; rs_new = _rdot(xp, r, r)
        if np.sqrt(rs_new) / bn < tol:
            return x, it, np.sqrt(rs_new) / bn
        p = r + (rs_new / rs) * p; rs = rs_new
    return x, maxit, np.sqrt(rs) / bn


def _cg(xp, A, b, x0, tol, maxit):
    """Conjugate gradients for SPD A (a callable); relative residual tol."""
    x = xp.zeros_like(b) if x0 is None else x0.copy()
    r = b - A(x); p = r.copy(); rs = float(xp.sum(r * r, dtype=np.float64)); bn = np.sqrt(float(xp.sum(b * b, dtype=np.float64))) or 1.0
    for it in range(1, maxit + 1):
        Ap = A(p); a = rs / float(xp.sum(p * Ap, dtype=np.float64))
        x = x + a * p; r = r - a * Ap
        rs_new = float(xp.sum(r * r, dtype=np.float64))
        if np.sqrt(rs_new) / bn < tol:
            return x, it, np.sqrt(rs_new) / bn
        p = r + (rs_new / rs) * p; rs = rs_new
    return x, maxit, np.sqrt(rs) / bn


def _fista(xp, S, t, lam, L, x0, tol, maxit):
    """min ||S a - t||^2 + lam ||a||_1 by FISTA (gradient 2 S (S a - t), step 1/L, L >= 2 lambda_max(S)^2); stops
    when the relative change of a is below tol."""
    x = xp.zeros_like(t) if x0 is None else x0.copy(); y = x.copy(); tk = 1.0
    for it in range(1, maxit + 1):
        g = 2 * S(S(y) - t)
        z = y - g / L; xn = xp.sign(z) * xp.maximum(xp.abs(z) - lam / L, 0)
        tn = float(1 + np.sqrt(1 + 4 * tk * tk)) / 2; y = xn + ((tk - 1) / tn) * (xn - x)
        dx = float(xp.sqrt(xp.sum((xn - x) ** 2, dtype=np.float64))); nx = float(xp.sqrt(xp.sum(xn * xn, dtype=np.float64))) or 1.0
        x, tk = xn, tn
        if dx / nx < tol:
            return x, it, dx / nx
    return x, maxit, dx / nx


# ---------------------------------------------------------------- helpers
def _np(a):
    return a.get() if hasattr(a, 'get') else np.asarray(a)


def _next_fast_len(n):
    """Smallest 2^a 3^b 5^c >= n (cuFFT-friendly)."""
    best = 1 << int(np.ceil(np.log2(n)))
    a = 1
    while a <= best:
        b = a
        while b <= best:
            c = b
            while c < n:
                c *= 5
            best = min(best, c)
            b *= 3
        a *= 2
    return best


def _cd_root(xp, b, cc, sbb):
    if xp is np:
        return cd_root(b, cc, sbb)
    disc = xp.sqrt(b * b + 4.0 * sbb * cc)
    return xp.where(b >= 0, 2.0 * cc / xp.maximum(b + disc, 1e-300), (-b + disc) / (2.0 * sbb))


def _secular(xp, R2, s, c, iters=16):
    """Per bin, the root lam > max_k s_k of phi(lam) = lam sum_k R2_k / (lam - s_k)^2 = c, phi falling from +inf to 0
    there. Newton on phi^(-1/2) - c^(-1/2), nearly linear in mu = lam - max s at the double pole, safeguarded by the
    bisection bracket: quadratic convergence, round-off in about 12 iterations. R2: (K, ...) = |R_bk|^2, s: (K, 1, ...),
    c: (...)."""
    smax = float(xp.max(s)); tot = R2.sum(0); d = s - smax
    cpos = xp.maximum(c, 1e-30)
    hi = (tot + xp.sqrt(tot * tot + 4 * cpos * smax * tot)) / (2 * cpos)       # phi(smax + hi) <= c
    lo = xp.zeros_like(hi); mu = hi
    for _ in range(iters):
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            q = 1.0 / (mu[None] - d); W = R2 * (q * q); A = W.sum(0); B = (W * q).sum(0)
            lam = smax + mu; phi = lam * A
            big = phi > c
            lo = xp.where(big, mu, lo); hi = xp.where(big, hi, mu)
            nt = mu + 2 * phi * (1 - xp.sqrt(phi / cpos)) / (A - 2 * lam * B)
        mu = xp.where((nt >= lo) & (nt <= hi), nt, 0.5 * (lo + hi))           # closed: a converged step stays
    return smax + mu
