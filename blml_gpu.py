"""
BLMLQuick GPU version using CuPy.

Mirrors the CPU BLMLQuick API (scaling-free fit; kinds/periods on fit();
block-Jacobi CD until max_rho < tol, no max_iter). Mixed sinc/Dirichlet
kernels via 1-D tables with min-circular-distance indexing -- the same
tables as the CPU path (built by blml_trivial), so results agree with CPU.

Usage on Colab (match the wheel to the runtime's CUDA major version
from nvidia-smi):
    !pip install cupy-cuda13x numba   # CUDA 13 runtime; use cupy-cuda12x on CUDA 12
    from blml_trivial import BLMLQuick
    q = BLMLQuick(engine="gpu").fit(X, [{0: dict(fc=0.073), 1: dict(fc=0.27, kind='dirichlet', period=24)}])
"""
import os
import warnings
import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
    try:                                   # J0, J1 on the device, for radial (ball) tables (blml_trivial.bessel_j)
        from cupyx.scipy.special import j0 as _j0, j1 as _j1
        _J01 = (_j0, _j1)
    except ImportError:
        _J01 = None
except ImportError:
    _HAS_CUPY = False
    _J01 = None


from blml_trivial import projected_front as _projected_front

@_projected_front
class _BLMLQuickCUDA:
    """BLMLQuick(engine='gpu'): the CUDA implementation, block-Jacobi CD. API mirrors CPU BLMLQuick.

    fit() is scaling-free: data, fc, periods in consistent units
    (the selector owns any IQR scaling, as on CPU).
    """

    def __init__(self, tol=1e-4, block_size=4096,
                 seed=0, verbose=False, dtype=None, bin_scale=0.1, max_sweeps=None, solver='cd', lbfgs_m=20):
        if not _HAS_CUPY:
            raise ImportError(
                "CuPy required. On Colab: !pip install cupy-cuda13x "
                "(CUDA 13 runtime) or cupy-cuda12x (CUDA 12 runtime) -- "
                "match nvidia-smi's CUDA major version")
        if solver not in ('cd', 'lbfgs'):
            raise ValueError("solver: 'cd' (block-Jacobi CD with Anderson mixing) or 'lbfgs'")
        self.solver, self.lbfgs_m = solver, int(lbfgs_m)   # complex / rank K fits (blml_trivial.lbfgs_psd)
        self.tol = tol
        self.max_sweeps = max_sweeps   # None: to tol; k: k sweeps, then e'Se = n^2 (as blml_trivial.BLMLQuick)
        self.block_size = block_size
        self.seed = seed
        self.bin_scale = bin_scale  # bin width h = bin_scale * n^{-1/4} / fc
        self.verbose = verbose
        self.dtype = np.dtype(dtype or os.environ.get('BLML_DTYPE') or 'float32')   # env BLML_DTYPE=float64 switches every job
        self._cp_dtype = cp.float64 if self.dtype == np.float64 else cp.float32

    def fit(self, X, modes, init=None, rebin_tol=1.0, signs=None, complex_g=False, sample_weight=None, _counts=None):
        """modes: the model, as on the CPU (blml_trivial.parse_modes); rank K = len(modes). On the GPU
        the modes share kinds and periods (one lattice) and may differ in fc. Complex fits use the
        same start search as the CPU (self.n_starts; self.starts_).
        signs: optional fixed sign of g (array per sample, majority per bin, or callable s(Z) at the
        bin centres; see blml_trivial.resolve_signs). Not combined with dep columns (the marginal has its own g).
        dep columns (modes): if any, the marginal BLML of the other dims is
        fitted in the same pass (same cutoffs, same bins; one kernel evaluation serves both), and
        predict_marginal / predict_ratio become available. Fitting the marginal on the joint's bins
        gives the same estimate as fitting it on its own: bins that differ only in `dep` have
        identical marginal kernel rows, and their coefficients split their merged bin's in
        proportion to the counts.
        init: a previous fit (either engine) with the same modes (warm start for
        streaming / sliding windows). At the optimum e_b = w_b / g(z_b), with
        g(z) = (1/n) sum_b e_b K(z, z_b) = sqrt(f(z)) and w_b the raw count of bin b.
          * While the target width bin_scale n^{-1/4} / fc stays within (1 + rebin_tol)^{1/4} of
            the one the lattice was built for (blml_trivial.keep_lattice), the lattice is kept: bins of retained points coincide and keep their coefficients,
            bins of removed points drop out, and each new bin starts at its exact
            one-coordinate minimizer given the others.
          * Otherwise the data are re-binned on the lattice for the current n (h ~ n^{-1/4})
            and every bin starts at w_b / g_prev(z_b), g_prev evaluated with the exact kernel."""
        from blml_trivial import parse_modes, _shared_kernel, modes_lattice, strip_lattice, lattice_fixed
        sw = None
        if _counts is None:
            d_in = np.atleast_2d(X).shape[1]
            h_fixed = modes_lattice(modes, d_in)
            if h_fixed is not None:                    # counts on a fixed lattice (as BLMLQuick._fit_lattice)
                lattice_fixed(modes, 'a fixed lattice (h in the modes)')
                Z = np.atleast_2d(np.asarray(X, dtype=np.float64))
                w = np.ones(len(Z)) if sample_weight is None else np.asarray(sample_weight, float)
                if w.shape != (len(Z),) or not np.all(w >= 0):
                    raise ValueError("sample_weight: (n,) non-negative counts")
                return self.fit(None, strip_lattice(modes), init=init, signs=signs, complex_g=complex_g, _counts=(Z, w, h_fixed))
            sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
            if sw is not None and (sw.shape != (len(np.atleast_2d(X)),) or not np.all(sw > 0)):
                raise ValueError("sample_weight: (n,) positive weights")
        specs, dep = parse_modes(modes, (_counts[0] if _counts is not None else np.atleast_2d(X)).shape[1])
        self._specs, self.spec_ = specs, modes
        kinds, periods, rank = specs[0]['kinds'], specs[0]['periods'], len(specs)
        umodes, modes = modes, (None if _shared_kernel(specs) else [dict(sp) for sp in specs])
        self.modes_ = modes                              # per-mode specs (None: one shared kernel), as on the CPU
        psd = bool(complex_g) or rank > 1
        if psd and signs is not None:
            raise ValueError("complex_g / rank > 1 / modes estimate the phases; do not pass signs as well")
        if psd and dep is not None:
            # conditional: joint and marginal as two separate rank-K fits (no fused dep= fit)
            return self._fit_psd_conditional(X, umodes, dep, complex_g, init, rebin_tol, sw, _counts)
        from blml_trivial import lattice_kernel, dirichlet_table, is_radial, resolve_signs
        # Quantize in float64 regardless of self.dtype (see CPU fit()).
        if _counts is not None:
            Zc, wc, hc = _counts
            keep_c = wc > 0
            n, d = float(wc[keep_c].sum()), Zc.shape[1]
        else:
            X = np.atleast_2d(np.asarray(X, dtype=np.float64))
            n_rows, d = X.shape
            n = n_rows if sw is None else float(sw.sum())     # point i counts sw_i times
        kinds = list(kinds) if kinds is not None else ['sinc'] * d
        if dep is not None:
            from blml_trivial import radial_groups
            for _, dims in radial_groups(kinds):
                if 0 < sum(j in dep for j in dims) < len(dims):
                    raise ValueError(f"a radial group over columns {dims} straddles dep and conditioning columns")
        periods = list(periods) if periods is not None else [None] * d
        rng = np.random.default_rng(self.seed)

        fc_arr = self._fc0(d)
        mode_fcs = mode_specs = None
        if modes is not None:
            mode_specs = []
            for md in modes:
                mk = list(md.get('kinds') or kinds); mp = list(md.get('periods') or periods)
                for k in range(d):        # one lattice: a periodic dim must use the lattice's period
                    if str(mk[k]).partition(':')[0] in ('dirichlet', 'pball') and (periods[k] is None or mp[k] != periods[k]):
                        raise ValueError(f"mode periodic in dim {k} needs the fit's period {periods[k]} there")
                mode_specs.append((np.broadcast_to(np.asarray(md['fc'], dtype=float), (d,)).copy(), mk, mp))
            mode_fcs = [ms[0] for ms in mode_specs]
            fc_arr = mode_fcs[0]                         # real start uses mode 0's kernel
        # Same binning as CPU: h = 0.1 * n^{-0.25} / fc (finest over the modes).
        fc_lat = np.max(mode_fcs, axis=0) if mode_fcs is not None else fc_arr
        from blml_trivial import lattice_target
        h_paper = lattice_target(self.bin_scale * (n ** -0.25) / fc_lat, [sp['kinds'] for sp in specs]) if _counts is None else hc
        from blml_trivial import keep_lattice
        # Re-bin (and start from g_prev) once the target width has moved too far (keep_lattice);
        # a fixed lattice: the caller's lattice, init's coefficients reused if it has the same one.
        if _counts is None:
            rebin = init is not None and not keep_lattice(init, h_paper, rebin_tol)
        else:
            rebin = init is not None and not np.allclose(getattr(init, '_h_vec', np.nan), hc)
        self.h_target_ = h_paper if (init is None or rebin) else init.h_target_
        if init is not None and not rebin:
            h_paper = np.asarray(init._h_vec, dtype=float)
        self.n_lattice_ = n if (init is None or rebin) else init.n_lattice_
        self.rebinned_, self.fc_lattice_ = rebin, fc_lat
        from blml_trivial import lattice_fold_specs
        kern = lattice_kernel(fc_lat, h_paper, kinds, periods, d, lattice_fold_specs(specs))
        self._fold = list(kern.pk.fold)
        if _counts is None:
            Xq = kern.quantize(X)
            Xq_c = np.ascontiguousarray(Xq)
            vdt = np.dtype((np.void, Xq_c.dtype.itemsize * Xq_c.shape[1]))
            _, idx, inv, counts = np.unique(Xq_c.view(vdt), return_index=True,
                                            return_inverse=True, return_counts=True)
            Z = Xq[idx]                      # (B, d) unique bin centers
            w = counts.astype(float) if sw is None else np.bincount(np.ravel(inv), weights=sw, minlength=len(idx))
            sg = resolve_signs(signs, Z, inverse=np.ravel(inv), n_samples=n_rows)
        else:
            sg = resolve_signs(signs, Zc)
            sg = None if sg is None else sg[keep_c]
            Z, w = Zc[keep_c], wc[keep_c]
            counts = w
            if np.abs(np.round(Z / hc) * hc - Z).max() > 1e-6 * hc.max():
                raise ValueError("bin centers must lie on the lattice h * integers")
        B = Z.shape[0]
        if sg is not None and dep is not None:
            raise ValueError("signs with dep=... (the marginal's g has its own signs)")
        h_vec = kern.h
        Zi = np.round(Z / h_vec).astype(np.int64)   # (B, d)

        # ---- kernel tables (padded), same values as the CPU path ----
        kinds0, periods0 = (mode_specs[0][1], mode_specs[0][2]) if mode_specs else (kinds, periods)
        from blml_trivial import warm_start_Se
        pre = None
        if (init is not None and not rebin and sg is None and dep is None and not psd and mode_specs is None):
            pre = warm_start_Se(init, Zi, fc_arr, kinds, periods, h_vec)
        Zspan = Zi if pre is None or not len(pre[2]) else np.vstack([Zi, pre[2]])   # tables span removed bins too
        Tmain = self._tables_gpu(fc_arr, kinds0, periods0, h_vec, Zspan, d)
        sbb_const = Tmain['sbb']
        diag_h = Tmain['diag']                           # K(0) per dim (marginal S_bb below)

        # ---- move to GPU ----
        Zi_gpu = self._ranked(Zi, Tmain)
        w_gpu = cp.asarray(w, dtype=self._cp_dtype)
        e_gpu = cp.ones(B, dtype=self._cp_dtype)
        threads = 256
        e_new_mask = None
        # The marginal (dep=...) warm-starts too when init was fitted with the same dep dims.
        dep_req = [] if dep is None else sorted({int(k) for k in np.atleast_1d(dep)})
        warm_marg = init is not None and dep_req and getattr(init, 'dep_', None) == dep_req
        em0, em_new_mask = None, None
        if init is not None and not rebin:
            zkeys = [z.tobytes() for z in np.ascontiguousarray(Zi)]
            prev = {z.tobytes(): v for z, v in zip(np.ascontiguousarray(init._Zi), init.e_hat_)}
            e0 = np.array([prev.get(k, np.nan) for k in zkeys], dtype=float)
            e_new_mask = np.isnan(e0)
            if sg is not None:
                e0 = np.abs(e0)
            e_gpu = cp.asarray(np.where(e_new_mask, 0.0, e0), dtype=self._cp_dtype)
            if warm_marg:
                prev_m = {z.tobytes(): v for z, v in zip(np.ascontiguousarray(init._Zi), init.e_marg_)}
                em0 = np.array([prev_m.get(k, np.nan) for k in zkeys], dtype=float)
                em_new_mask = np.isnan(em0)
        elif init is not None:
            g = init.sqrt_density(Z)
            if sg is not None:
                g = g * sg                                                # t-space: want s*g > 0
            e_new_mask = ~(g > 1e-3 * max(float(np.abs(g).max()), 1e-300))    # g ~ 0: w/g blows up; 1-D start
            e0 = w / np.where(e_new_mask, 1.0, g)
            e_gpu = cp.asarray(np.where(e_new_mask, 0.0, e0), dtype=self._cp_dtype)
            if warm_marg:
                rest_d = [k for k in range(d) if k not in dep_req]
                gm = init.sqrt_density(Z, e=init.e_marg_, dims=rest_d)
                em_new_mask = ~(gm > 1e-3 * max(float(gm.max()), 1e-300))
                em0 = w / np.where(em_new_mask, 1.0, gm)

        dep_set = [] if dep is None else sorted({int(k) for k in np.atleast_1d(dep)})
        if dep_set and (dep_set[0] < 0 or dep_set[-1] >= d or len(dep_set) == d):
            raise ValueError(f'dep {dep} must name some, but not all, of the {d} dims')
        depmask = sum(1 << k for k in dep_set)
        nfit = 2 if dep_set else 1               # fit 0: joint; fit 1: marginal of the other dims

        sg_gpu = None if sg is None else cp.asarray(sg, dtype=self._cp_dtype)

        def launch_block(Zj_gpu, m, J=None):
            # J: indices (into the B bins) of the block's columns; needed only for signs
            out = cp.empty((B, m), dtype=self._cp_dtype)
            out_m = cp.empty((B, m), dtype=self._cp_dtype) if nfit == 2 else out
            self._launch(Tmain, Zi_gpu, Zj_gpu, B, m, d, out, out_m, depmask)
            if sg_gpu is not None:      # fixed sign pattern: kernel D S D
                out *= sg_gpu[:, None] * sg_gpu[J][None, :]
            return (out, out_m) if nfit == 2 else (out,)

        def mem_cols(per_entry):
            # columns of a B-row kernel block such that the block(s) and their per-entry temporaries
            # (per_entry bytes) use at most a quarter of the GPU's free memory (as the sweep's blocks)
            free_bytes = cp.cuda.Device().mem_info[0]
            return max(1, min(self.block_size, int(0.25 * free_bytes // (nfit * max(B, 1) * per_entry))))

        def exact_Se(es):
            # Fresh S @ e for every fit, float64 accumulation (no incremental drift).
            accs = [cp.zeros(B, dtype=cp.float64) for _ in range(nfit)]
            Bs0 = min(mem_cols(np.dtype(self.dtype).itemsize + 8), B)   # the block and its float64 copy
            for s in range(0, B, Bs0):
                J = slice(s, min(s + Bs0, B))
                blks = launch_block(Zi_gpu[J], J.stop - J.start, J)
                for f in range(nfit):
                    accs[f] += blks[f].astype(cp.float64) @ es[f][J].astype(cp.float64)
            return [a.astype(self._cp_dtype) for a in accs]

        nn = self._cp_dtype(n)

        def residual(e, Se):
            rho = cp.abs(Se * e / (nn * w_gpu) - 1.0)
            rho = cp.where((e > 0) & (w_gpu > 0), rho, 0.0)
            return float(cp.max(rho))

        def objective(e, Se):
            e64 = e.astype(cp.float64)
            return float(0.5 * e64 @ Se.astype(cp.float64)
                         - n * cp.sum(w_gpu * cp.log(cp.maximum(e64, 1e-300)))) / n

        if nfit == 2:                                    # the marginal's S_bb: the units off the dep side (radial: whole)
            from blml_local import _unit_k0
            from blml_trivial import kernel_units
            k0u = _unit_k0(kinds0, fc_arr, periods0)
            sbb_m = float(np.prod([k0u[i] for i, (_, dims) in enumerate(kernel_units(kinds0, d)) if dims[0] not in dep_set]))
        else:
            sbb_m = None
        sbbs = [self._cp_dtype(sbb_const)] + ([self._cp_dtype(sbb_m)] if nfit == 2 else [])
        em_gpu = (cp.asarray(np.where(em_new_mask, 0.0, em0), dtype=self._cp_dtype) if em0 is not None
                  else cp.ones(B, dtype=self._cp_dtype))
        es = [e_gpu] + ([em_gpu] if nfit == 2 else [])
        masks = [e_new_mask] + ([em_new_mask] if nfit == 2 else [])
        Bc = min(mem_cols(np.dtype(self.dtype).itemsize + 8), max(B, 1))
        def add_cols(Zj_host, coef, sign, f_list):
            # Ses[f] += sign * S(bins, Zj) @ coef[f] for lattice points Zj (chunks of Bc columns)
            for s0 in range(0, len(Zj_host), Bc):
                Zj = self._ranked(Zj_host[s0:s0 + Bc], Tmain); m = len(Zj)
                blks = launch_block(Zj, m)
                for f in f_list:
                    Ses[f] += sign * (blks[f].astype(cp.float64) @ coef[f][s0:s0 + m].astype(cp.float64)).astype(self._cp_dtype)
        if pre is not None and e_new_mask is not None and not e_new_mask.all():
            # previous fit's exact S e on the shared bins; kernel columns only for removed and new bins
            Se0, newp, Zi_rem, e_rem = pre
            Ses = [cp.asarray(np.real(Se0), dtype=self._cp_dtype)]
            if len(Zi_rem):
                add_cols(Zi_rem, [cp.asarray(np.real(e_rem), dtype=self._cp_dtype)], -1.0, [0])
            newI = np.where(e_new_mask)[0]
            for s0 in range(0, len(newI), Bc):            # rows of the new bins (their e is 0 now)
                I = newI[s0:s0 + Bc]; blks = launch_block(Zi_gpu[I], len(I))
                Ses[0][cp.asarray(I)] = (blks[0].astype(cp.float64).T @ es[0].astype(cp.float64)).astype(self._cp_dtype)
        else:
            Ses = exact_Se(es)
        if any(mk is not None and mk.any() for mk in masks):
            # New bins: exact 1-D minimizer of the objective in e_b given the others,
            # e_b = argmin 1/2 S_bb e^2 + b e - n w_b log e with b = (Se)_b (e_b = 0 now).
            newcols = {}
            for f, mk in enumerate(masks):
                if mk is None or not mk.any():
                    continue
                J = cp.asarray(np.where(mk)[0])
                b_term, cc, s0 = Ses[f][J], nn * w_gpu[J], sbbs[f]
                disc = cp.sqrt(b_term * b_term + 4.0 * s0 * cc)
                es[f][J] = cp.where(b_term >= 0, 2.0 * cc / (b_term + disc), (-b_term + disc) / (2.0 * s0))
                newcols[f] = np.where(mk)[0]
            if len(newcols) == nfit and all(np.array_equal(v, newcols[0]) for v in newcols.values()):
                I = newcols[0]                             # add the new bins' columns, not a full pass
                add_cols(Zi[I], [es[f][cp.asarray(I)] for f in range(nfit)], 1.0, list(range(nfit)))
            else:
                Ses = exact_Se(es)
        self.n_new_bins_ = 0 if e_new_mask is None else int(e_new_mask.sum())

        # Block-Jacobi CD until max_rho < tol (no max_iter), for each fit; the fits share the
        # kernel blocks. Block size = self.block_size (default 256): each block costs one
        # kernel launch for the Gram block(s) plus ~8 cupy launches per fit for the
        # vectorized update, so small blocks are launch-overhead bound.
        # Block size changes only the trajectory, not the fixed point.
        # At least 8 blocks per sweep: with one block holding every bin (small B,
        # e.g. 1-D) the update is fully parallel and the safeguard must shrink it.
        # Large blocks amortize launch overhead (block 256 -> ~1250: 5x faster at n = 1e4, same
        # sweeps); capped at B/8 and by memory: the block's Gram output(s) plus about as much
        # again for temporaries may use at most a quarter of the GPU's free memory.
        free_bytes = cp.cuda.Device().mem_info[0]
        mem_cap = int(0.25 * free_bytes // (2 * nfit * B * np.dtype(self.dtype).itemsize))
        Bs = max(1, min(self.block_size, -(-B // 8), mem_cap))
        n_sweeps = 0
        # Step sizes tried by the block-level safeguard (largest first).
        steps = cp.asarray(0.5 ** np.arange(0, 31), dtype=cp.float64)

        def sweep(active, safeguard):
            perm = rng.permutation(B)
            for start in range(0, B, Bs):
                I = perm[start:start + Bs]
                m = len(I)
                I_gpu = cp.asarray(I)
                blks = launch_block(Zi_gpu[I_gpu], m, I_gpu)
                for f in active:
                    e, Se, sbb, Sblk = es[f], Ses[f], sbbs[f], blks[f]
                    e_old = e[I_gpu]
                    b_term = Se[I_gpu] - sbb * e_old
                    cc = nn * w_gpu[I_gpu]
                    disc = cp.sqrt(b_term * b_term + 4.0 * sbb * cc)
                    e_star = cp.where(b_term >= 0,
                                      2.0 * cc / (b_term + disc),
                                      (-b_term + disc) / (2.0 * sbb))
                    dlt = e_star - e_old
                    v = Sblk @ dlt                       # S_{:,I} d, needed for the update anyway
                    if safeguard[f]:
                        # Largest t in {1, 1/2, ...} with Phi(e + t d) <= Phi(e):
                        # dPhi(t) = t d'(Se)_I + t^2/2 d'S_II d - n sum w_I log(1 + t d/e).
                        # d is a descent direction (each coordinate moves toward its
                        # own 1-D minimizer), so such a t exists.
                        d64 = dlt.astype(cp.float64)
                        lin = d64 @ Se[I_gpu].astype(cp.float64)
                        quad = d64 @ v[I_gpu].astype(cp.float64)
                        logs = cp.log1p(steps[:, None] * (d64 / e_old)[None, :]) @ cc.astype(cp.float64)
                        ok = steps * lin + 0.5 * steps ** 2 * quad - logs <= 0
                        t = cp.where(ok.any(), steps[cp.argmax(ok)], 0.0).astype(self._cp_dtype)
                        e[I_gpu] = e_old + t * dlt
                        Se += t * v
                    else:
                        e[I_gpu] = e_star
                        Se += v

        # Parallel (Jacobi) block updates are fast but can overshoot when bins
        # are strongly coupled (1-D: each bin sees ~1/(fc h) neighbors). Plain
        # sweeps run until one increases Phi; that sweep is undone and redone
        # with the block-level step-size safeguard, which then stays on. The
        # objective is computed every sweep anyway, so the plain path is free.
        safeguard = [False] * nfit
        done = [False] * nfit
        conv = [False] * nfit
        sweeps_f = [0] * nfit
        max_rhos = [np.inf] * nfit
        rho_hists = [[] for _ in range(nfit)]
        obj_hist = []
        obj_prev = [objective(es[f], Ses[f]) for f in range(nfit)]
        cap = self.max_sweeps
        while not all(done):
            if cap is not None and n_sweeps >= cap:
                break
            n_sweeps += 1
            active = [f for f in range(nfit) if not done[f]]
            saves = {f: (es[f].copy(), Ses[f].copy()) for f in active}
            sweep(active, safeguard)
            obj_now = {f: objective(es[f], Ses[f]) for f in active}
            redo = [f for f in active if not safeguard[f] and obj_now[f] > obj_prev[f] + 1e-10 * abs(obj_prev[f])]
            for f in redo:
                es[f][...] = saves[f][0]; Ses[f][...] = saves[f][1]
                safeguard[f] = True
            if redo:
                sweep(redo, safeguard)
                for f in redo:
                    obj_now[f] = objective(es[f], Ses[f])
            exact = None
            for f in active:
                sweeps_f[f] += 1
                obj_prev[f] = obj_now[f]
                if not bool(cp.all(cp.isfinite(es[f]))):
                    max_rhos[f] = np.inf; rho_hists[f].append(np.inf); done[f] = True
                    continue
                rho = residual(es[f], Ses[f])
                if rho < self.tol:
                    exact = exact or exact_Se(es)
                    Ses[f][...] = exact[f]
                    rho = residual(es[f], Ses[f])
                max_rhos[f] = rho
                rho_hists[f].append(rho)
                if f == 0:
                    obj_hist.append(objective(es[0], Ses[0]) if rho < self.tol else obj_now[0])
                    if self.verbose:
                        print(f"  sweep {n_sweeps}: max|rho|={rho:.2e} Phi/n={obj_hist[-1]:.6f}", flush=True)
                if rho < self.tol:
                    conv[f] = done[f] = True
                # Stall guard (not an iteration cap): stop only if the residual has
                # improved by < 1% over the last 50 sweeps, e.g. when tol is below
                # what the working precision can reach. converged stays False.
                elif len(rho_hists[f]) > 50 and rho_hists[f][-1] > 0.99 * min(rho_hists[f][:-50]):
                    done[f] = True
        if cap is not None and not all(conv):            # capped: normalise each fit exactly (e'Se = n^2)
            Ses = exact_Se(es)
            for f in range(nfit):
                if conv[f] or not bool(cp.all(cp.isfinite(es[f]))):
                    continue
                sc = n / np.sqrt(max(float(es[f].astype(cp.float64) @ Ses[f].astype(cp.float64)), 1e-300))
                es[f] *= self._cp_dtype(sc); Ses[f] *= self._cp_dtype(sc)
                max_rhos[f] = residual(es[f], Ses[f]); conv[f] = max_rhos[f] < self.tol
        se_real = Ses[0] if (conv[0] or cap is not None) else None   # exact: met tol, or capped and recomputed
        e_gpu = es[0]
        n_sweeps, converged, max_rho, rho_hist = sweeps_f[0], conv[0], max_rhos[0], rho_hists[0]
        if nfit == 2:
            self.dep_ = dep_set
            self.e_marg_ = cp.asnumpy(es[1]).astype(self.dtype)
            self.marg_converged_, self.marg_sweeps_, self.marg_residual_ = conv[1], sweeps_f[1], max_rhos[1]
        else:
            self.dep_ = None
        safeguard = safeguard[0]
        self.complex_g_, self.rank_ = bool(complex_g), int(rank)
        self.mode_fcs_ = None
        if psd:
            # Rank-K (complex) BLML from the real solution (see blml_trivial._solve_modes_cd).
            K = int(rank)
            same = mode_specs is None or all(np.allclose(ms[0], mode_specs[0][0]) and ms[1] == mode_specs[0][1]
                                             and ms[2] == mode_specs[0][2] for ms in mode_specs)
            if same:
                launchers = [(lambda J, m: launch_block(Zi_gpu[J], m)[0], float(sbb_const))]   # one shared kernel
            else:
                launchers = [self._mode_launcher(ms[0], ms[1], ms[2], h_vec, Zi, Zi_gpu, B, d) for ms in mode_specs]
                self.mode_fcs_ = mode_fcs; self.mode_specs_ = mode_specs
            e0_warm = None
            if init is not None and getattr(init, 'e_hat_', None) is not None:
                # warm start: e_bk = w_b g_k(z_b) / f(z_b) from the previous fit (exact at its optimum)
                gprev = init.sqrt_density(Z) if getattr(init, 'mode_specs_', None) is None else \
                    np.stack([init._mode_sqrt_density(Z, k) for k in range(len(init.mode_specs_))], 1)
                gprev = gprev if gprev.ndim == 2 else gprev[:, None]
                if gprev.shape[1] == K:
                    fprev = np.maximum((np.abs(gprev) ** 2).sum(1), 1e-300)
                    e0_warm = w[:, None] * gprev / fprev[:, None]
                    e0_warm[np.sqrt(fprev) <= 1e-3 * np.sqrt(fprev.max())] = 0    # g ~ 0: w/g blows up
            solve = lambda e0: self._psd_solve(e_gpu, launchers, Zi_gpu, w_gpu, B, Bs, n, K, bool(complex_g), rng, e0=e0)
            starts = None; self.starts_ = None
            nh, ns = getattr(self, 'n_starts', (3, 2))
            if complex_g and e0_warm is None and nh + ns > 0:
                # same start search as the CPU (blml_trivial.complex_starts), on mode 0's band
                from blml_trivial import complex_starts
                if any(str(k).startswith(('dirichlet', 'pball')) for k in kinds0):
                    warnings.warn("complex_g: start search not implemented for periodic dims; using one start")
                else:
                    starts = complex_starts(Zi, w, h_vec, fc_arr, kinds0, nh, ns, seed=self.seed)
                    if starts is None:
                        warnings.warn("complex_g: FFT grid of the bin box too large for the start search; using one start")
            if starts:
                best, self.starts_ = None, []
                for family, sc, e0 in starts:
                    out = solve(e0[:, None])
                    ll = -n * out[5][-1] if out[5] and np.isfinite(out[5][-1]) else -np.inf
                    self.starts_.append(dict(family=family, start_score=sc, loglik=float(ll), converged=bool(out[2]),
                                             n_sweeps=int(out[1])))
                    if best is None or ll > best[0]:
                        best = (ll, out)
                out = best[1]
            else:
                out = solve(e0_warm)
            e_gpu, n_sweeps, converged, max_rho, rho_hist, obj_hist, safeguard = out
            SEp = getattr(self, '_SE_end', None)
            se_real = None if SEp is None else (SEp[:, 0] if int(rank) == 1 else SEp)

        e_out = cp.asnumpy(e_gpu)
        self.Se_ = None if se_real is None else cp.asnumpy(se_real) * (sg if sg is not None else 1.0)
        if sg is not None:
            e_out = e_out * sg
        self.signs_ = sg
        self.e_hat_ = e_out if psd else e_out.astype(self.dtype)
        self.Z_ = Z
        self.w_ = counts
        self.n_samples_ = n
        self.n_bins_ = B
        self.dim_ = d
        self.n_sweeps_ = n_sweeps
        self.converged_ = converged
        self.residual_ = max_rho
        self.safeguard_used_ = safeguard
        self.rho_history_ = rho_hist
        self.obj_history_ = obj_hist
        self.kinds_ = kinds
        self.periods_ = periods
        self._fc_arr = fc_arr
        self._h_vec = h_vec
        self._Zi = Zi
        return self

    def _fc0(self, d):
        """Cut-offs (d,) of mode 0 of the spec being fitted."""
        return np.asarray(self._specs[0]['fc'], dtype=float).copy()

    def _tables_gpu(self, fc_arr, kinds, periods, h_vec, Zi, d, want=()):
        """One kernel spec's value tables for the bins Zi (blml_kernels.ProductKernel, built on the device and cached
        there, one per kinds / periods), plus K(0) per dim ('diag', for the marginal's S_bb) and S_bb ('sbb')."""
        from blml_kernels import ProductKernel
        fold = getattr(self, '_fold', None) or [True] * d
        key = (tuple(kinds), tuple(periods), tuple(fold))
        pks = self.__dict__.setdefault('_pks', {})
        if key not in pks:
            pks[key] = ProductKernel(kinds, periods, fold, engine='gpu', f32=self._cp_dtype == cp.float32)
        T = pks[key].tables(np.asarray(fc_arr, float), periods, h_vec, Zi, want)
        if 'sbb' not in T:
            diag, sbb = pks[key].k0(fc_arr, periods)
            T['diag'], T['sbb'] = diag, sbb
        return T

    def _launch(self, T, Zi_gpu, Zj_gpu, B, m, d, out, out_m, depmask=0):
        """The block S(bins, Zj) into out (and the marginal into out_m with depmask): blml_local_gpu.cols. Zi_gpu,
        Zj_gpu: the coordinates ranked for T (self._ranked)."""
        from blml_local_gpu import cols
        cols(Zi_gpu, Zj_gpu, T, out, out_m, depmask, f32=self._cp_dtype == cp.float32)

    @staticmethod
    def _ranked(Z, T):
        """Lattice coordinates on the device as the kernels read them (rank columns: ranks)."""
        from blml_kernels import rank_coords
        return cp.asarray(rank_coords(np.ascontiguousarray(Z), T))

    def _mode_launcher(self, fc_k, kinds, periods, h_vec, Zi, Zi_gpu, B, d, dirichlet_table=None):
        """One mode's kernel (own fc, kinds, periods) on the shared lattice: (launch(J, m), sbb), J the block's bins.
        The mode's coordinates are ranked for its own tables (rank columns: each mode's table its own values)."""
        T = self._tables_gpu(fc_k, kinds, periods, h_vec, Zi, d)
        Zk = self._ranked(Zi, T) if any(v is not None for v in T['vals']) else Zi_gpu

        def launch(J, m):
            out = cp.empty((B, m), dtype=self._cp_dtype)
            self._launch(T, Zk, Zk[J], B, m, d, out, out)
            return out
        return launch, T['sbb']

    def _psd_solve(self, e_real, launchers, Zi_gpu, w_gpu, B, Bs, n, K, complex_g, rng, e0=None):
        """General rank-K solver (GPU): one launcher per distinct kernel (len 1 = shared kernel).
        Shared: closed-form row update e_b = t R_b/||R_b||. Per-mode kernels: e_bk = R_bk/(lam - s_k)
        with lam from lam sum_k |R_bk|^2/(lam - s_k)^2 = n w_b (vectorised bisection). Block-Jacobi
        with Anderson acceleration (fixed visiting order) and the likelihood safeguard."""
        shared = len(launchers) == 1
        rdt = self._cp_dtype
        cdt = (cp.complex64 if rdt == cp.float32 else cp.complex128) if complex_g else rdt
        acc_dt = cp.complex128 if complex_g else cp.float64
        sbbs = np.array([l[1] for l in launchers] * (K if shared else 1))
        sbbs_g = cp.asarray(sbbs, dtype=cp.float64)
        pn = getattr(self, 'phase_noise', 0.1)
        ws = None if e0 is None else np.asarray(e0, dtype=complex if complex_g else np.float64)
        ws = None if ws is None else (ws[:, None] if ws.ndim == 1 else ws)
        e0 = np.zeros((B, K), dtype=complex if complex_g else np.float64)
        if ws is not None:                       # warm start (init= or a start search): replaces the real start
            k0 = min(K, ws.shape[1]); e0[:, :k0] = ws[:, :k0]; er = np.abs(ws[:, 0])
        else:
            k0 = 1; er = cp.asnumpy(e_real).astype(np.float64)
            e0[:, 0] = er * (np.exp(1j * pn * rng.standard_normal(B)) if complex_g else 1.0)
        for k in range(k0, K):                   # missing modes: small noise (zero would stay zero)
            noise = ((rng.standard_normal(B) + 1j * rng.standard_normal(B)) / np.sqrt(2) if complex_g
                     else rng.standard_normal(B))
            e0[:, k] = pn * er * noise
        w64 = w_gpu.astype(cp.float64)

        def mm(blk, V):
            if complex_g:
                return blk @ V.real.astype(blk.dtype) + 1j * (blk @ V.imag.astype(blk.dtype))
            return blk @ V.astype(blk.dtype)

        def blocks_for(J, m):                    # J: the block's bins (each launcher reads its own coordinates)
            return [l[0](J, m) for l in launchers]

        def apply(blks, D):          # S restricted to the block's columns, times D (m, K) -> (B, K)
            if shared:
                return mm(blks[0], D)
            return cp.stack([mm(blks[k], D[:, k:k + 1])[:, 0] for k in range(K)], 1)

        def exact_Se(E):
            acc = cp.zeros((B, K), dtype=acc_dt)
            for s0 in range(0, B, Bs):
                J = slice(s0, min(s0 + Bs, B))
                acc += apply(blocks_for(J, J.stop - J.start), E[J]).astype(acc_dt)
            return acc.astype(cdt)

        def loglik(E, SE):
            F = (cp.abs(SE.astype(acc_dt)) ** 2).sum(1)
            tr = float(cp.real(cp.vdot(E.astype(acc_dt), SE.astype(acc_dt))))
            return float(w64 @ cp.log(cp.maximum(F, 1e-300)) - n * np.log(max(tr, 1e-300))) / n

        def residual(E, SE):
            E64, S64 = E.astype(acc_dt), SE.astype(acc_dt)
            en2 = (cp.abs(E64) ** 2).sum(1)
            dev = S64 * (en2 / (n * w64))[:, None] - E64
            return float(cp.max(cp.sqrt((cp.abs(dev) ** 2).sum(1) / cp.maximum(en2, 1e-300))))

        def row_update(R, EI, cc):
            if shared:
                s0 = float(sbbs[0])
                ar = cp.sqrt((cp.abs(R) ** 2).sum(1))
                t = 2.0 * cc / (ar + cp.sqrt(ar * ar + 4.0 * s0 * cc))
                en = cp.sqrt((cp.abs(EI) ** 2).sum(1))
                dirn = cp.where((ar > 0)[:, None], R / cp.where(ar > 0, ar, 1.0)[:, None], EI / cp.maximum(en, 1e-30)[:, None])
                return t[:, None] * dirn
            a2 = (cp.abs(R.astype(acc_dt)) ** 2)
            smax = float(sbbs.max()); lo = cp.zeros(len(cc)); hi = cp.ones(len(cc))
            phi = lambda x: (smax + x) * (a2 / (smax + x[:, None] - sbbs_g[None]) ** 2).sum(1)
            for _ in range(200):
                big = phi(hi) >= cc
                if not bool(big.any()):
                    break
                hi = cp.where(big, hi * 2, hi)
            for _ in range(60):
                mid = 0.5 * (lo + hi); up = phi(mid) > cc
                lo = cp.where(up, mid, lo); hi = cp.where(up, hi, mid)
            lam = smax + 0.5 * (lo + hi)
            return (R.astype(acc_dt) / (lam[:, None] - sbbs_g[None])).astype(cdt)

        E = cp.asarray(e0, dtype=cdt)
        SE = exact_Se(E)
        sc = n / np.sqrt(max(float(cp.real(cp.vdot(E.astype(acc_dt), SE.astype(acc_dt)))), 1e-300))
        E *= sc; SE *= sc                                  # S (c E) = c S E: no second pass
        self._SE_end = None
        if self.solver == 'lbfgs':
            from blml_trivial import lbfgs_psd
            E, SE, n_sweeps, converged, max_rho, rho_hist, obj_hist = lbfgs_psd(
                E, SE, w64, n, exact_Se, loglik, residual, self.tol, self.max_sweeps, self.lbfgs_m, self.verbose, cp)
            self._SE_end = SE
            return (E[:, 0] if K == 1 else E), n_sweeps, converged, max_rho, rho_hist, obj_hist, False
        steps = cp.asarray(0.5 ** np.arange(0, 31), dtype=cp.float64)
        aa_m = int(getattr(self, 'anderson', 5))
        perm_fixed = rng.permutation(B) if aa_m else None

        def sweep(safeguard):
            nonlocal E, SE
            perm = perm_fixed if perm_fixed is not None else rng.permutation(B)
            for start in range(0, B, Bs):
                I = cp.asarray(perm[start:start + Bs]); m = len(I)
                blks = blocks_for(I, m)
                EI = E[I]
                R = SE[I] - cp.asarray(sbbs, dtype=EI.real.dtype)[None] * EI
                d = (row_update(R, EI, n * w_gpu[I]) - EI).astype(cdt)
                V = apply(blks, d).astype(cdt)
                if safeguard:
                    S64, V64, E64, d64 = SE.astype(acc_dt), V.astype(acc_dt), E.astype(acc_dt), d.astype(acc_dt)
                    a0 = float(cp.real(cp.vdot(E64, S64)))
                    a1 = float(cp.real(cp.vdot(d64, S64[I])) + cp.real(cp.vdot(E64, V64)))
                    a2 = float(cp.real(cp.vdot(d64, V64[I])))
                    F = (cp.abs(S64[None] + steps[:, None, None] * V64[None]) ** 2).sum(2)
                    Ls = cp.log(cp.maximum(F, 1e-300)) @ w64 - n * cp.log(cp.maximum(a0 + a1 * steps + a2 * steps ** 2, 1e-300))
                    L0 = float(w64 @ cp.log(cp.maximum((cp.abs(S64) ** 2).sum(1), 1e-300))) - n * np.log(max(a0, 1e-300))
                    ok = Ls >= L0
                    st = float(steps[int(cp.argmax(ok))]) if bool(ok.any()) else 0.0
                    E[I] = EI + st * d; SE += st * V
                else:
                    E[I] = EI + d; SE += V

        def flat(a):
            return cp.concatenate([a.real.ravel(), a.imag.ravel()]) if complex_g else a.ravel()
        hG, hSG, hR = [], [], []
        self.aa_accepted_ = 0
        safeguard, n_sweeps, converged = False, 0, False
        rho_hist, obj_hist = [], []
        L_prev = loglik(E, SE)
        cap, max_rho = self.max_sweeps, residual(E, SE)
        while True:
            if cap is not None and n_sweeps >= cap:
                break
            n_sweeps += 1
            E_save, SE_save = E.copy(), SE.copy()
            sweep(safeguard)
            L_now = loglik(E, SE)
            if not safeguard and L_now < L_prev - 1e-10 * abs(L_prev):
                E, SE = E_save, SE_save
                safeguard = True
                sweep(safeguard)
                L_now = loglik(E, SE)
            if aa_m:
                hG.append(E.copy()); hSG.append(SE.copy()); hR.append(flat(E - E_save).astype(cp.float64))
                if len(hG) > aa_m + 1:
                    hG.pop(0); hSG.pop(0); hR.pop(0)
                if len(hG) >= 2:
                    dR = cp.stack([hR[j + 1] - hR[j] for j in range(len(hR) - 1)], 1)
                    G = cp.asnumpy(dR.T @ dR); b = cp.asnumpy(dR.T @ hR[-1])
                    gam = np.linalg.lstsq(G + 1e-12 * np.trace(G) * np.eye(len(G)), b, rcond=None)[0]
                    E_aa, SE_aa = hG[-1].copy(), hSG[-1].copy()
                    for j, gj in enumerate(gam):
                        E_aa -= float(gj) * (hG[j + 1] - hG[j]); SE_aa -= float(gj) * (hSG[j + 1] - hSG[j])
                    L_aa = loglik(E_aa, SE_aa) if bool(cp.all(cp.isfinite(E_aa))) else -np.inf
                    if L_aa >= L_now:
                        E, SE, L_now = E_aa, SE_aa, L_aa; self.aa_accepted_ += 1
                    else:
                        hG, hSG, hR = hG[-1:], hSG[-1:], hR[-1:]
            L_prev = L_now
            if not bool(cp.all(cp.isfinite(E))):
                max_rho = np.inf; rho_hist.append(max_rho); break
            max_rho = residual(E, SE)
            if max_rho < self.tol:
                SE = exact_Se(E); max_rho = residual(E, SE)
            rho_hist.append(max_rho); obj_hist.append(-L_now)
            if self.verbose and n_sweeps % 50 == 0:
                print(f"  psd sweep {n_sweeps}: max|rho|={max_rho:.2e} -L/n={-L_now:.6f}", flush=True)
            if max_rho < self.tol:
                converged = True; break
            if (len(rho_hist) > 500 and rho_hist[-1] > 0.99 * min(rho_hist[:-500])
                    and obj_hist[-501] - obj_hist[-1] < 1e-9 * abs(obj_hist[-1])):
                break
        if not converged and cap is not None and bool(cp.all(cp.isfinite(E))):   # capped: e'Se = n^2
            SE = exact_Se(E)
            sc = float(n / np.sqrt(max(float(cp.real(cp.vdot(E.astype(acc_dt), SE.astype(acc_dt)))), 1e-300)))
            E, SE = E * sc, SE * sc
            max_rho = residual(E, SE); converged = max_rho < self.tol
            obj_hist.append(-loglik(E, SE)); rho_hist.append(max_rho)
            self._SE_end = SE
        elif converged:
            self._SE_end = SE                              # exact: recomputed when the residual met tol
        if K == 1:
            E = E[:, 0]
        return E, n_sweeps, converged, max_rho, rho_hist, obj_hist, safeguard

    def fc_grad(self, Xq, units, **kw):
        """param_grad for the fc of each unit (a list of dims per unit)."""
        return self.param_grad(Xq, units, **kw)

    def param_grad(self, Xq, units, implicit_grad=False, eps_rel=0.0, mode='full', weights=None, acc=None):
        """d log f(x) / d log theta_u at Xq for each unit u: dims (its fc) or (dims, parameter name), e.g.
        ([1], 'period'); coefficients fixed and the density renormalised; as blml_trivial.BLMLQuick.param_grad,
        from the same kernel classes (derivative rows built on the device). The normaliser's derivative
        from lattice matvecs with the unit's table swapped for its derivative (the fit's CUDA kernel), g
        and dg from the exact kernel at Xq (on the GPU; radial groups on the host). Shared-kernel fits
        without dep columns. Returns (m, len(units)). implicit_grad, eps_rel, mode, weights, acc: as on the CPU
        (the one-sweep response, real rank 1; the gradient of log(f + eps); pooling over batches)."""
        from blml_trivial import (kernel_units, kernel_unit, fc_grad_from_parts, _split_kind, is_radial, radial_groups,
                                  sweep_response, pooled_fc_grad, _want_params, _diag_and_dlog, kernel_class, unit_theta,
                                  param_grad_modes, conditional_param_grad)
        if getattr(self, '_joint', None) is not None:      # rank-K / complex conditional: joint minus marginal
            if mode != 'full':
                raise ValueError("param_grad: pooling (mode='points' / 'lattice') is for joint fits")
            kw = dict(implicit_grad=implicit_grad, eps_rel=eps_rel, weights=weights)
            return conditional_param_grad(lambda X, u: self._joint.param_grad(X, u, **kw),
                                          lambda X, u: self._marg.param_grad(X, u, **kw), Xq, units, self.dep_, self.dim_)
        dep = getattr(self, 'dep_', None); per_mode = getattr(self, 'mode_specs_', None) is not None
        if dep or per_mode:                               # the general lattice path, with this engine's kernels
            if implicit_grad or mode != 'full':
                raise ValueError("param_grad: the fused conditional and per-mode kernels have no one-sweep response "
                                 "and no pooling")
            import functools
            import blml_local_gpu as G
            f32 = self._cp_dtype == cp.float32
            fns = dict(rows_fn=functools.partial(G.rows, f32=f32), pts_fn=functools.partial(G.point_rows, f32=f32))
            fold = getattr(self, '_fold', None) or [True] * self.dim_
            n = float(self.n_samples_)
            if per_mode:
                ms = [(fc, k, p, fold) for fc, k, p in self.mode_specs_]
                return param_grad_modes(self._Zi, self._h_vec, self.Z_, self.e_hat_, n, ms, Xq, units, eps_rel, **fns)
            d = self.dim_; rest = [j for j in range(d) if j not in dep]
            fcv = np.broadcast_to(np.asarray(self._fc_arr, float), (d,)); kinds = list(self.kinds_)
            per = list(self.periods_) if self.periods_ is not None else [None] * d
            joint = lambda X, u: param_grad_modes(self._Zi, self._h_vec, self.Z_, self.e_hat_, n,
                                                  [(fcv, kinds, per, fold)], X, u, eps_rel, **fns)
            marg = lambda X, u: param_grad_modes(np.ascontiguousarray(self._Zi[:, rest]), self._h_vec[rest],
                                                 self.Z_[:, rest], self.e_marg_, n,
                                                 [(fcv[rest], [kinds[j] for j in rest], [per[j] for j in rest],
                                                   [fold[j] for j in rest])], X, u, eps_rel, **fns)
            return conditional_param_grad(joint, marg, Xq, units, dep, d)
        if implicit_grad and (np.iscomplexobj(self.e_hat_) or self.e_hat_.ndim != 1
                         or getattr(self, 'signs_', None) is not None):
            raise ValueError("fc_grad(implicit_grad=True): real rank-1 fits without fixed signs")
        d, n = self.dim_, float(self.n_samples_)
        fc = np.broadcast_to(np.asarray(self._fc_arr, float), (d,)).copy()
        kinds, periods = list(self.kinds_), list(self.periods_)
        allu = kernel_units(kinds, d)
        want = _want_params(allu, units)
        E = self.e_hat_ if self.e_hat_.ndim == 2 else self.e_hat_[:, None]
        cx = np.iscomplexobj(E)
        E_gpu = cp.asarray(E, dtype=cp.complex128 if cx else cp.float64)
        Zi, h, B = self._Zi, self._h_vec, len(self._Zi)
        wl = list(want) if mode != 'points' else []
        T = self._tables_gpu(fc, kinds, periods, h, Zi, d, wl)
        f32 = self._cp_dtype == cp.float32

        def mv(grad):                                  # S E and, with grad, (dS_u E) per wanted unit: one GPU pass
            from blml_local_gpu import rows as grows
            parts = [E.real, E.imag] if cx else [np.asarray(E, float)]
            out = []
            for P in parts:
                rr = [grows(Zi, Zi, np.ascontiguousarray(P[:, k], float), T, grad=grad, f32=f32) for k in range(P.shape[1])]
                out.append((np.stack([r_[0] for r_ in rr], 1), np.stack([r_[1] for r_ in rr], 1)))
            if len(out) == 2:
                return out[0][0] + 1j * out[1][0], out[0][1] + 1j * out[1][1]
            return out[0]
        quad = lambda SE: float(cp.real(cp.vdot(E_gpu, SE))) / n ** 2
        Se_ = getattr(self, 'Se_', None)
        if wl:
            SE_h, dS = mv(True)
        SE = cp.asarray(np.asarray(Se_).reshape(E.shape) if Se_ is not None else (SE_h if wl else mv(False)[0]),
                        dtype=E_gpu.dtype)
        Zn = quad(SE)
        eps = float(eps_rel * cp.median((cp.abs(SE / n) ** 2).sum(1) / Zn)) if eps_rel else 0.0
        dSE = [cp.asarray(dS[:, :, q], dtype=E_gpu.dtype) for q in range(len(wl))]
        dZ = [quad(x) for x in dSE]
        if mode == 'lattice':
            return pooled_fc_grad(self, acc, want, allu, fc, periods, E.astype(np.complex128 if cx else np.float64),
                                  cp.asnumpy(SE), Zn, dZ, [cp.asnumpy(x) for x in dSE], n, implicit_grad)
        Xq = np.atleast_2d(np.asarray(Xq, dtype=float))
        m = len(Xq); U = len(want)
        wq = np.ones(m) if weights is None else np.asarray(weights, float)
        radial = any(is_radial(k) for k in kinds)
        xp = np if radial else cp
        Zc = self.Z_ if radial else cp.asarray(self.Z_)
        Ec = E.astype(np.complex128 if cx else np.float64) if radial else E_gpu
        g = np.zeros((m,) + E.shape[1:], dtype=complex if cx else float); dg = np.zeros((U,) + g.shape, dtype=g.dtype)
        step = max(1, int(2e7 // max(1, B * (len(allu) + 1))))
        for i0 in range(0, m, step):
            Xc = xp.asarray(Xq[i0:i0 + step])
            diffs = [Xc[:, k, None] - Zc[None, :, k] for k in range(d)]
            facs = [kernel_unit(kind, fc, dims, [diffs[k] for k in dims], periods, need_grad=False, xp=xp)[0]
                    for kind, dims in allu]
            G = facs[0]
            for F in facs[1:]:
                G = G * F
            v = G @ Ec / n
            g[i0:i0 + step] = cp.asnumpy(v) if xp is cp else v
            for u, (i, p) in enumerate(want):
                kind, dims = allu[i]
                dG = kernel_class(kind).eval_dlog(unit_theta(kind, fc, periods, dims), p, [diffs[k] for k in dims], xp)
                for j, F in enumerate(facs):
                    if j != i:
                        dG = dG * F
                v = dG @ Ec / n
                dg[u, i0:i0 + step] = cp.asnumpy(v) if xp is cp else v
            if implicit_grad:                               # d mean score / d e, kernel part
                gc = xp.asarray(g[i0:i0 + step, 0])
                va = G.T @ (xp.asarray(wq[i0:i0 + step]) * 2.0 * gc / (n * (gc ** 2 + eps)))
                v_acc = (v_acc if i0 else 0) + (cp.asnumpy(va) if xp is cp else va)
        if mode == 'points':
            F = np.maximum((np.abs(g) ** 2).sum(1), 1e-300) + eps
            cont = np.array([wq @ (2 * np.real((np.conj(g) * dg[u]).sum(1)) / F) for u in range(U)])
            return dict(cont=cont, W=float(wq.sum()), v=(v_acc if implicit_grad else None), Zi=self._Zi.copy())
        if implicit_grad:
            v_acc = v_acc * m / wq.sum()
        out = fc_grad_from_parts(g, dg, Zn, dZ, eps)
        if implicit_grad:
            e = np.asarray(E[:, 0], float); Se = cp.asnumpy(SE[:, 0]).real; w = np.asarray(self.w_, float)
            sbb, dsbb = _diag_and_dlog(allu, want, fc, periods)
            dT = sweep_response(e, Se, np.stack([cp.asnumpy(d[:, 0]).real for d in dSE]), w, n, sbb, dsbb)
            out = out + (dT @ (v_acc / m - 2.0 * Se / (n ** 2 * Zn)))[None, :]
        return out

    def _gram(self, Xq_gpu, Z_gpu, dims):
        """Continuous product kernel over ``dims`` between query and centers."""
        from blml_trivial import dirichlet_modes_scale
        G = cp.ones((Xq_gpu.shape[0], Z_gpu.shape[0]), dtype=self._cp_dtype)
        from blml_trivial import kernel_class
        for k in dims:
            diff = Xq_gpu[:, None, k] - Z_gpu[None, :, k]
            fck = float(self._fc_arr[k])
            if self.kinds_[k] == 'sinc':
                G *= (fck * cp.sinc(fck * diff)).astype(self._cp_dtype)
            elif str(self.kinds_[k]).partition(':')[0] == 'slits':   # fc sinc(fc t) - fi sinc(fi t)
                G *= kernel_class(self.kinds_[k]).eval(dict(fc=[fck]), [diff], cp).astype(self._cp_dtype)
            else:                                                   # dirichlet (with fill: the periodic two slits)
                P = float(self.periods_[k])
                G *= kernel_class(self.kinds_[k]).eval(dict(fc=[fck], period=P), [diff], cp).astype(self._cp_dtype)
        return G

    def predict(self, Xq):
        """Density f(x) = sum_k |(1/n) sum_b e_bk K(x, z_b)|^2 with the exact kernel (K = 1: g^2)."""
        if getattr(self, 'mode_specs_', None) is not None:    # each mode with its own kernel
            return sum(np.abs(self._mode_sqrt_density(Xq, k)) ** 2 for k in range(len(self.mode_specs_)))
        g = self.sqrt_density(Xq)
        return (np.abs(g) ** 2).sum(1) if g.ndim == 2 else np.abs(g) ** 2

    def _fit_psd_conditional(self, X, modes, dep, complex_g, init, rebin_tol, sw=None, counts=None):
        """Conditional rank-K (complex) BLML: joint fit on all columns and a separate marginal fit on the
        non-dep columns (same spec restricted to them); predict_ratio = joint / marginal. counts (Z, w, h): on a
        fixed lattice, the marginal's counts the joint's summed over the dep columns."""
        from blml_trivial import marginal_modes, strip_dep, with_lattice, marginal_counts
        mk = dict(tol=self.tol, block_size=self.block_size, seed=self.seed, dtype=self.dtype, bin_scale=self.bin_scale,
                  max_sweeps=self.max_sweeps)
        kw = dict(complex_g=complex_g, rebin_tol=rebin_tol)
        ij, im = getattr(init, '_joint', None), getattr(init, '_marg', None)
        if counts is not None:
            Z, w, h = counts; d = Z.shape[1]; rest = [k for k in range(d) if k not in dep]
            Zm, wm = marginal_counts(Z, w, rest)
            joint = _BLMLQuickCUDA(**mk).fit(Z, with_lattice(strip_dep(modes), h), init=ij, sample_weight=w, **kw)
            marg = _BLMLQuickCUDA(**mk).fit(Zm, with_lattice(marginal_modes(modes, rest), np.asarray(h)[rest]), init=im,
                                            sample_weight=wm, **kw)
        else:
            X = np.atleast_2d(np.asarray(X, dtype=np.float64)); d = X.shape[1]
            rest = [k for k in range(d) if k not in dep]
            joint = _BLMLQuickCUDA(**mk).fit(X, strip_dep(modes), init=ij, sample_weight=sw, **kw)
            marg = _BLMLQuickCUDA(**mk).fit(X[:, rest], marginal_modes(modes, rest), init=im, sample_weight=sw, **kw)
        self.__dict__.update({k: v for k, v in joint.__dict__.items()})
        self._joint, self._marg, self.dep_, self.spec_ = joint, marg, dep, modes
        return self

    def predict_marginal(self, Xrest):
        """Marginal density of the non-dep dims (fit with dep=...), at points given in those dims."""
        if getattr(self, 'dep_', None) is None:
            raise RuntimeError("fit with dep=... to get the marginal")
        if getattr(self, '_marg', None) is not None:           # rank-K / complex conditional
            return self._marg.predict(Xrest)
        Xrest = np.atleast_2d(np.asarray(Xrest, dtype=float))
        rest = [k for k in range(self.dim_) if k not in self.dep_]
        Xq = np.zeros((len(Xrest), self.dim_)); Xq[:, rest] = Xrest
        return self.sqrt_density(Xq, e=self.e_marg_, dims=rest) ** 2

    def sample_conditional(self, Xrest, n_samples=100, seed=0, n_prop=400):
        """Samples of the dep columns given the rest, on the GPU (blml_trivial.sample_conditional: any kinds, rank K)."""
        from blml_trivial import sample_conditional
        return sample_conditional(self, Xrest, n_samples, seed, n_prop, xp=cp, j01=_J01, dt=self._cp_dtype)

    def predict_ratio(self, Xq):
        """Conditional density of the dep dim(s) given the others, joint / marginal (fit with dep=...)."""
        Xq = np.atleast_2d(np.asarray(Xq, dtype=float))
        rest = [k for k in range(self.dim_) if k not in self.dep_]
        return self.predict(Xq) / np.maximum(self.predict_marginal(Xq[:, rest]), 1e-300)

    def _mode_sqrt_density(self, Xq, k):
        """g_k(x) for a fit with per-mode kernels (exact continuous kernel, evaluated on the host)."""
        from blml_trivial import _continuous_product_kernel
        fck, kk, pk = self.mode_specs_[k]
        Xq = np.atleast_2d(np.asarray(Xq, dtype=float)); out = np.empty(len(Xq), dtype=complex)
        Kf = _continuous_product_kernel(fck, kk, pk, self.dim_)
        for i in range(0, len(Xq), 4096):
            out[i:i + 4096] = Kf(Xq[i:i + 4096], self.Z_) @ self.e_hat_[:, k]
        return out / self.n_samples_

    def sqrt_density(self, Xq, e=None, dims=None):
        """g(x) = (1/n) sum_b e_b K(x, z_b) with the exact kernel (signed; f = g^2)."""
        if not hasattr(self, "e_hat_"):
            raise RuntimeError("Call fit() before predict().")
        Xq = np.atleast_2d(np.asarray(Xq, dtype=float))
        B = self.Z_.shape[0]
        Z_gpu = cp.asarray(self.Z_)
        e_src = self.e_hat_ if e is None else e
        from blml_trivial import is_radial, _continuous_product_kernel
        if any(is_radial(k) for k in self.kinds_):      # radial: exact kernel on the host (dims: a marginal's columns)
            dd = list(range(self.dim_)) if dims is None else list(dims)
            fcv = np.broadcast_to(np.asarray(self._fc_arr, float), (self.dim_,))
            Kf = _continuous_product_kernel(fcv[dd], [self.kinds_[k] for k in dd],
                                            None if self.periods_ is None else [self.periods_[k] for k in dd], len(dd))
            E = cp.asnumpy(e_src) if hasattr(e_src, 'get') else np.asarray(e_src)
            out = np.empty((len(Xq),) + E.shape[1:], dtype=E.dtype if E.dtype.kind == 'c' else float)
            for i in range(0, len(Xq), 4096):
                out[i:i + 4096] = Kf(Xq[i:i + 4096][:, dd], self.Z_[:, dd]) @ E
            return out / self.n_samples_
        e_gpu = cp.asarray(e_src, dtype=(cp.complex128 if np.iscomplexobj(e_src) else self._cp_dtype))
        dims = range(self.dim_) if dims is None else dims
        # Bounded tiles of (queries x bins): ~2^24 entries each (128 MB in float64, a few such
        # temporaries), whatever the number of queries or the fit's block size.
        Bs = min(self.block_size, B, 4096)
        Qc = max(1, (1 << 24) // Bs)
        out = np.empty((Xq.shape[0],) + tuple(e_gpu.shape[1:]), dtype=complex if e_gpu.dtype.kind == 'c' else float)
        for q in range(0, Xq.shape[0], Qc):
            Xq_gpu = cp.asarray(Xq[q:q + Qc])
            v_gpu = cp.zeros((len(Xq_gpu),) + tuple(e_gpu.shape[1:]), dtype=e_gpu.dtype)
            for s in range(0, B, Bs):
                be = min(s + Bs, B)
                G = self._gram(Xq_gpu, Z_gpu[s:be], dims)
                v_gpu += (G.astype(e_gpu.real.dtype) @ e_gpu[s:be]) if e_gpu.dtype.kind == 'c' else G @ e_gpu[s:be]
            out[q:q + Qc] = cp.asnumpy(v_gpu)
        return out / self.n_samples_

    def __repr__(self):
        r = getattr(self, 'rank_', None)
        return f"BLMLQuick(engine='gpu', tol={self.tol}" + (f", rank={r}, fc={self._fc_arr})" if r else ")")


