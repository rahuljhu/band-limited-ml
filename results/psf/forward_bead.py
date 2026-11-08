"""Pixel-forward BLML: same band-limited g (sinc expansion on the pixel lattice, band fc/2), but the
likelihood uses the pixel MASS  P_b = int_pixel g^2 (4x4 midpoint rule), not g^2 at the pixel centre.
So g^2 estimates the PSF itself, not PSF (*) rect. Initialised from BLMLQuick on counts, refined by L-BFGS.
Compares FT(g-hat) with and without the pixel model."""
import sys, json, numpy as np, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import blml_trivial as bt
from blml_trivial import make_modes
bt.DIRICHLET_LEGACY = False; torch.set_default_dtype(torch.float64)
D = 'results/psf/'; NA, LAM, PX = 1.3, 0.515, 0.065; FCP = 2 * NA / LAM * PX
k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
S = np.load(D + 'stack_photons.npy', mmap_mode='r'); im = np.asarray(S[5])
y0, x0 = np.load(D + 'beads_iso.npy')[k]; H = 15
roi = im[y0 - H:y0 + H + 1, x0 - H:x0 + H + 1].astype(float)
bg = np.median(np.concatenate([roi[0], roi[-1], roi[:, 0], roi[:, -1]]))
cnt = np.round(np.maximum(roi - bg, 0)).astype(np.int64)
rng = np.random.default_rng(0)                                  # same split as fit_bead_px.py
tr = rng.binomial(cnt, 0.64); rest = cnt - tr; va = rng.binomial(rest, 0.16 / 0.36); te = rest - va
tv = (tr + va).ravel().astype(float); te = te.ravel().astype(float)
ax = np.arange(-H, H + 1.0); X, Y = np.meshgrid(ax, ax); Z = np.column_stack([X.ravel(), Y.ravel()])
s4 = (np.arange(4) + 0.5) / 4 - 0.5; SX, SY = np.meshgrid(s4, s4)
SQ = (Z[:, None, :] + np.column_stack([SX.ravel(), SY.ravel()])[None]).reshape(-1, 2)   # 16 sub-points / pixel
K = lambda A, B, fc: (fc * np.sinc(fc * (A[:, None, 0] - B[None, :, 0]))) * (fc * np.sinc(fc * (A[:, None, 1] - B[None, :, 1])))
def fit(fc, forward):
    m = bt.BLMLQuick(dtype=np.float64).fit(Z, bt.with_lattice(make_modes([fc, fc], None, None, d=Z.shape[1]), 1.0), sample_weight=tv)
    e0 = np.zeros(len(Z)); idx = {tuple(z): i for i, z in enumerate(Z)}
    for zb, eb in zip(m.Z_, m.e_hat_): e0[idx[tuple(zb)]] = eb
    e0 /= m.n_samples_
    Phi = torch.tensor(K(SQ if forward else Z, Z, fc))            # g at sub-points (forward) or centres
    N, Nte = torch.tensor(tv), torch.tensor(te)
    def pmass(e):
        g2 = (Phi @ e) ** 2
        P = g2.reshape(len(Z), 16).mean(1) if forward else g2
        return P / P.sum()
    e = torch.tensor(e0, requires_grad=True)
    if forward:
        opt = torch.optim.LBFGS([e], max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-14, history_size=50,
                                line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad(); L = -(N * torch.log(pmass(e) + 1e-300)).sum() / N.sum(); L.backward(); return L
        for _ in range(5): opt.step(closure)
    with torch.no_grad():
        P = pmass(e); return e.detach().numpy(), float((Nte * torch.log(P)).sum() / Nte.sum())
def ftg(e, fc, nu):
    NX, NY = np.meshgrid(nu, nu); c = np.array([(cnt * X).sum(), (cnt * Y).sum()]) / cnt.sum()
    F = np.tensordot(e, np.exp(-2j * np.pi * ((Z[:, 0] - c[0])[:, None, None] * NX + (Z[:, 1] - c[1])[:, None, None] * NY)), 1)
    return np.where((np.abs(NX) <= fc / 2) & (np.abs(NY) <= fc / 2), np.abs(F), 0)
nu = np.linspace(-0.27, 0.27, 109); res = {}
out = {}
for fc in [0.5, FCP]:
    for fw in [False, True]:
        e, ll = fit(fc, fw); tag = f"{'forward' if fw else 'centre'}_fc{fc:.3f}"
        res[tag] = ll; out[tag] = ftg(e, fc, nu); print(tag, 'test pixel LL %.5f' % ll, flush=True)
json.dump(res, open(D + f'forward_bead{k}.json', 'w'), indent=1)
# figure: FT g centre vs forward (fc 0.5), ratio, and radial means for both fc
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(1, 4, figsize=(15, 3.9)); t = np.linspace(0, 2 * np.pi, 200); r0 = FCP / 2
a0, a1 = out[f'centre_fc{FCP:.3f}'], out[f'forward_fc{FCP:.3f}']; a0n, a1n = a0 / a0.max(), a1 / a1.max()
for a, A, ttl in [(axs[0], a0n, '(a) |FT ĝ| at fc=2NA/λ, ĝ² ≈ PSF ⊛ pixel'), (axs[1], a1n, '(b) same fc, pixel forward model (ĝ² ≈ PSF)')]:
    a.imshow(np.maximum(A, 1e-3), origin='lower', extent=[nu[0], nu[-1]] * 2, norm=LogNorm(1e-3, 1), cmap='viridis')
    a.plot(r0 * np.cos(t), r0 * np.sin(t), 'w--', lw=1); a.set_title(ttl); a.set_xlabel('$\\nu_x$ (cycles/px)')
axs[0].set_ylabel('$\\nu_y$')
rat = np.where(a0n > 1e-3, a1n / np.maximum(a0n, 1e-12), np.nan)
im = axs[2].imshow(rat, origin='lower', extent=[nu[0], nu[-1]] * 2, cmap='RdBu_r', vmin=0.5, vmax=1.5)
axs[2].plot(r0 * np.cos(t), r0 * np.sin(t), 'k--', lw=1); axs[2].set_title('(c) ratio (b)/(a)'); axs[2].set_xlabel('$\\nu_x$')
fig.colorbar(im, ax=axs[2], fraction=0.046)
NX, NY = np.meshgrid(nu, nu); R = np.hypot(NX, NY); rb = np.linspace(0, 0.27, 28); ix = np.digitize(R.ravel(), rb) - 1
rm = lambda A: np.array([A.ravel()[ix == i].mean() if (ix == i).any() else np.nan for i in range(len(rb) - 1)])
mid = 0.5 * (rb[1:] + rb[:-1])
for tag, col, ls, lab in [('centre_fc0.500', '#9ec3e6', '-', 'centre, fc 0.5'), ('forward_fc0.500', '#1f6fb4', '-', 'forward, fc 0.5'),
                          (f'centre_fc{FCP:.3f}', '#f2b880', '--', 'centre, fc 2NA/λ'), (f'forward_fc{FCP:.3f}', '#c46a00', '--', 'forward, fc 2NA/λ')]:
    A = out[tag]; axs[3].plot(mid, rm(A / A.max()), ls, color=col, lw=1.6, label=f"{lab}  (LL {res[tag]:.4f})")
sinc_ref = rm(np.sinc(NX) * np.sinc(NY))
axs[3].axvline(r0, color='0.7', lw=0.8); axs[3].text(r0, 1.5e-3, ' NA/λ', color='0.4', fontsize=8)
axs[3].set_yscale('log'); axs[3].set_ylim(1e-3, 1.2); axs[3].set_xlabel('|ν| (cycles/px)'); axs[3].set_title('(d) radial |FT ĝ|')
axs[3].legend(frameon=False, fontsize=7)
fig.suptitle(f'Bead {k}: removing the pixel blur from ĝ (λ = 0.515 µm assumed; dashed circle = pupil radius NA/λ)', fontsize=9)
fig.tight_layout(); fig.savefig(D + f'forward_bead{k}.png', dpi=170); fig.savefig(D + f'forward_bead{k}.pdf')
