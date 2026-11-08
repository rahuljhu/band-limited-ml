"""1-D demo: principled (plane-normalised) BLML fits to a windowed photon histogram, evaluated between
bin centres and beyond the window. Truth: slit pupil |nu| <= rho (rho = NA/lambda in cycles/px),
h = IFT(pupil * exp(i phase)), f = |h|^2 convolved with the 1-px pixel. Window: 31 px. n = 1e5 photons."""
import sys, numpy as np, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'results/psf'); import psfkit as pk; import blml_trivial as bt
torch.set_default_dtype(torch.float64)
RHO = pk.FC_PHYS / 2; FC = pk.FC_PHYS; H = 15; Z = np.arange(-H, H + 1.0)[:, None]; n = 10 ** 5
xs = np.linspace(-45, 45, 3601); dx = xs[1] - xs[0]
def truth(case):
    nu = np.linspace(-RHO, RHO, 4001); dnu = nu[1] - nu[0]
    phase = 2.0 * (2 * (nu / RHO) ** 2 - 1) if case == 'defocus' else 0 * nu
    xf = np.linspace(-200, 200, 32001); h = (np.exp(1j * phase)[None, :] * np.exp(2j * np.pi * xf[:, None] * nu[None, :])).sum(1) * dnu
    f = np.abs(h) ** 2; f /= f.sum() * (xf[1] - xf[0])
    fp = np.convolve(f, np.ones(80) / 80, mode='same')                       # 1-px box (80 samples per px)
    return lambda x: np.interp(x, xf, fp)
K = lambda A, B: FC * np.sinc(FC * (A[:, None] - B[None, :]))
def rank_k(N, k, e1):
    S = torch.tensor(K(Z[:, 0], Z[:, 0])); Nt = torch.tensor(N.astype(float)); nn = Nt.sum()
    a = 0.02 * torch.randn(k, len(Z), generator=torch.Generator().manual_seed(1)) / np.sqrt(len(Z)); a[0] = torch.tensor(e1); a.requires_grad_(True)
    opt = torch.optim.LBFGS([a], max_iter=1000, tolerance_grad=1e-12, tolerance_change=1e-15, history_size=50, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); u = a @ S; E = (a @ S * a).sum()
        L = -((Nt * torch.log((u ** 2).sum(0) + 1e-300)).sum() - nn * torch.log(E)) / nn; L.backward(); return L
    for _ in range(10): opt.step(closure)
    a = a.detach().numpy(); E = float((a @ K(Z[:, 0], Z[:, 0]) * a).sum())
    return lambda x: ((K(x, Z[:, 0]) @ a.T) ** 2).sum(1) / E   # density on R: sum_k g_k(x)^2 / plane energy
fig, axs = plt.subplots(2, 2, figsize=(13, 8.5))
for row, case in enumerate(['focus', 'defocus']):
    ft = truth(case); pb = np.array([ft(np.linspace(z - .5, z + .5, 41)).mean() for z in Z[:, 0]]); out = 1 - pb.sum()
    N = np.random.default_rng(0).multinomial(n, pb / pb.sum())
    m = bt.BLMLQuick(dtype=np.float64).fit(Z, bt.with_lattice(bt.make_modes(FC, None, None, d=Z.shape[1]), 1.0), sample_weight=N)
    e1 = np.zeros(len(Z)); idx = {float(z): i for i, z in enumerate(Z[:, 0])}
    for zb, eb in zip(m.Z_[:, 0], m.e_hat_): e1[idx[float(zb)]] = eb / m.n_samples_
    f1 = lambda x: m.predict(x[:, None]); f2 = rank_k(N, 2, e1)
    u = np.sqrt(N / n); c = np.linalg.lstsq(K(Z[:, 0], Z[:, 0]), u, rcond=1e-14)[0]; fw = lambda x: (K(x, Z[:, 0]) @ c) ** 2
    def klw(fn):                                                                # KL over window pixels (window-normalised)
        q = np.array([fn(np.linspace(z - .5, z + .5, 21)).mean() for z in Z[:, 0]]); q /= q.sum(); p = pb / pb.sum()
        return float((p * np.log(p / np.maximum(q, 1e-300))).sum())
    hist = (N + 0.5) / (N + 0.5).sum() * (1 - out)
    for col, logy in enumerate([False, True]):
        a = axs[row, col]
        a.axvspan(-H - .5, H + .5, color='0.94', zorder=0)
        a.bar(Z[:, 0], N / n * (1 - out), width=1.0, color='0.8', edgecolor='0.7', lw=0.3, label=f'histogram, {n:,} photons (KL {klw(lambda x: np.interp(x, Z[:, 0], hist, left=1e-12, right=1e-12)):.4f})')
        a.plot(xs, ft(xs), color='k', lw=2.2, label=f'truth |h|² ⊛ pixel ({out:.1%} of mass outside window)')
        a.plot(xs, f1(xs), color='#e08a00', lw=1.4, label=f'rank-1 BLML ĝ² (KL {klw(f1):.4f})')
        a.plot(xs, f2(xs), '--', color='#1f6fb4', lw=1.6, label=f'rank-2 BLML g₁²+g₂² (KL {klw(f2):.4f})')
        if logy:
            a.plot(xs, fw(xs) * (1 - out), ':', color='#b8322a', lw=1, label='windowed exact interpolation of √hist')
            a.set_yscale('log'); a.set_ylim(1e-7, 0.5)
        else:
            a.set_xlim(-25, 25)
        a.set_xlabel('x (pixels)'); a.set_ylabel('density (per px)')
        a.set_title(f"{'(a)' if row == 0 and col == 0 else '(b)' if row == 0 else '(c)' if col == 0 else '(d)'} "
                    f"{'in focus (h real, sign changes)' if case == 'focus' else 'defocus 2 rad (h complex)'}, {'linear' if not logy else 'log, ±45 px'}")
        if col == 1: a.legend(frameon=False, fontsize=7, loc='lower center')
    print(case, 'outside mass %.3f' % out, 'KL rank1 %.5f rank2 %.5f hist %.5f' % (klw(f1), klw(f2), klw(lambda x: np.interp(x, Z[:, 0], hist))))
fig.suptitle(f'1-D: band-limited fits to a windowed histogram (grey band = observed window, fc = 2NA/λ = {FC:.3f} cycles/px, n = {n:,})', fontsize=10)
fig.tight_layout(); fig.savefig('results/psf/demo1d.png', dpi=160); fig.savefig('results/psf/demo1d.pdf')
