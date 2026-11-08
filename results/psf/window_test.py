"""Does BLML beat the histogram when the window is big enough that truncation is negligible?
1-D slit-pupil PSF, f_pix = |h|^2 * box (band-limited at FC_PHYS), p_b = f_pix(z_b) exactly.
Scoring: KL(p || q) with q_b = ghat^2(z_b) renormalised over the window."""
import sys, numpy as np, torch
sys.path.insert(0, 'results/psf'); import psfkit as pk; import blml_trivial as bt
torch.set_default_dtype(torch.float64); FC = pk.FC_PHYS; RHO = FC / 2
def truth(case):
    nu = np.linspace(-RHO, RHO, 4001); dnu = nu[1] - nu[0]
    ph = 2.0 * (2 * (nu / RHO) ** 2 - 1) if case == 'defocus' else 0 * nu
    xf = np.linspace(-1500, 1500, 240001); dx = xf[1] - xf[0]
    h = np.zeros(len(xf), complex)
    for i in range(0, len(xf), 20000): h[i:i+20000] = (np.exp(1j*ph)[None]*np.exp(2j*np.pi*xf[i:i+20000, None]*nu[None])).sum(1)*dnu
    f = np.abs(h) ** 2; f /= f.sum() * dx; k = int(round(1 / dx))
    fp = np.convolve(f, np.ones(k) / k, mode='same'); return lambda x: np.interp(x, xf, fp)
def rank_k(Z, N, fc, k, seed=1):
    z = Z[:, 0]; S = torch.tensor(fc * np.sinc(fc * (z[:, None] - z[None]))); Nt = torch.tensor(N.astype(float)); n = Nt.sum()
    g = torch.Generator().manual_seed(seed); a = (torch.sqrt(Nt / n) @ torch.linalg.pinv(S, rtol=1e-3)).repeat(k, 1) + 0.05 * torch.randn(k, len(z), generator=g) / np.sqrt(len(z))
    a.requires_grad_(True); opt = torch.optim.LBFGS([a], max_iter=2000, tolerance_grad=1e-12, tolerance_change=1e-15, history_size=50, line_search_fn='strong_wolfe')
    def cl():
        opt.zero_grad(); u = a @ S; L = -((Nt * torch.log((u ** 2).sum(0) + 1e-300)).sum() - n * torch.log((u * a).sum())) / n; L.backward(); return L
    for _ in range(5): opt.step(cl)
    u = (a @ S).detach().numpy(); return (u ** 2).sum(0)
n = int(float(sys.argv[1])) if len(sys.argv) > 1 else 10 ** 5
for case in ['focus', 'defocus']:
    ft = truth(case)
    for H in [15, 50, 150]:
        Z = np.arange(-H, H + 1.0)[:, None]; z = Z[:, 0]; pb = ft(z); out = 1 - pb.sum(); p = pb / pb.sum()
        N = np.random.default_rng(0).multinomial(n, p)
        kl = lambda q: float((p * np.log(p / (np.maximum(q, 1e-300) / np.maximum(q, 1e-300).sum()))).sum())
        r = {'hist': kl(N + 0.5)}
        for fc in [FC, 0.5, 0.7]:
            r[f'r1@{fc:.2f}'] = kl(bt.BLMLQuick(dtype=np.float64).fit(Z, bt.with_lattice(bt.make_modes(fc, None, None, d=Z.shape[1]), 1.0), sample_weight=N).predict(Z))
        for k in [2, 3]: r[f'r{k}@{FC:.2f}'] = kl(rank_k(Z, N, FC, k))
        print(f'{case:8s} window {2*H+1:4d}px outside {out:.4f} | ' + '  '.join(f'{a} {b:.5f}' for a, b in r.items()), flush=True)
