"""1-D: solve S a = sqrt(hist) at bin centres (S_ij = K(z_i - z_j), sinc kernel at fc), then
g(x) = sum_i a_i K(x - z_i) evaluated between bin centres and beyond the window; density = g^2."""
import sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'results/psf'); import psfkit as pk
exec(open('results/psf/demo1d.py').read().split('K = lambda')[0].split('torch.set_default_dtype')[1].split('\n', 1)[1])  # RHO, FC, H, Z, n, xs, truth
K = lambda A, B: FC * np.sinc(FC * (A[:, None] - B[None, :]))
z = Z[:, 0]; S = K(z, z); KS = [10, 13, 16, 22]; CS = ['#1f6fb4', '#2a9d5c', '#e08a00', '#b8322a']; print('cond(S) = %.2e' % np.linalg.cond(S))
fig, axs = plt.subplots(2, 3, figsize=(16, 8))
for row, case in enumerate(['focus', 'defocus']):
    ft = truth(case); pb = np.array([ft(np.linspace(t - .5, t + .5, 41)).mean() for t in z])
    N = np.random.default_rng(0).multinomial(n, pb / pb.sum()); u = np.sqrt(N / n)
    lam, V = np.linalg.eigh(S); lam, V = lam[::-1], V[:, ::-1]
    def sol(k): return V[:, :k] @ ((V[:, :k].T @ u) / lam[:k])         # a using the top-k Slepian modes of S
    fits = {}
    for k in KS:
        a = sol(k); fits[k] = (lambda a: lambda x: K(x, z) @ a)(a)
        print(case, 'k=%2d |a|=%.1e  max|Sa-u|=%.1e  sum_b g(z_b)^2=%.3f  int_R g^2=a^T S a=%.3f'
              % (k, np.linalg.norm(a), np.abs(S @ a - u).max(), (fits[k](z) ** 2).sum(), a @ S @ a))
    for col, (lo, logy) in enumerate([(25, False), (45, True), (25, False)]):
        ax = axs[row, col]; ax.axvspan(-H - .5, H + .5, color='0.94', zorder=0); m = np.abs(xs) <= lo
        if col < 2:
            ax.bar(z, N / n, width=1.0, color='0.8', edgecolor='0.7', lw=0.3, label='histogram N_b/n')
            ax.plot(xs[m], ft(xs[m]) / pb.sum(), 'k', lw=2, label='truth (window-normalised)')
            for k, c in zip(KS, CS): ax.plot(xs[m], fits[k](xs[m]) ** 2, color=c, lw=1.1, label=f'g², top {k} modes of S')
            if logy: ax.set_yscale('log'); ax.set_ylim(1e-7, 1e3)
            ax.set_ylabel('density (per px)')
        else:
            ax.axhline(0, color='0.6', lw=.5)
            for k, c in zip(KS, CS): ax.plot(xs[m], fits[k](xs[m]), color=c, lw=1.1, label=f'g, top {k} modes')
            ax.plot(z, u, 'o', ms=3, color='#b8322a', mfc='w', label='√(N_b/n)')
            ax.set_ylim(-1.2 * u.max(), 1.5 * u.max())
        ax.set_xlabel('x (pixels)'); ax.legend(frameon=False, fontsize=7)
        ax.set_title(f"{case}: {['g², linear', 'g², log, ±45 px', 'g itself'][col]}")
fig.suptitle(f'S a = √hist (sinc kernel, fc = 2NA/λ = {FC:.3f}, 1-px bins, 31-px window, n = {n:,}); grey = window', fontsize=10)
fig.tight_layout(); fig.savefig('results/psf/demo_sqrt.png', dpi=150)
