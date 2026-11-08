"""Disk-band (jinc) vs box-band (sinc x sinc) BLML on bead pixels, pixel-centre vs pixel-forward likelihood.
jinc kernel K(r) = rho J1(2 pi rho r)/r, K(0) = pi rho^2  (FT of the disk |nu| <= rho, rho = fc/2) -- the
reproducing kernel of disk-band-limited functions, so the BLML system (S e) e = n w is unchanged.
On the integer pixel lattice r^2 = i^2 + j^2 is an integer: S is read from one cached 1-D table over r^2.
Plots |FT g|^2 (Parseval: integrates to 1 -> photon density over pupil coordinates)."""
import sys, json, time, numpy as np, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.special import j1
torch.set_default_dtype(torch.float64)
D = 'results/psf/'; NA, LAM, PX = 1.3, 0.515, 0.065; FCP = 2 * NA / LAM * PX; RHO = FCP / 2
k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
S_ = np.load(D + 'stack_photons.npy', mmap_mode='r'); im = np.asarray(S_[5])
y0, x0 = np.load(D + 'beads_iso.npy')[k]; H = 15
roi = im[y0 - H:y0 + H + 1, x0 - H:x0 + H + 1].astype(float)
bg = np.median(np.concatenate([roi[0], roi[-1], roi[:, 0], roi[:, -1]]))
cnt = np.round(np.maximum(roi - bg, 0)).astype(np.int64)
rng = np.random.default_rng(0)                                  # same split as fit_bead_px.py
tr = rng.binomial(cnt, 0.64); rest = cnt - tr; va = rng.binomial(rest, 0.16 / 0.36); te = rest - va
tv = (tr + va).ravel().astype(float); te = te.ravel().astype(float)
ax = np.arange(-H, H + 1); X, Y = np.meshgrid(ax, ax); Zi = np.column_stack([X.ravel(), Y.ravel()]); Z = Zi.astype(float)

def jinc(r, rho):
    r = np.asarray(r, float); out = np.full(r.shape, np.pi * rho ** 2)
    nz = r > 0; out[nz] = rho * j1(2 * np.pi * rho * r[nz]) / r[nz]; return out
def boxk(A, B, fc):
    return (fc * np.sinc(fc * (A[:, None, 0] - B[None, :, 0]))) * (fc * np.sinc(fc * (A[:, None, 1] - B[None, :, 1])))
def gram(kind, fc):
    if kind == 'box':
        return boxk(Z, Z, fc)
    d = Zi[:, None, :] - Zi[None, :, :]; r2 = (d ** 2).sum(-1)          # integer r^2 on the lattice
    table = jinc(np.sqrt(np.arange(r2.max() + 1)), fc / 2)             # cached: one entry per integer r^2
    return table[r2]
def kmat(kind, A, fc):                                                  # g at arbitrary points (forward model)
    if kind == 'box':
        return boxk(A, Z, fc)
    return jinc(np.hypot(A[:, None, 0] - Z[None, :, 0], A[:, None, 1] - Z[None, :, 1]), fc / 2)

def blml_dense(S, w, tol=1e-8):
    """CD on Phi(e) = 0.5 e'Se - n sum w log e; exact coordinate minimiser = positive root."""
    keep = w > 0; S = S[np.ix_(keep, keep)]; w = w[keep]; n = w.sum(); B = len(w)
    e = np.full(B, np.sqrt(n * w.mean() / S.diagonal().mean())); Se = S @ e; sbb = S.diagonal()
    for it in range(5000):
        for b in rng.permutation(B):
            bt_ = Se[b] - sbb[b] * e[b]; cc = n * w[b]
            new = 2 * cc / (bt_ + np.sqrt(bt_ ** 2 + 4 * sbb[b] * cc)) if bt_ >= 0 else (-bt_ + np.sqrt(bt_ ** 2 + 4 * sbb[b] * cc)) / (2 * sbb[b])
            Se += S[:, b] * (new - e[b]); e[b] = new
        if it % 10 == 9:
            Se = S @ e
            if np.max(np.abs(Se * e / (n * w) - 1)) < tol: break
    full = np.zeros(len(keep)); full[keep] = e / n; return full        # g = sum_b full_b K(x, z_b)

s4 = (np.arange(4) + 0.5) / 4 - 0.5; SX, SY = np.meshgrid(s4, s4)
SQ = (Z[:, None, :] + np.column_stack([SX.ravel(), SY.ravel()])[None]).reshape(-1, 2)
def fit(kind, fc, forward):
    t0 = time.time(); e0 = blml_dense(gram(kind, fc), tv)
    Phi = torch.tensor(kmat(kind, SQ, fc) if forward else gram(kind, fc))
    N, Nte = torch.tensor(tv), torch.tensor(te)
    def pmass(e):
        g2 = (Phi @ e) ** 2; P = g2.reshape(len(Z), 16).mean(1) if forward else g2; return P / P.sum()
    e = torch.tensor(e0, requires_grad=True)
    if forward:
        opt = torch.optim.LBFGS([e], max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-14, history_size=50,
                                line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad(); L = -(N * torch.log(pmass(e) + 1e-300)).sum() / N.sum(); L.backward(); return L
        for _ in range(5): opt.step(closure)
    with torch.no_grad():
        ll = float((Nte * torch.log(pmass(e))).sum() / Nte.sum())
    return e.detach().numpy(), ll, time.time() - t0

c = np.array([(cnt * X).sum(), (cnt * Y).sum()]) / cnt.sum()
nu = np.linspace(-0.2, 0.2, 161); NX, NY = np.meshgrid(nu, nu); dnu = nu[1] - nu[0]
def ftg2(e, kind, fc):
    F = np.tensordot(e, np.exp(-2j * np.pi * ((Z[:, 0] - c[0])[:, None, None] * NX + (Z[:, 1] - c[1])[:, None, None] * NY)), 1)
    band = (np.hypot(NX, NY) <= fc / 2) if kind == 'jinc' else ((np.abs(NX) <= fc / 2) & (np.abs(NY) <= fc / 2))
    P = np.where(band, np.abs(F) ** 2, 0); return P, float(P.sum() * dnu * dnu)   # Parseval check: ~1 (ROI-truncated)
res, maps = {}, {}
for kind in ['box', 'jinc']:
    for fw in [False, True]:
        e, ll, s = fit(kind, FCP, fw); P, mass = ftg2(e, kind, FCP); tag = f"{kind}_{'forward' if fw else 'centre'}"
        res[tag] = dict(ll=ll, s=s, ft_mass=mass); maps[tag] = P / mass   # forward fit: scale of e is free (P/sum P)
        print(f'{tag:14s} test pixel LL {ll:.5f}  int|FTg|^2 = {mass:.4f}  ({s:.1f}s)', flush=True)
va_ = va.ravel().astype(float)
def scan(kind):
    out = []
    for fc in np.round(np.geomspace(0.2, 0.6, 15), 4):
        e = blml_dense(gram(kind, fc), tr.ravel().astype(float)); g2 = (gram(kind, fc) @ e) ** 2; P = g2 / g2.sum()
        out.append((float((va_ * np.log(np.maximum(P, 1e-300))).sum() / va_.sum()), float(fc)))
    return out
for kind in ['box', 'jinc']:
    sc = scan(kind); b = max(sc); res[kind + '_scan'] = sc
    print(kind, 'val-best fc %.3f (val LL %.5f); phys %.3f' % (b[1], b[0], FCP), flush=True)
json.dump(res, open(D + f'jinc_bead{k}.json', 'w'), indent=1)
# figure
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(1, 5, figsize=(17, 3.7), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.25]))
t = np.linspace(0, 2 * np.pi, 200); vmax = max(m.max() for m in maps.values())
for a, tag, ttl in zip(axs, ['box_centre', 'box_forward', 'jinc_centre', 'jinc_forward'],
                       ['(a) box band, pixel-centre', '(b) box band, pixel-forward', '(c) disk band (jinc), pixel-centre',
                        '(d) disk band (jinc), pixel-forward']):
    a.imshow(maps[tag], origin='lower', extent=[nu[0], nu[-1]] * 2, cmap='magma', vmin=0, vmax=vmax)
    a.plot(RHO * np.cos(t), RHO * np.sin(t), 'w--', lw=0.9); a.set_title(ttl + f"\nLL {res[tag]['ll']:.4f}", fontsize=8.5)
    a.set_xlabel('$\\nu_x$ (cycles/px)'); a.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
axs[0].set_ylabel('$\\nu_y$ (cycles/px)')
R = np.hypot(NX, NY); rb = np.linspace(0, 0.2, 33); ix = np.digitize(R.ravel(), rb) - 1; mid = 0.5 * (rb[1:] + rb[:-1])
rm = lambda A: np.array([A.ravel()[ix == i].mean() for i in range(len(mid))])
for tag, col, ls in [('box_centre', '#9ec3e6', '-'), ('box_forward', '#1f6fb4', '-'), ('jinc_centre', '#f2b880', '-'),
                     ('jinc_forward', '#c46a00', '-')]:
    axs[4].plot(mid, rm(maps[tag]), ls, color=col, lw=1.8, label=tag.replace('_', ', '))
axs[4].plot([0, RHO, RHO], [1 / (np.pi * RHO ** 2)] * 2 + [0], '--', color='0.4', lw=1, label='ideal pupil: flat top')
axs[4].axvline(RHO, color='0.8', lw=0.8); axs[4].set_xlabel('|ν| (cycles/px)'); axs[4].set_ylabel('azimuthal mean of |FT ĝ|²')
axs[4].set_title('(e) photon density across the pupil'); axs[4].legend(frameon=False, fontsize=7.5)
fig.suptitle(f'Bead {k}: |FT ĝ|² at fc = 2NA/λ (λ = 0.515 µm assumed); dashed circle = NA/λ. |FT ĝ|² integrates to 1 (Parseval).',
             fontsize=9)
fig.tight_layout(); fig.savefig(D + f'jinc_bead{k}.png', dpi=170); fig.savefig(D + f'jinc_bead{k}.pdf')
