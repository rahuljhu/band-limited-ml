"""Window-truncated BLML on raw counts: f_true = g^2 band-limited on R^2 (bead + smooth background, no
edges); the data density is f_true * 1_W / int_W f_true. Fit: maximise sum_b N_b log(g(z_b)^2 / sum_W g^2)
over the sinc coefficients (one per window pixel), L-BFGS from the BLMLQuick on counts solution. No EM, no
background term. fc on val (grid <= 0.5), refit on train+val, score test.  Usage: python fit_bead_window.py <k>"""
import sys, json, time, numpy as np, torch
import blml_trivial as bt
from blml_trivial import make_modes
bt.DIRICHLET_LEGACY = False; torch.set_default_dtype(torch.float64)
D = 'results/psf/'; FC_PHYS = 2 * 1.3 / 0.515 * 0.065; GAIN_CORR = 1.438
k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
im = np.asarray(np.load(D + 'stack_photons.npy', mmap_mode='r')[5]); y0, x0 = np.load(D + 'beads_iso.npy')[k]; H = 15
cnt = np.round(np.maximum(im[y0 - H:y0 + H + 1, x0 - H:x0 + H + 1].astype(float) / GAIN_CORR, 0)).astype(np.int64)
rng = np.random.default_rng(0)                                          # same split as fit_bead_total.py
tr = rng.binomial(cnt, 0.64); rest = cnt - tr; va = rng.binomial(rest, 0.16 / 0.36); te = rest - va; tv = tr + va
ax = np.arange(-H, H + 1.0); X, Y = np.meshgrid(ax, ax); Z = np.column_stack([X.ravel(), Y.ravel()]); B = len(Z)
pll = lambda P, N: float((N.ravel() * np.log(np.maximum(P.ravel(), 1e-300))).sum() / N.sum())
Kmat = lambda fc: (fc[0] * np.sinc(fc[0] * (Z[:, None, 0] - Z[None, :, 0]))) * (fc[1] * np.sinc(fc[1] * (Z[:, None, 1] - Z[None, :, 1])))

def fit(fc, N, iters=4):
    m = bt.BLMLQuick(dtype=np.float64).fit(Z, bt.with_lattice(make_modes(list(fc), None, None, d=Z.shape[1]), 1.0), sample_weight=N.ravel().astype(float))
    idx = {tuple(z): i for i, z in enumerate(Z)}; a0 = np.zeros(B)
    for zb, eb in zip(m.Z_, m.e_hat_): a0[idx[tuple(zb)]] = eb / m.n_samples_
    P0 = m.predict(Z); P0 /= P0.sum()
    Kt = torch.tensor(Kmat(fc)); Nt = torch.tensor(N.ravel().astype(float)); a = torch.tensor(a0, requires_grad=True)
    def probs(a):
        g2 = (Kt @ a) ** 2; return g2 / g2.sum()
    opt = torch.optim.LBFGS([a], max_iter=300, tolerance_grad=1e-11, tolerance_change=1e-13, history_size=30, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); L = -(Nt * torch.log(probs(a) + 1e-300)).sum() / Nt.sum(); L.backward(); return L
    for _ in range(iters): opt.step(closure)
    with torch.no_grad(): P = probs(a).numpy()
    return P, P0, a.detach().numpy()

t0 = time.time(); grid = np.round(np.geomspace(0.12, 0.5, 14), 4); cache = {}
def vl(fc):
    if fc not in cache: cache[fc] = pll(fit(fc, tr)[0], va)
    return cache[fc]
fc = [max((vl((f, f)), f) for f in grid)[1]] * 2
for _ in range(2):
    for a_ in range(2):
        fc[a_] = max((vl(tuple(fc[:a_] + [f] + fc[a_ + 1:])), f) for f in grid)[1]
P, P0, a = fit(tuple(fc), tv); Pp, P0p, _ = fit((FC_PHYS, FC_PHYS), tv)
res = dict(bead=[int(y0), int(x0)], fc=[float(f) for f in fc], s=time.time() - t0, fits=len(cache),
           window_BLML=pll(P, te), plain_BLML_same_fc=pll(P0, te), window_BLML_phys=pll(Pp, te), plain_BLML_phys=pll(P0p, te),
           histogram=pll(np.maximum(tv, 0.5) / np.maximum(tv, 0.5).sum(), te))
tot = json.load(open(D + f'bead{k}_total.json')) if k == 0 else {}
print(json.dumps(res, indent=1))
if tot: print('reference (bead0_total / EM run): plain BLML selected %.5f | KDE %.5f | histogram %.5f | BLML+bg EM -5.34790' %
              (tot['BLML']['ll'], tot['KDE']['ll'], tot['histogram']['ll']))
json.dump(res, open(D + f'bead{k}_window.json', 'w'), indent=1); np.save(D + f'bead{k}_window_P.npy', P)
