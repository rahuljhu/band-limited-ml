"""Paper 3, Sec. 4.2 diagnostic: start complex BLML at the true amplitude (e_b ~ N_b / conj h(x_b)) and see whether
it stays. Compares sum_b N_b log f(x_b) for truth, default start, truth start; and phase rms error of each fit.
Usage: sec42_truthstart.py <case> <n> <seed>"""
from blml_trivial import with_lattice
import sys, json, types, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'results/psf')
src = open('results/complex/sec42_phase.py').read(); exec(src.split("case, n, seed")[0])
case, n, seed = sys.argv[1], int(float(sys.argv[2])), int(sys.argv[3]); coef = CASES[case]
k = np.fft.fftfreq(NF, d=1.0 / OS); KX, KY = np.meshgrid(k, k, indexing='ij'); r = np.hypot(KX, KY) / RHO; t = np.arctan2(KY, KX)
hf = np.fft.fftshift(np.fft.ifft2(np.where(r <= 1, np.exp(1j * phase_of(coef, r, t)), 0)))
hf /= np.sqrt((np.abs(hf) ** 2).sum() / OS ** 2)                     # unit integral of |h|^2 over R
h = hf[::OS, ::OS]; c = h.shape[0] // 2
hw = h[c - H:c + H + 1, c - H:c + H + 1].ravel(); p = np.abs(hw) ** 2; p /= p.sum()
g1 = np.arange(-H, H + 1.0); I, J = np.meshgrid(g1, g1, indexing='ij'); Z = np.c_[I.ravel(), J.ravel()]
N = np.random.default_rng(seed).multinomial(n, p)
names = [nm for nm, _, _ in ZN]; par = np.array([pp for _, _, pp in ZN]); keep = par != 'nuis'
true = np.array([coef.get(nm, 0.0) for nm in names])
L = 128; dx = 0.25; xs = np.arange(-L, L) * dx; X, Y = np.meshgrid(xs, xs, indexing='ij')
kk = np.fft.fftshift(np.fft.fftfreq(2 * L, d=dx)); QX, QY = np.meshgrid(kk, kk, indexing='ij')
qr = np.hypot(QX, QY) / RHO; qt = np.arctan2(QY, QX); ins = qr <= 0.9
A = np.stack([f(qr[ins], qt[ins]) for _, f, _ in ZN], 1)
def zern(m):
    g = m.sqrt_density(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    Ph = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g))); amp = np.abs(Ph[ins]); w = amp / amp.sum(); est = np.zeros(len(ZN))
    for _ in range(20):
        res = np.angle(Ph[ins] * np.exp(-1j * (A @ est))); est += np.linalg.lstsq(A * np.sqrt(w)[:, None], res * np.sqrt(w), rcond=None)[0]
    d1 = est[keep] - true[keep]; tw = est.copy(); tw[par == 'even'] *= -1; d2 = tw[keep] - true[keep]
    use = np.linalg.norm(d2) < np.linalg.norm(d1)
    return float(np.sqrt(np.mean((d2 if use else d1) ** 2))), bool(use), {nm: round(float(v), 3) for nm, v in zip(np.array(names)[keep], (tw if use else est)[keep]) if abs(v) > 0.03}
def ll(f): return float(np.sum(N[N > 0] * np.log(np.maximum(f[N > 0], 1e-300))))
out = dict(case=case, n=n, seed=seed, ll_truth=ll(np.abs(hw) ** 2))
for tag in sys.argv[4:] or ['default', 'truth_start']:
    init = None
    if tag == 'truth_start':
        e0 = np.where(N > 0, N / np.conj(np.where(hw == 0, 1, hw)), 0)
        init = types.SimpleNamespace(Z_=Z[N > 0], e_hat_=e0[N > 0])
    m = bt.BLMLQuick(dtype=np.float64, tol=1e-5).fit(Z, with_lattice(bt.make_modes([FC, FC], ['ball', 'ball'], None), 1.0), sample_weight=N, complex_g=True, init=init)
    gb = m.sqrt_density(Z.astype(float)); f = np.abs(gb) ** 2
    rms, tw, est = zern(m)
    # how far the fit moved from the truth: relative L2 distance of g at bins, up to global phase
    ph = np.vdot(gb, hw); ph = ph / abs(ph)
    dist = float(np.linalg.norm(gb * ph - hw) / np.linalg.norm(hw))
    out[tag] = dict(ll=ll(f), rms_err=rms, twin=tw, est=est, dist_from_truth=dist, conv=bool(m.converged_))
    print(case, tag, out[tag], 'll-ll_truth', out[tag]['ll'] - out['ll_truth'], flush=True)
json.dump(out, open(f'results/complex/sec42_init/{case}_{n}_{seed}.json', 'w'), indent=1)
