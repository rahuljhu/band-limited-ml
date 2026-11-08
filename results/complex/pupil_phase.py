"""Does complex BLML recover the pupil phase? Fit complex BLML to 2-D photon counts (pixel-centre sampling of |h|^2),
evaluate g on a fine grid, FFT -> estimated pupil P_hat(nu) (amplitude + phase inside |nu| <= rho).
Compare with the true pupil phase up to the inherent ambiguities of |h|^2: global phase, the twin
P(nu) -> conj P(-nu) (h -> conj h(-x)) and a linear phase (a shift, fixed here by pixel-centre sampling)."""
import sys, json, numpy as np
sys.path.insert(0, 'results/psf'); import blml_trivial as bt
exec(open('results/psf/complex2d.py').read().split("if __name__")[0])
def pupil_true(case, k):
    KX, KY = np.meshgrid(k, k, indexing='ij'); r = np.hypot(KX, KY) / RHO; th = np.arctan2(KY, KX)
    ph = {'airy': 0 * r, 'defocus': 2.0 * (2 * r ** 2 - 1), 'astig': 1.5 * r ** 2 * np.cos(2 * th),
          'coma': 1.0 * (3 * r ** 3 - 2 * r) * np.cos(th)}[case]
    return np.where(r <= 1, np.exp(1j * ph), 0), r
def field_case(case):
    k = np.fft.fftfreq(NF, d=1.0 / OS); P, r = pupil_true(case, k)
    return np.fft.fftshift(np.fft.ifft2(P))[::OS, ::OS]
H = 30; n = int(float(sys.argv[1])) if len(sys.argv) > 1 else 10 ** 6
res = {}
for case in ['defocus', 'astig', 'coma']:
    h = field_case(case); c0 = h.shape[0] // 2; pb = np.abs(h[c0 - H:c0 + H + 1, c0 - H:c0 + H + 1]) ** 2; p = (pb / pb.sum()).ravel()
    ii, jj = np.meshgrid(np.arange(-H, H + 1.0), np.arange(-H, H + 1.0), indexing='ij'); Z = np.c_[ii.ravel(), jj.ravel()]
    N = np.random.default_rng(0).multinomial(n, p)
    m = bt.BLMLQuick(dtype=np.float64, tol=1e-6).fit(Z, bt.with_lattice(bt.make_modes(FC, ['ball', 'ball'], None), 1.0), sample_weight=N, complex_g=True)
    # g on a fine grid -> pupil estimate
    L = 128; dx = 0.25; xs = (np.arange(-L, L) * dx); X, Y = np.meshgrid(xs, xs, indexing='ij')
    g = m.sqrt_density(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    Ph = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g))); k = np.fft.fftshift(np.fft.fftfreq(2 * L, d=dx))
    Pt, r = pupil_true(case, k); inside = r <= 0.9                                    # stay off the rim
    best = None
    for twin in [False, True]:
        Pc = np.conj(Ph[::-1, ::-1]) if twin else Ph
        if twin: Pc = np.roll(np.roll(Pc, 1, 0), 1, 1)                             # re-centre the flipped grid
        z = (Pc * np.conj(Pt))[inside]; z = z * np.exp(-1j * np.angle(z.sum()))       # remove global phase
        err = np.sqrt(np.mean(np.angle(z) ** 2)); best = min(best or (9, None), (err, twin))
    true_rms = np.sqrt(np.mean((np.angle(Pt[inside] * np.exp(-1j * np.angle(Pt[inside].sum()))) ** 2)))
    res[case] = dict(n=n, phase_rms_err_rad=float(best[0]), twin=bool(best[1]), true_phase_rms_rad=float(true_rms), converged=bool(m.converged_))
    print(case, res[case], flush=True)
    np.savez(f'results/complex/pupil_{case}.npz', Ph=Ph, Pt=Pt, k=k)
json.dump(res, open('results/complex/pupil_phase.json', 'w'), indent=1)
