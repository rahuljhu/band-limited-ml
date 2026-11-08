"""Paper 3: pupil phase from one image, on the whole frame, against phase retrieval (HIO).
Truth: the far field of a circular pupil (band 2NA/lambda) carrying an aberration (sec42_phase.py's Zernike cases),
sampled at the pixel centres of the whole 256 x 256 frame; n photons drawn from |h|^2 over the frame (no window).
Methods, every one scored by the rms error of the fitted Zernike coefficients (piston and tilt removed, the twin
allowed) and by the Poisson log-likelihood of its intensity on the frame:
  hio        30 HIO runs from random pupil phases (400 HIO + 100 error-reduction iterations; magnitude sqrt(N) on the
             frame, support the pupil disk), the run with the highest likelihood (what a practitioner would keep)
  blml_1     complex BLML (BLMLBinned, ball kernel at the true band), the engine's default start
  blml       complex BLML polished from the 3 best HIO runs and from 2 symmetry-breaking starts (the positive-root pupil
             times exp(+-i Z_j) for the even modes), the fit with the highest likelihood
  truth      complex BLML started at the true field (a reference: what the data prefer near the truth)
P3_KERNEL=pball: complex BLML with the periodic ball (kind 'pball', period the frame) instead of the ball on the plane --
the truth is periodic over the frame (an FFT on it), as HIO's model is; HIO's runs are the same (same seeds).
Usage: [P3_KERNEL=pball] p3_phase_frame.py <case> <n> <seed>  ->  results/complex/phase_frame[3]/<case>_<n>_<seed>.json"""
import sys, os, json, time, types, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'results/psf')
import blml_trivial as bt, psfkit as pk
from blml_binned import BLMLBinned
FC = pk.FC_PHYS; RHO = FC / 2; OS = 8; M = 256; NF = M * OS
ZN = [('piston', lambda r, t: 1 + 0 * r, 'nuis'), ('tilt_x', lambda r, t: r * np.cos(t), 'nuis'),
      ('tilt_y', lambda r, t: r * np.sin(t), 'nuis'),
      ('defocus', lambda r, t: 2 * r ** 2 - 1, 'even'), ('astig_0', lambda r, t: r ** 2 * np.cos(2 * t), 'even'),
      ('astig_45', lambda r, t: r ** 2 * np.sin(2 * t), 'even'),
      ('coma_x', lambda r, t: (3 * r ** 3 - 2 * r) * np.cos(t), 'odd'), ('coma_y', lambda r, t: (3 * r ** 3 - 2 * r) * np.sin(t), 'odd'),
      ('trefoil_x', lambda r, t: r ** 3 * np.cos(3 * t), 'odd'), ('trefoil_y', lambda r, t: r ** 3 * np.sin(3 * t), 'odd'),
      ('spherical', lambda r, t: 6 * r ** 4 - 6 * r ** 2 + 1, 'even')]
CASES = {'defocus': {'defocus': 1.0}, 'astig': {'astig_0': 1.0}, 'coma': {'coma_x': 0.5},
         'mixed': {'defocus': 0.4, 'astig_45': 0.5, 'coma_y': 0.3, 'spherical': 0.2}}
Zf = dict((nm, f) for nm, f, _ in ZN); par = np.array([p for _, _, p in ZN]); keep = par != 'nuis'
case, n, seed = sys.argv[1], int(float(sys.argv[2])), int(sys.argv[3]); coef = CASES[case]
OUT = 'results/complex/phase_frame2' if os.environ.get('P3_KERNEL', 'ball') == 'ball' else 'results/complex/phase_frame3'; os.makedirs(OUT, exist_ok=True); fn = f'{OUT}/{case}_{n}_{seed}.json'
phase_of = lambda r, t: sum(c * Zf[nm](r, t) for nm, c in coef.items())

# truth on the frame: the pixel-centre samples of the far field (period M px, so the FFT lattice is the frame)
k = np.fft.fftfreq(NF, d=1.0 / OS); KX, KY = np.meshgrid(k, k, indexing='ij'); r = np.hypot(KX, KY) / RHO; t = np.arctan2(KY, KX)
h = np.fft.fftshift(np.fft.ifft2(np.where(r <= 1, np.exp(1j * phase_of(r, t)), 0)))[::OS, ::OS]   # (M, M), centred on the frame
p = np.abs(h) ** 2; p /= p.sum()
N = np.random.default_rng(seed).multinomial(n, p.ravel()).reshape(M, M).astype(float)
nu = np.fft.fftfreq(M); NX, NY = np.meshgrid(nu, nu, indexing='ij'); pr = np.hypot(NX, NY) / RHO; pt = np.arctan2(NY, NX)
disk = pr <= 1; insP = disk & (pr <= 0.9); Az = np.stack([f(pr[insP], pt[insP]) for _, f, _ in ZN], 1)
true = np.array([coef.get(nm, 0.0) for nm, _, _ in ZN])


def score(P):                                  # Poisson log-likelihood per photon of f = |g|^2 on the frame
    f = np.abs(np.fft.ifft2(P * disk)) ** 2; f = f / f.sum(); m = N > 0
    return float(np.sum(N[m] * np.log(np.maximum(f[m], 1e-300))) / n)


CHK = (-1.0) ** np.add.outer(np.arange(M), np.arange(M))   # undoes the half-frame shift of a centred field
def zern(P):                                   # Zernike fit to the pupil phase (amplitude-weighted), twin allowed
    P = P * CHK
    amp = np.abs(P[insP]); w = amp / amp.sum(); est = np.zeros(len(ZN))
    for _ in range(20):
        res = np.angle(P[insP] * np.exp(-1j * (Az @ est))); est += np.linalg.lstsq(Az * np.sqrt(w)[:, None], res * np.sqrt(w), rcond=None)[0]
    tw = est.copy(); tw[par == 'even'] *= -1
    e1, e2 = [float(np.sqrt(np.mean((e[keep] - true[keep]) ** 2))) for e in (est, tw)]
    return (e1, est) if e1 <= e2 else (e2, tw)


REF = np.fft.fft2(h) * CHK                       # the true pupil (centring removed)
wt = (insP * np.abs(REF)).astype(float); wt /= wt.sum(); Ab = np.stack([np.ones(insP.sum()), NX[insP], NY[insP]], 1)
def phase_err(P):                              # rms pupil phase error (rad, amplitude-weighted over 0.9 rho), up to
    P = P * CHK; out = []                      # the ambiguities: global phase, tilt (a shift) and the twin conj(P(-nu))
    for Q in (P, np.conj(P[(-np.arange(M)) % M][:, (-np.arange(M)) % M])):
        d = np.angle(Q[insP] * np.conj(REF[insP])); w = wt[insP]; c = np.zeros(3)
        for _ in range(30):
            res = np.angle(np.exp(1j * (d - Ab @ c))); c += np.linalg.lstsq(Ab * np.sqrt(w)[:, None], res * np.sqrt(w), rcond=None)[0]
        out.append(float(np.sqrt(np.sum(w * np.angle(np.exp(1j * (d - Ab @ c))) ** 2) / w.sum())))
    return min(out)


Amag = np.sqrt(N)
def hio(P, n_hio, n_er, beta=0.9):
    for it in range(n_hio + n_er):
        g = np.fft.ifft2(P); s = np.sqrt((np.abs(g) ** 2).sum() / (Amag ** 2).sum())
        Pn = np.fft.fft2(Amag * s * np.exp(1j * np.angle(g)))
        P = np.where(disk, Pn, (P - beta * Pn) if it < n_hio else 0)
    return P * disk


KERN = os.environ.get('P3_KERNEL', 'ball')
MODES = bt.make_modes([FC, FC], ['ball', 'ball']) if KERN == 'ball' else [{(0, 1): dict(kind='pball', fc=FC, period=[M, M])}]
def blml(start=None):                          # start: a pupil; warm start e_b = N_b / conj(g_b)
    init = None
    if start is not None:
        g = np.fft.ifft2(start * disk); e = np.where(N > 0, N / np.conj(np.where(np.abs(g) < 1e-300, 1e-300, g)), 0)
        init = types.SimpleNamespace(e_hat_=e[None].astype(complex), shape_=(M, M))
    m = BLMLBinned(engine='cpu', tol=1e-6, max_sweeps=20000).fit(N, MODES, complex_g=True, init=init)
    return np.fft.fft2(np.asarray(m.Se_)[0]) * disk, m          # g at the bins is (S e)_b / n; its DFT on the disk is the pupil


def report(P, extra=None):
    err, est = zern(P); d = dict(rms_err=err, phase_err=phase_err(P), ll=score(P), est={nm: round(float(v), 4) for nm, v, k_ in zip([z[0] for z in ZN], est, keep) if k_})
    return dict(d, **(extra or {}))


t0 = time.time(); rng = np.random.default_rng(100 + seed); out = dict(case=case, n=n, seed=seed)
H = [hio(disk * np.exp(2j * np.pi * rng.random((M, M))), 400, 100) for _ in range(30)]
sc = np.array([score(P) for P in H]); order = np.argsort(-sc)
out['hio'] = report(H[order[0]], dict(all_rms=[zern(P)[0] for P in H], all_ll=sc.tolist()))
P1, m1 = blml(); out['blml_1'] = report(P1, dict(converged=bool(m1.converged_)))
G0 = np.zeros((M, M), complex) + Amag; P0 = np.fft.fft2(G0) * disk                 # positive-root pupil
starts = [H[j] for j in order[:3]] + [hio(P0 * np.exp(1j * s * Zf[nm](pr, pt)), 0, 100)
                                       for nm in ['defocus', 'astig_0', 'astig_45', 'spherical'] for s in (1, -1)][:2]
pol = [blml(S)[0] for S in starts]; ll_pol = [score(P) for P in pol]; jb = int(np.argmax(ll_pol))
out['blml'] = report(pol[jb], dict(start=['hio1', 'hio2', 'hio3', 'sym1', 'sym2'][jb], all_rms=[zern(P)[0] for P in pol], all_ll=ll_pol))
Hs = np.fft.fft2(h); Pt, _ = blml(Hs); out['truth'] = report(Pt)
out['s'] = time.time() - t0
json.dump(out, open(fn, 'w'), indent=1)
np.savez_compressed(f'{OUT}/{case}_{n}_{seed}_pupils.npz', hio=H[order[0]].astype(np.complex64), blml=pol[jb].astype(np.complex64),
                    blml_1=P1.astype(np.complex64), truth=Pt.astype(np.complex64))
if seed == 0:
    np.savez_compressed(f'{OUT}/{case}_{n}_{seed}_maps.npz', true=np.angle(np.fft.fftshift(Hs * CHK)), hio=np.angle(np.fft.fftshift(H[order[0]] * CHK)),
                        blml=np.angle(np.fft.fftshift(pol[jb] * CHK)), disk=np.fft.fftshift(disk), N=N, p=p)
print(case, n, seed, {k_: (round(v['rms_err'], 3), round(v['phase_err'], 3)) for k_, v in out.items() if isinstance(v, dict)}, f"{out['s']:.0f}s", flush=True)
