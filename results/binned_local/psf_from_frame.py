"""The PSF recovered from the full-frame complex BLML fit, by registering and averaging the beads of the fitted frame
(B), against the classical average of raw training crops (C) and the frame fit itself (A). Plane 5, the frame split
(seed 0, 70/10/20). Isolated beads (no other bright peak within 20 px, 25 px from the edges). Two-fold over beads: the
PSF is averaged on one half and scored on the other. Per bead, B and C refit position, brightness and a flat
background on the bead's training photons (Poisson ML, the PSF shifted in Fourier space; the fitted f is band-limited
below the pixel Nyquist, so the shift is exact up to window truncation), then the held-out photons of its 31 x 31
window are scored under the window-normalised model (the bead's brightness cancels). A is the frame fit renormalised
on the same window. Output results/binned_local/psf_from_frame.json / .npz"""
import json, time, numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.optimize import minimize
from scipy.spatial import cKDTree

S = '/private/tmp/claude-501/-Users-ra/e6cd45ce-b6f4-42cc-be9a-45c946facadf/scratchpad/gpu_data/psf_plane5.npy'
N = np.load(S).astype(np.int64)
r = np.random.default_rng(0); tr = r.binomial(N, 0.7); rest = N - tr; va = r.binomial(rest, 1 / 3); te = rest - va
F = np.load('results/binned/frame_complex.npz')['density'].astype(np.float64); F = F / F.sum() * tr.sum()  # in train photons
W, P = 31, 41; hw, hp = W // 2, P // 2

# ---------------------------------------------------------------- beads: bright, isolated, away from the edges
Sm = gaussian_filter(tr.astype(float), 1.0); bg = np.median(Sm); sd = 1.4826 * np.median(np.abs(Sm - bg))
pk = (Sm == maximum_filter(Sm, 7)) & (Sm > bg + 20 * sd)
y, x = np.nonzero(pk); d, _ = cKDTree(np.c_[y, x]).query(np.c_[y, x], k=2)
keep = (d[:, 1] > 20) & (y > 25) & (y < N.shape[0] - 26) & (x > 25) & (x < N.shape[1] - 26)
y, x = y[keep], x[keep]; nb = len(y); print('beads', nb, flush=True)

ky = np.fft.fftfreq(P)[:, None]; kx = np.fft.fftfreq(P)[None, :]
def shift(img, dy, dx):
    """img (P x P) moved by (dy, dx) pixels (band-limited, Fourier phase ramp)."""
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2j * np.pi * (ky * dy + kx * dx))))

def win(A, i, h):
    return A[y[i] - h:y[i] + h + 1, x[i] - h:x[i] + h + 1].astype(float)

def ring_bg(w):
    m = np.ones_like(w, bool); m[3:-3, 3:-3] = False; return float(np.median(w[m]))

def fit_bead(psf, cnt, x0=(0.0, 0.0)):
    """Poisson ML of (dy, dx, a, b) for counts cnt (W x W) under a * psf shifted + b; psf is P x P (sum 1 on W)."""
    def ab(dy, dx):
        q = shift(psf, dy, dx)[hp - hw:hp + hw + 1, hp - hw:hp + hw + 1]; q = np.maximum(q, 1e-12)
        a, b = max(cnt.sum() - ring_bg(cnt) * W * W, 1.0), max(ring_bg(cnt), 0.1)
        for _ in range(30):                                        # EM for the two linear components
            m = a * q + b; rr = cnt / m
            a, b = a * float((rr * q).sum()) / float(q.sum()), b * float(rr.sum()) / (W * W)
        return q, a, b
    def nll(s):
        q, a, b = ab(*s); m = a * q + b; return float((m - cnt * np.log(m)).sum())
    res = minimize(nll, x0, method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-6, maxiter=200))
    q, a, b = ab(*res.x); return res.x, a, b, q

def average(idx, src, init_psf=None, rounds=3):
    """Register the src windows (fitted f, or raw counts) of beads idx on the current mean and average."""
    wins = [win(src, i, hp) for i in idx]
    shifts = []
    for w in wins:                                                 # start: centroid above the ring background
        c = np.clip(w - ring_bg(w), 0, None); t = c.sum()
        yy, xx = np.mgrid[-hp:hp + 1, -hp:hp + 1]; shifts.append((float((c * yy).sum() / t), float((c * xx).sum() / t)))
    psf = init_psf
    for _ in range(rounds):
        acc = np.zeros((P, P)); wt = 0.0
        for w, (dy, dx) in zip(wins, shifts):
            b = ring_bg(w); al = shift(w - b, -dy, -dx); s = al[hp - hw:hp + hw + 1, hp - hw:hp + hw + 1].sum()
            acc += al; wt += s                                     # photon-weighted (sum of aligned windows)
        psf = acc / wt                                             # sums to ~1 on the W window
        shifts = [tuple(fit_bead(psf, w[hp - hw:hp + hw + 1, hp - hw:hp + hw + 1], (dy, dx))[0])
                  for w, (dy, dx) in zip(wins, shifts)]
    return psf

def score(m, cnt):
    p = m / m.sum(); return float((cnt * np.log(np.maximum(p, 1e-300))).sum()), float(cnt.sum())

t0 = time.time()
half = np.arange(nb) % 2
out = {k: [0.0, 0.0] for k in ('A_frame_fit', 'B_psf_from_fit', 'C_psf_from_raw_crops')}
per_bead = []; psfs = {}
for fold in (0, 1):
    build, test = np.nonzero(half == fold)[0], np.nonzero(half != fold)[0]
    pB = average(build, F); pC = average(build, tr.astype(float)); psfs[f'B{fold}'], psfs[f'C{fold}'] = pB, pC
    print(f'fold {fold}: PSFs from {len(build)} beads  {time.time() - t0:.0f}s', flush=True)
    for i in test:
        ct, ce = win(tr, i, hw), win(te, i, hw); row = dict(i=int(i), y=int(y[i]), x=int(x[i]), photons_test=float(ce.sum()))
        s, n = score(win(F, i, hw), ce); out['A_frame_fit'][0] += s; out['A_frame_fit'][1] += n; row['A'] = s / n
        for lab, p in (('B_psf_from_fit', pB), ('C_psf_from_raw_crops', pC)):
            _, a, b, q = fit_bead(p, ct); s, n = score(a * q + b, ce); out[lab][0] += s; out[lab][1] += n; row[lab[0]] = s / n
        per_bead.append(row)
res = {k: v[0] / v[1] for k, v in out.items()}
D = np.array([[r_['B'] - r_['A'], r_['C'] - r_['A'], r_['B'] - r_['C']] for r_ in per_bead]); wts = np.array([r_['photons_test'] for r_ in per_bead])
for j, lab in enumerate(('B-A', 'C-A', 'B-C')):
    m = np.average(D[:, j], weights=wts); se = np.sqrt(np.average((D[:, j] - m) ** 2, weights=wts) / len(D))
    res[lab] = dict(mean=float(m), se=float(se), beads_won=float(np.mean(D[:, j] > 0)))
res.update(beads=int(nb), window=W, time_s=time.time() - t0)
print(json.dumps(res, indent=1))
json.dump(dict(summary=res, per_bead=per_bead), open('results/binned_local/psf_from_frame.json', 'w'), indent=1)
np.savez_compressed('results/binned_local/psf_from_frame.npz', **psfs)
