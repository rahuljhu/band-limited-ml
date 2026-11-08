"""Paper 3, Sec. 4.2: starts for complex BLML on single-image phase retrieval. Two cheap generators, both on
the FFT lattice (O(M log M) per iteration, M = 128^2 grid): (a) HIO from random pupil phases, magnitude sqrt(N)
in the window, support = pupil disk; (b) symmetry breaking: positive-root pupil times exp(+-i delta Z_j) for the
even modes, then error reduction. Starts scored by Poisson log-likelihood on the bins; the top K of each family
(K = 3 HIO, 2 symbreak) are polished with complex BLML (warm start e_b = N_b / conj g_b) and compared with the truth start. Fits already in the output json are reused.
Usage: sec42_starts.py <case> <n> <seed>"""
import sys, json, time, numpy as np
src = open('results/complex/sec42_staircase.py').read(); exec(src.split("\nout = dict(")[0])
M = 128; nu = np.fft.fftfreq(M); NX, NY = np.meshgrid(nu, nu, indexing='ij')
pr = np.hypot(NX, NY) / RHO; pt = np.arctan2(NY, NX); disk = pr <= 1
ix = (Z[:, 0].astype(int) % M, Z[:, 1].astype(int) % M); Amag = np.sqrt(N.astype(float))
def g_of(P): return np.fft.ifft2(P)
def score(P):                                   # Poisson LL of f = |g|^2 normalised to unit mass on the lattice
    g = g_of(P * disk); f = np.abs(g) ** 2; f = f / f.sum()
    return float(np.sum(N[N > 0] * np.log(np.maximum(f[ix][N > 0], 1e-300))))
def project_data(g):
    gb = g[ix]; s = np.sqrt((np.abs(gb) ** 2).sum() / (Amag @ Amag))    # keep the overall scale
    g = g.copy(); g[ix] = Amag * s * np.exp(1j * np.angle(gb)); return g
def hio(P, n_hio, n_er, beta=0.9):
    for it in range(n_hio + n_er):
        Pn = np.fft.fft2(project_data(g_of(P)))
        P = np.where(disk, Pn, (P - beta * Pn) if it < n_hio else 0)
    return P * disk
insP = disk & (pr <= 0.9); Az = np.stack([f(pr[insP], pt[insP]) for _, f, _ in ZN], 1)
def zernP(P):
    amp = np.abs(P[insP]); w = amp / amp.sum(); est = np.zeros(len(ZN))
    for _ in range(20):
        res = np.angle(P[insP] * np.exp(-1j * (Az @ est))); est += np.linalg.lstsq(Az * np.sqrt(w)[:, None], res * np.sqrt(w), rcond=None)[0]
    d1 = est[keep] - true[keep]; tw = est.copy(); tw[par == 'even'] *= -1; d2 = tw[keep] - true[keep]
    return float(min(np.sqrt(np.mean(d1 ** 2)), np.sqrt(np.mean(d2 ** 2))))
rng = np.random.default_rng(100 + seed); t0 = time.time(); fam = {}
# (a) HIO, random phases
fam['hio'] = [hio(disk * np.exp(2j * np.pi * rng.random((M, M))), 400, 100) for _ in range(30)]
# (b) symmetry breaking from the positive root: P0 = band-limited sqrt of the counts
G0 = np.zeros((M, M), complex); G0[ix] = Amag; P0 = np.fft.fft2(G0) * disk
Zf = dict((nm, f) for nm, f, _ in ZN)
fam['symbreak'] = [hio(P0 * np.exp(1j * s * 1.0 * Zf[nm](pr, pt)), 0, 100)
                   for nm in ['defocus', 'astig_0', 'astig_45', 'spherical'] for s in (1, -1)]
t_gen = time.time() - t0
out = dict(case=case, n=n, seed=seed, t_generate=t_gen); K = dict(hio=3, symbreak=2)
fn = f'results/complex/sec42_init/starts_{case}_{n}_{seed}.json'
try: prev = json.load(open(fn))
except FileNotFoundError: prev = {}
if 'truth_start' in prev: out['truth_start'] = prev['truth_start']
else:
    ref = fit(np.where(N > 0, N / np.conj(np.where(hw == 0, 1, hw)), 0)[N > 0])
    out['truth_start'] = report('truth_start', ref); out['truth_start']['ll'] = llm(ref)
for name, Ps in fam.items():
    sc = np.array([score(P) for P in Ps]); order = np.argsort(-sc)
    out[name] = dict(start_scores=sc.tolist(), start_rms=[zernP(P) for P in Ps], polished=[])
    done = {round(r['start_score'], 3): r for r in prev.get(name, {}).get('polished', [])}
    for j in order[:K[name]]:
        if round(float(sc[j]), 3) in done: out[name]['polished'].append(done[round(float(sc[j]), 3)]); continue
        gb = g_of(Ps[j])[ix]
        m = fit(np.where(N > 0, N / np.conj(np.where(gb == 0, 1e-12, gb)), 0)[N > 0])
        r = report(f'{name}#{j}', m, dict(start_score=float(sc[j]), start_rms=out[name]['start_rms'][j])); r['ll'] = llm(m)
        out[name]['polished'].append(r)
out['t_total'] = time.time() - t0
json.dump(out, open(fn, 'w'), indent=1, default=str)
print('done', case, seed, 'gen %.1fs total %.0fs' % (t_gen, out['t_total']), flush=True)
