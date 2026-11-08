"""Paper 3: the pseudo-true (n -> infinity) slit-kernel geometry of the wide-window benchmark truth (p3_dbench.py,
tapered slits on 768 px, case sym or asym, no background): fit the expected counts 1e8 P with the two-slit kernel and a
flat background (N = 0) over the benchmark's fills and a fine cut-off grid; the best expected log-likelihood.
  python p3_dbench_pseudo.py <case>   -> results/dbench/pseudo_<case>.json"""
import os, sys, json, numpy as np
os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_SOLVER', 'lbfgs')
case = sys.argv[1]; sys.argv = sys.argv[:1]; sys.path.insert(0, '.')
import dslit_job as dj
W, NF, D_, A_ = 768, 1 << 16, 0.017, 0.005
nu = np.fft.fftfreq(NF)
slit = lambda c: np.where(np.abs(nu - c) < A_ / 2, np.cos(np.pi * np.abs(nu - c) / A_) ** 2, 0.0)
field = lambda A: np.fft.fftshift(np.fft.ifft(A))[NF // 2 - W // 2:NF // 2 + W // 2] * NF
A = slit(-D_ / 2) + (1.0 if case == 'sym' else 0.6 * np.exp(1j)) * slit(D_ / 2)
I = np.abs(field(A)) ** 2; P = I / I.sum(); T = 1e8 * P; dj.BFIX = np.array([np.nan])
full = np.abs(np.fft.ifft(A)) ** 2; out = dict(case=case, mass_outside=float(1 - I.sum() / (full.sum() * NF ** 2)), grid=[])
negH = float((P * np.log(P)).sum()); phys = D_ + A_
for w in (0.35, 0.45, 0.55, 0.7, 1.0):
    prev = None
    for f in [float(f) for f in dj.FC_GRID if 0.9 * phys <= f <= 1.6 * phys]:
        Q, prev, beta = dj.bg_fit(T, [f], (0,), True, 1, init=prev, max_sweeps=5000, tol=1e-7, fill=w)
        out['grid'].append(dict(fill=w, fc=f, ratio=f / phys, kl=negH - float((P * np.log(np.maximum(Q, 1e-300))).sum()), beta=beta))
for w in (0.35, 0.45, 0.55, 0.7, 1.0):
    b = min((g for g in out['grid'] if g['fill'] == w), key=lambda g: g['kl']); out[f'best_{w}'] = b
out['best'] = min(out['grid'], key=lambda g: g['kl'])
json.dump(out, open(f'results/dbench/pseudo_{case}.json', 'w'), indent=1)
print(case, 'outside %.1e' % out['mass_outside'], {w: ('%.2f' % out[f'best_{w}']['ratio'], '%.1e' % out[f'best_{w}']['kl']) for w in (0.35, 0.45, 0.55, 0.7, 1.0)})
