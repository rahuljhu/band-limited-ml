"""Compression of the clean photo at equal parameter counts: |g|^2 with g band-limited at half the Nyquist band (real
g >= 0: N/4 numbers; complex g: N/2) against the linear low-pass of f keeping N/4 and N/2 Fourier coefficients. PSNR
against the clean image. lam = 1e-8; complex alternation 20 outer steps. Output results/halfband_compression.json"""
import sys, os, json, numpy as np
sys.path.insert(0, '.')
import matplotlib; from PIL import Image
import blml_trivial as bt
from blml_binned import BLMLEmpirical, _np
img = np.asarray(Image.open(os.path.join(matplotlib.get_data_path(), 'sample_data', 'grace_hopper.jpg')).convert('L'), float)[40:552, :512]
f = img / img.mean(); L = 512; peak = f.max(); out = {}
psnr = lambda a: float(10 * np.log10(peak ** 2 / np.mean((a - f) ** 2)))
F = np.fft.fft2(f); nu = np.fft.fftfreq(L); NY, NX = np.meshgrid(nu, nu, indexing='ij')
low = lambda b: np.real(np.fft.ifft2(F * ((np.abs(NY) < b) & (np.abs(NX) < b))))
out['linear low-pass, N/4'] = psnr(low(0.25)); out['linear low-pass, N/2'] = psnr(low(0.5 / np.sqrt(2)))
w = f / f.sum() * 1e7
for lab, kw, ekw in (('|g|^2, real g >= 0 at half Nyquist, N/4', {}, {}), ('|g|^2, complex g at half Nyquist, N/2', dict(complex_g=True), dict(max_outer=20, outer_tol=0))):
    m = BLMLEmpirical(lam=1e-8, tol=1e-6, max_iter=20000, **ekw).fit(w, bt.make_modes([0.5, 0.5], ['sinc', 'sinc']), **kw)
    est = _np(m.density_); out[lab] = psnr(est / est.sum() * f.sum())
for k, v in out.items():
    print(f'{k:42s} {v:.2f} dB', flush=True)
json.dump(out, open('results/halfband_compression.json', 'w'), indent=1)
