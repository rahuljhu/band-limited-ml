"""Truths of sec41_table.py (far field of a slit / disk pupil), importable."""
import numpy as np
import psfkit as pk
FC = pk.FC_PHYS; RHO = FC / 2
def truth_1d(case, z):
    nu = np.linspace(-RHO, RHO, 2001); dnu = nu[1] - nu[0]
    ph = 2.0 * (2 * (nu / RHO) ** 2 - 1) if case == 'defocus' else 0 * nu
    return (np.exp(1j * ph)[None] * np.exp(2j * np.pi * z[:, None] * nu[None])).sum(1) * dnu
def truth_2d(case, H, OS=8, NF=2048):
    k = np.fft.fftfreq(NF, d=1.0 / OS); KX, KY = np.meshgrid(k, k, indexing='ij'); r = np.hypot(KX, KY) / RHO
    ph = 2.0 * (2 * r ** 2 - 1) if case == 'defocus' else 0 * r
    h = np.fft.fftshift(np.fft.ifft2(np.where(r <= 1, np.exp(1j * ph), 0)))[::OS, ::OS]; c = h.shape[0] // 2
    return h[c - H:c + H + 1, c - H:c + H + 1].ravel()
