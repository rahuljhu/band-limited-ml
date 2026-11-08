"""Paper 3, the aperture recovered from real double-slit records: |FT g| of complex BLML with the band and with the
two-slit kernel (fill and fc by 5-fold CV and the one-SE rule, background mode of N harmonics), against the published
slit geometry (shaded: slits of the Fresnel fit's width at +-d/2), for the single-mode photons and the electrons;
abscissa in units of the slit separation (1 / the fitted fringe period). Usage (repo root):
  BLML_DATA_DIR=~/data_repo/dslit_gpu .venv/bin/python results/p3_fig_slits_real.py <N>  -> figs/slits_real.pdf"""
import os, sys, json, numpy as np, matplotlib
NBG = int(sys.argv[1]) if len(sys.argv) > 1 else 3
os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_SOLVER', 'lbfgs')
sys.argv = sys.argv[:1]; sys.path.insert(0, '.')
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
import dslit_job as dj
from blml_binned import _np
from p3_buildup import setup
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3))
for a, name, lab in zip(ax, ('photon', 'electron'), ('photons, single mode', 'electrons')):
    x, W, bflat, trv, te = setup(name); u = np.random.default_rng(0).random(len(x)); part = np.where(u < 0.64, 0, np.where(u < 0.80, 1, 2))
    TV = np.bincount(x[part < 2], minlength=W).astype(float)
    th = json.load(open(f'results/dslit_1d/{name}/flat/{name}_fresnel.json'))['params']; P, r = th['P'], th['a_over_d']
    d_nu = 1.0 / P                                   # the slit separation in the field's frequency (cycles / px)
    band = json.load(open(f'results/p3synth_lbfgs/bg1/{name}/{name}_complex_bg.json'))['fc'][0]
    diag = json.load(open(f'results/p3_slits_diag/{name}_K1-N{NBG}.json'))
    fits = [('band', band, None, 1, 'C0'), (f"two-slit kernel, $w={diag['fill']:g}$", diag['fc'], diag['fill'], NBG, 'C3')]
    L = 16 * W; nu = np.fft.fftshift(np.fft.fftfreq(L))
    for v in (-0.5, 0.5):
        a.axvspan(v - r / 2, v + r / 2, color='0.85', lw=0)
    for nm, fc, fill, nb, col in fits:
        m = dj.bg_fit(TV, [fc], (nb,), True, 1, max_sweeps=5000, tol=1e-6, fill=fill)[1]
        g = np.asarray(_np(m.Se_))[0] / m.n_; A = np.abs(np.fft.fftshift(np.fft.fft(np.r_[g, np.zeros(L - W)])))
        a.plot(nu / d_nu, A / A.max(), color=col, lw=1, label=nm)
    a.set_xlim(-1.3, 1.3); a.set_yticks([0, 0.5, 1]); a.set_xlabel(r'aperture coordinate / slit separation')
    a.set_title(lab, fontsize=8, loc='left'); a.legend(fontsize=6.5, frameon=False, loc='upper right')
ax[0].set_ylabel(r'$|\hat g|$ (rel.)')
fig.tight_layout(pad=0.3)
fig.savefig('papers/submit/pra_complex/figs/slits_real.pdf'); fig.savefig('papers/submit/pra_complex/figs/slits_real.png', dpi=200)
print('done')
