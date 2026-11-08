"""Upsampling a natural image from block-summed counts (paper 7): BLML empirical (+sqrt h, complex), BLML binned
(complex) against nearest / bilinear / bicubic spline / Lanczos-3 / Fourier (sinc) interpolation.
Truth: matplotlib's grace_hopper.jpg, grey, 512 x 512, scaled to a mean of 50 photons per pixel; fine counts Poisson,
thinned into train / val / test (64 / 16 / 20%); train and val are block-summed r x r (r = 2, 4). Each method
reconstructs the fine expected train counts from the coarse train counts (fc, lam chosen on the coarse val counts);
scored on the interior fine pixels (inside the coarse lattice's centres) by: held-out fine test photons (log
probability per photon, the reconstruction normalised over the interior; negatives clipped and counted), PSNR against
the clean truth (scaled to train photons), and the fraction of negative pixels. Note: a coarse count is the block
integral of f; every method treats it as a sample of the box-blurred image, so none undoes the box blur.
Output results/upsample_natural.json, papers/paper7/figs/upsample_natural.png"""
import os, sys, json, time, numpy as np
sys.path.insert(0, '.')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import map_coordinates
import blml_trivial as bt
from blml_binned import BLMLBinned, BLMLEmpirical, _np

img = np.asarray(Image.open(os.path.join(matplotlib.get_data_path(), 'sample_data', 'grace_hopper.jpg')).convert('L'), float)
I = img[40:552, :512] / 255.0
L = I.shape[0]
LEVELS = (50.0, 2.0)                                            # mean photons per fine pixel: normal and low light


def block(A, r):
    return A.reshape(L // r, r, L // r, r).sum((1, 3)).astype(float)


def lanczos_matrix(m_c, r, a=3):
    """(L, m_c) Lanczos-a weights from coarse samples (centres r j + (r-1)/2) to fine pixel centres i."""
    u = (np.arange(L) - (r - 1) / 2) / r; j = np.arange(m_c)
    d = u[:, None] - j[None, :]; K = np.sinc(d) * np.sinc(d / a) * (np.abs(d) < a)
    return K / K.sum(1, keepdims=True)


def fourier_matrix(m_c, r):
    """(L, m_c) periodic band-limited (Dirichlet) interpolation from the coarse samples to the fine centres."""
    u = (np.arange(L) - (r - 1) / 2) / r; j = np.arange(m_c); d = u[:, None] - j[None, :]
    k = np.fft.fftfreq(m_c) * m_c
    return np.real(np.exp(2j * np.pi * np.outer(d.ravel(), k) / m_c).sum(1).reshape(d.shape)) / m_c


def blml_fine(m, r, origin):
    """The fit's f on the fine pixel centres: predict_grid(2r) lands on them every other point."""
    G = m.predict_grid(2 * r)                                   # points origin + k h / (2 r) = origin + k / 2
    o = int(round(2 * (np.ceil(origin) - origin)))               # first index on an integer fine position
    return G[o::2, o::2], int(np.ceil(origin))


res = {}; figs = {}
for level in LEVELS:
  lam_true = I / I.mean() * level
  r0 = np.random.default_rng(0); Nf = r0.poisson(lam_true)
  tr = r0.binomial(Nf, 0.64); rest = Nf - tr; va = r0.binomial(rest, 16 / 36); te = rest - va
  for r in (1, 2, 4):
      C, Cv = block(tr, r), block(va, r); mc = L // r; origin = (r - 1) / 2; ntr = float(C.sum())
      lo, hi = int(np.ceil(origin)), int(np.floor(L - 1 - origin)); sl = slice(lo, hi + 1)
      truth = lam_true[sl, sl] * 0.64; tst = te[sl, sl].astype(float)
      def metrics(rec, t_s):
          rec = np.asarray(rec, float); neg = float(np.mean(rec < 0))
          p = np.maximum(rec, 1e-6 * max(rec.mean(), 1e-12)); p = p / p.sum()
          mse = float(np.mean((rec - truth) ** 2)); psnr = 10 * np.log10(truth.max() ** 2 / mse)
          return dict(test_loglik=float((tst * np.log(p)).sum() / tst.sum()), psnr=float(psnr), neg_frac=neg,
                      min=float(rec.min()), time_s=t_s)
      out = {}; M = C / r ** 2                                     # coarse means per fine pixel
      u = (np.arange(L) - origin) / r; UU, VV = np.meshgrid(u, u, indexing='ij')
      from scipy.ndimage import gaussian_filter
      t = time.time(); bestg = None
      for sg in (0.0, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8):                 # Gaussian blur on the coarse means, sigma on val, then bicubic
          Mb = gaussian_filter(M, sg / r, mode='nearest') if r > 1 else gaussian_filter(M, sg, mode='nearest')
          pv = np.maximum(Mb, 1e-9); pv = pv / pv.sum(); sv = float((Cv * np.log(pv)).sum() / Cv.sum())
          if bestg is None or sv > bestg[0]:
              bestg = (sv, sg, Mb)
      recg = map_coordinates(bestg[2], [UU, VV], order=3, mode='nearest') if r > 1 else bestg[2]
      out['Gaussian blur (+bicubic)'] = dict(metrics(recg[sl, sl], time.time() - t), sigma_fine_px=float(bestg[1])); figs[(level, r, 'Gaussian blur (+bicubic)')] = recg
      t = time.time(); out['nearest'] = metrics(np.kron(M, np.ones((r, r)))[sl, sl], time.time() - t)
      for name, order in (('bilinear', 1), ('bicubic spline', 3)):
          t = time.time(); rec = map_coordinates(M, [UU, VV], order=order, mode='nearest'); out[name] = metrics(rec[sl, sl], time.time() - t)
          figs[(level, r, name)] = rec
      t = time.time(); A = lanczos_matrix(mc, r); rec = A @ M @ A.T; out['Lanczos-3'] = metrics(rec[sl, sl], time.time() - t); figs[(level, r, 'Lanczos-3')] = rec
      t = time.time(); A = fourier_matrix(mc, r); rec = A @ M @ A.T; out['Fourier (sinc)'] = metrics(rec[sl, sl], time.time() - t); figs[(level, r, 'Fourier (sinc)')] = rec
      figs[(level, r, 'nearest')] = np.kron(M, np.ones((r, r)))
      fcs = np.array([f_ for f_ in (0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5) if f_ <= 0.5 / r + 1e-12])
      # cycles per fine pixel; f's band = fc, capped at the coarse Nyquist 0.5 / r: above it the coarse data (val too) alias
      lams = [1e-8]                                               # lam is only a numerical stabiliser: fixed, not selected
      sel = None
      for name, kw in (('BLML empirical +sqrt h', {}), ('BLML empirical complex', dict(complex_g=True))):
          t = time.time(); best = None
          grid = [(fc, lams) for fc in fcs] if sel is None else \
              [(fc, lams) for fc in fcs if 0.6 * sel[0] <= fc <= 1.7 * sel[0]]
          ekw = dict(max_outer=20, outer_tol=0) if 'complex' in name else {}   # held-out gets worse past ~10-20 outer steps
          for fc, lg in grid:                                         # complex: around the +sqrt h choice
              modes = bt.make_modes([fc, fc], ['sinc', 'sinc'])
              prev = None
              for lam in lg:
                  m = BLMLEmpirical(lam=lam, tol=1e-6, max_iter=20000, **ekw).fit(C, modes, h=float(r), origin=origin, init=prev, **kw); prev = m
                  s = m.score(Cv)
                  if best is None or s > best[0]:
                      best = (s, m, fc, lam)
          _, m, fc, lam = best; sel = (fc, lam); F, o0 = blml_fine(m, r, origin); rec = F * ntr   # f per unit area: x n = counts / fine px
          out[name] = dict(metrics(rec[:hi - lo + 1, :hi - lo + 1], time.time() - t), fc=float(fc), lam=float(lam), fc_over_nyq=float(fc * r / 0.5))
          figs[(level, r, name)] = np.pad(rec, ((lo, 0), (lo, 0)), mode='edge')[:L, :L]
          print(level, r, name, json.dumps(out[name]), flush=True)
      t = time.time(); best = None
      j0 = int(np.argmin(np.abs(fcs - out['BLML empirical complex']['fc'])))
      for fc in fcs[max(j0 - 1, 0):j0 + 2]:                         # around the empirical choice (binned is the slow one)
          m = BLMLBinned(tol=1e-5, max_sweeps=500, ll_tol=1e-9).fit(C, bt.make_modes([fc, fc], ['sinc', 'sinc']), h=float(r), origin=origin, complex_g=True)
          p = _np(m.density_); p = p / p.sum(); s = float((Cv * np.log(np.maximum(p, 1e-300))).sum() / Cv.sum())
          if best is None or s > best[0]:
              best = (s, m, fc)
      _, m, fc = best; F, _ = blml_fine(m, r, origin); rec = F * ntr
      out['BLML binned complex'] = dict(metrics(rec[:hi - lo + 1, :hi - lo + 1], time.time() - t), fc=float(fc), fc_over_nyq=float(fc * r / 0.5))
      figs[(level, r, 'BLML binned complex')] = np.pad(rec, ((lo, 0), (lo, 0)), mode='edge')[:L, :L]
      for k, v in out.items():
          print(f"level={level} r={r} {k:24s} test {v['test_loglik']:.5f}  PSNR {v['psnr']:.2f}  neg {v['neg_frac']:.4f}  {v['time_s']:.1f}s"
                + (f"  fc/Nyq {v['fc_over_nyq']:.2f}" if 'fc_over_nyq' in v else ''), flush=True)
      res[f'{level}ph_r{r}'] = out
  json.dump(res, open('results/upsample_natural.json', 'w'), indent=1)

# figure: one crop (eye) per light level, rows r = 1, 2, 4
names = ['nearest', 'Gaussian blur (+bicubic)', 'bicubic spline', 'Lanczos-3', 'Fourier (sinc)', 'BLML empirical +sqrt h',
         'BLML empirical complex', 'BLML binned complex']
ys, xs = slice(150, 278), slice(170, 298)
for level in LEVELS:
    lam_true = I / I.mean() * level
    fig, ax = plt.subplots(3, len(names) + 1, figsize=(2.3 * (len(names) + 1), 7.6))
    for row, r in enumerate((1, 2, 4)):
        ref = lam_true * 0.64; v0, v1 = np.percentile(ref[ys, xs], [0.5, 99.5])
        ax[row, 0].imshow(ref[ys, xs], cmap='gray', vmin=v0, vmax=v1); ax[row, 0].set_title('truth' if row == 0 else '', fontsize=9)
        ax[row, 0].set_ylabel(f'{r}x{r} binned' if r > 1 else 'no binning (denoise)', fontsize=9)
        for j, n in enumerate(names):
            if (level, r, n) not in figs:
                ax[row, j + 1].axis('off'); continue
            ax[row, j + 1].imshow(figs[(level, r, n)][ys, xs], cmap='gray', vmin=v0, vmax=v1)
            o = res[f'{level}ph_r{r}'][n]; ax[row, j + 1].set_title(f"{n}\n{o['psnr']:.2f} dB, {o['test_loglik']:.4f}", fontsize=7)
    for a_ in ax.ravel():
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f'{level:g} photons per pixel (mean): PSNR vs clean truth, held-out photon log-likelihood', fontsize=10)
    plt.tight_layout(); os.makedirs('papers/paper7/figs', exist_ok=True)
    plt.savefig(f'papers/paper7/figs/upsample_natural_{int(level)}ph.png', dpi=120); plt.close(fig)
