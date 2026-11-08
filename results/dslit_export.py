"""Paper 3, double slit: put both datasets in one format for dslit_job.py (CPU or Colab), in ~/data_repo/dslit_gpu/:
  dslit_<set>_points.npy  (N, 3): arrival order, row, col of each particle in crop coordinates (sub-pixel), arrival order
  dslit_<set>_b.npy       the background shape on the crop lattice, normalised to sum 1
  dslit_<set>_axis.npy    the fringe axis (0 rows, 1 cols)
Sets
  photon     Glasgow heralded far field, single mode (results/dslit_prep.py: cluster centroids, streaks and hot pixel removed),
             full 512 x 512 frame, fringes along rows, background uniform (dark counts / stray light over the frame)
  photon_mm  the same, multimode (partially coherent) illumination
  electron   UNL build-up (Bach et al. 2013), the authors' sub-pixel electron locations (Master_list_Double_Slit_+056.3).
             Crop: rows 176-193 (the signal line, 18 rows) x cols 64-599 (inside the detector's active edge in both the
             line and the sidebands). Background shape and level: the column profile of the electrons in the sidebands (rows 140-169
             and 200-229, outside the line; Gaussian-smoothed, sigma 5 px), uniform over the 18 rows; sideband electrons
             are not used otherwise.
  dslit_<set>_b.npy / _beta.npy   background shapes (J, H, W) and weights (J,), NaN = fitted by EM:
             photon     flat + the multimode pattern (gaussian-smoothed, sigma 4 px), both weights fitted; the frame cropped
                        to columns 152-407 (the beam +-8 sigma)
             photon_mm  flat, fitted
             electron   the sideband column profile, level measured on the plateau 12-22 rows from the line (0.35)"""
import os, json, numpy as np
from scipy.ndimage import gaussian_filter1d
OUT = os.path.expanduser('~/data_repo/dslit_gpu/'); os.makedirs(OUT, exist_ok=True)
def save(name, pts, b, axis, beta):
    """b: (J, H, W) background shapes (each normalised to sum 1); beta: (J,) their weights, NaN where EM fits them."""
    b = np.asarray(b, float); b = b / b.sum((1, 2), keepdims=True)
    np.save(OUT + f'dslit_{name}_beta.npy', np.asarray(beta, float))
    np.save(OUT + f'dslit_{name}_points.npy', pts); np.save(OUT + f'dslit_{name}_b.npy', b); np.save(OUT + f'dslit_{name}_axis.npy', np.array(axis))
    print(name, 'particles', len(pts), 'lattice', b.shape[1:], 'fringe axis', axis, 'background components', len(b), 'beta', beta)
G = os.path.expanduser('~/data_repo/glasgow_dslit/')
from scipy.ndimage import gaussian_filter
MM = gaussian_filter(np.load(G + 'Heraded_DIffraction_MM_clean.npz')['counts'].astype(float), 4)   # 405k multimode photons
flat = np.ones((512, 512))
# single mode: flat floor (intensifier dark events) + the multimode pattern (photons heralded without the single-mode
# selection, which Aspden et al. describe as the multimode SPDC light; the far-from-beam counts follow the diffraction
# envelope along the rows, as this component does): shapes fixed, weights by EM
# crop: the 256 columns around the beam (centre col 280, beam sigma 15 px: +-8 sigma), the rows whole
X0, X1 = 152, 408
def crop(P):
    P = P[(P[:, 2] >= X0 - 0.5) & (P[:, 2] < X1 - 0.5)].copy(); P[:, 2] -= X0; return P
save('photon', crop(np.load(G + 'Heralded_Diffraction_SM_clean.npz')['photons']), [flat[:, X0:X1], MM[:, X0:X1]], 0, [np.nan, np.nan])
save('photon_mm', crop(np.load(G + 'Heraded_DIffraction_MM_clean.npz')['photons']), [flat[:, X0:X1]], 0, [np.nan])
ev = np.load(os.path.expanduser('~/data_repo/unl_dslit/electrons.npy'))          # image, x, y, err
R0, R1, C0, C1 = 176, 194, 64, 600
x, y = ev[:, 1], ev[:, 2]; line = (y >= R0 - 0.5) & (y < R1 - 0.5) & (x >= C0 - 0.5) & (x < C1 - 0.5)
side = ((y >= 140) & (y < 170)) | ((y >= 200) & (y < 230))
prof = gaussian_filter1d(np.histogram(x[side], bins=np.arange(C0, C1 + 1) - 0.5)[0].astype(float), 5, mode='nearest')
# level: the plateau 12-22 rows from the line centre (row 185; the counts per row flatten there, then drift down with
# the detector's 18 mm active area further out), outside the scored crop; Bach et al. fit a flat offset A0
near = (((y >= 163) & (y < 174)) | ((y >= 197) & (y < 208))) & (x >= C0 - 0.5) & (x < C1 - 0.5)
save('electron', np.column_stack([ev[line, 0], y[line] - R0, x[line] - C0]), [np.tile(prof, (R1 - R0, 1))], 1, [float(near.sum() * (R1 - R0) / 22 / line.sum())])
