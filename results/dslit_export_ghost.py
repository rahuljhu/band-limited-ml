"""Paper 3, double slit: the Glasgow ghost-diffraction record (single mode) in dslit_job.py's format, like
results/dslit_export.py does for the heralded records. In ghost diffraction the slits are in the other arm of the
photon pair: the camera photons never meet the slits, and the pattern forms only in the heralded coincidences.
Crop: the 256 columns around the beam (centre col 252), the rows whole; background flat, weight fitted (EM)."""
import os, numpy as np
OUT = os.path.expanduser('~/data_repo/dslit_gpu/'); G = os.path.expanduser('~/data_repo/glasgow_dslit/')
P = np.load(G + 'Ghost_Diffraction_SM_clean.npz')['photons']; X0, X1 = 124, 380
P = P[(P[:, 2] >= X0 - 0.5) & (P[:, 2] < X1 - 0.5)].copy(); P[:, 2] -= X0
b = np.ones((1, 512, X1 - X0)); b /= b.sum()
np.save(OUT + 'dslit_ghost_points.npy', P); np.save(OUT + 'dslit_ghost_b.npy', b); np.save(OUT + 'dslit_ghost_axis.npy', np.array(0))
np.save(OUT + 'dslit_ghost_beta.npy', np.array([np.nan]))
print('ghost particles', len(P), 'lattice', b.shape[1:])
