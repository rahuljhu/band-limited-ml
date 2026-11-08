"""Read the localize-psf example bead stack (zarr, CC-BY-4.0, Zenodo 10022862) and save the
widefield z-stack as photons: (ADU - 100) * 0.51 e-/ADU. Run in the isolated reader env."""
import sys, json, numpy as np, zarr
src, out = sys.argv[1], sys.argv[2]
g = zarr.open_group(src, mode='r')
a = np.asarray(g['cam1/widefield_blue'][0, 0, :, 0, 0])            # (z, y, x) uint16
ph = (a.astype(np.float32) - 100.0) * 0.51
print('stack', a.shape, 'ADU range', int(a.min()), int(a.max()))
np.save(out + '/stack_photons.npy', ph)
