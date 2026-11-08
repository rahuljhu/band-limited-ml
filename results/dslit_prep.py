"""Paper 3, single-photon double slit (Aspden, Padgett & Spalding 2016, doi 10.5525/gla.researchdata.281): photon events
in arrival order from the cumulative ICCD frames. Each ASCII frame is the running per-pixel count of above-threshold pixels
(thresholded and binarised per readout by the authors, then accumulated), so frame k minus frame k-1 holds readout k.
One photon lights a small cluster of adjacent pixels (the intensifier spot): within a readout, 8-neighbour pixel pairs are
about 470x more common than with readout labels shuffled. So photons are found by cleaning, reported step by step:
  1. negative increments (impossible in a cumulative record) are counted and dropped;
  2. photons: 8-connected clusters of a readout's pixels; position = the cluster centroid (sub-pixel), binned to the pixel by floor(c + 1/2);
  3. streaks / ion events: clusters wider than MAX_EXT pixels or larger than MAX_SIZE pixels (the extent distribution falls
     ~25x from 3 to 4 px and is flat beyond 5) are removed;
  4. hot pixels: pixels whose photon count is above 6 Poisson sigmas of a 9x9 median of the cleaned image and >= 3.
Usage: python results/dslit_prep.py <set>   (Heralded_Diffraction_SM | Heraded_DIffraction_MM | Ghost_Diffraction_SM)
Writes ~/data_repo/glasgow_dslit/<set>_clean.npz: photons (readout, y, x centroid) in arrival order, counts (512 x 512),
hot (mask), and <set>_clean.json (the log)."""
import sys, os, io, zipfile, json, numpy as np
from scipy.ndimage import median_filter, label, find_objects, center_of_mass
MAX_EXT, MAX_SIZE = 4, 12
D = os.path.expanduser('~/data_repo/glasgow_dslit/'); name = sys.argv[1]
z = zipfile.ZipFile(D + f'ex/{name}.zip'); fs = sorted([i for i in z.infolist() if not i.is_dir()], key=lambda i: i.filename)
prev = np.zeros((512, 512)); ph = []; neg = 0; npix = 0; streak = 0; ext_h = np.zeros(32, int)
for k, i in enumerate(fs):
    a = np.loadtxt(io.BytesIO(z.read(i))); d = a - prev; prev = a
    neg += int(-d[d < 0].sum()); A = (d > 0).astype(int); npix += int(A.sum())
    if not A.any(): continue
    L, n = label(A, structure=np.ones((3, 3))); sl = find_objects(L); sz = np.bincount(L.ravel())[1:]
    cm = np.array(center_of_mass(A, L, range(1, n + 1)))
    for j in range(n):
        e = max(sl[j][0].stop - sl[j][0].start, sl[j][1].stop - sl[j][1].start); ext_h[min(e, 31)] += 1
        if e > MAX_EXT or sz[j] > MAX_SIZE: streak += 1; continue
        ph.append((k, cm[j][0], cm[j][1]))
ph = np.array(ph); R = len(fs)
log = dict(set=name, readouts=R, pixels_raw=npix, negative_increments=neg, clusters=int(ext_h.sum()), streak_clusters=streak,
           cluster_extent_hist=ext_h[:12].tolist(), photons_after_streaks=len(ph))
iy, ix = (np.clip(np.floor(ph[:, j] + 0.5).astype(int), 0, 511) for j in (1, 2))   # not np.rint: half-to-even sends every .5 centroid (two-pixel clusters) to even rows
C = np.zeros((512, 512), int); np.add.at(C, (iy, ix), 1)
M = median_filter(C.astype(float), 9); hot = (C > M + 6 * np.sqrt(np.maximum(M, 1.0))) & (C >= 3)
keep = ~hot[iy, ix]; ph = ph[keep]; C[hot] = 0
per = np.bincount(ph[:, 0].astype(int), minlength=R)
log.update(hot_pixels=int(hot.sum()), photons_on_hot_pixels=int((~keep).sum()), photons_clean=len(ph),
           photons_per_readout_median=float(np.median(per)), photons_per_readout_max=int(per.max()))
np.savez_compressed(D + f'{name}_clean.npz', photons=ph, counts=C, hot=hot)
json.dump(log, open(D + f'{name}_clean.json', 'w'), indent=1); print(json.dumps(log))
