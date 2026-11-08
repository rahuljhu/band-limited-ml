"""Paper 3, Sec. 6 check: one-plane Zernike ML (plane 5 or 3 alone, zeta as in zstack.py), twin P(nu) vs
conj P(-nu) chosen on the other training planes, as for the one-plane BLML fit. Output zstack_oneplane_zern.json."""
import sys, json, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'results/complex')
import zstack as Z
L = 128; nu = np.fft.fftfreq(L); NX, NY = np.meshgrid(nu, nu, indexing='ij'); nu2 = NX ** 2 + NY ** 2
disk = nu2 <= Z.RHO ** 2; c0 = L // 2 - Z.H; zeta = Z.ZETA['zernike']
def pred(P, z):
    h = np.fft.fftshift(np.fft.ifft2(P * np.where(disk, np.exp(1j * Z.psi(nu2, (z - Z.FOCUS) * zeta * Z.DZ)), 0)))
    q = np.abs(h[c0:c0 + Z.W, c0:c0 + Z.W].ravel()) ** 2; return q / q.sum()
st = np.load('results/psf/stack_photons.npy', mmap_mode='r'); pos = np.load('results/psf/beads_iso.npy')
rows = []
for bead in range(53):
    y0, x0 = pos[bead]; H = Z.H
    cnt = {z: np.round(np.maximum(np.asarray(st[z, y0 - H:y0 + H + 1, x0 - H:x0 + H + 1], float) / Z.GAIN_CORR, 0))
           .astype(np.int64).ravel() for z in Z.TRAIN + Z.TEST}
    rng = np.random.default_rng(100 + bead); hold = {z: rng.binomial(cnt[z], 0.2) for z in Z.TEST}
    r = dict(bead=bead)
    for fp in (5, 3):
        P, _ = Z.ml_zernike(cnt[fp][None], [fp], zeta=zeta)
        cands = (P, np.conj(np.roll(np.flip(P, (0, 1)), 1, (0, 1))))
        lik = [sum(Z.score(pred(c, t), cnt[t], cnt[t])[0] for t in Z.TRAIN if t != fp) for c in cands]
        c = cands[int(np.argmax(lik))]
        r[f'p{fp}'] = {z: Z.score(pred(c, z), hold[z], cnt[z] - hold[z])[0] for z in Z.TEST}
    rows.append(r); print(bead, {k: round(float(np.mean(list(v.values()))), 4) for k, v in r.items() if k != 'bead'}, flush=True)
    json.dump(rows, open('results/complex/zstack_oneplane_zern.json', 'w'))
