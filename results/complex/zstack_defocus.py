"""Paper 3, Sec. 6 check: one defocused training plane (3 or 7) fitted alone by complex BLML, propagated to the test
planes. Twin chosen without other planes: the candidate (e or conj e) whose back-propagated focal intensity (plane 5)
is most compact (largest sum q^2), i.e. uses only the stage position. 'twin_lik' instead chooses on the other training
planes, for reference. Same thinning, floor and score as zstack.py. Output results/complex/zstack_defocus.json."""
import sys, json, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'results/complex')
import zstack as Z
Z.set_zeta(Z.ZETA['blml'])
st = np.load('results/psf/stack_photons.npy', mmap_mode='r'); pos = np.load('results/psf/beads_iso.npy')
rows = []
for bead in range(53):
    y0, x0 = pos[bead]; H = Z.H
    cnt = {z: np.round(np.maximum(np.asarray(st[z, y0 - H:y0 + H + 1, x0 - H:x0 + H + 1], float) / Z.GAIN_CORR, 0))
           .astype(np.int64).ravel() for z in Z.TRAIN + Z.TEST}
    rng = np.random.default_rng(100 + bead); hold = {z: rng.binomial(cnt[z], 0.2) for z in Z.TEST}
    r = dict(bead=bead)
    for fp in (3, 7):
        e, _ = Z.blml_stack(cnt[fp][None], [fp])
        cands = (e, np.conj(e))
        comp = [float((Z.predict_blml(c, [fp], Z.FOCUS) ** 2).sum()) for c in cands]
        lik = [sum(Z.score(Z.predict_blml(c, [fp], t), cnt[t], cnt[t])[0] for t in Z.TRAIN if t != fp) for c in cands]
        for rule, crit in (('compact', comp), ('twin_lik', lik)):
            c = cands[int(np.argmax(crit))]
            r[f'p{fp}_{rule}'] = {z: Z.score(Z.predict_blml(c, [fp], z), hold[z], cnt[z] - hold[z])[0] for z in Z.TEST}
        r[f'p{fp}_agree'] = int(np.argmax(comp) == np.argmax(lik))
    rows.append(r); print(bead, json.dumps({k: (np.mean(list(v.values())) if isinstance(v, dict) else v) for k, v in r.items()}), flush=True)
    json.dump(rows, open('results/complex/zstack_defocus.json', 'w'))
