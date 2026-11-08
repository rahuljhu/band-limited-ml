"""Paper 3, double slit: streaming BLML (BLMLBinnedStreaming, Adam on log fc per axis, implicit gradient) on the cleaned
photons in arrival order (results/dslit_prep.py), batches of RB readouts, so fc is chosen online by the prequential score.
Usage: python results/dslit_stream.py <set> <real|complex> [fcy0 fcx0] -> results/dslit/stream_<set>_<kind>.json"""
import sys, os, json, time, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_binned import BLMLBinnedStreaming
name, kind = sys.argv[1], sys.argv[2]; fc0 = [float(v) for v in sys.argv[3:5]] or [0.05, 0.1]
RB, BURN = 100, 1000
P = np.load(os.path.expanduser(f'~/data_repo/glasgow_dslit/{name}_clean.npz'))['photons']
iy, ix = (np.clip(np.floor(P[:, j] + 0.5).astype(int), 0, 511) for j in (1, 2)); rd = P[:, 0].astype(int)
m = BLMLBinnedStreaming(burn_in=BURN, lr=0.1, param_every=3, tol=1e-5, max_sweeps=3000, ll_tol=1e-9, workers=4,
                        batch_sweeps=None if kind == 'real' else 200)
modes = bt.make_modes(fc0, ['sinc', 'sinc']); t = time.time()
for b0 in range(0, rd.max() + 1, RB):
    s = (rd >= b0) & (rd < b0 + RB); C = np.zeros((512, 512)); np.add.at(C, (iy[s], ix[s]), 1)
    if m.n_batches_ == 0: m.partial_fit(C, modes, complex_g=(kind == 'complex'))
    else: m.partial_fit(C)
    hh = m.history_[-1]; print(f"n={hh['n']:.0f} preq={hh['preq']:.4f} fc={np.round(hh['fc'][0], 4)} {time.time() - t:.0f}s", flush=True)
json.dump(dict(set=name, kind=kind, fc0=fc0, history=m.history_), open(f'results/dslit/stream_{name}_{kind}.json', 'w'), indent=1)
