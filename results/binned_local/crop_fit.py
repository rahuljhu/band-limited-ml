"""513 x 513 native crop around bead 0 (plane 5), complex rank-1 BLMLBinned (ball, 2NA/lambda), train/val/test 70/10/20.
Saves density and coefficients for the figures; also the 31 and 51 px window fits of bead 0 on the same train counts."""
import sys, json, time, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_binned import BLMLBinned
FC = 2 * 1.3 / 0.515 * 0.065
N = np.load('/private/tmp/claude-501/-Users-ra/e6cd45ce-b6f4-42cc-be9a-45c946facadf/scratchpad/gpu_data/psf_plane5.npy').astype(np.int64)
y0, x0 = 661, 1068; H = 256; C = N[y0 - H:y0 + H + 1, x0 - H:x0 + H + 1]
r = np.random.default_rng(0); tr = r.binomial(C, 0.7); rest = C - tr; va = r.binomial(rest, 1 / 3); te = rest - va
score = lambda P, M: float((M * np.log(np.maximum(P, 1e-300))).sum() / M.sum())
modes = bt.make_modes([FC, FC], ['ball', 'ball']); out = {}
t = time.time(); m = BLMLBinned(tol=1e-4, verbose=True).fit(tr, modes, complex_g=True); out['crop'] = dict(time_s=time.time() - t, sweeps=m.n_sweeps_, conv=bool(m.converged_), residual=m.residual_)
f = m.density_; P = f / f.sum(); out['crop'].update(test=score(P, te), mass=float(f.sum()))
np.savez_compressed('results/binned_local/crop513.npz', density=f.astype(np.float32), train=tr.astype(np.int32), test=te.astype(np.int32),
                    rho=np.array(m.rho_history_, np.float32))
for h in (15, 25):
    W = tr[H - h:H + h + 1, H - h:H + h + 1]; Wt = te[H - h:H + h + 1, H - h:H + h + 1]
    mw = BLMLBinned(tol=1e-5).fit(W, modes, complex_g=True); Pw = mw.density_ / mw.density_.sum()
    Pc = f[H - h:H + h + 1, H - h:H + h + 1]; Pc = Pc / Pc.sum()
    out[f'win{2 * h + 1}'] = dict(test_window=score(Pw, Wt), test_window_cropfit=score(Pc, Wt), mass_in_window=float(mw.density_.sum()))
    np.save(f'results/binned_local/win{2 * h + 1}.npy', mw.density_.astype(np.float32))
print(json.dumps(out, indent=1)); json.dump(out, open('results/binned_local/crop_fit.json', 'w'), indent=1)
