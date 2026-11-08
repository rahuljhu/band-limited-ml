"""Paper 3: time of one three-mode fit (two-slit mode + two single-slit modes + flat background, p3_partial3.py's
model) against the window width W, per engine (cpu float64; gpu float32 / float64), to choose where crop-free fits run.
Data: a sharp-slit partially coherent far field like the multimode record's (V 0.15, background share 0.62, fringe
period ~56 px, a/d 0.18), n = 10^6 detections; the fit at the true (fc, fill), cold start.
  [P3_WIDTHS=2560,8704,...] python p3_wide_timing.py   -> $BLML_OUT_DIR/p3_wide_timing.json"""
import os, sys, json, time, numpy as np
sys.path.insert(0, os.environ.get('BLML_CODE_DIR', '.'))
os.environ.setdefault('BLML_ENGINE', 'cpu')
import dslit_job as dj
from blml_binned import BLMLBinned
OUT = os.environ.get('BLML_OUT_DIR', 'results'); os.makedirs(OUT, exist_ok=True); fn = f'{OUT}/p3_wide_timing.json'
WIDTHS = [int(v) for v in os.environ.get('P3_WIDTHS', '2560,8704,33280,131584').split(',')]
try:
    import cupy; ENGINES = [('cpu', np.float64), ('gpu', np.float32), ('gpu', np.float64)]
except ImportError:
    ENGINES = [('cpu', np.float64)]
P_, R, V, BG, N = 56.0, 0.18, 0.15, 0.62, 10 ** 6
d = 1 / P_; a = R * d; fc, w = d + a, 2 * a / (d + a)          # slits at +-d/2 (cycles / px), width a
out = json.load(open(fn)) if os.path.exists(fn) else []
for W in WIDTHS:
    NF = max(1 << 16, 8 * W); nu = np.fft.fftfreq(NF)
    fld = lambda c: np.fft.fftshift(np.fft.ifft((np.abs(nu - c) < a / 2).astype(float)))[NF // 2 - W // 2:NF // 2 + W // 2]
    g1, g2 = fld(-d / 2), fld(d / 2); I = V * np.abs(g1 + g2) ** 2 + (1 - V) * (np.abs(g1) ** 2 + np.abs(g2) ** 2)
    P = (1 - BG) * I / I.sum() + BG / W; P /= P.sum()
    T = np.bincount(np.random.default_rng(0).choice(W, size=N, p=P), minlength=W).astype(float)
    for eng, dt in ENGINES:
        if any(r['W'] == W and r['engine'] == eng and r['dtype'] == np.dtype(dt).name for r in out):
            continue
        modes = dj.bg_modes([fc], (W,), (0,), 1, w, incoh=2)
        t0 = time.time()
        m = BLMLBinned(tol=1e-6, max_sweeps=5000, ll_tol=1e-10, engine=eng, dtype=dt, solver='lbfgs').fit(T, modes, complex_g=True)
        s = time.time() - t0
        r = dict(W=W, engine=eng, dtype=np.dtype(dt).name, s=s, ll=float(m.loglik_), it=int(getattr(m, 'n_iter_', -1)),
                 stop=str(getattr(m, 'stop_reason_', '')))
        out.append(r); json.dump(out, open(fn, 'w'), indent=1); print(r, flush=True)
print('all done', flush=True)
