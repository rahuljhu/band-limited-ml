"""solver='lbfgs' on the GPU against the CPU (Colab T4: ./colab_job.sh start lbfgsgpu lbfgs_gpu_check.py T4).
A synthetic two-slit line (apodised slits, 512 bins, 8000 particles), float64 throughout. BLMLBinned: one complex mode,
the background mode (one harmonic) and rank 2 at fc = 0.0226, 0.045, 0.18, solvers cd and lbfgs, CPU and GPU;
BLMLQuick (counts on the pixel lattice): one complex mode at fc = 0.045, 0.18; BLMLBinnedStreaming with lbfgs. Writes
lbfgs_gpu_check.json (train log-likelihood per particle, iterations, seconds) to BLML_OUT_DIR."""
import os, json, time, numpy as np
import blml_trivial as bt
from blml_binned import BLMLBinned, BLMLBinnedStreaming, _np
OUT = os.environ.get('BLML_OUT_DIR', 'results/lbfgsgpu'); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0); W = 512; NF = 1 << 16; nu = np.fft.fftfreq(NF); c, a = 0.0085, 0.005
u = np.abs(np.abs(nu) - c) / (a / 2); Ap = np.where(u < 1, np.cos(np.pi * u / 2) ** 2, 0.0)
g = np.fft.fftshift(np.fft.ifft(Ap))[NF // 2 - W // 2:NF // 2 + W // 2]; f = np.abs(g) ** 2; f = 0.9 * f / f.sum() + 0.1 / W
C = rng.multinomial(8000, f).astype(float)
ll = lambda P: float((C * np.log(np.maximum(P / P.sum(), 1e-300))).sum() / C.sum())
bgm = lambda fc: [bt.make_modes([fc], ['sinc'])[0], {0: dict(kind='dirichlet', period=float(W), fc=dict(value=3 / W, trainable=False))}]
res = []
for fc in (0.0226, 0.045, 0.18):
    for lab, md in (('complex', bt.make_modes([fc], ['sinc'])), ('bg', bgm(fc)), ('rank2', bt.make_modes([fc], ['sinc'], rank=2))):
        for engine in ('cpu', 'gpu'):
            for solver in ('cd', 'lbfgs'):
                t = time.time()
                m = BLMLBinned(engine=engine, tol=1e-6, max_sweeps=30000, solver=solver).fit(C, md, complex_g=True)
                res.append(dict(est='binned', fc=fc, model=lab, engine=engine, solver=solver, ll=ll(_np(m.density_)),
                                it=int(m.n_sweeps_), s=time.time() - t, stop=m.stop_reason_))
                print(res[-1], flush=True)
occ = np.nonzero(C)[0]; X = occ[:, None].astype(float); w = C[occ]; Xv = np.arange(W)[:, None].astype(float)
for fc in (0.045, 0.18):
    md = bt.make_modes([fc], ['sinc']); md[0][0]['h'] = 1.0
    for engine in ('cpu', 'gpu'):
        for solver in ('cd', 'lbfgs'):
            t = time.time()
            m = bt.BLMLQuick(tol=1e-6, max_sweeps=30000, dtype=np.float64, solver=solver, engine=engine).fit(
                X, md, complex_g=True, sample_weight=w)
            res.append(dict(est='quick', fc=fc, model='complex', engine=engine, solver=solver, ll=ll(np.asarray(m.predict(Xv))),
                            it=int(m.n_sweeps_), s=time.time() - t))
            print(res[-1], flush=True)
batches = [rng.multinomial(400, f).astype(float) for _ in range(20)]
for engine in ('cpu', 'gpu'):
    t = time.time()
    m = BLMLBinnedStreaming(burn_in=1000, lr=0.05, param_every=1, tol=1e-6, max_sweeps=5000, engine=engine, solver='lbfgs').fit(
        batches, [{0: dict(kind='sinc')}], epochs=10, complex_g=True, tol=1e-4, tol_check='exact')
    res.append(dict(est='stream', engine=engine, solver='lbfgs', fc=float(m._fc_now()[0][0]), ll=ll(_np(m.density_)),
                    it=int(m.epochs_), s=time.time() - t))
    print(res[-1], flush=True)
json.dump(res, open(f'{OUT}/lbfgs_gpu_check.json', 'w'), indent=1)
print('done')
