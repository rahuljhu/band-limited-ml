"""Does the one-sweep response (implicit_grad) matter in the default streaming configuration (smoothing grad_eps 1e-3,
gradients pooled over 3 batches)? The 2-D benchmark of bench_ablation.py, n = 2e4, CPU, float32, burn-in 5000,
batches of 2000, two passes, the paper's lattice protocol (rebin_tol 1, 2 sweeps); five seeds, on and off.
Writes results/bench/igrad_onoff.json."""
import os, sys, json, time, numpy as np
sys.path.insert(0, '.')
src = open('results/bench_ablation.py').read()
exec(src[src.index('import'):src.index('Xtr, Xte = draw(N)')].replace("N = int(sys.argv[1])", "N = 20000"))
rows = []
for seed in range(5):
    rng = np.random.default_rng(seed)                    # draw() samples from the module-level rng
    Xtr, Xte = draw(20000), draw(2000)
    for resp in (False, True):
        t = time.time()
        s = BLMLQuickStreaming(search='auto', burn_in=5000, dtype=np.float32, implicit_grad=resp, seed=seed, rebin_tol=1.0,
                               rebin_sweeps=2).fit(Xtr, bt.make_modes(np.nan, ['sinc', 'sinc']), epochs=2, batch=2000)
        ll = float(np.mean(np.log(np.maximum(s.predict(Xte), 1e-300))))
        rows.append(dict(seed=seed, implicit_grad=resp, ll=ll, s=time.time() - t, fc=np.ravel(bt.modes_fc(s.modes_)[0]).tolist()))
        print(rows[-1], flush=True)
json.dump(rows, open('results/bench/igrad_onoff.json', 'w'), indent=1)
d = [r['ll'] for r in rows if r['implicit_grad']]; o = [r['ll'] for r in rows if not r['implicit_grad']]
print('on - off per seed:', np.round(np.array(d) - np.array(o), 4), 'mean', round(float(np.mean(np.array(d) - np.array(o))), 4))
