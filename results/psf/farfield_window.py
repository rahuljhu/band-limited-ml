"""Far field, in focus: f = FC * sinc(FC x)^2 = h^2, h real & band-limited. Fixed n, growing window.
Rank-1 binned BLML (coordinate descent) at fc = FC with a fixed sign pattern = signs of h (oracle);
plain constraint int_R g^2 = 1."""
import sys, time, numpy as np
sys.path.insert(0, 'results/psf'); import psfkit as pk; import blml_trivial as bt
FC = pk.FC_PHYS
for n in [int(float(a)) for a in sys.argv[1:]]:
    for H in [15, 50, 150, 500, 1500]:
        z = np.arange(-H, H + 1.0); pb = FC * np.sinc(FC * z) ** 2; p = pb / pb.sum(); sg = np.sign(np.sinc(FC * z) + 1e-12)
        kh, kb, ku, t0 = [], [], [], time.time()
        for s in range(3):
            N = np.random.default_rng(s).multinomial(n, p)
            kl = lambda q: float((p * np.log(p / (q / q.sum()))).sum())
            kh.append(kl(N + 0.5))
            m = bt.BLMLQuick(dtype=np.float64, tol=1e-6).fit(z[:, None], bt.with_lattice(bt.make_modes(FC, None, None, d=z[:, None].shape[1]), 1.0), sample_weight=N, signs=sg); kb.append(kl(np.maximum(m.predict(z), 1e-300)))
            m = bt.BLMLQuick(dtype=np.float64, tol=1e-6).fit(z[:, None], bt.with_lattice(bt.make_modes(FC, None, None, d=z[:, None].shape[1]), 1.0), sample_weight=N); ku.append(kl(np.maximum(m.predict(z), 1e-300)))
        print(f'n={n:>7d} window {2*H+1:5d}px outside {1-pb.sum():.5f} | histogram {np.mean(kh):.5f}  BLML signed {np.mean(kb):.5f} (ratio {np.mean(kb)/np.mean(kh):.2f})  BLML unsigned {np.mean(ku):.5f}  [{time.time()-t0:.0f}s, converged {m.converged_}]', flush=True)
