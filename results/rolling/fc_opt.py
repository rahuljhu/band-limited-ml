"""Paper 5: ISE-optimal common cut-off fc_min for each synthetic cell (truth, d, n <= 1e4), the yardstick for the
streaming cut-offs. BLMLQuick at a common cut-off fc in every dimension, on the synthetic study's samples
(4 replicates) and evaluation sets; fc on a log grid (ratio 1.1) around the streaming cut-offs, relative ISE
averaged over replicates, fc_min from a parabola through the best three grid points in log fc.
Writes results/synstream/fcopt_<cell>.json.   env: BLML_SYN_TRUTHS, BLML_SYN_D, BLML_SYN_N"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')
import numpy as np
import blml_trivial as bt
from blml_trivial import make_modes, BLMLQuick
from p1_truth import Truth
from synth_stream import eval_sets
bt.DIRICHLET_LEGACY = False
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, '..', 'synstream')
TRUTHS = os.environ.get('BLML_SYN_TRUTHS', 'T1,T2,T3').split(',')
D_LIST = [int(v) for v in os.environ.get('BLML_SYN_D', '2,4,8').split(',')]
N_LIST = [int(v) for v in os.environ.get('BLML_SYN_N', '1000,10000').split(',')]


def rel_ise(m, ev):
    return float(np.mean((m.predict(ev['Xq']) - ev['fq']) ** 2 / ev['qq'])) / ev['int_f2']


for kind in TRUTHS:
    for d in D_LIST:
        truth = Truth(kind, d); ev = None
        for n in N_LIST:
            nm = f'{kind}_d{d}_n{n}'; p = os.path.join(OUT, f'fcopt_{nm}.json')
            if os.path.exists(p):
                continue
            ev = ev or eval_sets(truth, d)
            st = json.load(open(os.path.join(OUT, nm + '.json')))['reps']
            c = float(np.exp(np.median([np.mean(np.log(r['stream']['fc'])) for r in st])))
            Xs = [truth.sample(np.random.default_rng([3, d, n, rep]), n) for rep in range(len(st))]
            grid, ise, t0 = {}, {}, time.time()
            def at(k):                                   # mean relative ISE at fc = c * 1.1**k
                if k not in ise:
                    fc = c * 1.1 ** k; grid[k] = fc
                    ise[k] = float(np.mean([rel_ise(BLMLQuick(dtype=np.float64).fit(X, make_modes(fc, ['sinc'] * d)), ev) for X in Xs]))
                return ise[k]
            k = 0
            while True:                                  # walk downhill on the grid until a bracketed minimum
                lo, mid, hi = at(k - 1), at(k), at(k + 1)
                if mid <= lo and mid <= hi:
                    break
                k += -1 if lo < hi else 1
            y = np.log([lo, mid, hi]); a = (y[0] + y[2] - 2 * y[1]) / 2; b = (y[2] - y[0]) / 2
            kmin = k - b / (2 * a) if a > 0 else k
            res = dict(truth=kind, d=d, n=n, fc_min=float(c * 1.1 ** kmin), fc_grid_best=grid[k],
                       grid={f'{grid[j]:.5f}': ise[j] for j in sorted(ise)}, s=time.time() - t0)
            json.dump(res, open(p, 'w'))
            print(f"{nm}: fc_min {res['fc_min']:.3f} (stream centre {c:.3f}; {len(ise)} grid points, {res['s']:.0f}s)", flush=True)
print('ALL DONE')
