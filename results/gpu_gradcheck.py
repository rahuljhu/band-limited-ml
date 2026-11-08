"""GPU fc_grad vs CPU fc_grad on small fits (sinc 2-D, complex, sinc + periodic 3-D). A few seconds."""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np
warnings.filterwarnings('ignore')
import blml_trivial as bt
from blml_trivial import BLMLQuick
rng = np.random.default_rng(0)
X2 = np.c_[rng.normal(size=3000), rng.normal(size=3000) * 0.7]; Q = X2[:300]
hr = rng.integers(0, 24, 3000).astype(float); X3 = np.c_[np.sin(2 * np.pi * hr / 24) + 0.5 * rng.normal(size=3000), hr, rng.normal(size=3000)]
for name, X, md, cx, units in [('sinc 2d', X2, bt.make_modes([1.5, 1.2], ['sinc', 'sinc']), False, [[0], [1]]),
                               ('complex', X2, bt.make_modes([1.5, 1.2], ['sinc', 'sinc']), True, [[0], [1]]),
                               ('sinc+dir 3d', X3, bt.make_modes([1.5, 0.3, 0.6], ['sinc', 'dirichlet', 'sinc'], [None, 24.0, None]), False, [[0], [2]])]:
    m = bt.BLMLQuick(dtype=np.float64, tol=1e-6).fit(X, md, complex_g=cx)
    gc = m.fc_grad(X[:300], units)
    g = BLMLQuick(engine='gpu', dtype=np.float64, max_sweeps=0).fit(m.Z_, bt.with_lattice(md, m._h_vec), sample_weight=m.w_, complex_g=cx, init=m)
    gg = g.fc_grad(X[:300], units)
    print(f'GRADCHECK {name}: max|gpu - cpu| {np.abs(gg - gc).max():.2e} (scale {np.abs(gc).max():.1f})', flush=True)
