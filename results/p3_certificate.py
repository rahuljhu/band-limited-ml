"""Paper 3, the certificate on the ladder's data: the unequal slits of p3_ladder.py (no background, n = 1e5, data set 0,
the same draws), one complex mode and no background mode, so that the modes share a kernel and the certificate
lambda_max(D^1/2 S D^1/2)/n of Theorem 2 applies. Fits: the band at the true cut-off from the default start and from
the true field; the two-slit kernel at the true geometry from six start seeds and from the true field. Prints and
writes results/p3_certificate.json: log-likelihood per particle, certificate, aperture error and amplitude ratio.
Run from the repo root."""
import os, sys, json, subprocess, numpy as np
os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_COLLAPSE', '1'); os.environ.setdefault('BLML_SOLVER', 'lbfgs')
S = 'results/p3_certificate_body.py'                       # p3_ladder.py's set-up (truth, draws, scores), then the fits below
src = open('p3_ladder.py').read(); src = src[:src.index('# kde')].replace("if os.path.exists(fn) and not RESCORE:\n    sys.exit(0)", "")
open(S, 'w').write(src + r'''
from types import SimpleNamespace
from blml_binned import BLMLBinned
import blml_trivial as bt
def fit1(fill=None, init=None, seed=0):
    sig = bt.make_modes([FC0], ['sinc' if fill is None else f'slits:{float(fill)!r}'])[0]
    m = BLMLBinned(tol=1e-9, max_sweeps=20000, ll_tol=1e-12, engine=dj.ENGINE, solver=dj.SOLVER, seed=seed).fit(TV, [sig], complex_g=True, init=init)
    Q = np.maximum(dj._np(m.density_), 1e-300); Q /= Q.sum(); fm = field_metrics(np.asarray(dj._np(m.Se_))[0] / m.n_)
    return dict(loglik=float((TV * np.log(Q)).sum() / TV.sum()), cert=float(m.certificate()), aperture_err=fm['aperture_err'], rho=fm['rho'])
E0 = np.zeros((1, W), complex); E0[0] = TV * gtrue / np.maximum(np.abs(gtrue) ** 2, 1e-300); true0 = SimpleNamespace(e_hat_=E0, shape_=(W,))
out = {'band, default start': fit1(), 'band, start at the true field': fit1(init=true0)}
out.update({f'two-slit kernel, seed {s}': fit1(W0, seed=s) for s in range(6)})
out['two-slit kernel, start at the true field'] = fit1(W0, init=true0)
json.dump(out, open('results/p3_certificate.json', 'w'), indent=1)
for k, v in out.items():
    print(f"{k:42s} loglik/n {v['loglik']:.7f}  cert {v['cert']:.4f}  aperture err {v['aperture_err']:.3f}  rho {v['rho']:.3f}")
''')
subprocess.run([sys.executable, S, 'asym', '0', '100000', '0'], check=True, env=dict(os.environ, LD_OUT='results/p3_certificate_tmp'))
