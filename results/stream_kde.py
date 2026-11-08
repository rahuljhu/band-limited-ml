"""Paper 5: streaming Gaussian KDE with its bandwidth learned online, against streaming BLML on the same stream.
Streaming KDE follows BLMLQuickStreaming's protocol (blml_rolling._StreamingBase) without the BCD step: burn-in (the
first batch; h from Scott's rule), then per batch: 1. prequential score log f_h(x) of the batch under the current KDE
of the held points (out of sample); 2. its gradient in log h (per dimension; exact: a KDE's coefficients are the data,
so there is no refit response), pooled over param_every batches; 3. the batch's points are added; 4. every
param_every batches, Adam on log h (blml_maps.adam_step, the streaming BLML lr and betas).
References at each checkpoint n: batch Gaussian KDE with h chosen on a validation sample (likelihood CV over a grid),
the oracle-best h on the test sample (the best a Gaussian KDE can do at that n), and Scott's rule (h ~ n^-1/(d+4)).
Truths: p1_truth T1, T2 (band-limited, fc 0.4) and T3 (correlated normal), d = 1, 2. Scores: excess held-out loss
E_test[log f_true - log f_hat] (KL, nats) and, in 1-d, the integrated squared error ISE = int (f_hat - f)^2 (the
n^-4/5 rate of a Gaussian KDE is for MISE; KL is dominated by the tails), by quadrature on a fine grid; log-log slopes
in n. For ISE the oracle h minimises ISE. Output results/stream_kde.json"""
import sys, json, time, numpy as np
sys.path.insert(0, '.')
import blml_trivial as bt
from blml_maps import adam_step
from blml_rolling import BLMLQuickStreaming
from p1_truth import Truth

N0, B, NMAX, NTEST, NVAL = 2000, 1000, 64000, 5000, 5000
CHECK = [2000, 4000, 8000, 16000, 32000, 64000]
LR, BETAS, PE = 0.1, (0.9, 0.999), 3


def kde_logpdf(Xtr, Xq, h, chunk=2000):
    """Gaussian product KDE log density at Xq (log-sum-exp over the training points)."""
    out = np.empty(len(Xq)); d = Xtr.shape[1]; c = -0.5 * d * np.log(2 * np.pi) - np.log(h).sum() - np.log(len(Xtr))
    for i in range(0, len(Xq), chunk):
        U = (Xq[i:i + chunk, None, :] - Xtr[None, :, :]) / h; L = -0.5 * (U ** 2).sum(2)
        mx = L.max(1, keepdims=True); out[i:i + chunk] = (mx[:, 0] + np.log(np.exp(L - mx).sum(1))) + c
    return out


def kde_grad(Xtr, Xq, h, chunk=1000):
    """mean over Xq of d log f_h(x) / d log h_j: E_w[u_j^2 / h_j^2] - 1 with weights w_i = K_h(x - x_i) / sum."""
    g = np.zeros(len(h)); tot = 0.0
    for i in range(0, len(Xq), chunk):
        U = (Xq[i:i + chunk, None, :] - Xtr[None, :, :]) / h; L = -0.5 * (U ** 2).sum(2)
        W = np.exp(L - L.max(1, keepdims=True)); W = W / W.sum(1, keepdims=True)
        g += ((W[:, :, None] * U ** 2).sum(1) - 1.0).sum(0); tot += len(U)
    return g / tot


def scott(X):
    n, d = X.shape; return X.std(0) * n ** (-1.0 / (d + 4))


class KDEStreaming:
    def __init__(self, lr=LR, betas=BETAS, param_every=PE):
        self.lr, self.betas, self.pe = lr, betas, param_every
        self.X = None; self.h = None; self.pool = []; self.adam = [0.0, 0.0, 0]; self.nb = 0; self.preq_ = []; self.h_path = []

    def partial_fit(self, X):
        if self.X is None:
            self.X = X.copy(); self.h = scott(X); self.h_path.append((len(self.X), self.h.copy())); return self
        self.preq_.append(float(kde_logpdf(self.X, X, self.h).mean()))
        self.pool.append(kde_grad(self.X, X, self.h))
        self.X = np.vstack([self.X, X]); self.nb += 1
        if self.nb % self.pe == 0:
            G = np.mean(self.pool, 0); self.pool = []
            self.h = np.asarray(adam_step(self.h, G, self.adam, 'scale', self.lr, self.betas, dlog=True), float)
        self.h_path.append((len(self.X), self.h.copy()))
        return self


def run(kind, d, seed):
    T = Truth(kind, d); rng = np.random.default_rng([seed, d, ord(kind[1])])
    Xs = T.sample(rng, NMAX); Xte = T.sample(rng, NTEST); Xva = T.sample(rng, NVAL)
    lt = np.log(T.pdf(Xte)); out = dict(kind=kind, d=d, seed=seed, n=[], stream_kde=[], stream_blml=[], kde_val=[], kde_oracle=[],
                                       kde_scott=[], h_stream=[], h_val=[], h_oracle=[], h_scott=[])
    sk = KDEStreaming(); sb = BLMLQuickStreaming(burn_in=N0, lr=LR, param_every=PE, seed=seed)
    modes = bt.make_modes([0.3] * d, ['sinc'] * d)
    t = time.time(); pos = 0
    for n in range(N0, NMAX + 1, B):
        Xb = Xs[pos:n] if pos == 0 else Xs[pos:n]
        sk.partial_fit(Xb)
        if pos == 0:
            sb.partial_fit(Xb, modes)
        else:
            sb.partial_fit(Xb)
        pos = n
        if n in CHECK:
            Xn = Xs[:n]; h0 = scott(Xn); grid = [h0 * 2.0 ** (k / 4) for k in range(-12, 5)]
            ex = lambda lp: float(np.mean(lt - lp))
            vals = [kde_logpdf(Xn, Xva, h).mean() for h in grid]; hv = grid[int(np.argmax(vals))]
            tests = [ex(kde_logpdf(Xn, Xte, h)) for h in grid]; ho = grid[int(np.argmin(tests))]
            out['n'].append(n); out['stream_kde'].append(ex(kde_logpdf(Xn, Xte, sk.h))); out['kde_val'].append(ex(kde_logpdf(Xn, Xte, hv)))
            out['kde_oracle'].append(min(tests)); out['kde_scott'].append(ex(kde_logpdf(Xn, Xte, h0)))
            out['stream_blml'].append(ex(np.log(np.maximum(sb.predict(Xte), 1e-300))))
            if d == 1:                                              # ISE by quadrature (trapezoid on a fine grid)
                xg = np.linspace(-80, 80, 4001)[:, None]; fg = T.pdf(xg); dx = xg[1, 0] - xg[0, 0]
                ise = lambda fh: float(np.trapezoid((fh - fg) ** 2, dx=dx))
                ig = [ise(np.exp(kde_logpdf(Xn, xg, h))) for h in grid]
                for k_, v_ in (('ise_stream_kde', ise(np.exp(kde_logpdf(Xn, xg, sk.h)))), ('ise_kde_val', ise(np.exp(kde_logpdf(Xn, xg, hv)))),
                               ('ise_kde_oracle', min(ig)), ('ise_kde_scott', ise(np.exp(kde_logpdf(Xn, xg, h0)))),
                               ('ise_stream_blml', ise(sb.predict(xg)))):
                    out.setdefault(k_, []).append(v_)
                out.setdefault('h_oracle_ise', []).append([float(grid[int(np.argmin(ig))][0])])
                print(f"      ISE: streamKDE {out['ise_stream_kde'][-1]:.3e}  KDE(val h) {out['ise_kde_val'][-1]:.3e}  KDE(oracle) {out['ise_kde_oracle'][-1]:.3e}  "
                      f"Scott {out['ise_kde_scott'][-1]:.3e}  streamBLML {out['ise_stream_blml'][-1]:.3e}", flush=True)
            for k_, v_ in (('h_stream', sk.h), ('h_val', hv), ('h_oracle', ho), ('h_scott', h0)):
                out[k_].append([float(x) for x in np.atleast_1d(v_)])
            print(f"{kind} d={d} s={seed} n={n:6d}  excess loss: streamKDE {out['stream_kde'][-1]:.5f}  KDE(val h) {out['kde_val'][-1]:.5f}  "
                  f"KDE(oracle) {out['kde_oracle'][-1]:.5f}  Scott {out['kde_scott'][-1]:.5f}  streamBLML {out['stream_blml'][-1]:.5f}   "
                  f"h stream/oracle/Scott {sk.h[0]:.3f}/{ho[0]:.3f}/{h0[0]:.3f}  {time.time() - t:.0f}s", flush=True)
    ln = np.log(out['n'])
    out['slopes'] = {k_: float(np.polyfit(ln, np.log(np.maximum(out[k_], 1e-12)), 1)[0])
                     for k_ in ('stream_kde', 'kde_val', 'kde_oracle', 'kde_scott', 'stream_blml')}
    if d == 1:
        out['slopes'].update({k_: float(np.polyfit(ln, np.log(out[k_]), 1)[0]) for k_ in
                              ('ise_stream_kde', 'ise_kde_val', 'ise_kde_oracle', 'ise_kde_scott', 'ise_stream_blml')})
    out['h_slope'] = {k_: float(np.polyfit(ln, np.log([h[0] for h in out[k_]]), 1)[0]) for k_ in
                      ('h_stream', 'h_oracle', 'h_scott') + (('h_oracle_ise',) if d == 1 else ())}
    print(f"{kind} d={d} s={seed} slopes (excess loss vs n): " + '  '.join(f'{k_} {v_:+.2f}' for k_, v_ in out['slopes'].items())
          + ' | h slopes: ' + '  '.join(f'{k_} {v_:+.3f}' for k_, v_ in out['h_slope'].items()), flush=True)
    return out


if __name__ == '__main__':
    res = []
    for d in (1, 2):
        for kind in ('T1', 'T2', 'T3'):
            for seed in (0, 1):
                res.append(run(kind, d, seed))
                json.dump(res, open('results/stream_kde.json', 'w'), indent=1)
