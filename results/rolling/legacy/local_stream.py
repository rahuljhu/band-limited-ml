"""Prototype (paper 5): streaming BLML with touched-bin updates. Sinc kernels, joint density, real rank 1, CPU.
Cost, worst case (every point in its own bin, B = n): one epoch O(d n^2) time -- about two sweeps of the
one-shot solver -- and O(n) memory (coefficients, counts, lattice; no kernel block is ever stored).
Per batch of b points (I: the bins they fall in, |I| <= b), each term O(d b n_t):
  1. score: g and its d log-fc derivatives at the points, one pass over the bins (prefix/suffix products:
     O(d) per point-bin pair for the value and all d derivatives);
  2. rows (S e)_I and (dS/dlog fc_k e)_I, one pass over the bins, same trick;
  3. one block coordinate-descent update of e_I (rows reused from 2; within-batch coupling O(d b^2));
  4. normaliser Q = e'Se tracked exactly through the updates, e <- e n / sqrt(Q);
  5. every fc_every batches an Adam step on log fc; Q(fc_new)/Q(fc) from two more row passes over I.
The fixed-c normaliser derivative e'dSe and Q(fc_new)/Q(fc) are mass-weighted estimates on the batches' bins
(weight batch count / total count: bins are reached in proportion to their mass); the one-sweep response
of c is taken on the touched bins. No pass over the whole lattice after the burn-in.
Epochs continue: the same coefficients and cut-offs, counts not added again; a revisited batch is scored
leave-batch-out (its share bc_i / cnt_i of each touched coefficient removed: e_i is ~ proportional to the
bin's count), so the fc gradient stays out-of-sample. The lattice is the burn-in's (no re-binning).
Usage: local_stream.py   (2-D trajectory test; writes results/synstream/local_trace.json)"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')
import numpy as np
from numba import njit, prange
import blml_trivial as bt
from blml_trivial import make_modes, BLMLQuick, SincKernel

bt.DIRICHLET_LEGACY = False
SK = SincKernel()
STEPS = 0.5 ** np.arange(12)


@njit(parallel=True, cache=True)
def _lat_rows(ZA, ZB, w, T, dT, J, grad):
    """s_r = sum_j S(ZA_r, ZB_j) w_j on the lattice (tables T, offsets -J..J), and with grad its
    d/dlog fc_k (rows dT); O(d) per pair for the value and all d derivatives; memory O(R d)."""
    R, d = ZA.shape; M = ZB.shape[0]
    s = np.zeros(R); ds = np.zeros((R, d))
    for r in prange(R):
        kv = np.empty(d); pre = np.empty(d + 1); suf = np.empty(d + 1); acc = 0.0; dacc = np.zeros(d)
        for j in range(M):
            wj = w[j]
            if wj == 0.0:
                continue
            pre[0] = 1.0
            for k in range(d):
                kv[k] = T[k, ZA[r, k] - ZB[j, k] + J]; pre[k + 1] = pre[k] * kv[k]
            acc += pre[d] * wj
            if grad:
                suf[d] = 1.0
                for k in range(d - 1, -1, -1):
                    suf[k] = suf[k + 1] * kv[k]
                for k in range(d):
                    dacc[k] += dT[k, ZA[r, k] - ZB[j, k] + J] * pre[k] * suf[k + 1] * wj
        s[r] = acc
        for k in range(d):
            ds[r, k] = dacc[k]
    return s, ds


@njit(parallel=True, cache=True)
def _cont_rows(A, Bp, w, fc, grad):
    """s_r = sum_j K(A_r - Bp_j) w_j with the continuous sinc product kernel K = prod_k fc_k sinc(fc_k t_k),
    and with grad d/dlog fc_k (factor fc_k cos(pi fc_k t_k)); O(d) per pair; memory O(R d)."""
    R, d = A.shape; M = Bp.shape[0]
    s = np.zeros(R); ds = np.zeros((R, d))
    for r in prange(R):
        kv = np.empty(d); dk = np.empty(d); pre = np.empty(d + 1); suf = np.empty(d + 1)
        acc = 0.0; dacc = np.zeros(d)
        for j in range(M):
            wj = w[j]
            if wj == 0.0:
                continue
            pre[0] = 1.0
            for k in range(d):
                x = A[r, k] - Bp[j, k]; t = np.pi * fc[k] * x
                kv[k] = fc[k] if abs(t) < 1e-12 else np.sin(t) / (np.pi * x)
                if grad:
                    dk[k] = fc[k] * np.cos(t)
                pre[k + 1] = pre[k] * kv[k]
            acc += pre[d] * wj
            if grad:
                suf[d] = 1.0
                for k in range(d - 1, -1, -1):
                    suf[k] = suf[k + 1] * kv[k]
                for k in range(d):
                    dacc[k] += dk[k] * pre[k] * suf[k + 1] * wj
        s[r] = acc
        for k in range(d):
            ds[r, k] = dacc[k]
    return s, ds


class LocalStream:
    def __init__(self, fc0, burn_in=1000, lr=0.1, betas=(0.9, 0.999), fc_every=3, eps_rel=1e-3, train=True, q_rate=0.1):
        self.fc = np.asarray(fc0, float).copy(); self.burn_in, self.lr, self.betas = burn_in, lr, betas
        self.fc_every, self.eps_rel, self.train, self.q_rate = fc_every, eps_rel, train, q_rate
        self.n = 0.0; self.history_, self.preq_ = [], []
        self._adam, self._nb, self._acc = [np.zeros_like(self.fc), np.zeros_like(self.fc), 0], 0, []

    # ---------------------------------------------------------------- lattice
    def _tables(self, fc):
        T = np.stack([SK.lattice(dict(fc=[fc[k]]), self.h[[k]], self.J) for k in range(self.d)])
        dT = np.stack([SK.lattice_dlog(dict(fc=[fc[k]]), 'fc', self.h[[k]], self.J) for k in range(self.d)])
        return T, dT

    def _rows(self, I, w, fc, grad=False, ZB=None):
        T, dT = self._tables(fc)
        return _lat_rows(np.ascontiguousarray(self.Zi[I]), self.Zi if ZB is None else ZB, w, T, dT, self.J, grad)

    def _bins(self, X, add):
        """Bin index of each point; add=True appends unseen bins (e = 0) -- epoch 1 only."""
        Zi = np.round(self.qk.quantize(X) / self.h).astype(np.int64)
        idx = np.empty(len(Zi), np.int64); new = []
        for r, z in enumerate(map(tuple, Zi)):
            j = self.key.get(z)
            if j is None:
                j = self.key[z] = len(self.key) + 0; new.append(z)
            idx[r] = j
        if new:
            self.Zi = np.vstack([self.Zi, np.array(new, np.int64)])
            self.e = np.r_[self.e, np.zeros(len(new))]; self.cnt = np.r_[self.cnt, np.zeros(len(new))]
            span = int(np.abs(self.Zi.max(0) - self.Zi.min(0)).max())
            self.J = max(self.J, span)
        return idx

    # ---------------------------------------------------------------- burn-in
    def _burn(self, Xb):
        m = BLMLQuick(dtype=np.float64).fit(Xb, make_modes(self.fc, ['sinc'] * Xb.shape[1]))
        self.d, self.h, self.n, self.qk = Xb.shape[1], np.asarray(m._h_vec, float), float(len(Xb)), m._fit_kernel
        self.Zi = np.ascontiguousarray(np.asarray(m._Zi, np.int64)); self.e = np.asarray(m.e_hat_, float).copy()
        self.cnt = np.asarray(m.w_, float).copy()
        self.key = {z: i for i, z in enumerate(map(tuple, self.Zi))}
        self.J = max(int(np.abs(self.Zi.max(0) - self.Zi.min(0)).max()), 1)
        self.Q = self.n ** 2

    # ---------------------------------------------------------------- one batch
    def _batch(self, X, revisit):
        t0 = time.time(); d, fc, b = self.d, self.fc, len(X)
        idx = self._bins(X, add=not revisit)
        I, bc = np.unique(idx, return_counts=True)
        Zc = self.Zi * self.h
        # 1. score at the points (leave-batch-out on a revisit)
        g, dg = _cont_rows(X, Zc, self.e / self.n, fc, self.train)
        if revisit:
            share = bc / self.cnt[I] * self.e[I] / self.n
            g0, dg0 = _cont_rows(X, np.ascontiguousarray(Zc[I]), share, fc, self.train)
            g, dg = g - g0, dg - dg0
        f = g ** 2
        self.preq_.append(float(np.mean(np.log(np.maximum(f, 1e-300)))))
        # 2. rows of the touched bins (reused by the coordinate update)
        SeI, dSeI = self._rows(I, self.e, fc, grad=self.train)
        # normaliser drift correction: the batch reaches bin i with probability ~ cnt_i / n, so
        # Qhat = (n/b) sum_I bc_i e_i (Se)_i / cnt_i estimates e'Se without bias; pull log Q toward it
        old = self.cnt[I] > 0
        if old.any():
            Qhat = self.n / b * float((bc[old] * self.e[I][old] * SeI[old] / self.cnt[I][old]).sum()) * b / bc[old].sum()
            self.Q *= np.exp(self.q_rate * (np.log(max(Qhat, 1e-300)) - np.log(self.Q)))
            self._renorm(); SeI, dSeI = SeI * self._last_sc, dSeI * self._last_sc
        if self.train:
            eps = self.eps_rel * float(np.median(f))
            cont = (2 * g[:, None] * dg / (f + eps)[:, None]).sum(0)
            a = 2 * g / (self.n * (f + eps))
            vI = _cont_rows(np.ascontiguousarray(Zc[I]), X, a, fc, False)[0]    # d score / d e_I
            cntI = self.cnt[I] + (0 if revisit else bc)
            wI = bc / cntI
            eI, sbb = self.e[I], float(np.prod(fc))
            R = SeI - sbb * eI; den = 2 * sbb * eI + R; den = np.where(np.abs(den) > 1e-300, den, 1e-300)
            dTr = -(eI[:, None] * (dSeI - sbb * eI[:, None]) + (eI ** 2 * sbb)[:, None]) / den[:, None]
            dTr *= (eI > 0)[:, None]
            resp = dTr.T @ (vI - b * 2 * SeI / self.Q)
            self._acc.append(dict(cont=cont, resp=resp, W=b, dQ=(wI * eI) @ dSeI, Qs=(wI * eI) @ SeI))
        # 3. counts (epoch 1), coordinate update of e_I
        if not revisit:
            self.n += b; np.add.at(self.cnt, idx, 1.0)
        self._cd(I, SeI)
        self._nb += 1
        # 5. Adam step on log fc
        if self.train and self._nb % self.fc_every == 0:
            A = self._acc; W = sum(x['W'] for x in A)
            grad = (sum(x['cont'] for x in A) + sum(x['resp'] for x in A)) / W - sum(x['dQ'] for x in A) / sum(x['Qs'] for x in A)
            st = self._adam; b1, b2 = self.betas
            st[2] += 1; st[0] = b1 * st[0] + (1 - b1) * grad; st[1] = b2 * st[1] + (1 - b2) * grad ** 2
            new = fc * np.exp(self.lr * (st[0] / (1 - b1 ** st[2])) / (np.sqrt(st[1] / (1 - b2 ** st[2])) + 1e-12))
            wI = bc / self.cnt[I]
            on = self._rows(I, self.e, new)[0]; oo = self._rows(I, self.e, fc)[0]
            self.Q *= ((wI * self.e[I]) @ on) / ((wI * self.e[I]) @ oo)
            self.fc = new; self._acc = []
            self._renorm()
        self.history_.append(dict(n=self.n, fc=self.fc.tolist(), preq=self.preq_[-1], s=time.time() - t0))

    def _cd(self, I, SeI):
        """One parallel coordinate update per block of <= 512 touched bins; later blocks' rows corrected."""
        sbb = float(np.prod(self.fc)); SeI = SeI.copy()
        for s0 in range(0, len(I), 512):
            Jb = I[s0:s0 + 512]; SeJ = SeI[s0:s0 + 512]
            b_ = SeJ - sbb * self.e[Jb]; cc = self.n * self.cnt[Jb]
            disc = np.sqrt(b_ * b_ + 4 * sbb * cc)
            es = np.where(b_ >= 0, 2 * cc / (b_ + disc), (-b_ + disc) / (2 * sbb))
            dl = es - self.e[Jb]
            ZJ = np.ascontiguousarray(self.Zi[Jb])
            SJd = self._rows(Jb, dl, self.fc, ZB=ZJ)[0]                         # S_JJ dl, O(d |J|^2)
            # block safeguard (as the solver's): largest t in 1, 1/2, ... with no increase of
            # Phi = e'Se/2 - sum cc log e along e_J + t dl (coupled bins of one block can overshoot)
            lin, quad = dl @ SeJ, dl @ SJd
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.where(self.e[Jb] > 0, dl / self.e[Jb], np.inf)
            t = 0.0
            for tt in STEPS:
                logs = cc @ np.log1p(np.minimum(tt * r, 1e300))
                if tt * lin + 0.5 * tt * tt * quad - logs <= 0:
                    t = tt; break
            dl = t * dl
            self.Q += 2 * dl @ SeJ + dl @ (t * SJd)
            self.e[Jb] = self.e[Jb] + dl
            if s0 + 512 < len(I):
                rest = I[s0 + 512:]
                SeI[s0 + 512:] += self._rows(rest, dl, self.fc, ZB=ZJ)[0]
        self._renorm()

    def _renorm(self):
        sc = self.n / np.sqrt(max(self.Q, 1e-300)); self.e *= sc; self.Q = self.n ** 2; self._last_sc = sc

    # ---------------------------------------------------------------- driver
    def fit(self, X, epochs=2, batch=500, seed=0):
        X = np.asarray(X, float); N = len(X)
        for ep in range(epochs):
            order = np.random.default_rng([seed, ep]).permutation(N)
            if ep == 0:
                self._burn(X[order[:self.burn_in]]); start = self.burn_in
                self.history_.append(dict(n=self.n, fc=self.fc.tolist(), burn_in=True))
            else:
                start = 0
            for i in range(start, N, batch):
                self._batch(np.ascontiguousarray(X[order[i:i + batch]]), revisit=ep > 0)
            self.history_[-1]['epoch_end'] = ep
        return self

    def predict(self, X):
        return _cont_rows(np.ascontiguousarray(np.asarray(X, float)), self.Zi * self.h, self.e / self.n, self.fc, False)[0] ** 2


def rel_ise(model, ev):
    return float(np.mean((model.predict(ev['Xq']) - ev['fq']) ** 2 / ev['qq'])) / ev['int_f2']


if __name__ == '__main__':
    from p1_truth import Truth
    from synth_stream import eval_sets
    N, D, BS = 10000, 2, 500
    out = {}
    for kind in ('T1', 'T2'):
        truth = Truth(kind, D); ev = eval_sets(truth, D)
        for rep in range(2):
            X = truth.sample(np.random.default_rng([3, D, N, rep]), N)
            for f0 in (0.2, 0.8):
                t0 = time.time(); s = LocalStream(np.full(D, f0), burn_in=1000).fit(X, epochs=5, batch=BS, seed=rep)
                t = time.time() - t0; r = rel_ise(s, ev)
                out[f'{kind}/start{f0}/{rep}'] = dict(trace=s.history_, rel_ise=r, s=t, fc=s.fc.tolist())
                print(f'{kind} rep {rep} start {f0}: fc {np.round(s.fc, 3)} rel ISE {r:.2e} ({t:.1f}s, 5 epochs)', flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'synstream', 'local_trace.json'), 'w'))
