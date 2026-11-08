"""Paper 2: joint density of a charge/discharge price pair (GPU).  ./colab_job.sh start p2pairs p2_pairs.py

Every pair of hours t1 < t2 < t1 + 24 is an observation: charge price p1 = price(t1), discharge
price p2 = price(t2), with charge hour h1 = hour(t1) (periodic), gap D = t2 - t1, and yesterday's
prices at both hours, l1 = price(t1 - 24), l2 = price(t2 - 24). Times are jittered within the hour
(the same jitter as the hourly data), so h1 and D are continuous. Target (p1, p2), at most 6-D:
    p(p1, p2 | c) = p(p1 | h1, D, l1, l2) * p(p2 | p1, h1, D, l1, l2)      ('chain', exact)
Day split with two-day blocks: a test charge day D and the day after are held out, so no test
pair shares a price with training. Validation likewise. Pairs overlap (each hour is in ~46), so
training and validation are random subsets (50k / 10k pairs) and all tests are per charge day.
One JSON per model with the per-pair log densities of the test pairs.
"""
import os, sys, json, time, traceback, subprocess

OUT = os.environ.get('BLML_OUT_DIR', 'outputs_p2pairs')
CODE = os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('BLML_DATA_DIR')
os.makedirs(os.path.join(OUT, 'pairs'), exist_ok=True)
def pip(*pkgs):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *pkgs], check=False)
if os.environ.get('BLML_ENGINE', 'gpu') == 'gpu':      # Colab: GPU and baseline packages
    if subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
        pip('cupy-cuda13x', 'numba')
    for mod, pkg in [('xgboostlss', 'xgboostlss'), ('tabpfn', 'tabpfn')]:
        if subprocess.run([sys.executable, '-c', f'import {mod}'], capture_output=True).returncode:
            pip(pkg)
sys.path.insert(0, CODE)
import numpy as np
import blml_trivial as bt
from blml_trivial import make_modes, modes_fc
import caiso_data
import p2_models as M

bt.DIRICHLET_LEGACY = False
N_TR, N_VA = 50000, 10000
ETA = 0.85          # battery round-trip efficiency: spread s = ETA p2 - p1 (discharge revenue - charge cost)
PT_FRAC, N_DRAWS = float(os.environ.get('BLML_PT_FRAC') or 0), 100   # point metrics (off by default; the paper reports LL):
                                                                       # medians of N_DRAWS joint draws on PT_FRAC of the test charge days

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_pairs(cols, ts):
    t = ts.astype('datetime64[h]').astype(np.int64)
    pos = {v: i for i, v in enumerate(t)}
    h_int = (t - ts.astype('datetime64[D]').astype('datetime64[h]').astype(np.int64)).astype(float)
    u = np.mod(cols['hour'] - h_int, 24.0)                  # within-hour jitter of each row
    u = np.where(u > 1.0, 0.0, u)
    tau = t + u
    price, day = cols['price'], ts.astype('datetime64[D]')
    rows = []
    for i in range(len(t)):
        il1 = pos.get(t[i] - 24)
        if il1 is None:
            continue
        for g in range(1, 24):
            j = pos.get(t[i] + g); jl = pos.get(t[i] + g - 24)
            if j is not None and jl is not None:
                rows.append((i, j, il1, jl))
    I, J, L1, L2 = map(np.array, zip(*rows))
    return dict(p1=price[I], p2=price[J], hour=cols['hour'][I], hour2=cols['hour'][J], gap=tau[J] - tau[I],
                l1=price[L1], l2=price[L2], day1=day[I], day2=day[J], t1=ts[I])


class Chain:
    """p(p1 | c) p(p2 | p1, c) from two one-target conditional density models."""
    def __init__(self, make):
        self.a, self.b = make(), make()
    @staticmethod
    def ca(P, idx):
        return {'hour': P['hour'][idx], 'gap': P['gap'][idx], 'lag24': P['l1'][idx], 'lag2': P['l2'][idx]}
    def cb(self, P, idx):
        return {**self.ca(P, idx), 'p1': P['p1'][idx]}
    def fit(self, P, tr, va):
        self.a.fit(self.ca(P, tr), P['p1'][tr], self.ca(P, va), P['p1'][va])
        self.b.fit(self.cb(P, tr), P['p2'][tr], self.cb(P, va), P['p2'][va])
        return self
    def logpdf(self, P, idx):
        return self.a.logpdf(self.ca(P, idx), P['p1'][idx]), self.b.logpdf(self.cb(P, idx), P['p2'][idx])
    def sample(self, P, idx, n, rng):
        """Joint draws by the chain: p1 ~ p(p1 | c), then one p2 ~ p(p2 | p1, c) per p1 draw."""
        s1 = self.a.sample(self.ca(P, idx), n, rng)                              # (m, n)
        cb = {k: np.repeat(v, n) for k, v in self.ca(P, idx).items()}; cb['p1'] = s1.ravel()
        s2 = self.b.sample(cb, 1, rng)[:, 0].reshape(s1.shape)
        return np.stack([s1, s2], -1)



class PairBLML:
    """One fused BLML fit of (p1, p2, h1, D, l1, l2) with dep=[p1, p2]: the joint of all six dims and
    the marginal of the four conditioning dims in one pass; p(p1, p2 | c) = joint / marginal.
    Cutoffs by the selector on the validation pairs (same ratio objective, two dependent dims)."""
    def _X(self, P, idx):
        return np.column_stack([P['p1'][idx], P['p2'][idx], P['hour'][idx], P['gap'][idx], P['l1'][idx], P['l2'][idx]])
    def fit(self, P, tr, va):
        from selector import select
        from blml_trivial import BLMLQuick
        kinds = ['sinc', 'sinc', 'dirichlet', 'sinc', 'sinc', 'sinc']
        periods = [None, None, 24.0, None, None, None]
        Xa, Xb = self._X(P, tr), self._X(P, va)
        t0 = time.time()
        sel, _ = select(Xa, make_modes(np.nan, kinds, periods, dep=[0, 1]), search='auto', seed=0, engine='gpu', X_val=Xb)
        fc = modes_fc(sel)[0]
        self.fc, self.select_s = [float(v) for v in fc], time.time() - t0
        self.m = BLMLQuick(engine='gpu', seed=0).fit(np.concatenate([Xa, Xb]), make_modes(self.fc, kinds, periods, dep=[0, 1]))
        return self
    def logpdf(self, P, idx):
        lp = np.log(np.maximum(self.m.predict_ratio(self._X(P, idx)), 1e-300))
        return lp, np.zeros_like(lp)          # joint log density of (p1, p2); no chain split
    def sample(self, P, idx, n, rng):
        """Joint draws of (p1, p2) given the covariates (importance resampling, no grid)."""
        y, self.ess_frac = self.m.sample_conditional(self._X(P, idx)[:, 2:], n_samples=n, seed=int(rng.integers(1 << 31)))
        return y


class PairBLMLStream(PairBLML):
    """Streaming BLML of the same 6-D conditional (blml_rolling.BLMLQuickStreaming, dep = (p1, p2)): the training
    and validation pairs arrive in batches of whole charge days, in a random day order (split_groups = (charge
    day, discharge day): batches take whole charge days, and the burn-in selection holds out two-day blocks,
    since pairs of adjacent days share prices); burn-in selection on the first burn_in pairs, then touched-bin
    updates and Adam steps on the cut-offs; later epochs continue from the current fit."""
    def __init__(self, batch_pairs=4000, burn_in=None, epochs=None, seed=0, search=None, on_epoch=None):
        self.batch_pairs, self.seed, self.on_epoch = batch_pairs, seed, on_epoch
        self.burn_in = int(burn_in or os.environ.get('BLML_STREAM_BURN_IN', 10000))
        self.epochs = int(epochs or os.environ.get('BLML_STREAM_EPOCHS', 6))
        self.search = search or os.environ.get('BLML_STREAM_SEARCH', 'auto')
    def fit(self, P, tr, va):
        from blml_rolling import BLMLQuickStreaming
        kinds = ['sinc', 'sinc', 'dirichlet', 'sinc', 'sinc', 'sinc']
        periods = [None, None, 24.0, None, None, None]
        idx = np.concatenate([tr, va]); X = self._X(P, idx)
        dnum = lambda d: d.astype('datetime64[D]').astype(np.int64)      # day labels: consecutive integers
        G = np.c_[dnum(P['day1'][idx]), dnum(P['day2'][idx])]
        t0 = time.time()
        dt = os.environ.get('BLML_STREAM_DTYPE')        # float32 on GPUs with slow float64 (T4); default float64
        fc0 = os.environ.get('BLML_STREAM_FC0')         # JSON (K, 6): each mode's starting fc (rank K = rows; no
        rank = int(os.environ.get('BLML_STREAM_RANK', 1))   # burn-in search); else rank K with the search's fc
        if fc0:
            F0 = np.asarray(json.loads(fc0), float)
            modes = make_modes(F0, kinds, periods, rank=len(F0), dep=[0, 1])
        else:
            modes = make_modes(np.nan, kinds, periods, rank=rank, dep=[0, 1])
        self.m = BLMLQuickStreaming(engine=os.environ.get('BLML_ENGINE', 'gpu'), search=self.search,
                                    burn_in=self.burn_in, seed=self.seed, **({} if dt is None else dict(dtype=np.dtype(dt))))
        self.m.fit(X, modes, epochs=self.epochs, batch=self.batch_pairs, split_groups=G, on_epoch=self.on_epoch)
        self.fc, self.fit_s = [float(v) for v in modes_fc(self.m.modes_)[0]], time.time() - t0
        self.select_s = self.fit_s          # (field name kept for the result files: the whole streaming fit)
        return self


class PairKDE(M.CondKDE):
    """Nonparametric baseline: Nadaraya-Watson conditional KDE of (p1, p2) given (h1, D, l1, l2) --
    a product Gaussian kernel in (p1, p2) weighted by kernels on the covariates (von Mises on h1,
    Gaussian on the standardized others); all six bandwidths by coordinate ascent on validation LL.
    Queries are processed in chunks (GPU memory)."""
    CH = 1024

    def _lp2(self, cov, Y, kappa, h):
        m = len(Y); out = []
        for i in range(0, m, self.CH):
            c = {k: np.asarray(v)[i:i + self.CH] for k, v in cov.items()}
            W = self._w(c, kappa)
            z1 = (M.xp.asarray(Y[i:i + self.CH, 0])[:, None] - self.ytr[None, :, 0]) / h[0]
            z2 = (M.xp.asarray(Y[i:i + self.CH, 1])[:, None] - self.ytr[None, :, 1]) / h[1]
            num = (W * M.xp.exp(-0.5 * (z1 * z1 + z2 * z2))).sum(1) / (2 * np.pi * h[0] * h[1])
            out.append(M.to_np(M.xp.log(M.xp.maximum(num / W.sum(1), 1e-300))))
        return np.concatenate(out)

    def fit(self, P, tr, va):
        ca = Chain.ca
        cov_tr, cov_va = ca(P, tr), ca(P, va)
        Ytr = np.column_stack([P['p1'][tr], P['p2'][tr]]); Yva = np.column_stack([P['p1'][va], P['p2'][va]])
        self.covs = list(cov_tr)
        self.ctr = {c: M.xp.asarray(cov_tr[c]) for c in self.covs}
        self.ytr = M.xp.asarray(Ytr)
        kgrid = np.geomspace(0.3, 3000, 18)
        hgrid = [np.std(Ytr[:, j]) * len(Ytr) ** (-1 / 6) * np.geomspace(0.03, 2, 14) for j in range(2)]
        kappa, h = [10.0] * len(self.covs), [hgrid[0][6], hgrid[1][6]]
        score = lambda kap, hh: float(self._lp2(cov_va, Yva, kap, hh).mean())
        best = score(kappa, h)
        t0 = time.time()
        for _ in range(3):
            for i in range(len(self.covs)):
                for k in kgrid:
                    kap = list(kappa); kap[i] = k
                    sc = score(kap, h)
                    if sc > best:
                        best, kappa = sc, kap
            for j in range(2):
                for hv in hgrid[j]:
                    hh = list(h); hh[j] = hv
                    sc = score(kappa, hh)
                    if sc > best:
                        best, h = sc, hh
        self.kappa, self.h, self.select_s = kappa, [float(v) for v in h], time.time() - t0
        self.fc = {'kappa': [float(k) for k in kappa], 'h': self.h}
        both = np.concatenate([tr, va])                   # refit on train + validation
        self.ctr = {c: M.xp.asarray(ca(P, both)[c]) for c in self.covs}
        self.ytr = M.xp.asarray(np.column_stack([P['p1'][both], P['p2'][both]]))
        return self

    def logpdf(self, P, idx):
        lp = self._lp2(Chain.ca(P, idx), np.column_stack([P['p1'][idx], P['p2'][idx]]), self.kappa, self.h)
        return lp, np.zeros_like(lp)


class PairKNN(M.CondKNN):
    """Nonparametric baseline: k nearest neighbors in covariate space (h1 circular, others
    standardized), then a product Gaussian KDE of their (p1, p2); k and the bandwidth scale tuned on
    validation. Neighbors are found once (sorted, up to the largest k) in query chunks."""
    KS, CS, CH = [10, 25, 50, 100, 200, 400], [0.25, 0.5, 1.0, 1.5], 512

    def _sorted_nbrs(self, cov, kmax):
        out = []
        m = len(next(iter(cov.values())))
        for i in range(0, m, self.CH):
            c = {k: np.asarray(v)[i:i + self.CH] for k, v in cov.items()}
            d2 = 0
            for name in self.covs:
                if name not in M.PERIODS:
                    dd = np.abs(c[name][:, None] - self.ctr[name][None, :]) / (4 * M.SCALES[name][1])
                else:
                    dd = np.abs(c[name][:, None] - self.ctr[name][None, :]) / M.PERIODS[name]
                    dd = np.minimum(dd, 1.0 - dd)
                d2 = d2 + dd * dd
            idx = np.argpartition(d2, kmax - 1, axis=1)[:, :kmax]
            o = np.argsort(np.take_along_axis(d2, idx, 1), axis=1)
            out.append(np.take_along_axis(idx, o, 1))
        return np.concatenate(out)

    def _lp(self, nb, Y, k, c):
        N = self.ytr[nb[:, :k]]                                        # (m, k, 2)
        h = c * N.std(1) * k ** (-1 / 6) + 1e-6                        # (m, 2)
        z = (Y[:, None, :] - N) / h[:, None, :]
        dens = np.exp(-0.5 * (z * z).sum(-1)).mean(1) / (2 * np.pi * h[:, 0] * h[:, 1])
        return np.log(np.maximum(dens, 1e-300))

    def fit(self, P, tr, va):
        cov_tr, cov_va = Chain.ca(P, tr), Chain.ca(P, va)
        self.covs = list(cov_tr)
        self.ctr = {c: np.asarray(cov_tr[c]) for c in self.covs}
        self.ytr = np.column_stack([P['p1'][tr], P['p2'][tr]])
        Yva = np.column_stack([P['p1'][va], P['p2'][va]])
        t0 = time.time()
        nb = self._sorted_nbrs(cov_va, max(self.KS))
        best = (-np.inf, None)
        for k in self.KS:
            for c in self.CS:
                sc = float(self._lp(nb, Yva, k, c).mean())
                if sc > best[0]:
                    best = (sc, (k, c))
        self.k, self.c = best[1]
        self.select_s, self.fc = time.time() - t0, {'k': int(self.k), 'c': float(self.c)}
        both = np.concatenate([tr, va])                                # refit on train + validation
        self.ctr = {c: np.asarray(Chain.ca(P, both)[c]) for c in self.covs}
        self.ytr = np.column_stack([P['p1'][both], P['p2'][both]])
        return self

    def logpdf(self, P, idx):
        nb = self._sorted_nbrs(Chain.ca(P, idx), self.k)
        lp = self._lp(nb, np.column_stack([P['p1'][idx], P['p2'][idx]]), self.k, self.c)
        return lp, np.zeros_like(lp)


class PairFlow:
    """Neural spline flow (zuko NSF) for the joint conditional density p(p1, p2 | hour, gap, lag24, lag2) in one
    model: features = (p1, p2), context = (sin, cos of the hour, gap, lag24, lag2), all standardised on the training
    pairs; 5 spline transforms, two hidden layers of 128, Adam, early stopping on the validation pairs."""
    def _ctx(self, P, idx):
        h = 2 * np.pi * P['hour'][idx] / 24.0
        return np.column_stack([np.sin(h), np.cos(h), P['gap'][idx], P['l1'][idx], P['l2'][idx]])
    def _y(self, P, idx):
        return np.column_stack([P['p1'][idx], P['p2'][idx]])
    def fit(self, P, tr, va):
        import torch, zuko
        torch.manual_seed(0); self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        C, Y = self._ctx(P, tr), self._y(P, tr)
        self.cm, self.cs, self.ym, self.ys = C.mean(0), C.std(0) + 1e-9, Y.mean(0), Y.std(0) + 1e-9
        T = lambda a: torch.tensor(a, dtype=torch.float32, device=self.dev)
        Ct, Yt = T((C - self.cm) / self.cs), T((Y - self.ym) / self.ys)
        Cv, Yv = T((self._ctx(P, va) - self.cm) / self.cs), T((self._y(P, va) - self.ym) / self.ys)
        self.flow = zuko.flows.NSF(features=2, context=5, transforms=5, hidden_features=[128, 128]).to(self.dev)
        opt = torch.optim.Adam(self.flow.parameters(), lr=1e-3); best, state, bad = np.inf, None, 0
        for epoch in range(300):
            perm = torch.randperm(len(Yt), device=self.dev)
            for i in range(0, len(Yt), 512):
                b = perm[i:i + 512]; loss = -self.flow(Ct[b]).log_prob(Yt[b]).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                v = -self.flow(Cv).log_prob(Yv).mean().item()
            if v < best - 1e-4:
                best, bad = v, 0; state = {k: x.detach().clone() for k, x in self.flow.state_dict().items()}
            else:
                bad += 1
                if bad >= 20:
                    break
        self.flow.load_state_dict(state); self.epochs = epoch + 1
        return self
    def logpdf(self, P, idx):
        import torch
        with torch.no_grad():
            C = torch.tensor((self._ctx(P, idx) - self.cm) / self.cs, dtype=torch.float32, device=self.dev)
            Y = torch.tensor((self._y(P, idx) - self.ym) / self.ys, dtype=torch.float32, device=self.dev)
            lp = np.concatenate([self.flow(C[i:i + 20000]).log_prob(Y[i:i + 20000]).cpu().numpy() for i in range(0, len(Y), 20000)])
        lp = lp - np.log(self.ys).sum()
        return lp, np.zeros_like(lp)
    def sample(self, P, idx, n, rng):
        import torch
        torch.manual_seed(int(rng.integers(2 ** 31)))
        with torch.no_grad():
            C = torch.tensor((self._ctx(P, idx) - self.cm) / self.cs, dtype=torch.float32, device=self.dev)
            Z = self.flow(C).sample((n,)).cpu().numpy()                      # (n, m, 2)
        return np.transpose(Z, (1, 0, 2)) * self.ys + self.ym


MODELS = {
    'BLML-joint': lambda: PairBLML(), 'BLML-joint-stream': lambda: PairBLMLStream(),
    'KDE-joint': lambda: PairKDE(), 'KNN-joint': lambda: PairKNN(),
    'TabPFN-chain': lambda: Chain(M.TabPFN), 'MDN-chain': lambda: Chain(M.MDN), 'GMM-chain': lambda: Chain(M.CondGMM),
    'XGBoostLSS-chain': lambda: Chain(M.XGBoostLSS), 'NSF-joint': lambda: PairFlow(),
}

def free_gpu():
    import gc; gc.collect()
    try:
        import cupy as cp; cp.get_default_memory_pool().free_all_blocks(); cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass


def split(P):
    """The day split with two-day blocks (module docstring): (tr, va, te, te_days, pt, pt_days), fixed seed."""
    days = np.unique(P['day1'])
    rng = np.random.default_rng(0)
    one = np.timedelta64(1, 'D')
    te_days = rng.choice(days[:-1], int(0.2 * len(days)), replace=False)
    held = np.union1d(te_days, te_days + one)
    rest = np.setdiff1d(days[:-1], held)
    va_days = rng.choice(rest, int(0.2 * len(rest)), replace=False)
    va_held = np.union1d(va_days, va_days + one)
    off = lambda d: ~np.isin(d, held)
    te = np.where(np.isin(P['day1'], te_days))[0]
    va = np.where(np.isin(P['day1'], va_days) & off(P['day2']))[0]
    tr = np.where(off(P['day1']) & off(P['day2']) & ~np.isin(P['day1'], va_held) & ~np.isin(P['day2'], va_held))[0]
    tr = rng.choice(tr, min(N_TR, len(tr)), replace=False); va = rng.choice(va, min(N_VA, len(va)), replace=False)
    pt_days = rng.choice(te_days, int(PT_FRAC * len(te_days)), replace=False)
    pt = te[np.isin(P['day1'][te], pt_days)]
    return tr, va, te, te_days, pt, pt_days


def load_pairs():
    """The pairs and the scales the models use (as main)."""
    cols = caiso_data.load(DATA); ts = caiso_data.timestamps(DATA)
    P = build_pairs(cols, ts)
    for k, src in [('lag24', 'l1'), ('lag2', 'l2'), ('p1', 'p1'), ('gap', 'gap')]:
        M.SCALES[k] = (float(np.median(P[src])), float(np.std(P[src])))
    return P


def main():
    P = load_pairs()
    tr, va, te, te_days, pt, pt_days = split(P)
    if len(pt):
        lag = np.column_stack([P['l1'][pt], P['l2'][pt]]); tru = np.column_stack([P['p1'][pt], P['p2'][pt]])
        l1l, sl = np.abs(lag - tru).sum(1), np.abs((ETA * lag[:, 1] - lag[:, 0]) - (ETA * tru[:, 1] - tru[:, 0]))
        ref = dict(n_point=len(pt), l1_mean=float(l1l.mean()), l1_median=float(np.median(l1l)),
                   spread_mae=float(sl.mean()), spread_medae=float(np.median(sl)))
        json.dump(ref, open(os.path.join(OUT, 'pairs', 'lag24_reference.json'), 'w'))
    log(f'{len(P["p1"])} pairs; train {len(tr)}, validation {len(va)}, test {len(te)} ({len(te_days)} charge days)')
    if len(pt):
        log(f'point metrics on {len(pt)} pairs ({len(pt_days)} charge days); lag24 reference {ref}')
    only = os.environ.get('BLML_P2_MODELS')
    for name in (only.split(',') if only else MODELS):
        path = os.path.join(OUT, 'pairs', name + '.json')
        if os.path.exists(path):
            continue
        t0 = time.time()
        try:
            m = MODELS[name]().fit(P, tr, va)
            l1, l2 = m.logpdf(P, te)
            rec = dict(model=name, n_train=len(tr), n_val=len(va), n_test=len(te), fit_predict_s=time.time() - t0,
                       ll=float(np.mean(l1 + l2)), ll_p1=float(np.mean(l1)), ll_p2=float(np.mean(l2)),
                       per_pair=dict(lp1=np.round(l1, 4).tolist(), lp2=np.round(l2, 4).tolist()),
                       day1=[str(d) for d in P['day1'][te]], hour1=np.floor(P['hour'][te]).astype(int).tolist(),
                       gap=np.round(P['gap'][te]).astype(int).tolist(),
                       p1=np.round(P['p1'][te], 3).tolist(), p2=np.round(P['p2'][te], 3).tolist())
            if len(pt):    # point metrics (medians: optimal for absolute error) on a subset of the test charge days
                t1 = time.time()
                Y = m.sample(P, pt, N_DRAWS, np.random.default_rng(1))                  # (m, N_DRAWS, 2)
                med = np.median(Y, 1); smed = np.median(ETA * Y[:, :, 1] - Y[:, :, 0], 1)
                truth = np.column_stack([P['p1'][pt], P['p2'][pt]]); s_true = ETA * truth[:, 1] - truth[:, 0]
                l1 = np.abs(med - truth).sum(1); se = np.abs(smed - s_true)
                rec.update(point_s=time.time() - t1, n_point=len(pt), l1_mean=float(l1.mean()), l1_median=float(np.median(l1)),
                           spread_mae=float(se.mean()), spread_medae=float(np.median(se)),
                           point=dict(p1=np.round(med[:, 0], 3).tolist(), p2=np.round(med[:, 1], 3).tolist(), s=np.round(smed, 3).tolist()),
                           point_idx=[int(i) for i in np.searchsorted(te, pt)])
                if hasattr(m, 'ess_frac'):
                    rec['ess_frac'] = float(m.ess_frac)
                log(f"{name}: L1 mean {rec['l1_mean']:.2f} median {rec['l1_median']:.2f}; spread MAE {rec['spread_mae']:.2f} "
                    f"median {rec['spread_medae']:.2f} ({rec['point_s']:.0f}s)")
            for part, mm in ((('a', m.a), ('b', m.b)) if hasattr(m, 'a') else (('joint', m),)):
                for attr in ('fc', 'select_s', 'fit_s'):
                    if hasattr(mm, attr):
                        rec[f'{attr}_{part}'] = getattr(mm, attr)
            log(f"{name}: LL={rec['ll']:.4f} (p1 {rec['ll_p1']:.4f}, p2|p1 {rec['ll_p2']:.4f}) ({rec['fit_predict_s']:.0f}s)")
        except Exception as e:
            rec = dict(model=name, error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc()[-3000:])
            log(f'{name}: FAILED {rec["error"]}')
        m = None; free_gpu()                       # release cached GPU memory before the next model
        with open(path + '.tmp', 'w') as f:
            json.dump(rec, f, default=float)
        os.replace(path + '.tmp', path)
    log('ALL DONE')

if __name__ == '__main__':
    main()
