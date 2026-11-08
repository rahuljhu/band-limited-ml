"""Standard density-estimation benchmarks POWER (6-d) and GAS (8-d), with the preprocessing and splits of
Papamakarios et al. (2017, MAF), so test log-likelihoods are comparable with the normalizing-flow literature (nats per
point on the standardised data). Data: the MAF archive on Zenodo (record 1161203, data.tar.gz), downloaded on the VM.
Methods, every one with its smoothing chosen on the validation split:
  blml      cut-off by the selector's auto search on a subsample of SUB (2e4) training points (validation: SUBV points of the
            validation split), then the one-shot fit on all training points at that cut-off and at the cut-off
            scaled by (n / SUB)^(1/(d+4)) (the selected band grows with n for densities outside the class)
  blml_w    the same after whitening (PCA rotation and scaling of the training data; log |det W| added back); the
            full-size fits run only if its subsample validation score beats blml's
  blml_stream  the cut-off learned while streaming: burn-in search on n/5 (at most 2e4) points, batches of n/50 in [2000, 10000],
            up to 12 passes (stops when the coefficients meet the one-shot residual test and the cut-offs are stationary;
            held-out score logged every pass)
  kde       Gaussian product KDE on all training points, bandwidth multiplier of Scott's rule chosen on the validation
            split over a grid
Run: ./colab_job.sh start uci uci_bench_job.py T4   (one JSON per dataset and method in OUT/uci; resumable)
UCI_NTRAIN=n: smoke test on n training points."""
import os, sys, json, time, subprocess, tarfile, pickle
OUT = os.environ.get('BLML_OUT_DIR', 'outputs_uci'); CODE = os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(OUT, 'uci'), exist_ok=True)
ENGINE = os.environ.get('UCI_ENGINE', 'gpu')                # 'cpu': a laptop run (float64, numba kernels)
if ENGINE == 'gpu' and subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda13x', 'numba'], check=True)
sys.path.insert(0, CODE)
import numpy as np
cp = __import__('cupy') if ENGINE == 'gpu' else None
DT = np.float32 if ENGINE == 'gpu' else np.float64
import pandas as pd
from blml_trivial import make_modes, modes_fc, BLMLQuick
from blml_rolling import BLMLQuickStreaming
from selector import select
import p1_estimators as E
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
SUB, SUBV = int(os.environ.get('UCI_SUB', '20000')), int(os.environ.get('UCI_SUBV', '20000'))
EPOCHS = int(os.environ.get('UCI_EPOCHS', '12'))          # pass cap; stops earlier at the one-shot residual test (1e-4)
NTRAIN = int(os.environ.get('UCI_NTRAIN', '0'))
STREAM_FC = json.loads(os.environ['UCI_STREAM_FC']) if os.environ.get('UCI_STREAM_FC') else None   # start (no burn-in search)
METHODS = ('kde', 'blml', 'blml_w', 'blml_stream')
SKIP = set(filter(None, os.environ.get('UCI_SKIP', '').split(',')))    # dataset_method runs to leave out           # 0: the full training split
ROOT = os.environ.get('UCI_ROOT', '/content/maf_data')


def fetch():
    if 'UCI_ROOT' not in os.environ and not os.path.exists(ROOT + '/power/data.npy'):
        os.makedirs(ROOT, exist_ok=True)
        subprocess.run(['wget', '-q', '-O', '/content/data.tar.gz',
                        'https://zenodo.org/record/1161203/files/data.tar.gz'], check=True)
        with tarfile.open('/content/data.tar.gz') as t:
            t.extractall('/content/maf_tmp')
        top = os.path.dirname(os.path.dirname(subprocess.run(['bash', '-c', 'find /content/maf_tmp -path "*power/data.npy"'],
                                                             capture_output=True, text=True).stdout.split()[0]))
        subprocess.run(['bash', '-c', f'cp -r {top}/* {ROOT}/'], check=True)


def power():
    """MAF preprocessing: shuffle (seed 42), drop columns 1 and 3, add the published noise, 10% test, 10% of the rest
    validation, standardise on train + validation."""
    rng = np.random.RandomState(42); data = np.load(ROOT + '/power/data.npy'); rng.shuffle(data); N = data.shape[0]
    data = np.delete(data, 3, axis=1); data = np.delete(data, 1, axis=1)
    noise = np.hstack((0.001 * rng.rand(N, 1), 0.01 * rng.rand(N, 1), rng.rand(N, 3), np.zeros((N, 1))))
    data = data + noise
    nt = int(0.1 * N); te = data[-nt:]; data = data[:-nt]; nv = int(0.1 * data.shape[0]); va = data[-nv:]; tr = data[:-nv]
    tv = np.vstack((tr, va)); mu, s = tv.mean(0), tv.std(0)
    return (tr - mu) / s, (va - mu) / s, (te - mu) / s


def read_old_pickle(path):
    """The MAF archive's GAS frame was pickled by pandas 0.2x: its index classes lived in pandas.core.index (and
    Int64Index / Float64Index, removed in pandas 2); map them to their current homes."""
    from pandas.compat.pickle_compat import Unpickler
    moved = {'pandas.core.index': 'pandas.core.indexes.base', 'pandas.indexes.base': 'pandas.core.indexes.base',
             'pandas.indexes.range': 'pandas.core.indexes.range', 'pandas.core.indexes.numeric': 'pandas.core.indexes.base'}

    class Old(Unpickler):
        def find_class(self, module, name):
            if name in ('Int64Index', 'Float64Index', 'UInt64Index', 'NumericIndex'):
                module, name = 'pandas.core.indexes.base', 'Index'
            return super().find_class(moved.get(module, module), name)
    with open(path, 'rb') as fh:
        return pd.DataFrame(Old(fh, encoding='latin1').load())


def gas():
    """MAF preprocessing: drop Meth, Eth, Time; drop columns with correlation > 0.98 to another; standardise; 10% test,
    10% of the rest validation (no shuffle)."""
    data = read_old_pickle(ROOT + '/gas/ethylene_CO.pickle').drop(['Meth', 'Eth', 'Time'], axis=1)
    def corr_counts(d):
        return (d.corr() > 0.98).to_numpy().sum(1)
    B = corr_counts(data)
    while np.any(B > 1):
        data = data.drop(data.columns[np.where(B > 1)[0][0]], axis=1); B = corr_counts(data)
    data = ((data - data.mean()) / data.std()).to_numpy()
    nt = int(0.1 * data.shape[0]); te = data[-nt:]; tr = data[:-nt]; nv = int(0.1 * tr.shape[0]); va = tr[-nv:]; tr = tr[:-nv]
    return tr, va, te


def hepmass():
    """MAF preprocessing, quirks included so the numbers compare: the signal class (label 1) only, label dropped, the
    test split's last column dropped (its header has one name too many), standardised on train, features whose smallest value recurs more than 5 times dropped (d = 21), the last
    10% of train as validation."""
    from collections import Counter
    a = pd.read_csv(ROOT + '/hepmass/1000_train.csv', index_col=False); b = pd.read_csv(ROOT + '/hepmass/1000_test.csv', index_col=False)
    a = a[a[a.columns[0]] == 1].drop(a.columns[0], axis=1)
    b = b[b[b.columns[0]] == 1].drop(b.columns[0], axis=1); b = b.drop(b.columns[-1], axis=1)
    mu, s = a.mean(), a.std(); a, b = (a - mu) / s, (b - mu) / s
    drop = [f for f in a.columns if np.array([v for _, v in sorted(Counter(a[f]).items())])[0] > 5]
    a, b = a.drop(drop, axis=1).to_numpy(), b.drop(drop, axis=1).to_numpy()
    nv = int(0.1 * len(a))
    return a[:-nv], a[-nv:], b


def ll(pdf, X, chunk=20000):
    return float(np.mean(np.concatenate([np.log(np.maximum(E.to_np(pdf(X[i:i + chunk])), 1e-300)) for i in range(0, len(X), chunk)])))


def blml_runs(name, tr, va, te, tag, logdet=0.0, gate=None):
    d = tr.shape[1]; rng = np.random.default_rng(0)
    Xs = tr[rng.choice(len(tr), min(SUB, len(tr)), replace=False)]; Vs = va[rng.choice(len(va), min(SUBV, len(va)), replace=False)]
    t0 = time.time(); sel, vll = select(Xs, make_modes(np.nan, ['sinc'] * d), search='auto', seed=0, engine=ENGINE, X_val=Vs)
    fc0 = np.asarray(modes_fc(sel)[0], float); ts = time.time() - t0
    out = dict(dataset=name, method=tag, d=d, n_train=len(tr), sub=len(Xs), fc_sub=fc0.tolist(), select_s=ts,
               sub_val=float(vll) + logdet, fits=[])
    if gate is not None and out['sub_val'] < gate:      # loses already on the subsample: skip the full-size fits
        log(f"{name} {tag}: subsample val {out['sub_val']:.3f} < {gate:.3f}, full fits skipped")
        out.update(test=np.nan, val=np.nan, skipped=True); return out
    for scale in (1.0, (len(tr) / len(Xs)) ** (1.0 / (d + 4))):
        t0 = time.time(); fc = fc0 * scale
        m = BLMLQuick(engine=ENGINE, dtype=DT).fit(tr, make_modes(list(fc), ['sinc'] * d, None)); tf = time.time() - t0
        out['fits'].append(dict(scale=scale, fc=fc.tolist(), fit_s=tf, val=ll(m.predict, va) + logdet, test=ll(m.predict, te) + logdet))
        log(f"{name} {tag} scale {scale:.2f}: val {out['fits'][-1]['val']:.3f} test {out['fits'][-1]['test']:.3f} ({tf:.0f}s)")
        del m
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
    best = max(out['fits'], key=lambda r: r['val']); out.update(test=best['test'], val=best['val'])
    return out


def stream_run(name, tr, va, te):
    """Streaming BLML: burn-in (the selector's auto search) on the first min(n/5, 2e4) points of a random order, then
    batches of n/50 (between 2000 and 10000: enough cut-off steps at full size) for up to EPOCHS passes; the library's
    rebuild threshold (the lattice is rebuilt when its target width moves by a factor 2: n grows 16x or fc doubles, so
    once or twice from the burn-in to the full data) with two full sweeps after each rebuild, so the coefficients do
    not lag the cut-off gradient, Adam on log fc; scored on the validation and test splits."""
    d = tr.shape[1]; X = tr[np.random.default_rng(0).permutation(len(tr))]
    Vs = va[np.random.default_rng(0).choice(len(va), min(SUBV, len(va)), replace=False)]; hist = []; t0 = time.time()

    def on_epoch(ep, s):                        # progress: the cut-offs and the score on the validation subsample
        h = s.history_[-1] if s.history_ else {}
        hist.append(dict(epoch=ep + 1, t=time.time() - t0, fc=np.ravel(modes_fc(s.modes_)[0]).tolist(), val_sub=ll(s.predict, Vs),
                         residual=h.get('residual'), params_stationary=h.get('params_stationary')))
        log(f"{name} blml_stream epoch {ep + 1}: val_sub {hist[-1]['val_sub']:.3f} residual {h.get('residual')} "
            f"stationary {h.get('params_stationary')} fc {np.round(hist[-1]['fc'], 2)} ({hist[-1]['t']:.0f}s)")
    s = BLMLQuickStreaming(engine=ENGINE, search='auto', dtype=DT, burn_in=min(len(X) // 5, 20000), seed=0,
                           rebin_sweeps=2).fit(
        X, make_modes(STREAM_FC if STREAM_FC is not None else np.nan, ['sinc'] * d), epochs=EPOCHS, batch=min(max(len(X) // 50, 2000), 10000), tol=1e-4, on_epoch=on_epoch)
    tf = time.time() - t0
    out = dict(dataset=name, method='blml_stream', fc=np.ravel(modes_fc(s.modes_)[0]).tolist(), fit_s=tf, epochs=s.epochs_,
               converged=s.converged_, history=hist, val=ll(s.predict, va), test=ll(s.predict, te))
    log(f"{name} blml_stream: val {out['val']:.3f} test {out['test']:.3f} ({tf:.0f}s, {s.epochs_} epochs, converged {s.converged_})")
    return out


def kde_run(name, tr, va, te):
    h0 = E.scott(tr); G = tr if cp is None else cp.asarray(tr); grid = np.geomspace(0.01, 2.0, 24)
    vv = va[np.random.default_rng(1).choice(len(va), min(SUBV, len(va)), replace=False)]
    t0 = time.time(); vals = [ll(lambda Z: E.kde_gauss(G, Z, c * h0), vv) for c in grid]; c = float(grid[int(np.argmax(vals))])
    out = dict(dataset=name, method='kde', c=c, val=float(max(vals)), test=ll(lambda Z: E.kde_gauss(G, Z, c * h0), te), fit_s=time.time() - t0)
    log(f"{name} kde c={c:.3f}: test {out['test']:.3f}")
    return out


fetch()
for name, loader in (('power', power), ('gas', gas), ('hepmass', hepmass)):
    if all(f'{name}_{m}' in SKIP for m in METHODS):
        continue
    tr, va, te = loader()
    if NTRAIN:                                  # smoke test: a random subset of the training split (validation, test as published)
        tr = tr[np.random.default_rng(3).choice(len(tr), min(NTRAIN, len(tr)), replace=False)]
    log(f'{name}: train {tr.shape}, validation {va.shape}, test {te.shape}')
    for method in METHODS:
        path = os.path.join(OUT, 'uci', f'{name}_{method}.json')
        if os.path.exists(path) or f'{name}_{method}' in SKIP:
            continue
        if method == 'kde':
            out = kde_run(name, tr, va, te)
        elif method == 'blml_stream':
            out = stream_run(name, tr, va, te)
        elif method == 'blml':
            out = blml_runs(name, tr, va, te, 'blml')
        else:                                   # whitening: x -> W (x - m); density picks up log |det W|
            m = tr.mean(0); C = np.cov(tr.T); ev, U = np.linalg.eigh(C); W = (U / np.sqrt(ev)).T
            T = lambda X: (X - m) @ W.T
            ref = json.load(open(os.path.join(OUT, 'uci', f'{name}_blml.json'))).get('sub_val')
            out = blml_runs(name, T(tr), T(va), T(te), 'blml_w', logdet=float(np.log(np.abs(np.linalg.det(W)))), gate=ref)
        json.dump(out, open(path, 'w'), indent=1, default=float)
log('done')
