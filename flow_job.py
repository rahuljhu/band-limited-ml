"""Normalizing-flow baseline for the merged BLML paper: a neural spline flow (zuko NSF: 5 spline transforms, two
hidden layers of 128, Adam, early stopping on a held-out fifth of the sample, data standardised per dimension) on the
synthetic truths of the rate and non-smooth studies, scored by the same relative ISE (defensive importance sampling)
and KL (draws from f) as BLML and the kernel estimators; then the joint 6-d price pair (p2_pairs.py, model NSF-joint).
  synthetic: T1 (band-limited, separable), T3 (correlated normal), L (Laplace, kink), B (Beta(2,2), edges);
             d in {2, 4, 8}, n in {1e3, 1e4}, plus n = 1e5 for T1 and T3 at d = 8; two replicates.
Run: BLML_DATA_SRC=~/data_repo/caiso ./colab_job.sh start flow flow_job.py T4   (one JSON per cell; resumable)
BLML_FLOW_PART=plateau: fixed architectures of three sizes against n on T1 at d = 4 (into OUT/plateau)."""
import os, sys, json, time, subprocess
OUT = os.environ.get('BLML_OUT_DIR', 'outputs_flow'); CODE = os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(OUT, 'syn'), exist_ok=True)
for pkg, mod in (('zuko', 'zuko'), ('cupy-cuda13x', 'cupy')):
    if subprocess.run([sys.executable, '-c', f'import {mod}'], capture_output=True).returncode:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)
sys.path.insert(0, CODE)
import numpy as np
import torch, zuko
from p1_truth import Truth, NonSmooth, DefensiveProposal
import p1_estimators as E
M_F, M_Q = 100000, 200000
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def truth_of(kind, d):
    return NonSmooth(kind, d) if kind in ('L', 'B') else Truth(kind, d)


def eval_sets(truth, kind, d):
    """As the studies that produced the BLML and KDE numbers: defensive scale 6 for T1, 2 for T3, L and B."""
    rng = np.random.default_rng(20_000 + d)
    Xf = truth.sample(rng, M_F); q = DefensiveProposal(truth, scale=6.0 if kind == 'T1' else 2.0); Xq = q.sample(rng, M_Q)
    ff = truth.pdf(Xf)
    return dict(Xf=Xf, ff=ff, Xq=Xq, fq=truth.pdf(Xq), qq=q.pdf(Xq), int_f2=float(ff.mean()))


def fit_flow(X, seed, transforms=5, hidden=(128, 128), max_epochs=1000):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m, s = X.mean(0), X.std(0) + 1e-9; Z = (X - m) / s; idx = rng.permutation(len(Z)); k = int(0.8 * len(Z))
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    tr, va = T(Z[idx[:k]]), T(Z[idx[k:]])
    flow = zuko.flows.NSF(features=X.shape[1], transforms=transforms, hidden_features=list(hidden)).to(dev)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3); best, state, bad, bs = np.inf, None, 0, min(4096 if k > 200000 else 512, max(64, k // 8))
    for epoch in range(max_epochs):
        perm = torch.randperm(k, device=dev)
        for i in range(0, k, bs):
            loss = -flow().log_prob(tr[perm[i:i + bs]]).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v = -flow().log_prob(va).mean().item()
        if v < best - 1e-4:
            best, bad = v, 0; state = {a: b.detach().clone() for a, b in flow.state_dict().items()}
        else:
            bad += 1
            if bad >= 30:
                break
    flow.load_state_dict(state)

    def pdf(Xq):
        Zq = torch.tensor((np.asarray(Xq) - m) / s, dtype=torch.float32, device=dev)
        with torch.no_grad():
            lp = np.concatenate([flow().log_prob(Zq[i:i + 50000]).cpu().numpy() for i in range(0, len(Zq), 50000)])
        return np.exp(lp.astype(np.float64) - np.log(s).sum())
    return pdf, epoch + 1


if os.environ.get('BLML_FLOW_PART') == 'plateau':
    # Fixed architectures against sample size: each size should level off at its own approximation floor (T1, d = 4,
    # where BLML has runs to n = 1e6). Sizes: tiny (1 transform, 8 hidden), small (2, 32 x 2), the baseline (5, 128 x 2).
    os.makedirs(os.path.join(OUT, 'plateau'), exist_ok=True)
    SIZES = {'tiny': (1, (8,)), 'small': (2, (32, 32)), 'base': (5, (128, 128))}
    truth, kind, d = Truth('T1', 4), 'T1', 4; ev = eval_sets(truth, kind, d)
    for n in (1000, 10000, 100000, 1000000):
        for size, (tf, hid) in SIZES.items():
            path = os.path.join(OUT, 'plateau', f'{size}_n{n}.json')
            if os.path.exists(path):
                continue
            reps = []
            for rep in range(2 if n < 1000000 else 1):
                X = truth.sample(np.random.default_rng([9, d, n, rep]), n); t0 = time.time()
                pdf, ep = fit_flow(X, rep, tf, hid, max_epochs=1000 if n < 1000000 else 150)
                r = E.metrics(pdf(ev['Xf']), ev['ff'], pdf(ev['Xq']), ev['fq'], ev['qq'], ev['int_f2'])
                r.update(fit_s=time.time() - t0, epochs=ep); reps.append(r)
                log(f"plateau {size} n={n} rep {rep}: rel_ise {r['rel_ise']:.4g} ({r['fit_s']:.0f}s, {ep} epochs)")
            json.dump(dict(truth=kind, d=d, n=n, size=size, transforms=tf, hidden=list(hid), reps=reps),
                      open(path, 'w'), indent=1, default=float)
    log('done'); sys.exit()

CELLS = [(k, d, n) for k in ('T1', 'T3', 'L', 'B') for d in (2, 4, 8) for n in (1000, 10000)] + [('T1', 8, 100000), ('T3', 8, 100000)]
for kind, d, n in CELLS:
    path = os.path.join(OUT, 'syn', f'{kind}_d{d}_n{n}.json')
    if os.path.exists(path):
        continue
    truth = truth_of(kind, d); ev = eval_sets(truth, kind, d); reps = []
    for rep in range(2):
        X = truth.sample(np.random.default_rng([7, d, n, rep]), n); t0 = time.time()
        pdf, ep = fit_flow(X, rep)
        r = E.metrics(pdf(ev['Xf']), ev['ff'], pdf(ev['Xq']), ev['fq'], ev['qq'], ev['int_f2'])
        r.update(fit_s=time.time() - t0, epochs=ep); reps.append(r)
        log(f"{kind} d={d} n={n} rep {rep}: rel_ise {r['rel_ise']:.4g} kl {r.get('kl', float('nan')):.4g} ({r['fit_s']:.0f}s, {ep} epochs)")
    json.dump(dict(truth=kind, d=d, n=n, model='NSF', reps=reps), open(path, 'w'), indent=1, default=float)

# the joint price pair: the same split and scoring as the other pair models
env = dict(os.environ, BLML_P2_MODELS='NSF-joint', BLML_OUT_DIR=OUT)
subprocess.run([sys.executable, os.path.join(CODE, 'p2_pairs.py')], env=env, check=False)
log('done')
