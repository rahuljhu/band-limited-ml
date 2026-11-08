"""Paper 5: cut-off trajectories of streaming BLML on 2-D synthetic truths (T1, T2; fc_true = 0.4), n = 1e4.
(a) default: burn-in selection on min(n/2, 5000) points, then streaming, 2 epochs (as synth_stream.py);
(b) given starts fc0 in {0.2, 0.8} (trainable, no burn-in search), burn-in 1000, 5 epochs.
Batches of n/20; Adam on log fc, fc_every = 3. Writes results/synstream/fc_trace.json."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')
import numpy as np
import blml_trivial as bt
from blml_trivial import make_modes, modes_fc
from blml_rolling import BLMLQuickStreaming
from p1_truth import Truth
RS = dict(rebin_tol=float(os.environ.get('BLML_REBIN_TOL', 1.0)))   # 1: the paper's runs (library default 15 since the paper)
if os.environ.get('BLML_REBIN_SWEEPS') is not None: RS['rebin_sweeps'] = int(os.environ['BLML_REBIN_SWEEPS'])   # 2: the paper's runs

bt.DIRICHLET_LEGACY = False
N, D = 10000, 2


def run(X, tmpl, burn, epochs, rep):
    s = BLMLQuickStreaming(engine='cpu', search='auto', burn_in=burn, seed=rep, dtype=np.float64, **RS).fit(
        X, tmpl, epochs=epochs, batch=N // 20)
    seen, tr = 0, []
    for h in s.history_:                       # per batch after the burn-in (re-bin entries carry no fc)
        if 'fc' in h and not h.get('burn_in'):
            seen += N // 20; tr.append(dict(seen=burn + seen, fc=h['fc']))
        elif h.get('burn_in'):
            tr.append(dict(seen=burn, fc=h['fc']))
    return tr


out = {}
for kind in ('T1', 'T2'):
    truth = Truth(kind, D)
    for rep in range(4):
        X = truth.sample(np.random.default_rng([3, D, N, rep]), N)       # the synthetic study's samples
        out[f'{kind}/default/{rep}'] = run(X, make_modes(np.nan, ['sinc'] * D), min(N // 2, 5000), 2, rep)
        for f0 in (0.2, 0.8):
            out[f'{kind}/start{f0}/{rep}'] = run(X, make_modes(f0, ['sinc'] * D), 1000, 5, rep)
        print(kind, rep, {k: np.round(v[-1]['fc'], 3).tolist() for k, v in out.items() if k.startswith(kind) and k.endswith(f'/{rep}')}, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'synstream', 'fc_trace.json'), 'w'))
print('ALL DONE')
