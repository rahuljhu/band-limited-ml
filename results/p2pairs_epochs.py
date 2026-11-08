"""Paper 2 joint density (p2_pairs.py, BLML-joint-stream) with the test log-likelihood after every streaming epoch and
the burn-in's time. Same pairs, split and model as p2_pairs; env BLML_STREAM_SEARCH ('auto' as the paper, or the
library default 'shared'), BLML_STREAM_BURN_IN (10000 as the paper; the library default 5000), BLML_STREAM_EPOCHS (6),
BLML_ENGINE (cpu / gpu), BLML_STREAM_DTYPE (float32 on a T4), BLML_STREAM_FC0 (JSON (K, 6) starting fc: rank K) and
BLML_TAG (a suffix for the output name). Writes results/p2pairs_epochs_<search>[_b<burn-in>].json."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
os.environ.setdefault('BLML_ENGINE', 'cpu')
import numpy as np
import p2_pairs as PP

P = PP.load_pairs()
tr, va, te, *_ = PP.split(P)
search = os.environ.get('BLML_STREAM_SEARCH', 'auto')
bi = os.environ.get('BLML_STREAM_BURN_IN')
print(f'{len(P["p1"])} pairs; train {len(tr)}, validation {len(va)}, test {len(te)}; burn-in search {search}, '
      f'burn-in {bi or 10000}', flush=True)
rows, t0 = [], time.time()
Xte = PP.PairBLMLStream._X(None, P, te)                  # the test inputs (p1, p2, h1, D, l1, l2)


def on_epoch(ep, s):
    t = time.time()
    ll = float(np.mean(np.log(np.maximum(s.predict_ratio(Xte), 1e-300))))
    rows.append(dict(epoch=ep + 1, test_ll=ll, elapsed_s=round(t - t0, 1), predict_s=round(time.time() - t, 1),
                     fc=[[round(float(v), 4) for v in row] for row in PP.modes_fc(s.modes_)]))
    print(f"epoch {ep + 1}: test LL {ll:.4f} ({rows[-1]['elapsed_s']:.0f}s elapsed; fc {rows[-1]['fc']})", flush=True)


m = PP.PairBLMLStream(search=search, on_epoch=on_epoch).fit(P, tr, va)
burn = next((h for h in m.m.history_ if h.get('burn_in')), {})
out = dict(search=search, n_train=len(tr), n_val=len(va), n_test=len(te), burn_in_s=round(burn.get('s', 0.0), 1),
           fit_s=round(m.fit_s, 1), per_epoch=rows)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'p2pairs_epochs_{search}' + (f'_b{bi}' if bi else '') + os.environ.get('BLML_TAG', '') + '.json'), 'w'), indent=1)
print(f"burn-in {out['burn_in_s']:.0f}s of {out['fit_s']:.0f}s fit", flush=True)
