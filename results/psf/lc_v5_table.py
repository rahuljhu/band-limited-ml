"""Paper 3 learning-curve table from results/psf/v5/lc/*.json: excess loss (KL, nats / photon) of each method, estimated
on real data with the KT histogram as anchor. The held-out score of method m differs from the truth's by KL(p || p_m), so
KL_m = KL_hist - (LL_m - LL_hist), and the histogram's expected KL at n photons is computed, not assumed: Monte Carlo
over multinomial(n, p~) draws of KL(p~ || KT histogram), with p~ the bead's KT histogram from all its photons
(about 50-600 per pixel; the asymptote (B - 1) / 2n fails at small n). Averaged over seeds; slopes in log n over n >= 30k.
Writes papers/paper3/tables/lc_excess.tex and papers/paper3/numbers_lc.tex. Run from the repo root."""
import sys, json, glob, collections, numpy as np
sys.path.insert(0, 'results/psf'); import psfkit as pk
L = [json.load(open(f)) for f in glob.glob('results/psf/v5/lc/*.json')]
M = [('real_ball', 'real BLML'), ('complex_ball', 'complex BLML'), ('KDE', 'KDE'), ('KDE+bg', 'KDE + background')]
def kl_hist(p, n, draws=200, seed=0):
    g = np.random.default_rng(seed); v = []
    for _ in range(draws):
        q = pk.histogram(g.multinomial(n, p)); v.append(float((p * np.log(p / q)).sum()))
    return float(np.mean(v))
G = collections.defaultdict(list)
for r in L: G[(r['bead'], r['n_target'])].append(r)
rows, ex = [], collections.defaultdict(dict)
for (b, nt) in sorted(G):
    rs = G[(b, nt)]; p = pk.histogram(pk.bead_counts(b)[0]); n = int(np.mean([r['n'] for r in rs]))
    kh = np.mean([kl_hist(p, r['n'], seed=r['seed']) for r in rs])
    e = {m: kh - np.mean([r[m]['test_ll'] - r['histogram']['test_ll'] for r in rs]) for m, _ in M}; ex[b][n] = e
    best = min(e, key=e.get)
    rows.append(f"{b} & {n:,}".replace(',', '{,}') + f" & {kh:.5f} & "
                + ' & '.join((r'\textbf{%.5f}' if m == best else '%.5f') % e[m] for m, _ in M) + f" & {e['KDE'] / e['real_ball']:.1f} \\\\")
open('papers/paper3/tables/lc_excess.tex', 'w').write(
    '\\begin{tabular}{rr r rrrr r}\n\\toprule\nbead & $n$ & histogram & ' + ' & '.join(l for _, l in M)
    + ' & KDE / real \\\\\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')
sl = {m: [] for m, _ in M}
for b, d in ex.items():
    ns = sorted(v for v in d if v >= 25000)
    if len(ns) >= 2:
        for m, _ in M: sl[m].append(np.polyfit(np.log(ns), np.log([max(d[v][m], 1e-7) for v in ns]), 1)[0])
top = {b: d[max(d)] for b, d in ex.items()}
num = dict(LcSlopeRe='%.1f' % np.median(sl['real_ball']), LcSlopeCx='%.1f' % np.median(sl['complex_ball']),
           LcSlopeKDE='%.1f' % np.median(sl['KDE']), LcSlopeKDEbg='%.1f' % np.median(sl['KDE+bg']),
           LcRatioTop='%.1f--%.1f' % (min(t['KDE'] / t['real_ball'] for t in top.values()), max(t['KDE'] / t['real_ball'] for t in top.values())))
open('papers/paper3/numbers_lc.tex', 'w').write(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in num.items()))
print(open('papers/paper3/tables/lc_excess.tex').read(), num, {m: np.round(v, 2) for m, v in sl.items()})
