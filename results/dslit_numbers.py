"""Paper 3, double-slit section: \\newcommand numbers from the result files (photons: results/dslit_final, electrons:
results/dslit_electron; cleaning logs in ~/data_repo/glasgow_dslit; the export in ~/data_repo/dslit_gpu).
Writes papers/paper3/numbers_dslit.tex. Run from the repo root after results/dslit_analyze.py."""
import os, json, numpy as np
DATA = os.path.expanduser('~/data_repo/dslit_gpu/'); G = os.path.expanduser('~/data_repo/glasgow_dslit/')
SRC = dict(photon='results/dslit_final', electron='results/dslit_electron'); TAG = dict(photon='Ph', electron='El')
num = {}
def fmt(v, d=4): return f'{v:.{d}f}'
for name, D in SRC.items():
    t = TAG[name]; R = {p: json.load(open(f'{D}/{name}_{p}.json')) for p in ('complex', 'positive', 'kde', 'fraunhofer', 'fresnel', 'hist')}
    pts = np.load(DATA + f'dslit_{name}_points.npy'); SH = np.load(DATA + f'dslit_{name}_b.npy').shape[-2:]; axis = int(np.load(DATA + f'dslit_{name}_axis.npy'))
    ij = [np.clip(np.floor(pts[:, k] + 0.5).astype(int), 0, SH[k - 1] - 1) for k in (1, 2)]; te = np.random.default_rng(0).random(len(pts)) >= 0.80
    LP = {p: np.log(np.maximum(np.load(f'{D}/{name}_{p}_P.npy').astype(float), 1e-300))[ij[0][te], ij[1][te]] for p in R}
    fr = R['fraunhofer']['params']; phys = (1 + fr['a_over_d']) / fr['P']
    num[f'{t}N'] = f'{len(pts):,}'.replace(',', '{,}'); num[f'{t}NTest'] = f'{int(te.sum()):,}'.replace(',', '{,}')
    num[f'{t}Period'] = fmt(fr['P'], 1); num[f'{t}AD'] = fmt(fr['a_over_d'], 3); num[f'{t}Phys'] = fmt(phys, 4)
    for p in ('complex', 'positive'):
        num[f'{t}Fc{p.capitalize()}'] = fmt(R[p]['fc'][axis], 4); num[f'{t}Ratio{p.capitalize()}'] = fmt(R[p]['fc'][axis] / phys, 2)
    num[f'{t}Cert'] = fmt(R['complex']['certificate'], 2)
    for p, lab in (('complex', 'Cx'), ('positive', 'Pos'), ('kde', 'KDE'), ('fraunhofer', 'Fr'), ('fresnel', 'Fresnel'), ('hist', 'Hist')):
        num[f'{t}LL{lab}'] = fmt(LP[p].mean(), 4)
        if p != 'complex':
            d = LP[p] - LP['complex']; num[f'{t}D{lab}'] = fmt(-d.mean(), 4); num[f'{t}SE{lab}'] = fmt(d.std(ddof=1) / np.sqrt(len(d)), 4)
    num[f'{t}BetaCx'] = fmt(float(np.sum(R['complex']['beta'])), 2)
    # |FT g| along the fringes at xi/d = -1/2, 0, +1/2 (complex; and the positive root at 0 and +-1)
    pad = 8 * SH[axis]; xi = np.fft.fftfreq(pad) * fr['P']
    for p, lab in (('complex', 'Cx'), ('positive', 'Pos')):
        A = np.abs(np.fft.fft(np.load(f'{D}/{name}_{p}_line.npy'), pad)); A = A / A.max()
        h = {c: float(A[np.abs(xi - c) < 0.12].max()) for c in (-1.0, -0.5, 0.0, 0.5, 1.0)}
        num[f'{t}Slit{lab}'] = f'{min(h[-0.5], h[0.5]):.2f}--{max(h[-0.5], h[0.5]):.2f}'; num[f'{t}Centre{lab}'] = fmt(h[0.0], 2)
        num[f'{t}Side{lab}'] = fmt(max(h[-1.0], h[1.0]), 2)
num['ElBeta'] = fmt(float(np.load(DATA + 'dslit_electron_beta.npy')[0]), 2)
lg = json.load(open(G + 'Heralded_Diffraction_SM_clean.json'))
num.update(PhPixels=f"{lg['pixels_raw']:,}".replace(',', '{,}'), PhClusters=f"{lg['clusters']:,}".replace(',', '{,}'),
           PhStreaks=str(lg['streak_clusters']), PhHot=str(lg['hot_pixels']), PhReadouts=f"{lg['readouts']:,}".replace(',', '{,}'))
open('papers/paper3/numbers_dslit.tex', 'w').write(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in num.items()))
print(num)
