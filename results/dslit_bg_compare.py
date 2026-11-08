"""Paper 3, double slit: the background as a fixed dirichlet mode (dslit_job.py real_bg / complex_bg, BLML_BG_HARM)
against the EM background (positive / complex / KDE / references), on the same split. Paired test differences per
particle (log P_A - log P_B at its pixel; SE = sd / sqrt(n_test)) to complex BLML with the EM background.
Collapsed (1-d) fits (BLML_COLLAPSE=1) score per particle on the line: compare them only with each other (pass the
same directory twice).
Usage: python results/dslit_bg_compare.py <set> <EM dir> <bg-mode dir>"""
import os, sys, json, numpy as np
name, DEM, DBG = sys.argv[1:4]; DATA = os.path.expanduser('~/data_repo/dslit_gpu/')
pts = np.load(DATA + f'dslit_{name}_points.npy'); SH = np.load(DATA + f'dslit_{name}_b.npy').shape[-2:]
axis = int(np.load(DATA + f'dslit_{name}_axis.npy'))
ij = [np.clip(np.floor(pts[:, k] + 0.5).astype(int), 0, SH[k - 1] - 1) for k in (1, 2)]
te = np.random.default_rng(0).random(len(pts)) >= 0.80
runs = [(f'{p} (EM)', DEM, p) for p in ('complex', 'positive', 'kde', 'fraunhofer', 'fresnel', 'hist')] + \
       [(f'{p} (bg mode)', DBG, p) for p in ('complex_bg', 'real_bg')]
R, LP = {}, {}
for lab, D, p in runs:
    if os.path.exists(f'{D}/{name}_{p}.json') and os.path.exists(f'{D}/{name}_{p}_P.npy'):
        R[lab] = json.load(open(f'{D}/{name}_{p}.json'))
        Pm = np.load(f'{D}/{name}_{p}_P.npy').astype(float)            # (H, W), or (W,) collapsed across the fringes
        idx = (ij[axis][te],) if Pm.ndim == 1 else (ij[0][te], ij[1][te])
        LP[lab] = np.log(np.maximum(Pm, 1e-300))[idx]
ref = 'complex (EM)'
fr = R['fraunhofer (EM)']['params']; phys = (1 + fr['a_over_d']) / fr['P']
print(f'{name}: {te.sum()} test particles; physical band of f along the fringes {phys:.4f} cycles/px')
out = []
for lab in sorted(LP, key=lambda k: -LP[k].mean()):
    d = LP[lab] - LP[ref]; se = d.std(ddof=1) / np.sqrt(len(d)); r = R[lab]
    fc = r.get('fc'); fca = None if fc is None else fc[axis if len(fc) > 1 else 0]; beta = float(np.sum(r.get('beta', [0.0])))
    out.append(dict(model=lab, test_ll=float(LP[lab].mean()), diff=float(d.mean()), se=float(se), beta=beta,
                    fc=fc, fc_ratio=None if fca is None else fca / phys, harmonics=r.get('bg_harmonics'), s=r.get('s')))
    print(f"{lab:22s} {LP[lab].mean():.5f}  vs complex (EM) {d.mean():+.5f} +- {se:.5f}  beta {beta:.3f}"
          + ('' if fca is None else f"  fc {np.round(fc, 4).tolist()} ({fca / phys:.2f} x physical)")
          + ('' if r.get('bg_harmonics') is None else f"  bg harmonics {r['bg_harmonics']}"))
json.dump(out, open(f'{DBG}/{name}_compare.json', 'w'), indent=1)
