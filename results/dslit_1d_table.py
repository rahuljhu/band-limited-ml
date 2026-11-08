"""Paper 3, double slit, 1-d protocol (results/dslit_1d_runs.sh): test LL per particle, paired difference to complex
BLML with the background mode N = 1 (+- SE over test particles), beta, fc / physical band ((1 + a/d) / P of the
flat-background Fraunhofer fit), stop reason. Usage: python results/dslit_1d_table.py [set ...] -> results/dslit_1d/table.json"""
import os, sys, json, numpy as np
SETS = sys.argv[1:] or ['photon', 'electron', 'photon_mm']; DATA = os.path.expanduser('~/data_repo/dslit_gpu/'); R0 = 'results/dslit_1d'
RUNS = [('complex, bg N=1', 'bg1', 'complex_bg'), ('complex, bg N=0', 'bg0', 'complex_bg'), ('complex, bg N=2', 'bg2', 'complex_bg'),
        ('KDE, flat bg', 'flat', 'kde'), ('adaptive KDE, flat bg', 'flat', 'akde'), ('Fresnel, flat bg', 'flat', 'fresnel'), ('Fraunhofer, flat bg', 'flat', 'fraunhofer'),
        ('Fresnel + blur, flat bg', 'flat', 'fresnel_blur'), ('Fraunhofer + blur, flat bg', 'flat', 'fraunhofer_blur'),
        ('histogram', 'flat', 'hist')]
out = {}
for name in SETS:
    pts = np.load(DATA + f'dslit_{name}_points.npy'); SH = np.load(DATA + f'dslit_{name}_b.npy').shape[-2:]
    axis = int(np.load(DATA + f'dslit_{name}_axis.npy'))
    x = np.clip(np.floor(pts[:, 1 + axis] + 0.5).astype(int), 0, SH[axis] - 1)
    te = np.random.default_rng(0).random(len(pts)) >= 0.80
    fr = json.load(open(f'{R0}/{name}/flat/{name}_fraunhofer.json'))['params']; phys = (1 + fr['a_over_d']) / fr['P']
    LP, R = {}, {}
    for lab, d, p in RUNS:
        f = f'{R0}/{name}/{d}/{name}_{p}'
        if os.path.exists(f + '.json'):
            R[lab] = json.load(open(f + '.json')); LP[lab] = np.log(np.maximum(np.load(f + '_P.npy').astype(float), 1e-300))[x[te]]
    ref = LP['complex, bg N=1']; rows = []
    print(f'\n{name}: {te.sum()} test particles, physical band of f {phys:.4f} cycles/px')
    for lab in sorted(LP, key=lambda k: -LP[k].mean()):
        dd = LP[lab] - ref; se = dd.std(ddof=1) / np.sqrt(len(dd)); r = R[lab]; fc = r.get('fc')
        rows.append(dict(model=lab, test_ll=float(LP[lab].mean()), diff=float(dd.mean()), se=float(se), beta=float(np.sum(r.get('beta', [0]))),
                         fc=None if fc is None else fc[0], fc_ratio=None if fc is None else fc[0] / phys, stop=r.get('stop')))
        print(f"  {lab:20s} {LP[lab].mean():.5f}  {dd.mean():+.5f} +- {se:.5f}  beta {rows[-1]['beta']:.3f}"
              + ('' if fc is None else f"  fc {fc[0]:.4f} ({fc[0] / phys:.2f} x)") + ('' if r.get('stop') in (None, 'tol') else f"  [{r['stop']}]"))
    out[name] = dict(n_test=int(te.sum()), phys=phys, rows=rows)
json.dump(out, open(f'{R0}/table.json', 'w'), indent=1)
