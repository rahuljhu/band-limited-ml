"""Paper 3, double slit on the paper's split under the current protocol (5-fold CV, one-SE rule, L-BFGS; runs of
p3_synth_job.py in results/p3synth_lbfgs; the physics references and the histogram from results/dslit_1d/<set>/flat):
test log-likelihood per particle, the paired difference to complex BLML with the background mode N = 1 (+- SE over
test particles), the background weight and the selected cut-off relative to the physical band.
-> papers/submit/pra_complex/tables/dslit_<set>.tex and results/p3_dslit_table2.json. Run from the repo root."""
import os, json, numpy as np
DATA = os.path.expanduser('~/data_repo/dslit_gpu/'); R = 'results/p3synth_lbfgs'; R0 = 'results/dslit_1d'
RUNS = [('complex BLML, $N=1$', f'{R}/bg1', 'complex_bg'), ('complex BLML, $N=0$', f'{R}/bg0', 'complex_bg'),
        ('complex BLML, $N=2$', f'{R}/bg2', 'complex_bg'), ('complex BLML, flat background', f'{R}/flat', 'complex'),
        ('streaming, $N=0$', f'{R}/stream0', 'stream_bg'),
        ('streaming, $N=1$', f'{R}/stream1', 'stream_bg'), ('KDE', f'{R}/flat', 'kde'), ('adaptive KDE', f'{R}/flat', 'akde'),
        ('Fresnel', 'OLD', 'fresnel'), ('Fraunhofer', 'OLD', 'fraunhofer'), ('histogram', 'OLD', 'hist')]
out = {}
for name in ('photon', 'ghost', 'electron', 'photon_mm'):
    pts = np.load(DATA + f'dslit_{name}_points.npy'); SH = np.load(DATA + f'dslit_{name}_b.npy').shape[-2:]
    axis = int(np.load(DATA + f'dslit_{name}_axis.npy')); x = np.clip(np.floor(pts[:, 1 + axis] + 0.5).astype(int), 0, SH[axis] - 1)
    te = np.random.default_rng(0).random(len(pts)) >= 0.80
    fr = json.load(open(f'{R0}/{name}/flat/{name}_fraunhofer.json'))['params']; phys = (1 + fr['a_over_d']) / fr['P']
    LP, J = {}, {}
    for lab, d, p in RUNS:
        d = f'{R0}/{name}/flat' if d == 'OLD' else f'{d}/{name}'
        f = f'{d}/{name}_{p}'
        if os.path.exists(f + '.json') and os.path.exists(f + '_P.npy'):
            J[lab] = json.load(open(f + '.json')); LP[lab] = np.log(np.maximum(np.load(f + '_P.npy').astype(float), 1e-300))[x[te]]
    # the two-slit kernel with the cut-off, fill and background harmonics chosen by CV (p3_slits_diag.py, paper split)
    import glob, sys; sys.path.insert(0, '.'); os.environ.setdefault('BLML_ENGINE', 'cpu'); os.environ.setdefault('BLML_SOLVER', 'lbfgs')
    import dslit_job as dj
    C = {json.load(open(f))['config']: json.load(open(f)) for f in glob.glob(f'results/p3_slits_diag/{name}_K1-N[0-9].json')}
    if C:
        c = C[max(C, key=lambda k: C[k]['cv_best'])]; TV = np.bincount(x[~te], minlength=SH[axis]).astype(float)
        Q, m, beta = dj.bg_fit(TV, [c['fc']], (c['N'],), True, 1, max_sweeps=5000, tol=1e-6, fill=c['fill'])
        lab = f"two-slit kernel, CV ($N={c['N']}$)"; LP[lab] = np.log(np.maximum(Q, 1e-300))[x[te]]
        J[lab] = dict(beta=[beta], fc=[c['fc']], fill=c['fill'])
    ref = LP['complex BLML, $N=1$']; rows = []
    for lab in sorted(LP, key=lambda k: -LP[k].mean()):
        dd = LP[lab] - ref; r = J[lab]; fc = r.get('fc'); fc = None if fc is None else (fc[0] if isinstance(fc, list) else fc)
        rows.append(dict(model=lab, test=float(LP[lab].mean()), diff=float(dd.mean()), se=float(dd.std(ddof=1) / np.sqrt(len(dd))),
                         beta=float(np.sum(r.get('beta', [0]))), ratio=None if fc is None else fc / phys, fill=r.get('fill')))
    out[name] = dict(n_test=int(te.sum()), phys=phys, rows=rows)
    L = [r'\begin{tabular}{lrrrr}', r'\toprule', r'model & test LL & $-$ BLML $N{=}1$ ($\pm$ SE) & $\beta$ & $f_c$/physical \\', r'\midrule']
    for r in rows:
        dcol = '--' if r['model'] == 'complex BLML, $N=1$' else f"{r['diff']:+.4f} ({r['se']:.4f})"
        fcc = '--' if r['ratio'] is None else f"{r['ratio']:.2f}" + (f" ($w={r['fill']:g}$)" if r['fill'] not in (None, 1.0) else '')
        bet = '--' if r['model'] == 'histogram' else f"{r['beta']:.2f}"
        L.append(f"{r['model']} & {r['test']:.4f} & {dcol} & {bet} & {fcc} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    open(f'papers/submit/pra_complex/tables/dslit_{name}.tex', 'w').write('\n'.join(L) + '\n')
    print(f'\n{name} (test {te.sum()}, phys {phys:.4f})'); [print('  ', l) for l in L[4:-2]]
json.dump(out, open('results/p3_dslit_table2.json', 'w'), indent=1)
