"""Paper 3 tables and numbers from the result files:
  results/psf/v5/bead*.json            -> papers/paper3/tables/beads_em.tex   (EM background, ball kernel, 53 beads)
  results/complex/sec42/*_1000000_*    -> papers/paper3/tables/phase_single.tex (single start)
  results/complex/sec42_init/starts_*  -> (5-start pipeline, truth start)
and \\newcommand numbers appended to papers/paper3/numbers_em.tex. Run from the repo root."""
import os, json, glob, numpy as np

P3 = 'papers/paper3/'
R = [json.load(open(f)) for f in sorted(glob.glob('results/psf/v5/bead*.json'),
                                        key=lambda s: int(s.split('bead')[1].split('.')[0]))]
hist = np.array([r['histogram_test_ll'] for r in R])
kde = np.array([json.load(open(f"results/p7_beads/b{r['bead']}_full_0.json"))['kde']['test'] for r in R])   # same split and floor
rows, num = [], {}
for key, lab in (('real_ball', 'real'), ('complex_ball', 'complex')):
    d = np.array([r[key]['test_ll'] for r in R]) - hist
    fc = np.array([np.ravel(r[key]['fc'])[0] for r in R])
    pi = np.array([r[key]['pi'] for r in R]); cert = np.array([r[key]['cert'] for r in R])
    q = lambda v, f='{:.3f}': '{}\\,[{}, {}]'.format(*(f.format(x) for x in np.quantile(v, [0.5, 0.25, 0.75])))
    dk = np.array([r[key]['test_ll'] for r in R]) - kde
    rows.append(f"{lab} BLML & {(d > 0).sum()}/{len(d)} & {np.median(d):+.4f} & {(dk > 0).sum()}/{len(dk)} & {np.median(dk):+.4f} & {q(fc)} & {q(pi)} & {q(cert)} \\\\")
    tag = 'Re' if lab == 'real' else 'Cx'
    num.update({f'Win{tag}': f'{(d > 0).sum()}/{len(d)}', f'Gain{tag}': f'{np.median(d):.4f}',
                f'Fc{tag}': f'{np.median(fc):.2f}', f'WinKde{tag}': f'{(dk > 0).sum()}/{len(dk)}', f'GainKde{tag}': f'{np.median(dk):+.4f}', f'Pi{tag}': f'{np.median(pi):.2f}', f'Cert{tag}': f'{np.median(cert):.2f}'})
dd = np.array([r['real_ball']['test_ll'] - r['complex_ball']['test_ll'] for r in R])
num.update(RealOverCx=f'{(dd > 0).sum()}/{len(dd)}', RealOverCxGain=f'{np.median(dd):.4f}')
pos = np.array([r['pos'] for r in R]); rad = np.hypot(*(pos - pos.mean(0)).T)
num['FcFieldCorr'] = f"{np.corrcoef(rad, [np.ravel(r['complex_ball']['fc'])[0] for r in R])[0, 1]:.2f}"
open(P3 + 'tables/beads_em.tex', 'w').write(
    "\\begin{tabular}{lccccccc}\n\\toprule\n"
    "& beats histogram & median gain & beats KDE & median gain & selected $f_c$ & background $\\pi$ & certificate \\\\\n"
    "& (beads) & (nats/photon) & (beads) & (nats/photon) & (cycles/px) & & $\\lambda_{\\max}/n$ \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

cases = [('coma', 'coma'), ('astig', 'astigmatism'), ('defocus', 'defocus'), ('mixed', 'mixed (4 modes)')]
prow = []
for c, lab in cases:
    single = [json.load(open(f))['rms_err'] for f in sorted(glob.glob(f'results/complex/sec42/{c}_1000000_*.json'))]
    ts, best, dl = [], [], []
    for f in sorted(glob.glob(f'results/complex/sec42_init/starts_{c}_1000000_*.json')):
        r = json.load(open(f)); pol = r['hio']['polished'] + r['symbreak']['polished']
        b = max(pol, key=lambda p: p['ll_minus_truth'])
        ts.append(r['truth_start']['rms_err']); best.append(b['rms_err'])
        dl.append(b['ll_minus_truth'] - r['truth_start']['ll_minus_truth'])
    fmt = lambda v, f='{:.3f}': ', '.join(f.format(x) for x in v)
    prow.append(f"{lab} & {fmt(single)} & {fmt(best)} & {fmt(dl, '{:+.0f}')} & {fmt(ts)} \\\\")
open(P3 + 'tables/phase_single.tex', 'w').write(
    "\\begin{tabular}{lcccc}\n\\toprule\n"
    "aberration & single start & 5 starts, best LL & $\\Delta$LL vs.\\ truth start & truth start \\\\\n"
    "& rms error (rad) & rms error (rad) & (nats) & rms error (rad) \\\\\n\\midrule\n"
    + "\n".join(prow) + "\n\\bottomrule\n\\end{tabular}\n")
open(P3 + 'numbers_em.tex', 'w').write(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in num.items()))
print('wrote', P3 + 'tables/beads_em.tex', P3 + 'tables/phase_single.tex', P3 + 'numbers_em.tex'); print(num)

# full-frame fits cropped to the bead windows (results/p3_frame_crop.py), paired against the window KDE
FR = json.load(open('results/p3_frame_crop.json')); fnum = {}
for l, t in (('complex', 'Cx'), ('real', 'Re'), ('empirical', 'Em')):
    d = np.array([x[f'{l}-kde_window'][0] for x in FR]); z = d / np.array([x[f'{l}-kde_window'][1] for x in FR])
    fnum.update({f'Frame{t}Wins': int((z > 2).sum()), f'Frame{t}Ties': int((abs(z) <= 2).sum()), f'Frame{t}Losses': int((z < -2).sum()),
                 f'Frame{t}Med': f'{np.median(d):+.4f}'})
fr = json.load(open('results/binned/frame.json'))
fnum.update(FrameCxSweeps=f"{fr['complex']['sweeps']:,}".replace(',', '{,}'), FramePhotons=f"{fr['photons'] / 1e6:.0f}")
open(P3 + 'numbers_em.tex', 'a').write(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in fnum.items()))
print(fnum)

# the background as one more mode on the frame (frame_bg_gpu_job.py, T4 float32), paired against the signal-only fit
if os.path.exists('results/frame_bg/oneshot_bg.json') and 'complex_bg-complex' in FR[0]:
    ob = json.load(open('results/frame_bg/oneshot_bg.json'))
    d = np.array([x['complex_bg-complex'][0] for x in FR]); z = d / np.array([x['complex_bg-complex'][1] for x in FR])
    bnum = dict(FrameBgWins=int((z > 2).sum()), FrameBgTies=int((abs(z) <= 2).sum()), FrameBgLosses=int((z < -2).sum()),
                FrameBgMed=f'{np.median(d):+.4f}', FrameBgShare=f"{100 * ob['shares'][1]:.1f}",
                FrameBgDTest=f"{ob['test'] - fr['complex']['test']:+.1e}".replace('e-05', '\\times10^{-5}'))
    open(P3 + 'numbers_em.tex', 'a').write(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in bnum.items()))
    print(bnum)

# band selection on the frame (frame_fc_gpu_job.py: fc grid, warm-started; frame_bg_gpu_job.py stream: fc learned)
# and the free blur on the double-slit references (dslit_job.py fraunhofer_blur / fresnel_blur, photons, flat bg)
FC = sorted((json.load(open(f)) for f in glob.glob('results/frame_fc/fc_*.json')), key=lambda d: d['mult'])
if FC:
    best = max(FC, key=lambda d: d['val']); at = {d['mult']: d for d in FC}
    st = json.load(open('results/frame_bg/stream_bg.partial.json'))['epochs'][-1]
    D = 'results/dslit_1d/photon/flat/photon_'; ph = {p: json.load(open(D + p + '.json')) for p in
                                                      ('fraunhofer', 'fraunhofer_blur', 'fresnel', 'fresnel_blur')}
    cnum = dict(FrameFcLo=f"{FC[0]['mult']:.1f}", FrameFcHi=f"{FC[-1]['mult']:.2f}", FrameFcBest=f"{best['mult']:.1f}",
                FrameFcGap=f"{1e4 * (best['val'] - at[1.0]['val']):.0f}\\times10^{{-4}}",
                FrameFcWinLoss=f"{1e3 * (best['val'] - at[1.22]['val']):.0f}\\times10^{{-3}}",
                FrameCert=f"{at[1.0]['certificate']:.2f}", FrameStreamFc=f"{st['fc'][0] / at[1.0]['fc']:.2f}",
                FrameStreamEpochs=st['epoch'],
                PhBlurSig=f"{ph['fraunhofer_blur']['params']['sigma']:.1f}",
                PhVisFr=f"{ph['fraunhofer']['params']['V']:.2f}", PhVisFrBlur=f"{ph['fraunhofer_blur']['params']['V']:.2f}",
                PhBlurDMax=f"{max(abs(ph[p + '_blur']['test_ll'] - ph[p]['test_ll']) for p in ('fraunhofer', 'fresnel')) * 1e5:.0f}\\times10^{{-5}}")
    open(P3 + 'numbers_em.tex', 'a').write(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in cnum.items()))
    print(cnum)

# bead windows with the background as a smooth second mode (results/p3_window_bgmode.py), against the window KDE
if os.path.exists('results/p3_window_bgmode.json'):
    wb = json.load(open('results/p3_window_bgmode.json'))
    wnum = dict(WinBgWins=wb['wins'], WinBgTies=wb['ties'], WinBgLosses=wb['losses'], WinBgN=len(wb['rows']),
                WinBgMed=f"{wb['median']:+.4f}", WinBgShare=f"{100 * np.median([r['bg_share'] for r in wb['rows']]):.0f}")
    open(P3 + 'numbers_em.tex', 'a').write(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in wnum.items()))
    print(wnum)
