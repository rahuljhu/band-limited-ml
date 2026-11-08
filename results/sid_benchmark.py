"""Real low-light raw images (SID Sony, Chen et al. 2018): BLML on each Bayer channel's own sub-lattice (R, G1, G2, B;
spacing 2 px) against demosaicing + the best Gaussian blur. Scenes 10003 / 10011 / 10022, exposures 0.1 s and 0.04 s;
a 512 x 512 raw crop (the most textured block of the long exposure, away from saturation).
Real data only: electrons = (ADU - black) x the per-channel photon-transfer gain (results/sid_photon_transfer.json),
clipped at 0 (the one change to the data: read noise makes dark pixels negative and a Poisson model needs counts >= 0;
its bias is reported as clip_bias_e, the mean added by the clip). No floor, no rounding, no thinning: frame _00 is
fitted, frame _01 (an independent exposure) selects fc, lam and sigma. Crop: the most textured 512 x 512 block among
the brightest quarter of blocks (so the clip bias is small). Truth: the 10 s exposure's own electrons, normalised
(no fitted scale; long_vs_short_fit records short = a long + b as a check of the offset).
Metrics, per method: each channel's estimate and the 10 s frame normalised to distributions over the unsaturated sites
  BLML predictions are photosite probabilities (f integrated over each 1 x 1 px pixel), not values at pixel centres.
  denoise   at each channel's own sites: PSNR of the normalised maps; LL = mean log-probability of the 10 s photons
  interp    green at the G2 sites predicted from the G1 sub-lattice alone (and G1 from G2): the same two scores there
            (sites the fit never saw, real held-out photons)
Output results/sid_benchmark.json, papers/paper7/figs/sid_<scene>_<exposure>.png"""
import os, sys, json, time, numpy as np, rawpy
sys.path.insert(0, '.')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter, laplace
import blml_trivial as bt
from blml_binned import BLMLBinned, BLMLEmpirical, _np

D = __import__('os').path.expanduser('~/data_repo/sid'); PT = json.load(open('results/sid_photon_transfer.json'))
CH = {'R': (0, 0), 'G1': (0, 1), 'G2': (1, 0), 'B': (1, 1)}; W = 512
FCS = [0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25]   # cycles per px, capped at each channel sub-lattice Nyquist 0.25 (above: aliased)
LAMS = [1e-8]; SIG = [0.0, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8]


def raw(f):
    return rawpy.imread(f'{D}/{f}.ARW').raw_image_visible.astype(float) - 512


def crop_origin(Lg):
    """The most textured W x W block (even origin) of the long exposure with < 0.1% saturated sites."""
    cands = []
    for y0 in range(0, Lg.shape[0] - W, W // 2):
        for x0 in range(0, Lg.shape[1] - W, W // 2):
            blk = Lg[y0:y0 + W, x0:x0 + W]
            if np.mean(blk > 15000) > 1e-3:
                continue
            e = float(np.mean(np.abs(laplace(gaussian_filter(blk, 2.0)))) / (blk.mean() + 1))
            cands.append((float(blk.mean()), e, y0 - y0 % 2, x0 - x0 % 2))
    bright = np.quantile([c_[0] for c_ in cands], 0.75)
    best = max((c_ for c_ in cands if c_[0] >= bright), key=lambda c_: c_[1])
    return best[2], best[3]


def sub(A, c):
    cy, cx = CH[c]; return A[cy::2, cx::2]


def full_grid(m, c):
    """A sub-lattice fit (h = 2, origin = the channel's offset) on every pixel of the crop: predict_grid(2) steps 1 px
    from the channel's first site; pixels before it take the edge value."""
    cy, cx = CH[c]; G = m.predict_grid(2); out = np.zeros((W, W))
    out[cy:cy + G.shape[0], cx:cx + G.shape[1]] = G[:W - cy, :W - cx]
    if cy:
        out[0] = out[1]
    if cx:
        out[:, 0] = out[:, 1]
    return out


def pixel_grid(m, c, pixel=1.0, q=4):
    """The fit's probability per photosite (the integral of f over each 1 x 1 px pixel, Simpson at 0.25 px; see
    results/halfnyq_vs_hist.py pixel_integrated) at every pixel of the crop: a sub-lattice fit evaluated on a grid
    h q / pixel times finer, Simpson-weighted around every pixel centre (step q fine points = 1 px)."""
    import itertools
    cy, cx = CH[c]; h = float(m.h_[0]); r = int(round(h * q / pixel)); G = m.predict_grid(r)
    w1 = np.ones(q + 1); w1[1:-1:2] = 4; w1[2:-1:2] = 2; w1 = w1 / w1.sum(); k = q // 2
    Gp = np.pad(G, [(k, k)] * 2, mode='constant', constant_values=np.nan); acc = 0.0; ws = 0.0
    for oy, ox in itertools.product(range(-k, k + 1), repeat=2):
        v = Gp[k + oy:k + oy + G.shape[0]:q, k + ox:k + ox + G.shape[1]:q]; w = w1[oy + k] * w1[ox + k]; ok = ~np.isnan(v)
        acc = acc + np.where(ok, v, 0.0) * w; ws = ws + ok * w
    P = acc / ws * pixel ** 2; out = np.zeros((W, W))
    out[cy:cy + P.shape[0], cx:cx + P.shape[1]] = P[:W - cy, :W - cx]
    if cy:
        out[0] = out[1]
    if cx:
        out[:, 0] = out[:, 1]
    return out


def interp_grid(M, c, order):
    """Spline interpolation of a channel's sub-lattice values M onto every pixel of the crop."""
    cy, cx = CH[c]; yy, xx = np.meshgrid((np.arange(W) - cy) / 2, (np.arange(W) - cx) / 2, indexing='ij')
    return map_coordinates(M, [yy, xx], order=order, mode='nearest')


def loglik(pred, cnt, floor):
    p = np.maximum(pred + floor, 1e-6); p = p / p.sum()
    return float((cnt * np.log(p)).sum()), float(cnt.sum())


res = {}
for scene in ('10003', '10011', '10022'):
    Lfull = raw(f'{scene}_00_10s'); y0, x0 = crop_origin(Lfull); Lc = Lfull[y0:y0 + W, x0:x0 + W]
    for expo in ('0.1s', '0.04s'):
        t0 = time.time(); pt = PT[f'{scene}_{expo}']
        g = {ch['ch']: ch['e_per_adu'] for ch in pt['channels']}; rd = {ch['ch']: ch['read_adu'] for ch in pt['channels']}
        A = raw(f'{scene}_00_{expo}')[y0:y0 + W, x0:x0 + W]; Bf = raw(f'{scene}_01_{expo}')[y0:y0 + W, x0:x0 + W]
        cnt, val, tst, floor, truth = {}, {}, {}, {}, {}; clip_bias = {}; intercept = {}
        for c, (cy, cx) in CH.items():
            k = f'{cy}{cx}'; floor[c] = 0.0
            ea = sub(A, c) * g[k]; cnt[c] = np.clip(ea, 0, None); clip_bias[c] = float(cnt[c].mean() - ea.mean())
            val[c] = np.clip(sub(Bf, c) * g[k], 0, None); tst[c] = val[c]
            mean_short = (sub(A, c) + sub(Bf, c)) / 2 * g[k]; Ls = sub(Lc, c); ok = Ls < 15000
            a_, b_ = np.polyfit(Ls[ok], mean_short[ok], 1)                 # diagnostic only: is the long frame offset?
            intercept[c] = dict(a=float(a_), b_e=float(b_), b_over_mean=float(b_ / max(mean_short[ok].mean(), 1e-9)))
            truth[c] = np.clip(Lc * g[k], 0, None)                          # the 10 s frame's electrons, every pixel
        unsat = Lc < 15000                                                   # saturated sites are left out of every score
        out = {}

        def evaluate(name, pred):
            """pred[c]: the channel's expected electrons on every pixel of the crop, from frame _00. Scored against the
            10 s frame with no fitted scale: per channel (and set of sites) both are normalised to distributions over the
            unsaturated sites (x the number of sites: mean 1). PSNR of the normalised maps (peak = the 10 s frame's max);
            LL = the 10 s photons' mean log-probability under the normalised estimate."""
            def parts(Pv, Tv, m):
                Pv, Tv = Pv[m], Tv[m]; q = np.maximum(Pv, 1e-9 * max(Pv.mean(), 1e-12)); q = q / q.sum(); t = Tv / Tv.sum()
                return q * len(q), t * len(t), float((Tv * np.log(q)).sum()), float(Tv.sum())
            se = ne = si = ni = 0.0; ll = n_ = lli = ni_ = 0.0; pk = pki = 0.0
            for c in CH:
                q, t, l_, m_ = parts(sub(pred[c], c), sub(truth[c], c), sub(unsat, c))
                se += float(((q - t) ** 2).sum()); ne += q.size; pk = max(pk, float(t.max())); ll += l_; n_ += m_
            for c_from, c_to in (('G1', 'G2'), ('G2', 'G1')):            # green at the other green sub-lattice
                q, t, l_, m_ = parts(sub(pred[c_from + '->'], c_to), sub(truth[c_to], c_to), sub(unsat, c_to))
                si += float(((q - t) ** 2).sum()); ni += q.size; pki = max(pki, float(t.max())); lli += l_; ni_ += m_
            out[name] = dict(denoise_psnr=float(10 * np.log10(pk ** 2 / (se / ne))), interp_psnr=float(10 * np.log10(pki ** 2 / (si / ni))),
                             denoise_ll_10s=ll / n_, interp_ll_10s=lli / ni_, time_s=time.time() - tt)
            print(f"{scene} {expo} {name:26s} denoise {out[name]['denoise_psnr']:.2f} dB  LL {out[name]['denoise_ll_10s']:.5f} | "
                  f"interp {out[name]['interp_psnr']:.2f} dB  LL {out[name]['interp_ll_10s']:.5f}  {out[name]['time_s']:.1f}s", flush=True)

        def val_score(Psub, c):
            l_, m_ = loglik(Psub, val[c], floor[c]); return l_ / m_

        # ---- baselines: spline demosaic (per channel) + Gaussian blur, sigma on val
        for name, order in (('bilinear + blur', 1), ('bicubic + blur', 3)):
            tt = time.time(); pred = {}
            for c in CH:
                M = cnt[c] - floor[c]; best = None
                for sg in SIG:
                    Mb = gaussian_filter(M, sg / 2) if sg else M
                    s = val_score(Mb, c)
                    if best is None or s > best[0]:
                        best = (s, Mb)
                pred[c] = interp_grid(best[1], c, order)
                if c in ('G1', 'G2'):
                    pred[c + '->'] = pred[c]
            evaluate(name, pred)
        # ---- BLML: empirical (+sqrt h, complex) and binned complex, per channel, (fc, lam) on val
        for name in ('BLML empirical +sqrt h', 'BLML empirical complex', 'BLML binned complex'):
            tt = time.time(); pred = {}
            for c, (cy, cx) in CH.items():
                best = None
                grid = [(fc, LAMS) for fc in FCS]
                if name == 'BLML empirical complex':                 # around the +sqrt h choice for this channel
                    f0, l0 = out['BLML empirical +sqrt h params'][c]['fc'], out['BLML empirical +sqrt h params'][c]['lam']
                    grid = [(fc, LAMS) for fc in FCS if 0.6 * f0 <= fc <= 1.7 * f0]
                if name == 'BLML binned complex':                    # the slow one: the empirical choice and its neighbours
                    j0 = FCS.index(out['BLML empirical complex params'][c]['fc'])
                    grid = [(fc, LAMS) for fc in FCS[max(j0 - 1, 0):j0 + 2]]
                for fc, lg in grid:
                    modes = bt.make_modes([fc, fc], ['sinc', 'sinc'])
                    if name == 'BLML binned complex':
                        cands = [BLMLBinned(tol=1e-5, max_sweeps=500, ll_tol=1e-9).fit(cnt[c], modes, h=2.0, origin=(cy, cx), complex_g=True)]
                    else:
                        cands, prev = [], None
                        for lam in lg:
                            m = BLMLEmpirical(lam=lam, tol=1e-5, max_iter=20000, **(dict(max_outer=20, outer_tol=0) if name == 'BLML empirical complex' else {})).fit(cnt[c], modes, h=2.0, origin=(cy, cx), init=prev,
                                                                                    complex_g=(name == 'BLML empirical complex'))
                            cands.append(m); prev = m
                    for m in cands:
                        Fsub = sub(pixel_grid(m, c), c) * cnt[c].sum()            # photosite probability x n (scale cancels in the scores)
                        s = val_score(Fsub, c)
                        if best is None or s > best[0]:
                            best = (s, m, fc, getattr(m, 'lam', None))
                m = best[1]; pred[c] = pixel_grid(m, c) * cnt[c].sum()
                if c in ('G1', 'G2'):
                    pred[c + '->'] = pred[c]
                out.setdefault(name + ' params', {})[c] = dict(fc=best[2], lam=best[3])
            evaluate(name, pred)
            if name == 'BLML empirical complex':
                keep_pred = pred
        res[f'{scene}_{expo}'] = dict(out, crop=[int(y0), int(x0)], clip_bias_e=clip_bias, long_vs_short_fit=intercept,
                                      mean_e={c: float(np.mean(np.clip(sub(A, c), None, None)) * g[f'{CH[c][0]}{CH[c][1]}']) for c in CH})
        print(f"{scene} {expo} crop {y0},{x0}: mean e- per site " + ' '.join(f"{c} {res[f'{scene}_{expo}']['mean_e'][c]:.2f}" for c in CH)
              + '  clip bias ' + ' '.join(f"{c} {clip_bias[c]:.2f}" for c in CH)
              + '  | short = a long + b: b/mean ' + ' '.join(f"{c} {intercept[c]['b_over_mean']:+.3f}" for c in CH), flush=True)
        # ---- figure: linear RGB (white-balanced on the truth), gamma 1/2.2
        def rgb(P):
            Gm = (P['G1'] + P['G2']) / 2; img = np.stack([P['R'], Gm, P['B']], -1)
            wb = np.array([truth_img['R'].mean(), (truth_img['G1'].mean() + truth_img['G2'].mean()) / 2, truth_img['B'].mean()])
            img = img / wb; img = np.clip(img / np.percentile(rgb_ref, 99.5), 0, 1); return img ** (1 / 2.2)
        raw_pred = {c: interp_grid(cnt[c] - floor[c], c, 1) for c in CH}
        truth_img = {c: interp_grid(sub(truth[c], c), c, 1) for c in CH}       # the 10 s mosaic demosaiced for display
        wb_ = np.array([truth_img['R'].mean(), (truth_img['G1'].mean() + truth_img['G2'].mean()) / 2, truth_img['B'].mean()])
        rgb_ref = np.stack([truth_img['R'], (truth_img['G1'] + truth_img['G2']) / 2, truth_img['B']], -1) / wb_   # display scale
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
        for a_, (tl, P) in zip(ax, (('short frame, bilinear demosaic', raw_pred), ('BLML empirical complex', keep_pred), ('10 s exposure (truth)', truth_img))):
            a_.imshow(rgb(P)); a_.set_title(tl, fontsize=10); a_.set_xticks([]); a_.set_yticks([])
        fig.suptitle(f'SID {scene}, {expo} ({pt["mean_photoelectrons_per_px"]:.1f} e-/px mean, read noise ~{np.mean([rd[k] * g[k] for k in g]):.1f} e-)', fontsize=10)
        plt.tight_layout(); os.makedirs('papers/paper7/figs', exist_ok=True); plt.savefig(f'papers/paper7/figs/sid_{scene}_{expo}.png', dpi=110); plt.close(fig)
        json.dump(res, open('results/sid_benchmark.json', 'w'), indent=1)
        print(f'{scene} {expo} done in {time.time() - t0:.0f}s', flush=True)
