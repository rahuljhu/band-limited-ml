"""Photon transfer on SID Sony raw pairs (two independent short exposures of one scene): per Bayer channel, bins of
pixels by the long-exposure brightness; var((a - b) / sqrt 2) against mean((a + b) / 2) above black: slope 1 / gain
(ADU per electron -> e/ADU = 1 / slope), intercept read noise^2 (ADU). Output results/sid_photon_transfer.json"""
import json, numpy as np, rawpy
D = __import__('os').path.expanduser('~/data_repo/sid'); out = {}
for s in ('10003', '10011', '10022'):
    L = rawpy.imread(f'{D}/{s}_00_10s.ARW').raw_image_visible.astype(float) - 512
    for t in ('0.1s', '0.04s'):
        a = rawpy.imread(f'{D}/{s}_00_{t}.ARW').raw_image_visible.astype(float) - 512
        b = rawpy.imread(f'{D}/{s}_01_{t}.ARW').raw_image_visible.astype(float) - 512
        rows = []
        for cy, cx in ((0, 0), (0, 1), (1, 0), (1, 1)):
            A, B, R = a[cy::2, cx::2].ravel(), b[cy::2, cx::2].ravel(), L[cy::2, cx::2].ravel()
            q = np.quantile(R, np.linspace(0.02, 0.98, 40)); idx = np.digitize(R, q)
            m = np.array([(A[idx == k] + B[idx == k]).mean() / 2 for k in range(1, 40)])
            v = np.array([np.var(A[idx == k] - B[idx == k]) / 2 for k in range(1, 40)])
            ok = (m > 0.5) & (m < 0.9 * 16000)
            sl, ic = np.polyfit(m[ok], v[ok], 1); rows.append(dict(ch=f'{cy}{cx}', e_per_adu=1 / sl, read_adu=float(np.sqrt(max(ic, 0))), max_mean=float(m.max())))
        mean_e = float(np.mean(a) * np.mean([r_['e_per_adu'] for r_ in rows]))
        out[f'{s}_{t}'] = dict(channels=rows, mean_photoelectrons_per_px=mean_e,
                               long_over_short=float(np.mean(L) / max((np.mean(a) + np.mean(b)) / 2, 1e-9)))
        print(s, t, ' '.join(f"{r_['ch']}: {r_['e_per_adu']:.2f} e/ADU read {r_['read_adu']:.2f}" for r_ in rows),
              f"| mean {mean_e:.2f} e/px, long/short {out[f'{s}_{t}']['long_over_short']:.0f}", flush=True)
json.dump(out, open('results/sid_photon_transfer.json', 'w'), indent=1)
