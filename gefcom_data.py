"""GEFCom2014 price track (Hong et al. 2016): hourly zonal price with day-ahead forecasts of
zonal and total load, 2011-01-01 .. 2013-12-17. Fifteen tasks, each a day-ahead forecast of
24 hourly prices (99 quantiles, pinball loss) using all data before the target day.
Raw files: ~/data_repo/gefcom2014/raw/GEFCom2014 Data/P/Price (unzipped GEFCom2014-P_V2.zip)."""
import os, csv
import numpy as np

RAW = os.path.expanduser(os.environ.get('GEFCOM_DIR', '~/data_repo/gefcom2014/raw/GEFCom2014 Data/P/Price'))
PERIODS = {'hour': 24, 'dow': 7, 'moy': 12}

def _read(path):
    with open(path, newline='') as f:
        rows = list(csv.reader(f))[1:]
    out = []
    for r in rows:
        if not r or not r[0].strip():
            continue
        d, t = r[1].split()
        ts = np.datetime64(f'{d[4:8]}-{d[0:2]}-{d[2:4]}T{int(t.split(":")[0]):02d}', 'h')
        out.append((ts, float(r[2]), float(r[3]), float(r[4]) if len(r) > 4 and r[4].strip() else np.nan))
    return out

KEYS = ('price', 'zload', 'tload', 'hour', 'dow', 'moy', 'lag24', 'lag168')

def cache(out_dir, raw=RAW):
    """Write the parsed data as gefcom_*.npy (shipped to Colab with the CAISO arrays)."""
    cols, ts, tasks = load(raw)
    for k in KEYS:
        np.save(os.path.join(out_dir, f'gefcom_{k}.npy'), cols[k])
    np.save(os.path.join(out_dir, 'gefcom_ts.npy'), ts.astype('int64'))
    np.save(os.path.join(out_dir, 'gefcom_targets.npy'), np.stack([te for _, _, te in tasks]))

def load(raw=RAW):
    """cols (dict of arrays over all hours) and tasks [(k, train_idx, test_idx)]. `raw` is
    the unzipped price-track folder, or a folder holding the gefcom_*.npy cache."""
    if os.path.exists(os.path.join(raw, 'gefcom_price.npy')):
        cols = {k: np.load(os.path.join(raw, f'gefcom_{k}.npy')) for k in KEYS}
        ts = np.load(os.path.join(raw, 'gefcom_ts.npy')).astype('datetime64[h]')
        tg = np.load(os.path.join(raw, 'gefcom_targets.npy'))
        return cols, ts, [(k + 1, np.arange(168, tg[k][0]), tg[k]) for k in range(len(tg))]
    rows = _read(os.path.join(raw, 'Task 15', 'Task15_P.csv'))
    ts = np.array([r[0] for r in rows]); price = np.array([r[3] for r in rows])
    pos = {t: i for i, t in enumerate(ts)}
    targets = []
    for k in range(1, 16):
        tk = [r[0] for r in _read(os.path.join(raw, f'Task {k}', f'Task{k}_P.csv'))[-24:]]
        targets.append(np.array([pos[t] for t in tk]))
    with open(os.path.join(raw, 'Solution to Task15', 'Solution to Task15_P.csv'), newline='') as f:
        sol = [float(r[2]) for r in list(csv.reader(f))[1:] if r and r[0].strip()]
    price[targets[-1]] = sol
    assert not np.isnan(price).any() and len(ts) % 24 == 0 and str(ts[0])[-2:] == '00'
    day = ts.astype('datetime64[D]')
    cols = dict(price=price, zload=np.array([r[2] for r in rows]), tload=np.array([r[1] for r in rows]),
                hour=(np.arange(len(ts)) % 24).astype(float),   # 24 rows per day (DST 2013-03-10 mislabels one)
                dow=((day.astype(int) + 3) % 7).astype(float),          # 1970-01-01 was a Thursday
                moy=(ts.astype('datetime64[M]').astype(int) % 12).astype(float))
    # Lagged prices, known at forecast time for a day-ahead forecast: the same hour one day and
    # one week earlier (24 rows per day, so a row lag of 24 is the same hour of the previous day). The first week has no lag-168 and is dropped.
    for L in (24, 168):
        lag = np.full(len(price), np.nan); lag[L:] = price[:-L]; cols[f'lag{L}'] = lag
    tasks = [(k + 1, np.arange(168, targets[k][0]), targets[k]) for k in range(15)]
    return cols, ts, tasks

def leaderboard(xlsx=None):
    """{team: {task: pinball}} from the provisional leaderboard (P-score-* sheets)."""
    import pandas as pd
    xlsx = xlsx or os.path.join(os.path.dirname(RAW.rstrip('/')).rsplit('/P', 1)[0], 'Provisional_Leaderboard_V2.xlsx')
    long = pd.read_excel(xlsx, sheet_name='P-score')
    return long
