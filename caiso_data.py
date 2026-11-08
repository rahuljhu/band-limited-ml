"""CAISO hourly price data with calendar covariates, jittered for BLML.

Every recorded quantity is quantized (hour, day, weekday and month are integers;
price is recorded to 1e-5 $/MWh). BLML would read that quantization as real
high-frequency structure, so each column gets uniform jitter over one
quantization step: x + U(0, step), wrapped for periodic columns.

The .npy files hold price, hour and day-of-year in time order, starting
2023-06-29 00:00 with one year wrap. Calendar dates are rebuilt from that, so
day-of-week and month are exact (the earlier formula dow = (doy-180+3) % 7 was
off for every 2024 row).
"""
import os
import numpy as np

DATA_DIR = os.environ.get('BLML_DATA_DIR', os.path.expanduser('~/data_repo/caiso'))
PRICE_STEP = 1e-5
START_YEAR = 2023

# (period, kind) per covariate; price is the non-periodic dependent variable.
PERIODS = {'hour': 24, 'doy': 365, 'dow': 7, 'moy': 12}


def _dates(doy):
    """datetime64[D] per row from day-of-year in time order (one wrap per new year)."""
    year = START_YEAR + np.concatenate([[0], np.cumsum(np.diff(doy) < -300)])
    jan1 = (year - 1970).astype('datetime64[Y]').astype('datetime64[D]')
    return jan1 + (doy.astype(np.int64) - 1)


def load(data_dir=DATA_DIR, legacy=False):
    """Return a dict of jittered columns: price, hour, doy, dow, moy.

    legacy=True reproduces the earlier notebook exactly: same seeds, the old
    approximate day-of-week, non-leap month bounds, and no price jitter.
    """
    p = np.load(os.path.join(data_dir, 'caiso_3d_price.npy'))
    h = np.load(os.path.join(data_dir, 'caiso_3d_hour.npy'))
    dy = np.load(os.path.join(data_dir, 'caiso_3d_doy.npy'))
    rng = np.random.default_rng(1)
    hj = (h + rng.random(len(h))) % 24
    dyj = (dy + rng.random(len(dy))) % 365
    if legacy:
        dow = (dy - 180 + 3) % 7
        bounds = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
        moy = np.digitize(dy, [0] + bounds)                   # 1..12
        price = p
    else:
        dates = _dates(dy)
        dow = (dates.astype(np.int64) + 3) % 7                # Mon=0 (1970-01-01 was a Thursday)
        moy = dates.astype('datetime64[M]').astype(np.int64) % 12 + 1
        price = p + np.random.default_rng(4).random(len(p)) * PRICE_STEP
    rng = np.random.default_rng(2); dowj = (dow + rng.random(len(dow))) % 7
    rng = np.random.default_rng(3); moyj = (moy + rng.random(len(moy))) % 12
    cols = {'price': price, 'hour': hj, 'doy': dyj, 'dow': dowj, 'moy': moyj}
    # Lagged prices, known at day-ahead forecast time: the price L hours earlier, looked up by
    # timestamp (two days have 23 hours); a missing hour falls back to the one before it.
    ts = timestamps(data_dir)
    pos = {t: i for i, t in enumerate(ts)}
    for L in (24, 168):
        lag = np.full(len(price), np.nan)
        for i, t in enumerate(ts):
            j = pos.get(t - np.timedelta64(L, 'h'), pos.get(t - np.timedelta64(L + 1, 'h')))
            if j is not None:
                lag[i] = price[j]
        cols[f'lag{L}'] = lag
    return cols


def timestamps(data_dir=DATA_DIR):
    """datetime64[h] per row (local clock hour, as recorded)."""
    h = np.load(os.path.join(data_dir, 'caiso_3d_hour.npy'))
    dy = np.load(os.path.join(data_dir, 'caiso_3d_doy.npy'))
    return _dates(dy).astype('datetime64[h]') + h.astype('timedelta64[h]')


def rolling_folds(data_dir=DATA_DIR, test_months=('2024-06', '2024-07', '2024-08',
                                                  '2024-09', '2024-10', '2024-11')):
    """Rolling-origin folds: train on every row before the test month, test on it."""
    ts = timestamps(data_dir)
    month = ts.astype('datetime64[M]')
    folds = []
    for m in test_months:
        m = np.datetime64(m, 'M')
        folds.append((m.astype(str), np.where(month < m)[0], np.where(month == m)[0]))
    return folds


def split(n, seed=0, frac=0.8):
    """The random 80/20 row split used so far (interpolation, not forecasting)."""
    idx = np.random.default_rng(seed).permutation(n)
    return idx[:int(frac * n)], idx[int(frac * n):]


def split_days(ts, seed=0, frac=0.8):
    """Random 80/20 split of calendar days: whole days are held out, so no test hour has
    training hours from the same day (stationary, but no within-day interpolation)."""
    day = ts.astype('datetime64[D]')
    days = np.unique(day)
    te_days = np.random.default_rng(seed).choice(days, len(days) - int(frac * len(days)), replace=False)
    is_te = np.isin(day, te_days)
    return np.where(~is_te)[0], np.where(is_te)[0]


def design(cols, covariates):
    """(X, kinds, periods) for p(price | covariates); price is column 0."""
    X = np.stack([cols['price']] + [cols[c] for c in covariates], axis=1)
    kinds = ['sinc'] + ['dirichlet'] * len(covariates)
    periods = [None] + [PERIODS[c] for c in covariates]
    return X, kinds, periods
