"""CPU speed and memory baseline in 5-d (REFACTOR_KERNELS.md stage 0; rerun after every stage). Each (method, n)
runs in its own process: wall time, peak RSS during the fit above the RSS just before it (the parent samples the
child's RSS every 20 ms), bins B, sweeps
or steps. Methods: one-shot BLMLQuick at fixed fc (time per sweep), streaming BLMLQuickStreaming with fc trained
(time per epoch), streaming with a mapped sinc unit 5 -> 2 (W trained). A method stops growing n once one run
exceeds BENCH_CAP seconds. Prints log-log slopes vs n. Writes results/bench_cpu5d.json (or BENCH_OUT).
    NUMBA_NUM_THREADS=2 python results/bench_cpu5d.py           # all
    python results/bench_cpu5d.py one <method> <n>             # one run (internal)"""
import os, sys, json, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np

D = 5
NS = [int(v) for v in os.environ.get('BENCH_NS', '2000,4000,8000,16000,32000').split(',')]
CAP = float(os.environ.get('BENCH_CAP', 120))
OUT = os.environ.get('BENCH_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bench_cpu5d.json'))


def data(n, seed=0):
    """A correlated 3-component Gaussian mixture in 5-d, unit-ish scale."""
    rng = np.random.default_rng(seed); A = np.linalg.qr(rng.normal(size=(D, D)))[0] * np.array([1, .8, .6, .5, .4])
    mu = rng.normal(size=(3, D)) * 1.5; c = rng.integers(0, 3, n)
    return mu[c] + rng.normal(size=(n, D)) @ A.T


def rss_of(pid):
    """Current resident set of a process, MB (ps: KB)."""
    o = subprocess.run(['ps', '-o', 'rss=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
    return float(o) / 1024 if o else None


def run_one(method, n):
    """The child prints START / END around the timed fit; the parent samples its RSS in between."""
    p = subprocess.Popen([sys.executable, __file__, 'one', method, str(n)], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    import threading
    state = dict(base=None, peak=0.0, on=False, done=False)

    def sample():
        while not state['done']:
            r = rss_of(p.pid)
            if r is not None and state['on']:
                state['peak'] = max(state['peak'], r)
            time.sleep(0.02)
    th = threading.Thread(target=sample, daemon=True); th.start()
    res = None
    for line in p.stdout:
        if line.startswith('START'):
            state['base'] = rss_of(p.pid); state['peak'] = state['base']; state['on'] = True
        elif line.startswith('END'):
            state['on'] = False
        elif line.startswith('{'):
            res = json.loads(line)
    p.wait(); state['done'] = True
    if res is None:
        print(method, n, 'FAILED', p.stderr.read()[-800:], flush=True)
        return None
    res['mem_mb'] = state['peak'] - state['base']; res['rss_base_mb'] = state['base']
    return res


def one(method, n):
    import warnings; warnings.filterwarnings('ignore')
    from blml_trivial import BLMLQuick
    from blml_rolling import BLMLQuickStreaming
    X = data(n); Xw = data(2000, 1)
    fc = 0.8
    if method == 'oneshot':                                   # compile on a small fit first
        BLMLQuick().fit(Xw, [{j: dict(fc=fc) for j in range(D)}])
    else:
        _stream(method, Xw, fc)
    print('START', flush=True); time.sleep(0.1); t = time.time()
    if method == 'oneshot':
        m = BLMLQuick().fit(X, [{j: dict(fc=fc) for j in range(D)}])
        r = dict(B=int(m.n_bins_), sweeps=int(m.n_sweeps_))
    else:
        s = _stream(method, X, fc)
        r = dict(B=int(len(s.ff.cnt)), steps=sum(1 for h in s.history_ if 'preq' in h))
    r.update(s=time.time() - t, n=n, method=method)
    time.sleep(0.05); print('END', flush=True)
    print(json.dumps(r), flush=True)


def _stream(method, X, fc):
    from blml_rolling import BLMLQuickStreaming
    if method == 'stream':
        modes = [{j: dict(fc=dict(value=fc, trainable=True)) for j in range(D)}]
    else:                                                     # projected: 5 inputs -> k = 2, W trained
        modes = [{tuple(range(D)): dict(kind='sinc', k=2, fc=[fc, fc])}]
    return BLMLQuickStreaming(burn_in=1000, lr=0.05, param_every=1).fit(X, modes, epochs=1, batch=500)


def main():
    out = {}
    for method in ('oneshot', 'stream', 'projected'):
        rows = []
        for n in NS:
            r = run_one(method, n)
            if r is None:
                break
            rows.append(r)
            per = r['s'] / max(r.get('sweeps', 1), 1) if method == 'oneshot' else r['s']
            print(f"{method:9s} n={n:6d} B={r['B']:6d} {'sweeps' if method == 'oneshot' else 'steps'}="
                  f"{r.get('sweeps', r.get('steps'))} time {r['s']:7.2f}s ({'per sweep' if method == 'oneshot' else 'epoch'}"
                  f" {per:.3f}s) peak mem +{r['mem_mb']:.0f} MB", flush=True)
            if r['s'] > CAP:
                break
        if len(rows) >= 3:
            ln = np.log([r['n'] for r in rows])
            tt = [r['s'] / max(r.get('sweeps', 1), 1) if method == 'oneshot' else r['s'] for r in rows]
            sl_t = np.polyfit(ln, np.log(tt), 1)[0]
            mm = [max(r['mem_mb'], 1.0) for r in rows]; sl_m = np.polyfit(ln, np.log(mm), 1)[0]
            sl_b = np.polyfit(ln, np.log([r['B'] for r in rows]), 1)[0]
            print(f"{method}: slope vs n -- time {sl_t:.2f}, memory {sl_m:.2f}, bins {sl_b:.2f}", flush=True)
            out[method] = dict(rows=rows, slope_time=sl_t, slope_mem=sl_m, slope_bins=sl_b)
        else:
            out[method] = dict(rows=rows)
        json.dump(out, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'one':
        one(sys.argv[2], int(sys.argv[3]))
    else:
        main()
