"""Paper 1 experiments (GPU). Run on Colab via:  ./colab_job.sh start p1 p1_experiments.py

Each experiment writes outputs/<name>.json and is skipped if that file exists,
so a restarted job resumes. Order: fast systems experiments first, then the
rate study (R1), which is the long one.
"""
import os, sys, json, time, subprocess, math

OUT = os.environ.get('BLML_OUT_DIR', 'outputs_p1')
CODE = os.environ.get('BLML_CODE_DIR', os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT, exist_ok=True)
if subprocess.run([sys.executable, '-c', 'import cupy'], capture_output=True).returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda13x', 'numba'], check=True)
sys.path.insert(0, CODE)

import numpy as np
import cupy as cp
import blml_trivial as bt
from blml_trivial import make_modes
from blml_trivial import BLMLQuick
from blml_trivial import BLMLQuick
from p1_truth import Truth, DefensiveProposal, FC_TRUE
import p1_estimators as E
import caiso_data

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def path(name):
    return os.path.join(OUT, name + '.json')
def done(name):
    return os.path.exists(path(name))
def save(name, obj):
    os.makedirs(os.path.dirname(path(name)), exist_ok=True)
    tmp = path(name) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path(name))
    log(f'saved {name}')
def gpu_sync():
    cp.cuda.Stream.null.synchronize()
def timed(f):
    gpu_sync(); t0 = time.time(); r = f(); gpu_sync(); return r, time.time() - t0
def synth(n, d, seed):
    return Truth('T1', d).sample(np.random.default_rng(seed), n)
def warmup():
    for dt in (np.float32, np.float64):
        BLMLQuick(engine='gpu', dtype=dt).fit(synth(500, 2, 0), make_modes(FC_TRUE, ['sinc'] * 2, None))
def monotone(h, rtol=1e-6):
    h = np.asarray(h)
    return bool(np.all(np.diff(h) <= rtol * np.abs(h[:-1]) + 1e-12))

log(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
warmup()
bt.DIRICHLET_LEGACY = False

# ======================================================== systems experiments
RUN_SYS = os.environ.get('BLML_P1_PARTS', 'all') not in ('rate', 'bign')
if RUN_SYS and not done('expA_traces'):
    out = {}
    for n, d in [(3000, 1), (3000, 2), (10000, 3), (3000, 8)]:
        m, t = timed(lambda: BLMLQuick(engine='gpu', dtype=np.float64).fit(synth(n, d, 1), make_modes(FC_TRUE, ['sinc'] * d, None)))
        out[f'n{n}_d{d}'] = dict(n=n, d=d, time_s=t, sweeps=m.n_sweeps_, B=m.n_bins_,
                                 rho=m.rho_history_, obj=m.obj_history_, obj_monotone=monotone(m.obj_history_))
    save('expA_traces', out)

if RUN_SYS and not done('expB_time_vs_n'):
    out = {'n': [], 'B': [], 'gpu_f32_s': [], 'gpu_f64_s': [], 'sweeps': []}
    for n in [1000, 3000, 10000, 30000, 100000]:
        X = synth(n, 3, 2)
        m32, t32 = timed(lambda: BLMLQuick(engine='gpu').fit(X, make_modes(FC_TRUE, ['sinc'] * 3, None)))
        m64, t64 = timed(lambda: BLMLQuick(engine='gpu', dtype=np.float64).fit(X, make_modes(FC_TRUE, ['sinc'] * 3, None)))
        for k, v in zip(out, [n, m64.n_bins_, t32, t64, m64.n_sweeps_]):
            out[k].append(v)
        log(f'B: n={n} B={m64.n_bins_} f32 {t32:.2f}s f64 {t64:.2f}s')
    save('expB_time_vs_n', out)

if RUN_SYS and not done('expCE_dim'):
    out = []
    for d in [1, 2, 3, 4, 6, 8, 12, 16]:
        X = synth(3000, d, 3)
        m, t = timed(lambda: BLMLQuick(engine='gpu', dtype=np.float64).fit(X, make_modes(FC_TRUE, ['sinc'] * d, None)))
        out.append(dict(d=d, n=3000, B=m.n_bins_, B_over_n=m.n_bins_ / 3000, time_s=t, sweeps=m.n_sweeps_,
                        obj_monotone=monotone(m.obj_history_)))
        log(f'C/E: d={d} B={m.n_bins_} t={t:.2f}s sweeps={m.n_sweeps_}')
    for d in [1, 2, 3, 4, 6, 8, 12, 16]:          # timing where kernel work dominates launches
        X = synth(30000, d, 3)
        m, t = timed(lambda: BLMLQuick(engine='gpu', dtype=np.float64).fit(X, make_modes(FC_TRUE, ['sinc'] * d, None)))
        out.append(dict(d=d, n=30000, B=m.n_bins_, B_over_n=m.n_bins_ / 30000, time_s=t, sweeps=m.n_sweeps_,
                        obj_monotone=monotone(m.obj_history_)))
        log(f'C/E: n=30000 d={d} B={m.n_bins_} t={t:.2f}s sweeps={m.n_sweeps_}')
    save('expCE_dim', out)

if RUN_SYS and not done('expF_block'):
    X = synth(10000, 3, 4)
    out = {}
    for bs in [32, 64, 128, 256, 512]:
        m, t = timed(lambda: BLMLQuick(engine='gpu', dtype=np.float64, block_size=bs).fit(X, make_modes(FC_TRUE, ['sinc'] * 3, None)))
        out[bs] = dict(sweeps=m.n_sweeps_, time_s=t, obj_monotone=monotone(m.obj_history_),
                       final_obj=m.obj_history_[-1])
        log(f'F: bs={bs} sweeps={m.n_sweeps_} t={t:.2f}s monotone={out[bs]["obj_monotone"]}')
    save('expF_block', out)

if RUN_SYS and not done('expM_memory'):
    pool = cp.get_default_memory_pool()
    out = []
    for d in [3, 8]:
        for n in [1000, 3000, 10000, 30000, 100000]:
            X = synth(n, d, 5)
            pool.free_all_blocks()
            m = BLMLQuick(engine='gpu').fit(X, make_modes(FC_TRUE, ['sinc'] * d, None))
            peak = pool.total_bytes()
            out.append(dict(d=d, n=n, B=m.n_bins_, gpu_bytes=int(peak),
                            dense_gram_bytes=int(m.n_bins_) ** 2 * 4, data_bytes=n * d * 8))
            log(f'M: d={d} n={n} B={m.n_bins_} peak {peak/1e6:.1f} MB vs dense {m.n_bins_**2*4/1e9:.2f} GB')
    save('expM_memory', out)

if RUN_SYS and not done('expD_precision'):
    out = {}
    X = synth(3000, 3, 6); Xt = synth(20000, 3, 7)
    ref = BLMLQuick(engine='gpu', dtype=np.float64, tol=1e-9).fit(X, make_modes(FC_TRUE, ['sinc'] * 3, None))
    pr = ref.predict(Xt)
    big = pr > 1e-3 * pr.max()
    for dt, tols in [('float32', [1e-4, 1e-6]), ('float64', [1e-4, 1e-7])]:
        for tol in tols:
            m = BLMLQuick(engine='gpu', dtype=getattr(np, dt), tol=tol).fit(X, make_modes(FC_TRUE, ['sinc'] * 3, None))
            p = m.predict(Xt)
            rel = np.abs(p - pr) / np.maximum(pr, 1e-300)
            out[f'{dt}_tol{tol:g}'] = dict(max_rel=float(rel.max()), max_rel_where_f_gt_1em3max=float(rel[big].max()),
                                           max_abs=float(np.abs(p - pr).max()), sweeps=m.n_sweeps_,
                                           converged=m.converged_)
    save('expD_precision', out)

if RUN_SYS and not done('expS_cpu_vs_gpu'):
    # The one CPU comparison in the paper (Table S).
    out = []
    for n in [3000, 10000, 30000]:
        X = synth(n, 3, 8); Xt = synth(5000, 3, 9)
        mc, tc = timed(lambda: BLMLQuick().fit(X, make_modes(FC_TRUE, ['sinc'] * 3, None)))
        pc, tpc = timed(lambda: mc.predict(Xt))
        mg, tg = timed(lambda: BLMLQuick(engine='gpu').fit(X, make_modes(FC_TRUE, ['sinc'] * 3, None)))
        pg, tpg = timed(lambda: mg.predict(Xt))
        out.append(dict(workload=f'synthetic d=3 n={n}', B=mg.n_bins_, cpu_fit_s=tc, gpu_fit_s=tg,
                        cpu_pred_s=tpc, gpu_pred_s=tpg, fit_speedup=tc / tg, pred_speedup=tpc / tpg,
                        test_ll_cpu=float(np.log(np.maximum(pc, 1e-300)).mean()),
                        test_ll_gpu=float(np.log(np.maximum(pg, 1e-300)).mean())))
        log(f'S: n={n} fit {tc:.1f}s vs {tg:.2f}s ({tc/tg:.0f}x)')
    cols = caiso_data.load(os.environ.get('BLML_DATA_DIR'))
    X2, kinds, periods = caiso_data.design(cols, ['hour'])
    tr, va = caiso_data.split(len(X2))
    fc = [0.0731, 0.2671]
    mc, tc = timed(lambda: BLMLQuick().fit(X2[tr], make_modes(fc, kinds, periods)))
    pc, tpc = timed(lambda: mc.predict(X2[va]))
    mg, tg = timed(lambda: BLMLQuick(engine='gpu').fit(X2[tr], make_modes(fc, kinds, periods)))
    pg, tpg_first = timed(lambda: mg.predict(X2[va]))   # includes one-time GPU setup
    pg, tpg = timed(lambda: mg.predict(X2[va]))         # steady state
    out.append(dict(workload='CAISO f(price, hour) n=10001', B=mg.n_bins_, cpu_fit_s=tc, gpu_fit_s=tg,
                    gpu_pred_first_call_s=tpg_first, cpu_pred_s=tpc, gpu_pred_s=tpg, fit_speedup=tc / tg, pred_speedup=tpc / tpg,
                    test_ll_cpu=float(np.log(np.maximum(pc, 1e-300)).mean()),
                    test_ll_gpu=float(np.log(np.maximum(pg, 1e-300)).mean())))
    save('expS_cpu_vs_gpu', out)

if RUN_SYS and not done('expG_caiso3d'):
    cols = caiso_data.load(os.environ.get('BLML_DATA_DIR'))
    X3, kinds, periods = caiso_data.design(cols, ['hour', 'doy'])
    tr, va = caiso_data.split(len(X3))
    m, t = timed(lambda: BLMLQuick(engine='gpu').fit(X3[tr], make_modes([0.0731, 0.2671, 0.3299], kinds, periods)))
    save('expG_caiso3d', dict(fit_s=t, B=m.n_bins_, n=len(tr), sweeps=m.n_sweeps_))

if RUN_SYS and not done('expBin_check'):
    out = []
    for d in [1, 2]:
        t = Truth('T1', d)
        h = 0.5; g1 = np.arange(-200, 200 + h / 2, h)
        G = g1[:, None] if d == 1 else np.stack(np.meshgrid(g1, g1, indexing='ij'), -1).reshape(-1, 2)
        fg = t.pdf(G); int_f2 = h ** d * (fg ** 2).sum()
        for n in [1000, 10000]:
            for bs in [0.1, 0.025]:
                errs = []
                for rep in range(5):
                    X = t.sample(np.random.default_rng(rep), n)
                    m = BLMLQuick(engine='gpu', dtype=np.float64, bin_scale=bs).fit(X, make_modes(FC_TRUE, ['sinc'] * d, None))
                    errs.append(h ** d * ((m.predict(G) - fg) ** 2).sum() / int_f2)
                out.append(dict(d=d, n=n, bin_scale=bs, rel_ise=float(np.mean(errs)), rel_ise_sd=float(np.std(errs))))
    save('expBin_check', out)

# ======================================================== R1: rate study
if os.environ.get('BLML_P1_PARTS', 'all') == 'systems':
    log('ALL DONE (systems only)')
    sys.exit(0)
D_LIST = [int(v) for v in os.environ.get('BLML_P1_DLIST', '1,2,3,4,6,8').split(',')]
N_LIST = [1000, 3000, 10000, 30000, 100000]
FC_MULTS = [float(v) for v in os.environ.get('BLML_P1_FC_MULTS', '1,2').split(',')]   # x truth.fc_true (T3: x reference cutoff 2)
M_F, M_Q, M_V = 100000, 200000, 20000

def eval_sets(truth, d):
    """Points for the metrics. d <= 2: fine-grid ISE. d > 2: defensive
    importance sampling."""
    rng = np.random.default_rng(10_000 + d)
    Xf = truth.sample(rng, M_F)
    ev = dict(Xf=Xf, ff=truth.pdf(Xf))
    if d <= 2:
        # BLML's error is band-limited (spacing 0.5 would be exact), but the KDE
        # baselines are not and their bandwidths reach ~0.4 at n = 1e5, so the grid
        # must resolve them too: Riemann sums of Gaussians are accurate to
        # exp(-2 pi^2 (sigma/h)^2), negligible for h <= sigma/2.
        if truth.kind == 'T3':                  # narrow normal: fine grid, short range
            h, R = (0.01, 10) if d == 1 else (0.05, 10)
        else:
            h, R = (0.05, 400) if d == 1 else (0.25, 80)
        g1 = np.arange(-R, R + h / 2, h)
        G = g1[:, None] if d == 1 else np.stack(np.meshgrid(g1, g1, indexing='ij'), -1).reshape(-1, 2)
        fg = truth.pdf(G)
        # NOTE: metrics() averages, so this records ISE / len(G); analyze_p1.py
        # multiplies d <= 2 ISE values by len(G) (16001 for d=1, 410881 for d=2).
        # Kept as-is so every R1 file in a run uses the same convention.
        ev.update(Xq=G, fq=fg, qq=np.full(len(G), 1.0 / h ** d))
        ev['int_f2'] = float(h ** d * (fg ** 2).sum())
    else:
        q = DefensiveProposal(truth, scale=2.0 if truth.kind == 'T3' else 6.0)
        Xq = q.sample(rng, M_Q)
        ev.update(Xq=Xq, fq=truth.pdf(Xq), qq=q.pdf(Xq), int_f2=float(ev['ff'].mean()))
    qv = DefensiveProposal(truth, scale=2.0 if truth.kind == 'T3' else 6.0)
    Xv = qv.sample(rng, M_V)
    ev.update(Xv=Xv, fv=truth.pdf(Xv), qv=qv.pdf(Xv))
    return ev

def evaluate(fn, ev):
    t0 = time.time()
    r = E.metrics(E.to_np(fn(ev['Xf'])), ev['ff'], E.to_np(fn(ev['Xq'])), ev['fq'], ev['qq'], ev['int_f2'])
    r['eval_s'] = time.time() - t0
    return r

def val_ise(fn, ev):
    return float(np.mean((E.to_np(fn(ev['Xv'])) - ev['fv']) ** 2 / ev['qv']))

C_GRID = np.geomspace(0.2, 3.0, 12)

TRUTHS = [] if os.environ.get('BLML_P1_PARTS') == 'bign' else os.environ.get('BLML_P1_TRUTHS', 'T1,T2').split(',')
for truth_kind in TRUTHS:
    for d in D_LIST:
        names = [f'r1/{truth_kind}_d{d}_n{n}' for n in N_LIST]
        if all(done(nm) for nm in names):
            continue
        truth = Truth(truth_kind, d)
        ev = eval_sets(truth, d)
        for n, nm in zip(N_LIST, names):
            if done(nm):
                continue
            reps = 8 if (n < 100000 or d <= 3) else 4
            rows = []
            for rep in range(reps):
                X = truth.sample(np.random.default_rng([0 if truth_kind == 'T1' else 1, d, n, rep]), n)
                Xg = cp.asarray(X)
                row = {}
                for mult in FC_MULTS:
                    m, tfit = timed(lambda: BLMLQuick(engine='gpu', dtype=np.float64).fit(X, make_modes(mult * truth.fc_true, ['sinc'] * d, None)))
                    r = evaluate(m.predict, ev)
                    r.update(fit_s=tfit, B=m.n_bins_, sweeps=m.n_sweeps_, converged=m.converged_,
                             obj_monotone=monotone(m.obj_history_))
                    row[f'blml_fc{mult:g}'] = r
                    row[f'sinc_kde_fc{mult:g}'] = evaluate(lambda Z: E.kde_sinc(Xg, Z, mult * truth.fc_true), ev)
                h0 = E.scott(X)
                row['kde_scott'] = evaluate(lambda Z: E.kde_gauss(Xg, Z, h0), ev)
                c_lscv = E.lscv_gauss(X, C_GRID)
                row['kde_lscv'] = evaluate(lambda Z: E.kde_gauss(Xg, Z, c_lscv * h0), ev)
                row['kde_lscv']['c'] = float(c_lscv)
                c_or = min(C_GRID, key=lambda c: val_ise(lambda Z: E.kde_gauss(Xg, Z, c * h0), ev))
                row['kde_oracle'] = evaluate(lambda Z: E.kde_gauss(Xg, Z, c_or * h0), ev)
                row['kde_oracle']['c'] = float(c_or)
                h6 = E.scott(X, order=6)
                c6 = min(C_GRID, key=lambda c: val_ise(lambda Z: E.kde6(Xg, Z, c * h6), ev))
                row['kde6_oracle'] = evaluate(lambda Z: E.kde6(Xg, Z, c6 * h6), ev)
                row['kde_adaptive'] = evaluate(lambda Z: E.kde_adaptive(Xg, Z, c_or * h0), ev)
                row['knn'] = evaluate(lambda Z: E.knn_density(Xg, Z), ev)
                rows.append(row)
                b = row[f'blml_fc{FC_MULTS[0]:g}']
                log(f"R1 {truth_kind} d={d} n={n} rep={rep}: relISE blml={b['rel_ise']:.2e} "
                    f"kde_oracle={row['kde_oracle']['rel_ise']:.2e} (B={b['B']}, fit {b['fit_s']:.1f}s)")
            save(nm, dict(truth=truth_kind, d=d, n=n, reps=rows, grid_points=len(ev['Xq']) if d <= 2 else 0))

# ======================================================== big-n: BLML only (asymptotic slope)
if os.environ.get('BLML_P1_PARTS') == 'bign':
    for d in [4, 6, 8]:
        truth = Truth('T1', d)
        ev = eval_sets(truth, d)
        for n, reps in [(300000, 4), (1000000, 3)]:
            nm = f'big/T1_d{d}_n{n}'
            if done(nm):
                continue
            rows = []
            for rep in range(reps):
                X = truth.sample(np.random.default_rng([0, d, n, rep]), n)
                row = {}
                for mult in FC_MULTS:
                    m, tfit = timed(lambda: BLMLQuick(engine='gpu', dtype=np.float64).fit(X, make_modes(mult * truth.fc_true, ['sinc'] * d, None)))
                    r = evaluate(m.predict, ev)
                    r.update(fit_s=tfit, B=m.n_bins_, sweeps=m.n_sweeps_, converged=m.converged_)
                    row[f'blml_fc{mult:g}'] = r
                    del m
                    cp.get_default_memory_pool().free_all_blocks()
                rows.append(row)
                log(f"BIG T1 d={d} n={n} rep={rep}: relISE fc1={row['blml_fc1']['rel_ise']:.3e} fc2={row['blml_fc2']['rel_ise']:.3e} (fit {row['blml_fc2']['fit_s']:.0f}s)")
            save(nm, dict(truth='T1', d=d, n=n, reps=rows, big=True))
log('ALL DONE')
