"""Speed and memory of the single MNIST node vs n (refactor stage 5 on a GPU). One run per n (results/node_mnist1.py:
k = 4, shrinkage-LDA start, burn-in fc search, one epoch), each in its own process; GPU memory sampled every 0.5 s by
nvidia-smi during the run (peak minus the idle level); each result is written to BENCH_DIR/node_<engine>_<n>.json as
soon as it finishes and a run whose file exists is skipped, so a lost VM loses at most the run in progress.
    BENCH_NS=10000,20000,60000 BENCH_ENGINE=gpu BENCH_DIR=out python results/bench_node_gpu.py"""
import os, sys, json, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
NS = [int(v) for v in os.environ.get('BENCH_NS', '10000,20000,60000').split(',')]
ENG = os.environ.get('BENCH_ENGINE', 'gpu')
OUT = os.environ.get('BENCH_DIR', os.path.join(HERE, 'bench_node'))
os.makedirs(OUT, exist_ok=True)


def gpu_mem():
    o = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    return float(o.stdout.strip().splitlines()[0])


for n in NS:
    path = os.path.join(OUT, f'node_{ENG}_{n}.json')
    if os.path.exists(path):
        print(f'n={n}: done already, skipped', flush=True); continue
    env = dict(os.environ, NODE_N=str(n), NODE_EPOCHS='1', NODE_K='4', NODE_START='ldas', NODE_FC0='auto',
               NODE_ENGINE=ENG, NODE_SEED='1', NODE_OUT=os.path.join(OUT, f'.run_{n}.json'))
    samp, base = None, None
    if ENG == 'gpu':
        base = gpu_mem()
        mfile = os.path.join(OUT, f'.mem_{n}.txt')
        samp = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-lms', '500'],
                                stdout=open(mfile, 'w'))
    t = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, 'node_mnist1.py')], env=env, capture_output=True, text=True)
    wall = time.time() - t
    if samp is not None:
        samp.terminate(); samp.wait()
    if p.returncode != 0:
        print(f'n={n}: FAILED\n{p.stderr[-2000:]}', flush=True); break
    r = json.load(open(env['NODE_OUT']))
    r.update(wall_s=round(wall, 1), engine=ENG, n=n)
    if samp is not None:
        vals = [float(v) for v in open(mfile).read().split() if v.strip()]
        r.update(gpu_mem_base_mb=base, gpu_mem_peak_mb=max(vals) if vals else None,
                 gpu_mem_used_mb=(max(vals) - base) if vals else None)
    json.dump(r, open(path, 'w'), indent=1)
    print(f"n={n}: acc {r['acc']} fit {r['s']}s wall {r['wall_s']}s" +
          (f" GPU mem +{r['gpu_mem_used_mb']:.0f} MB (peak {r['gpu_mem_peak_mb']:.0f})" if samp is not None else ''), flush=True)
print('DONE', flush=True)
