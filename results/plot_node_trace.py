"""Angles to the planted plane and the projected outputs' fc against the Adam step, from the 'trace' field the node
scripts save (results/node_stream_k5.py). Usage: plot_node_trace.py out.pdf run.json [run.json ...]."""
import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
out, files = sys.argv[1], sys.argv[2:]
runs = [(f, k, v) for f in files for k, v in json.load(open(f)).items() if v.get('trace')]
fig, axs = plt.subplots(2, len(runs), figsize=(2.6 * len(runs), 4.2), squeeze=False, sharex=True)
for i, (f, k, v) in enumerate(runs):
    A = np.array([t['angles'] for t in v['trace']]); F = np.array([t['fc'] for t in v['trace']])
    axs[0, i].plot(A); axs[0, i].set_ylim(0, 90); axs[0, i].set_title(f"{k}\nk={v['k']}, acc {v['acc']:.2f}", fontsize=7)
    axs[1, i].semilogy(F); axs[1, i].set_xlabel('Adam step')
axs[0, 0].set_ylabel('principal angles (deg)'); axs[1, 0].set_ylabel('fc of projected outputs')
fig.tight_layout(); fig.savefig(out); print('wrote', out)
