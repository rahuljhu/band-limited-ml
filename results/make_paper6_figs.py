"""Paper 6 figures from results/node_stage1.{json,npz}: (1) the 2-d projections of held-out points per structure
and method, coloured by class; (2) held-out Bayes accuracy by method and structure. Run from the repo root."""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P6 = 'papers/paper6/'
os.makedirs(P6 + 'figs', exist_ok=True)
R = json.load(open('results/node_stage1.json'))
Z = np.load('results/node_stage1_proj.npz')
KINDS = [('blobs', 'shifted means'), ('rings', 'rings'), ('xor', 'quadrant XOR'), ('checker', 'checkerboard')]
METHODS = ['LDA (+PCA)', 'SIR', 'SAVE', 'HSIC (KDR-style)', 'dependence node']
LABEL = {'LDA (+PCA)': 'LDA', 'SIR': 'SIR', 'SAVE': 'SAVE', 'HSIC (KDR-style)': 'HSIC (KDR-style)',
         'dependence node': 'dependence node', 'truth': 'true plane'}
C0, C1 = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.spines.top': False, 'axes.spines.right': False})

# (1) projections
cols = ['truth', 'LDA (+PCA)', 'SAVE', 'HSIC (KDR-style)', 'dependence node']
fig, axs = plt.subplots(len(KINDS), len(cols), figsize=(7.2, 6.0))
for i, (kind, title) in enumerate(KINDS):
    y = Z[f'{kind}|y']
    for j, m in enumerate(cols):
        ax = axs[i, j]; U = Z[f'{kind}|{m}']; U = (U - U.mean(0)) / U.std(0)
        ax.scatter(U[y == 0, 0], U[y == 0, 1], s=2, c=C0, alpha=0.55, lw=0)
        ax.scatter(U[y == 1, 0], U[y == 1, 1], s=2, c=C1, alpha=0.55, lw=0)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.set_aspect('equal')
        for sp in ax.spines.values():
            sp.set_color('0.8')
        if i == 0:
            ax.set_title(LABEL[m])
        if j == 0:
            ax.set_ylabel(title)
        if m != 'truth':
            ax.text(0.97, 0.03, f"{R[kind][m]['acc']:.2f}", transform=ax.transAxes, ha='right', va='bottom', fontsize=7,
                    color='0.25', bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
fig.tight_layout(pad=0.4); fig.savefig(P6 + 'figs/projections.pdf'); plt.close(fig)

# (2) accuracy bars
fig, ax = plt.subplots(figsize=(6.4, 2.4))
colors = ['#b9c6d6', '#8fa5bf', '#1baf7a', '#9b6fd0', '#eb6834']
w = 0.16
for j, m in enumerate(METHODS):
    acc = [R[k][m]['acc'] for k, _ in KINDS]
    ax.bar(np.arange(len(KINDS)) + (j - 2) * w, acc, w, color=colors[j], label=LABEL[m])
ax.axhline(0.5, color='0.5', lw=0.7, ls=':')
ax.set_xticks(np.arange(len(KINDS))); ax.set_xticklabels([t for _, t in KINDS])
ax.set_ylabel('held-out accuracy'); ax.set_ylim(0.4, 1.0)
ax.legend(fontsize=6.5, frameon=False, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.18))
fig.tight_layout(); fig.savefig(P6 + 'figs/accuracy.pdf'); plt.close(fig)
print('wrote', P6 + 'figs/projections.pdf', P6 + 'figs/accuracy.pdf')

# streaming node against the ambient dimension (results/node_stream_dims.py)
SD = 'results/node_stream_dims.json'
if os.path.exists(SD):
    S = json.load(open(SD))
    starts = ['true plane', 'SAVE', 'random']
    rows = []
    for score in ('cond_loglik', 'dependence'):
        for kind, kt in (('checker', 'checkerboard'), ('xor', 'quadrant XOR')):
            for D in (5, 10, 20, 50):
                rr = [S.get(f'{D}|{score}|{kind}|{st}') for st in starts]
                if not all(rr):
                    continue
                cell = lambda r: (r'\textbf{%.2f}' if r['acc'] >= 0.85 else '%.2f') % r['acc'] + ' ({:.0f}$^\\circ$)'.format(max(r['angles']))
                sc = 'cond.\\ log-lik.' if score == 'cond_loglik' else 'dependence'
                rows.append(f"{sc} & {kt} & {D} & " + ' & '.join(cell(r) for r in rr) + r' \\')
    open(P6 + 'tables/stream_dims.tex', 'w').write(
        "\\begin{tabular}{llrccc}\n\\toprule\n& & & \\multicolumn{3}{c}{start of $W$} \\\\\n\\cmidrule(lr){4-6}\n"
        "score & structure & $D$ & true plane & SAVE & random \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    print('wrote', P6 + 'tables/stream_dims.tex')

# output width k: k = 2 (node_stream_dims.json) against k = 5 with fine and auto start scales
def _load(*fs):
    d = {}
    for f in fs:
        if os.path.exists(f):
            d.update(json.load(open(f)))
    return d
K5 = _load('results/node_stream_k5.json')
K5a = _load('results/node_stream_k5_auto.json')
if K5 and os.path.exists(SD):
    rows = []
    variants = lambda D, kind: (('2', '5000', S, f'{D}|cond_loglik|{kind}|'), ('5 fine', '5000', K5, f'{D}|{kind}|'),
                                ('5 auto', '5000', K5a, f'{D}|{kind}|'))
    for kind, kt in (('rings', 'rings'), ('checker', 'checkerboard'), ('xor', 'quadrant XOR')):
        for D in (5, 10, 20):
            for lab, n, src, pre in variants(D, kind):
                rr = [src.get(pre + st) for st in ('true plane', 'SAVE', 'random')]
                if any(rr):
                    rows.append(f"{kt} & {D} & {lab} & {n} & " + ' & '.join('--' if r is None else
                                ('\\textbf{%.2f}' if r['acc'] >= 0.85 else '%.2f') % r['acc'] for r in rr) + r' \\')
    open(P6 + 'tables/stream_k.tex', 'w').write(
        "\\begin{tabular}{lrlrccc}\n\\toprule\n& & & & \\multicolumn{3}{c}{start of $W$} \\\\\n\\cmidrule(lr){5-7}\n"
        "structure & $D$ & outputs & $n$ & true plane & SAVE & random \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    print('wrote', P6 + 'tables/stream_k.tex')
