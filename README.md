# Band-limited maximum likelihood (BLML)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24063290.svg)](https://doi.org/10.5281/zenodo.24063290)

Code for band-limited maximum likelihood density estimation: a density is estimated as f = Σ_k |g_k|², each g_k
real or complex with its spectrum in a set given by a kernel (a band, two slits, a disk, their periodic versions on
a window, ...). The coefficients solve the stationarity conditions of the likelihood on a lattice of bins, by
coordinate descent or L-BFGS, on the CPU (NumPy / numba) or the GPU (CuPy). This repository holds the code only;
it accompanies two papers:

- R. Agarwal and A. Agarwal, *Taking Band-Limited Maximum Likelihood to High Dimensions: Linear Memory, the
  Parametric Rate, and a Self-Tuning Cut-off* (arXiv, 2026).
- R. Agarwal and A. Agarwal, *Complex band-limited maximum likelihood: estimating the wave behind detected quanta*
  (arXiv, 2026).

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # add cupy-cuda12x for the GPU engine
```

The modules import one another by name; run scripts from the repository root.

## Library

| module | contents |
|---|---|
| `blml_trivial.py` | estimators on points (`BLMLQuick`), kernel classes (`sinc`, `dirichlet`, `slits`, `ball`, `pball`) with their gradients, model specification (`make_modes`, `parse_modes`), the certificate |
| `blml_binned.py` | estimators on counts on a full lattice (`BLMLBinned`, FFT mat-vec; `BLMLBinnedStreaming`; `BLMLEmpirical`) |
| `blml_kernels.py` | the product kernel on a lattice: steps, binning, value tables for the row kernels |
| `blml_local.py`, `blml_local_gpu.py` | pair-accumulation kernels (numba, CUDA) |
| `blml_gpu.py` | the point estimators on the GPU |
| `selector.py` | cut-off selection by held-out likelihood |
| `blml_rolling.py` | sliding-window and streaming estimators |
| `blml_maps.py`, `blml_net.py`, `blml_node.py`, `blml_transform.py` | input maps and composite models |

Kernel kinds, per column or per group of columns (a tuple key):

| kind | band |
|---|---|
| `sinc` | an interval of width `fc` |
| `dirichlet` | the harmonics of a period `period` up to `fc` (`fill` < 1: two slits on the circle) |
| `slits` | two slits, full width `fc`, open fraction `fill` |
| `ball` | a disk (ball) of diameter `fc` over a group of columns |
| `pball` | the disk of harmonics of a periodic window (`period` per column): the ball made periodic |

```python
import numpy as np
from blml_binned import BLMLBinned

# counts on a 31 x 31 window, a pupil-limited field periodic on the window, complex root
m = BLMLBinned(solver='lbfgs').fit(counts, [{(0, 1): dict(kind='pball', fc=0.33, period=[31, 31])}], complex_g=True)
f = m.density_                    # the fitted density on the bins
g = m.Se_[0] / m.n_               # the fitted root (field) at the bins
print(m.loglik_, m.certificate()) # mean log-likelihood; lambda_max / n (<= 1: global optimum, shared kernel)
```

A background is one more mode with a narrow band, e.g. `{(0, 1): dict(kind='pball', fc=3 / 31, period=[31, 31])}`;
several modes sharing one aperture describe a partially coherent field.

## Tests

`results/pball_parity.py [cpu|gpu]` and `results/pslits_parity.py [cpu|gpu]` check the kernels against their
explicit formulas, the row / column / point kernels (numba, CUDA) against the kernels, and fits across estimators.

## Reproducing the papers

The scripts that produced the papers' results are included as they were run (root-level `p*_*.py`, `dslit_job.py`,
and the scripts under `results/`); their outputs, the papers and the data are not. They read public data
(the double-slit records of Aspden et al. and Bach et al., the bead data of `localize-psf`, CAISO and GEFCom
prices, UCI data sets; see the papers for sources) from `~/data_repo/...` or a path in the script, write results
under `results/`, and the figure and table scripts write into `papers/...`. Several runs used a GPU on Colab
(`colab_job.sh`, `run_colab*.sh`).

## Citing

Please cite the papers above and the archived code: R. Agarwal and A. Agarwal, *band-limited-ml*, Zenodo,
https://doi.org/10.5281/zenodo.24063290 (all versions; v1.0: https://doi.org/10.5281/zenodo.24063291).

## License

MIT (see `LICENSE`).
