# Lasso Correlation

[![Tests](https://github.com/YuqiHuang2003/lasso-correlation/actions/workflows/tests.yml/badge.svg)](https://github.com/YuqiHuang2003/lasso-correlation/actions/workflows/tests.yml)

Code and saved results for the report *Column-Order Sensitivity of Lasso Support Recovery under Correlated Features* (COMP6704 individual assignment, PolyU, Fall 2026).

The study solves the Lasso

$$\min_w \frac{1}{2n}\|y-Xw\|_2^2+\lambda\|w\|_1$$

with ISTA, FISTA, cyclic coordinate descent (C99 kernel), scikit-learn and skglm. All solvers stop on the same normalized duality gap. It measures how feature correlation slows each solver and when the support stops changing as the gap shrinks. It then permutes the input columns, with the objective held fixed, to test whether the support at convergence is unique.

## Main results

- At correlation 0.9, ISTA, FISTA and CD take 17 to 44 times as long as at correlation 0, and skglm 1.4 to 3.2 times as long. ISTA and CD converge linearly near the solution, and the ISTA iteration count matches the rate $(L/\mu_A)\ln 100$ set by the curvature on the selected support.
- On synthetic designs, all solvers return the same support once the gap reaches $10^{-6}$.
- On UJIIndoorLoc features solved to a gap of $10^{-10}$, reordering the columns changes the support on 18 of 40 targets with scikit-learn and 25 of 40 with skglm. F1 moves by up to 9.63 percentage points. Every change lies inside a group of nearly proportional columns found before fitting, and the outputs agree once coefficients are summed within each group.
- On Online News and Leukemia, every equicorrelation submatrix has full column rank, so the solution is unique, and no column order changes the support.

## Where each result in the report comes from

| Report item | Saved results | Script that produces them |
| --- | --- | --- |
| Sec. 3.1, sample Lipschitz constant | computed on the fly | `scripts/compute_lipschitz.py` |
| Table 1, Figure 1, Figure 2 | `results/v2/synthetic_runs.csv`, `results/v2/summary.json` | `scripts/run_v2.py synthetic`, then `scripts/analyze_v2.py` (figures in `figures/v2/`) |
| Sec. 5.1, convergence rate | `results/v2/synthetic_runs.csv` | `scripts/convergence_rate.py` |
| Table 2 | `results/v2/real_unbounded.json` | `scripts/run_real_unbounded.py`, all four solvers without a time limit (needs `make data` and GCC) |
| Table 3 | `results/workshop/confirm/` | `scripts/run_workshop_followup.py confirm`, then `scripts/analyze_workshop.py confirm` |
| Sec. 5.3, sklearn against skglm on 40 UJI targets | `results/workshop/real/analysis.json` | `scripts/run_workshop_followup.py real`, then `scripts/analyze_workshop.py real` |
| Tables 4 and 6, Sec. 5.4 group analysis | `results/column_permutation_v1/` | `scripts/run_column_permutation_v1.py`, then `scripts/analyze_column_permutation_v1.py` |
| News and gene control targets | `results/reference_audit_v1/confirm/` | `scripts/run_reference_audit_v1.py confirm`, then `scripts/analyze_reference_audit_v1.py confirm` |
| Table 5 | `results/equicorrelation_rank_check.json` | `scripts/check_equicorrelation_rank.py` |
| Table 7 | `results/uji_preprocessing_control_v1/` | `scripts/run_uji_preprocessing_control_v1.py` |

The group scan of Sec. 3.3 is `near_proportional_groups_chunked` in `src/lasso_study/audit_data_v1.py`. The solvers are in `src/lasso_study/solvers.py`, `src/lasso_study/cd_kernel.c` and `src/lasso_study/modern_solver.py`. The duality gap certificate is `certificate` in `src/lasso_study/solvers.py`.

## Installation

The experiments used Linux and Python 3.13. Pinned versions are in the requirements files. GCC is needed for the C coordinate descent kernel.

```bash
git clone https://github.com/YuqiHuang2003/lasso-correlation.git
cd lasso-correlation
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
make install-modern PYTHON=.venv/bin/python   # skglm, numba, llvmlite into .modern_deps/
make test PYTHON=.venv/bin/python
```

`uv venv --python 3.13 .venv` works as well when the system Python lacks `venv`.

## Quick checks (seconds, no download)

These read saved results or rebuild synthetic designs from fixed seeds.

```bash
.venv/bin/python scripts/compute_lipschitz.py    # Sec. 3.1: L from about 5.8 to 30.6
.venv/bin/python scripts/convergence_rate.py     # Sec. 5.1: iterations per two decades of the gap
python3 scripts/check_snapshot.py                # saved results match their recorded bytes
```

## Checks that need the public datasets

`make data` downloads Online News Popularity and UJIIndoorLoc from UCI and Leukemia from LIBSVM, and checks them against the recorded source files. The datasets are not stored in this repository.

```bash
make data PYTHON=.venv/bin/python
.venv/bin/python scripts/check_equicorrelation_rank.py   # Table 5, about one minute
.venv/bin/python scripts/verify_column_permutation_v1.py # checks all 784 saved column order outputs
```

`check_equicorrelation_rank.py` reuses the saved reference coefficients for 16 targets and refits the other 56 with the reference settings of `configs/reference_audit_v1.json`. It writes `results/equicorrelation_rank_check.json`.

## Full reruns

Each experiment has a frozen configuration in `configs/` and a protocol in `docs/`. Run timing experiments one at a time with one BLAS thread, as the `Makefile` sets through `OPENBLAS_NUM_THREADS` and related variables. The run scripts reuse cached results whose signature matches, so write a rerun to a fresh copy of the repository to obtain new timings. All saved results were produced on one machine (Intel Core Ultra X7 358H, OpenBLAS with Haswell kernels, Python 3.13). On a CPU whose BLAS uses other kernels, such as AVX-512, the response $y=X\beta^\star+\varepsilon$ of a few UJI targets can differ in its last bits. The runners then print a warning and continue on the regenerated $y$, and a bitwise mismatch with the saved references is recorded without stopping the column order study. Iteration counts, supports and F1 of ISTA, FISTA and CD do not change across machines. skglm supports on UJI can change slightly, which is the order and rounding effect that the column order study measures.

Detailed guides, in Chinese, are [docs/reproduction_v2.md](docs/reproduction_v2.md) for Tables 1 and 2 and [docs/workshop_reproduction.md](docs/workshop_reproduction.md) for Table 3. The column order protocol is [docs/column_permutation_protocol_v1.md](docs/column_permutation_protocol_v1.md).

## Repository layout

```text
src/lasso_study/   data generators, solvers, duality gap certificate, group scan
scripts/           experiment, analysis and verification scripts
configs/           frozen experiment settings
results/           saved outputs for every table in the report
figures/           figures generated from saved results
docs/              protocols, findings and data source records
tests/             small correctness tests that need no datasets
```

`results/README.md` lists which saved files are included. Dense per-iteration trajectories of the earliest runs are left out to keep the repository small.

## Data and licenses

UJIIndoorLoc and Online News Popularity are from the UCI Machine Learning Repository under CC BY 4.0. The Leukemia matrix is from the LIBSVM dataset page. See [NOTICE.md](NOTICE.md) and [DATA.md](DATA.md).
