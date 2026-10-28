# Lasso Correlation

[![Tests](https://github.com/YuqiHuang2003/lasso-correlation/actions/workflows/tests.yml/badge.svg)](https://github.com/YuqiHuang2003/lasso-correlation/actions/workflows/tests.yml)

Reproducible experiments on **feature correlation, Lasso optimization efficiency, and sparse support recovery**.

The project combines controlled synthetic designs, semi-synthetic responses on public feature matrices, high-accuracy numerical certificates, and audits of support-recovery evaluation. The current research focus is whether different numerical representations of the same Lasso target change variable-level conclusions. Research reports are currently in Chinese; this repository guide is in English.

## Start here

- [Latest research brief: column-order sensitivity](report/column_permutation_v1/column_order_zh.pdf)
- [Detailed findings and reproduction steps](docs/column_permutation_findings_v1_zh.md)
- [Frozen column-order protocol](docs/column_permutation_protocol_v1.md) and [independent review](docs/column_permutation_review_v1.md)
- [Research roadmap](TODO.md)
- [Snapshot coverage](results/README.md) and [data preparation](DATA.md)

## Latest result

The column-order study holds **the mathematical objective, response, penalty, feature values, and original variable identities fixed**. It changes only the input column order and maps fitted coefficients back before evaluating support and F1.

For each target and implementation, the primary comparison uses the original order and five fixed permutations. A separate original-order repeat checks reproducibility. All **784 outputs** passed the common accuracy check; all **112 original-order repeats** and **112 comparisons with historical original-order references** passed their respective checks.

At the primary support threshold, `abs(coef) > 1e-6`:

| Feature design | Implementation | Targets | Support changes across orders | F1 range > 0.02 | Maximum F1 range |
| --- | --- | ---: | ---: | ---: | ---: |
| UJIIndoorLoc | scikit-learn cyclic CD | 40 | 18 | 18 | 7.66 percentage points |
| UJIIndoorLoc | skglm | 40 | 26 | 21 | 9.63 percentage points |
| Online News | scikit-learn cyclic CD | 8 | 0 | 0 | 0 |
| Online News | skglm | 8 | 0 | 0 | 0 |
| Leukemia feature matrix | scikit-learn cyclic CD | 8 | 0 | 0 | 0 |
| Leukemia feature matrix | skglm | 8 | 0 | 0 | 0 |

Every observed support change was inside an X-only near-proportional group fixed before fitting. The 10 UJI targets that had passed the earlier two-reference consensus check remained stable under the tested permutations. The other two design sources also remained stable.

These are findings about the **specified subsamples, preprocessing, responses, regularization levels, and finite set of orders**. The paired targets, permutations, and implementations are not independent datasets. Small duality gaps or two-implementation agreement do not prove uniqueness or correct recovery of the generating support. Column-order effects and permutation-based handling have prior work; this project does not claim to introduce a new solver.

## Installation and tests

The recorded environment uses Linux and Python 3.13.13. The dependency versions used by the experiments are pinned in the three requirements files. GCC is required for the optional C coordinate-descent checks.

```bash
git clone git@github.com:YuqiHuang2003/lasso-correlation.git
cd lasso-correlation
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
make test PYTHON=.venv/bin/python
```

The default tests use small fixtures and mocked experiment runners. They require no original datasets and do not launch research runs. Additional small data-generator and solver checks are explicit:

```bash
make test-data test-solvers PYTHON=.venv/bin/python
```

Optional modern solver and report dependencies:

```bash
make install-modern PYTHON=.venv/bin/python
make test-modern PYTHON=.venv/bin/python
make install-report PYTHON=.venv/bin/python
```

The modern stack is installed into the ignored `.modern_deps/` directory with `--no-deps`, sharing the pinned base NumPy/SciPy/scikit-learn environment. Run `make help` for the available targets. No Make target starts a full research experiment implicitly.

## Data and evidence verification

Original datasets are **not included**. The frozen source manifests and attribution records are retained under `docs/`. The explicit data-preparation command downloads from the official sources, checks content hashes, and restores the historical source-record bytes needed by the saved provenance chain:

```bash
make data PYTHON=.venv/bin/python
```

New acquisition timestamps are recorded separately from the historical source manifests. See [DATA.md](DATA.md) for cache behavior, licenses, and details.

The committed scientific snapshot can be checked without scientific Python dependencies or raw data:

```bash
python3 scripts/check_snapshot.py
```

After preparing data and installing the modern stack, the latest experiment's metadata preflight checks the frozen inputs and planned calls without fitting, warming up a solver, or generating new data:

```bash
make check-column-metadata PYTHON=.venv/bin/python
```

To recompute the numerical verification, analysis, independent analysis checks, and Chinese brief:

```bash
.venv/bin/python scripts/verify_column_permutation_v1.py
.venv/bin/python scripts/analyze_column_permutation_v1.py
.venv/bin/python scripts/verify_column_permutation_v1.py --analysis-only
.venv/bin/python scripts/build_column_permutation_brief_v1.py
```

These commands do not refit the experiment. They **do update derived verification and analysis files**. Use a separate checkout if you want to retain the committed snapshot byte-for-byte. Snapshot checks will correctly report changed files after regeneration. Report rebuilding requires the report dependencies.

Actual numerical reruns are separate: follow the [protocol](docs/column_permutation_protocol_v1.md), use a new versioned output directory, prepare and lock inputs before the first fit, and keep numerical timing jobs sequential. Existing runs with matching signatures are cached results, not new timings.

## Repository layout

```text
src/lasso_study/    Data generators, certificates, proximal/CD solvers, and audits
scripts/           Versioned experiment, verification, analysis, and report entry points
tests/             Small correctness, provenance, mapping, and runner tests
configs/           Frozen experimental settings and environment records
docs/              Protocols, findings, literature checks, and source manifests
results/           Complete latest evidence, baseline dependencies, and older summaries
figures/           Scientific figures generated from saved results
report/            Chinese reports and editable report sources
repository_snapshot.json  SHA-256 inventory of the copied scientific snapshot
```

The original numerical source files and selected evidence retain their bytes because provenance checks use SHA-256 hashes. Repository setup, data bootstrap, documentation, and CI are added around them. Historical environment records may contain the original machine's installation paths; these are provenance metadata, not installation requirements.

## Earlier stages and snapshot scope

The repository also retains the earlier correlation/sparsity study, synthetic confirmation results, external negative results, and the UJI preprocessing controls. Useful entry points are [the course-stage reproduction guide](docs/reproduction_v2.md), [workshop findings](docs/workshop_findings_zh.md), and [external reference-audit findings](docs/reference_audit_findings_v1_zh.md).

This is a curated research snapshot. It contains the **complete latest column-order evidence and its historical baseline dependencies**. The older dense checkpoint trajectories, local environments, original course handout, original datasets, generated font files, and duplicate ZIP bundles are omitted. Therefore, a complete rebuild of every historical report or packaging command may require the original full archives or a new experimental run. [results/README.md](results/README.md) records the exact boundary.

## Attribution and contributions

See [NOTICE.md](NOTICE.md) for dataset, embedded-font, and dependency attribution. No project-wide open-source license has been selected yet. Third-party licenses remain separate.

Contributions should preserve frozen evidence and use new versioned stages for changed numerical protocols; see [CONTRIBUTING.md](CONTRIBUTING.md). Publication novelty and claims are assessed separately from reproducibility and observed numerical effects.
