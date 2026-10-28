# lasso-correlation: use Python 3.13, or set PYTHON=.venv/bin/python.
PYTHON ?= python3
.DEFAULT_GOAL := help

# Numerical jobs must not inherit a machine-dependent BLAS/OpenMP thread count.
export OPENBLAS_NUM_THREADS := 1
export OMP_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export NUMEXPR_NUM_THREADS := 1
export NUMBA_NUM_THREADS := 1

# Pure arithmetic, small fixtures and mocked runners; no real optimizer calls,
# data downloads, historical result reads, native compilation or PDF generation.
SMALL_TESTS := \
	tests/test_prepare_data.py \
	tests/test_audit_data_v1.py \
	tests/test_column_analysis_v1.py \
	tests/test_column_permutation_v1.py \
	tests/test_column_runner_v1.py \
	tests/test_uji_preprocessing_control_v1.py \
	tests/test_workshop_runner.py

.PHONY: help install install-modern install-report test test-data test-solvers test-modern data check-column-metadata

help:
	@printf '%s\n' \
	  'install                Install the frozen base requirements in the selected Python environment' \
	  'install-modern         Add isolated optional skglm/Numba packages in .modern_deps/' \
	  'install-report         Install optional report-building Python requirements' \
	  'test                   Run lightweight correctness checks (no datasets needed)' \
	  'test-data              Run additional generated/temporary-data unit tests' \
	  'test-solvers           Run small real solver checks; needs GCC and writes build/cd_kernel.so' \
	  'test-modern            Run small real skglm checks; needs install-modern and performs JIT warmup' \
	  'data                   Explicitly download/check all three frozen source datasets' \
	  'check-column-metadata  Read-only column-study preflight; needs modern packages, raw data and frozen artifacts' \
	  'No target starts a full study, prepares formal experiment inputs, or builds a PDF.'

install:
	$(PYTHON) -m pip install -r requirements.txt

# --no-deps preserves the base scientific stack installed from requirements.txt.
install-modern:
	$(PYTHON) -m pip install --no-deps --target .modern_deps -r requirements-modern.txt

install-report:
	$(PYTHON) -m pip install -r requirements-report.txt

test:
	$(PYTHON) -m pytest -q $(SMALL_TESTS)

test-data:
	$(PYTHON) -m pytest -q tests/test_data.py tests/test_data_v2.py tests/test_workshop_data.py

test-solvers:
	$(PYTHON) -m pytest -q tests/test_solvers.py

test-modern:
	$(PYTHON) -m pytest -q tests/test_modern_solver.py

# Opt-in network access. The existing scripts verify content hashes, reuse valid
# cached downloads, and retain their original source/attribution records.
data:
	$(PYTHON) scripts/prepare_data.py

# No regeneration, warmup, fitting, or output writes. This is intentionally not
# a dependency of test/CI: an evidence-light checkout cannot satisfy this check.
check-column-metadata:
	$(PYTHON) scripts/run_column_permutation_v1.py --check-only
