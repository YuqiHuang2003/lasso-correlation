# Contributing to lasso-correlation

This is a research repository with frozen numerical evidence. Python 3.13 on
Linux is the tested target. The Chinese research reports and their recorded
protocols define the interpretation of the saved results; repository cleanup
does not create a new experiment or an independent replication.

## Local checks

Create a separate environment and install the existing version pins:

```sh
python3.13 -m venv .venv
. .venv/bin/activate
make install
make test
```

`pyproject.toml` supplies the `src/` and `scripts/` import paths for pytest.
An editable package install is unnecessary. `make test` runs an explicit list
of 39 arithmetic, small-fixture and mocked-runner checks. It needs no raw data,
saved results, optional modern solver, or report packages. CI uses this command.

Additional checks are opt-in:

```sh
make test-data        # Generated and temporary-file data tests; no downloads
make test-solvers     # Small real solver problems; GCC compiles the C kernel
make install-modern  # Isolated .modern_deps; leaves the base pins in place
make test-modern     # Small skglm problems, including one-time JIT warmup
```

The default `pytest` collection includes the real solver tests. Use `make test`
for the dependency-light CI scope. Do not add a data download, full study,
formal-input preparation, PDF build, or historical-result verification to that
scope. Preserve the test files and the configurations they read when exporting
the repository.

## Frozen evidence and new experiments

- Preserve frozen numerical source, configuration, stored coefficient and result
  files byte for byte. Even a cosmetic source rename or reformat changes the
  hashes used by the existing signatures. Add repository scaffolding around
  those files rather than rewriting them.
- A changed algorithm, dependency stack, data transformation, target grid,
  tolerance or budget needs an independent protocol/configuration version and
  result directory. State the question and failure rules before fitting; lock
  code, source data, permutations and other prepared inputs as required by that
  protocol. Do not overwrite old evidence or relabel old runs as new ones.
- Keep numerical timing stages serial and single threaded. Do not run tests,
  verification, plots, report generation or another timed study concurrently.
  Record setup and warmup costs separately, retain actual library retries and
  partial failures, and honor the frozen soft-budget and no-retry rules.
- Retain ineligible, timed-out and uncertain outputs. Keep precision, observed
  budget, numerical consensus, repeat QC and statistical recovery distinct.
  An observed range does not describe a complete mathematical solution set.

Run the smallest relevant checks for a change. Include its purpose, affected
protocol/version, validation and remaining limitations in a proposed change.

## Data and metadata preflight

`make data` explicitly downloads the three source datasets through the existing
scripts, which check frozen content hashes and reuse valid cached files. Review
the source records and third-party terms before sharing data. Keep raw datasets,
virtual environments, `.modern_deps/`, compiled binaries and local caches out of
the repository.

After installing the optional modern stack, downloading the source data and
restoring the frozen evidence bundle, `make check-column-metadata` hashes the
column-study inputs and checks the registered call grid. It does not regenerate
designs, warm a solver, fit, or write results. It is expected to fail when that
evidence is absent or its source/dependency signature differs. Passing ordinary
unit tests does not imply that a historical study has been reproduced.

## Reports

The primary reports are in Chinese. Keep their terminology, denominators and
claims consistent with the independently verified artifacts, including negative
results and post-hoc limitations. Editorial changes do not require new numerical
runs. Optional Python report tools use `requirements-report.txt`; TeX sources
may also require a separate TeX/font installation. Neither is installed or run
by the lightweight CI workflow.
