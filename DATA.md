# Data preparation

No original input matrix, CSV archive, or LIBSVM dataset is committed. This includes both original and merged Leukemia inputs. Numerical coefficient outputs and preset synthetic generating coefficients are research artifacts, not original clinical or positioning labels.

| Design | Official source | Frozen manifest |
| --- | --- | --- |
| Online News Popularity | [UCI](https://archive.ics.uci.edu/dataset/332/online+news+popularity) | `docs/data_manifest_v2_raw.json` |
| UJIIndoorLoc | [UCI](https://archive.ics.uci.edu/dataset/310/ujiindoorloc) | `docs/data_manifest_v2_benchmarks_ujiindoorloc.json` |
| Leukemia | [LIBSVM collection](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#leukemia) | `docs/data_manifest_v2_benchmarks_leukemia.json` |

## Recommended entry point

```bash
make data PYTHON=.venv/bin/python
```

Or prepare one source:

```bash
.venv/bin/python scripts/prepare_data.py --dataset news
.venv/bin/python scripts/prepare_data.py --dataset uji
.venv/bin/python scripts/prepare_data.py --dataset leukemia
```

The repository bootstrap calls the original download helpers in temporary staging directories only when files are missing. It verifies complete content hashes before installing files. An existing file with an unexpected hash or an unsafe path is rejected rather than silently overwritten. A complete valid cache is usable offline.

The old experiments also hashed their `source.json` records. Freshly generated timestamps in those records would otherwise change their identity even when dataset content matches. The bootstrap therefore restores the **byte-identical historical manifests** from `docs/`; it records new acquisition events separately in ignored `data/acquisition_log.json`. Historical `retrieved_at_utc` values describe the recorded snapshot, not a claim about when you downloaded it.

The original `download_data.py` and `download_benchmarks.py` scripts remain unchanged as part of the frozen source. Use `prepare_data.py` to prepare an evidence-preserving fresh checkout. A changed official dataset is rejected and should be studied as a new version, not patched into the recorded experiments.

## Attribution and interpretation

The source records retain the original authors, official links, content hashes, and licensing information. The two UCI records identify CC BY 4.0. The checked LIBSVM page did not clearly establish redistribution permission for the Leukemia snapshot; its software license is not used as a substitute for dataset permission.

Later recovery audits generate new sparse regression responses on these matrices. They do not use the original news outcome, geographical coordinates, or leukemia class labels as generating truth. Dataset sampling, preprocessing, fixed versus changing X, and the meaning of each seed are specified in the versioned protocols.

See [NOTICE.md](NOTICE.md) and the [benchmark-source notes](docs/benchmark_sources_v2.md) for additional attribution.
