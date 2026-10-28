# Evidence included in this repository

This repository is a curated research snapshot, not a copy of the entire development workspace. All selected scientific files were copied byte-for-byte and are listed in [`repository_snapshot.json`](../repository_snapshot.json).

| Stage | Included files | Included size (MiB) | Coverage |
| --- | ---: | ---: | --- |
| `v2` | 30 | 10.72 | Summaries and required dependency files; older dense trajectories omitted |
| `workshop/pilot` | 11 | 2.32 | Summaries and required dependency files; older dense trajectories omitted |
| `workshop/confirm` | 11 | 12.23 | Summaries and required dependency files; older dense trajectories omitted |
| `workshop/real` | 140 | 6.11 | Summaries and required dependency files; older dense trajectories omitted |
| `reference_audit_v1` | 172 | 21.34 | Summaries and required dependency files; older dense trajectories omitted |
| `uji_preprocessing_control_v1` | 179 | 7.34 | Complete JSON/CSV/NPZ evidence |
| `column_permutation_v1` | 2,459 | 70.09 | Complete JSON/CSV/NPZ evidence |

## Complete latest stage

`column_permutation_v1/` contains all 784 output records and coefficient files, 784 call reservations, the target/call manifest, 13 permutation registries and original-coordinate group records, 56 dataset records, preparation and invocation ledgers, both independent verifications, analysis, and CSV tables. Its 672 primary outputs and 112 original-order repeats are kept separately. Zero-change targets and stable controls are included.

The latest `environment.json/frozen_input_hashes` names 179 inputs. The 176 non-raw files are included at their original relative paths; the three raw datasets must be prepared separately. These dependencies include all 40 old UJI targets and 16 selected external-control targets.

## Historical stages

Older summaries, reference-uncertainty records, external negative results, reported failures, and applicable verification snapshots are retained. Most old dense per-iteration run logs, checkpoint trajectories, and unrelated coefficient files are omitted. A saved historical verification records the original full run; its presence does not mean its entire input closure is included here.

Consequently, a complete rebuild of an older analysis/report/ZIP may need the full original archive or a new versioned numerical run. The GitHub snapshot is intended to support inspection and independent verification of the complete latest column-order study, while documenting earlier stages and their limitations.

Original data, local dependency environments, generated fonts, binary build products, course handouts, task instructions, and duplicate delivery ZIPs are excluded. Data manifests in `docs/` are retained and used by `scripts/prepare_data.py`.

## Verification

Run `python3 scripts/check_snapshot.py` to check copied bytes without datasets. Recomputing numerical verification requires the pinned optional solver stack and prepared source data; see the root README. Recomputed verification/analysis files can differ in timestamps and diagnostic timings even when their mathematical results agree.
