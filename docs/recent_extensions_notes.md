# Recent sparse optimization extensions verified from primary sources

Checked 28 October 2026. Recommended additions to the core review: **Larsson et al. (2023)** and **Żogała-Siudem and Jaroszewicz (2024; online 2023)**. Dupuis and Tardivel (2024) is supplementary reading. These papers motivate discussion and future work; none of their algorithms or datasets has been added to this project's experiments.

## Core addition 1: hybrid optimization for SLOPE

**Johan Larsson, Quentin Klopfenstein, Mathurin Massias, Jonas Wallin. “Coordinate Descent for SLOPE.” AISTATS 2023, PMLR 206:4802–4821.** [Publisher record](https://proceedings.mlr.press/v206/larsson23a.html); [paper](https://proceedings.mlr.press/v206/larsson23a/larsson23a.pdf), Sections 2–3, Tables 2–3.

- **Method:** combines proximal gradient steps with cluster coordinate updates because the sorted L1 penalty is nonseparable; ordinary coordinate updates can stall.
- **Synthetic benchmarks:** `(n,p,k,density)` = `(200,20000,20,1)`, `(20000,200,40,1)`, `(200,200000,20,.001)`; feature correlations `0.6^|j-j'|`.
- **Real datasets:** bcTCGA `(536,17322)`, news20 `(19996,1355191)`, rcv1 `(20242,44504)`, Rhee2006 `(842,360)`.
- **Comparators:** PGD, FISTA, Anderson PGD, ADMM, Newt-ALM, and an oracle CD baseline requiring solution clusters.
- **Finding:** the hybrid improves measured SLOPE performance across many tested settings; results are conditional on problem size and regularization.
- **Relation to our study:** supports investigating how penalty structure and problem geometry affect solver choice. SLOPE replaces a single L1 weight with sorted weights; constant weights recover Lasso after matching loss scaling. Our cyclic Lasso CD is **not** this cluster algorithm. This paper does not establish that SLOPE improves recovery in our block-correlation experiment.

## Core addition 2: screening for the ordinary Lasso path

**Barbara Żogała-Siudem, Szymon Jaroszewicz. “Variable screening for Lasso based on multidimensional indexing.” Data Mining and Knowledge Discovery 38:49–78 (2024). Published online 27 August 2023.** [Publisher record](https://link.springer.com/article/10.1007/s10618-023-00950-8); [paper](https://link.springer.com/content/pdf/10.1007/s10618-023-00950-8.pdf), Sections 3.4, 4.1–4.2. [DOI](https://doi.org/10.1007/s10618-023-00950-8).

- **Method:** MI-Lasso uses correlation queries to screen the next entering predictor along a homotopy path, rather than solving only a prescribed lambda grid.
- **Data:** rat-eye expression predicting TRIM32 `(120,31098)`; Eurostat approximately `(800,8 million)`, including 100 semi-synthetic responses with three known predictors and an infant-mortality illustration.
- **Comparators:** rat-eye dynamic GAP screening with active warm starts in gsroptim; Eurostat Strong Rules, DPP, GAP, and unscreened Lasso.
- **Important limits:** exact-index queries support the safety theorem; the experimental approximate Faiss index can miss candidates. Index construction (33 minutes 27 seconds) is excluded from reported solve times under repeated-model reuse. Eurostat recovery permits correlated surrogates above correlation `.95`, unlike our exact-support F1.
- **Relation to our study:** directly concerns ordinary Lasso and highly correlated predictors. It motivates screening and setup-cost accounting, but its path, disk-storage, and index-reuse setting does not imply a speedup for our single-lambda dense CPU solves.

## Supplementary reading: SLOPE solution paths

**Xavier Dupuis, Patrick Tardivel. “The Solution Path of SLOPE.” AISTATS 2024, PMLR 238:775–783.** [Publisher record](https://proceedings.mlr.press/v238/dupuis24a.html); [paper](https://proceedings.mlr.press/v238/dupuis24a/dupuis24a.pdf), Sections 1, 5, Tables 2–4.

- Establishes continuous piecewise-linear paths under strictly decreasing positive penalty weights and develops an exact path algorithm.
- Data: red Wine Quality `(1599,11)` and Riboflavin `(71,4088)`.
- Compares full OSCAR paths with genlasso; single-parameter SLOPE solves with FISTA, Anderson PGD, ADMM, and hybrid CD. At stringent dual-gap accuracy, the path method wins the reported Wine cases; hybrid CD wins the weaker-regularization Riboflavin case. There is no universal winner.
- Useful for separating path computation and parameter selection from fixed-lambda solver comparison. It does not justify replacing our validation protocol or describing our implementation as SLOPE.

## Placement in the report

Use the direct Lasso screening work in the paragraph on recent computational developments. Use hybrid SLOPE in a short paragraph explaining how extensions alter the penalty and the applicable coordinate updates. Mention the 2024 path paper only if space permits. Preserve the distinction between evidence from these papers and the results of our three implemented Lasso algorithms.
