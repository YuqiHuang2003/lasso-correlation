# Recent Lasso solver sources

Checked 2026-10-28. These notes cover the two pathwise-solver sources assigned to this subtask. The screening-paper review is maintained separately. No new numerical experiments were run.

## 2023 glmnet update

Tay, Narasimhan and Hastie published in **Journal of Statistical Software 106(1), 1–31**, on **2023-03-23**; DOI **10.18637/jss.v106.i01**. This is a formally published software/methodology paper, despite the earlier 2021 preprint.

The method extends elastic-net paths across GLM families using weighted least-squares subproblems, cyclic coordinate descent, warm starts, strong rules and KKT checks. Gaussian loss with alpha=1 contains ordinary Lasso. In **§3.3, Fig.1, pp.7–9**, the timing experiment compares glmnet's specialized Fortran implementation with its generic family-object implementation, averaging five simulations. It is **not** an ISTA/FISTA/CD comparison. **§6, pp.23–25** demonstrates prediction assessment on **UCI Spambase**, a spam/non-spam classification task; reported measures include AUC, deviance, classification error, MSE and MAE. The official dataset has **4,601 observations and 57 predictors**.

Use this source for practical path computation, model selection and implementation costs. Its software examples do not establish a controlled correlation-versus-recovery conclusion for our experiment.

Sources: [publisher and publication date](https://www.jstatsoft.org/article/view/v106i01), [primary full paper](https://www.jstatsoft.org/article/download/v106i01/4459), [official Spambase dimensions](https://archive.ics.uci.edu/dataset/94/spambase).

## 2024 adelie preprint

Status: **arXiv:2405.08631v1, 2024-05-14**; no checked journal reference. Cite as a **preprint**.

**§6.2, Figs.6–7:** ordinary Lasso versus **glmnet, sparsegl, gglasso, grpnet**; three-trial average **100-lambda path** runtime. **Table 1, PDF p.18:**

| Dataset | n | Original feature count | Task |
| --- | ---: | ---: | --- |
| Leukemia | 72 | 7,129 | Classification |
| UJIIndoorLoc | 19,937 | 465 | Longitude regression |
| Prostate | 102 | 6,033 | Classification |

Fig.6 uses globally equicorrelated Gaussian features, **ρ=0,0.5,0.95**, **n=100/1,000**, SNR=3, alternating-sign exponentially decaying coefficients. Horizontal p=G: **100–100,000** for n=100, **10–10,000** for n=1,000. Group-Lasso separately uses polynomial-expanded **p=3G**; ordinary Lasso has singleton groups.

**Figure reading, not a theorem:** Gaussian Fig.6(a) often shows slower high-ρ fits; ρ=0/0.5 are frequently similar. Binomial Fig.6(b) is not uniformly monotone: at n=100, large p, adelie/glmnet can be faster for ρ=0.95. Panels use different vertical ranges. The authors conclude similar adelie/glmnet performance.

Our distinction is block correlation, exactly sparse truth, and recovery-versus-accuracy analysis; their whole-path warm-start timing is a direct precedent, not evidence that our topic is novel.

Sources: [arXiv status](https://arxiv.org/abs/2405.08631), [primary PDF, §6 and Table 1](https://arxiv.org/pdf/2405.08631), [searchable full text, §6.1–6.2](https://arxiv.org/html/2405.08631v1#S6), [author's publication listing](https://hastie.su.domains/publications.html).

Original figure assets inspected: [Fig.6(a), Gaussian](https://arxiv.org/html/2405.08631v1/lasso_sim_gaussian_time.svg), [Fig.6(b), Binomial](https://arxiv.org/html/2405.08631v1/lasso_sim_binomial_time.svg).

## Report use

Keep the 2023 glmnet paper in the modern-methods discussion. Include the 2024 adelie preprint as the closer comparison for correlation-sensitive Lasso path experiments, explicitly retaining its publication status. These sources support related-work positioning; their results are not measurements of our implementations or our Online News Popularity task. No need to reproduce Group Lasso to cite the directly relevant §6.2 ordinary-Lasso comparison.
