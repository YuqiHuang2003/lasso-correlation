# 2023 年后相关设计与 Lasso 恢复研究核查

核查日期：2026 年 10 月 28 日（Asia/Shanghai）。候选按与本项目“普通 Lasso、特征相关性、统计恢复”的联系排序。出版年份依据期刊或原始预印本记录，不使用网页抓取或文件上传时间。以下均为文献核查，没有复现其数据或实验；BibTeX 片段单独保存在 `report/recent_recovery.bib`，尚未合并主报告引用库。

## 推荐顺序与范围

| 候选 | 年份与状态 | 直接联系 | 建议 |
| --- | --- | --- | --- |
| Stallrich 等，*An optimal design framework for lasso sign recovery* | 2025，JRSS B 正式论文；预印本首发 2023 | 普通 Lasso、设计相关结构、符号恢复概率 | 优先纳入，最贴近“恢复不只取决于相关性大小” |
| Buhai，*Lasso and Partially-Rotated Designs* | 2025，arXiv 预印本；作者主页也标为 Preprint | 普通 Lasso、相关列、限制特征值与预测保证 | 可纳入，清楚注明预印本和理论设计限制 |
| Minsker 与 Shen，*The impact of contamination and correlated design on the Lasso: An average case analysis* | 2025，Statistics & Probability Letters 正式论文 | 分组相关设计、平均情形、污染与恢复/预测 | 可补充，必须说明使用增广鲁棒目标 |
| Alrashdi 与 Alrasheedi，*Square-root lasso under correlated regressors: Tight statistical analysis with a wireless communications application* | 2024，AIMS Mathematics 正式论文 | 相关设计与稀疏信号性能，但主要为左侧相关 | 次选；不能当作本项目普通 Lasso 的直接理论依据 |

## 1 Stallrich 等 2025

**准确元数据：** Jonathan W. Stallrich, Kade Young, Maria L. Weese, Byran J. Smucker, David J. Edwards。*An optimal design framework for lasso sign recovery*. Journal of the Royal Statistical Society Series B: Statistical Methodology, **87(5):1457–1480, 2025**。2025 年 5 月 20 日在线发表，正式卷期为 2025 年 11 月；预印本 2023 年 3 月首发，2025 年修订。[期刊全文](https://academic.oup.com/jrsssb/article/87/5/1457/8139083)；[DOI](https://doi.org/10.1093/jrsssb/qkaf026)；[预印本历史](https://arxiv.org/abs/2303.16843)。引用键 `stallrich2025`。

**设计与任务：** 研究两水平、因素多于实验次数的超饱和设计，使用高斯误差的稀疏主效应模型；构造设计并计算 Lasso 路径上的符号恢复概率，并非在某个公开观测数据集上做预测。Section 5 的三个案例为 `(n,p)=(9,10)` 的未知/已知符号，以及 `(14,20)` 的已知符号；已知符号例的目标小正相关约为 0.04、0.045。

**结论边界：** 在其设计准则与假设下，已知效应方向和未知方向的理想结构不同。不能把摘要压缩成“正相关总能改善恢复”：全文 Theorem 3 对未知符号的零相关局部最优还要求式 (19)，一般全局最优表述含 conjecture。定位：Sections 2–3、Theorems 2–3、Sections 5.1–5.3。

**本项目如何使用：** 可用来解释为什么需要同时声明支持分布、系数符号及恢复指标，不能只按总体相关系数排列“好坏”。本项目使用连续高斯分组设计、混合正负真系数及每组至多一个信号，并未构造文中的最优两水平设计，也未计算其精确符号恢复概率。

## 2 Buhai 2025

**准确元数据：** Rares-Darius Buhai。*Lasso and Partially-Rotated Designs*. **arXiv:2505.11093，2025**，v1 于 2025 年 5 月 16 日提交。核查作者主页与 arXiv，当前按**预印本**处理；没有核实到其 NeurIPS 2025 正式发表记录。[原始论文](https://arxiv.org/abs/2505.11093)；[全文 HTML](https://arxiv.org/html/2505.11093v1)；[作者发表列表](https://raresbuhai.com/)。引用键 `buhai2025`。

**设计与任务：** 19 页理论稿，不以真实数据集或实测算法时间为主要证据。部分列相对其余列作随机旋转，保留两部分内部的任意相关结构；随机性用于控制两部分之间的相对方向。研究带高斯噪声的稀疏线性回归。

**结论边界：** 在列范数、部分旋转、样本量与支持所在部分的限制特征值条件下，给出高概率限制特征值及 Lasso 预测误差保证，使保证不依赖另一部分内部相关性。它不是“任意相关设计都可恢复支持”的定理。定位：Definition 1.2、Theorem 1.5、Corollary 1.6、Models 1.7–1.8。

**本项目如何使用：** 强调相关性发生在活跃变量之间、活跃与非活跃变量之间，或非活跃变量之间，含义不同。我们的等相关分组没有施加该随机旋转，因而不能用这篇定理直接证明本次高相关设置的恢复表现；也不能把其预测误差结论改写为精确支持恢复结论。

## 3 Minsker 与 Shen 2025

**准确元数据：** Stanislav Minsker, Yiqiu Shen。*The impact of contamination and correlated design on the Lasso: An average case analysis*. **Statistics & Probability Letters 223:110417，2025 年 8 月**。期刊 DOI 与 Crossref 登记的年份、卷号、文章号一致；作者公开稿页脚的 2025 年 3 月为投稿稿日期，不代替正式发表日期。[出版方页面](https://www.sciencedirect.com/science/article/abs/pii/S0167715225000628)；[DOI](https://doi.org/10.1016/j.spl.2025.110417)；[作者全文](https://stasminsker.github.io/papers/contaminated-lasso.pdf)。引用键 `minsker2025`。

**设计与任务：** Appendix F（作者稿 pp.23–24）使用 `n=1000,p=1100` 的合成数据，220 个大小为 5 的相关块，组内相关 `1/log(p)≈0.14`，污染比例从 0 到 15%，10 次重复。比较普通 Lasso 与显式估计稀疏污染向量的增广鲁棒 Lasso；未使用公开真实数据。

**结论边界：** 固定设计满足 coherence、元素幅度及稀疏度等条件，对随机支持和符号的“典型”稀疏向量给出平均情形保证；更强的符号/支持恢复还需信号幅度条件。不是对全部支持或强相关 `ρ=0.9` 的无条件保证。定位：Assumptions 2.1–2.3、Theorems 2.2、2.4–2.5。

**本项目如何使用：** 它的分组设计与我们的受控实验有可比较之处，但研究还引入污染，并改变优化变量和目标。可用于说明相关设计的保证依赖支持模型及误差类型；本项目未测鲁棒性或污染，因此不声称复现其结论。

## 4 Alrashdi 与 Alrasheedi 2024

**准确元数据：** Ayed M. Alrashdi, Masad A. Alrasheedi。*Square-root lasso under correlated regressors: Tight statistical analysis with a wireless communications application*. **AIMS Mathematics 9(11):32872–32903，2024**；2024 年 11 月 7 日接受，11 月 20 日发表。[期刊页面](https://aimspress.com/article/doi/10.3934/math.20241573)；[正式全文](https://www.aimspress.com/aimspress-data/math/2024/11/PDF/math-09-11-1573.pdf)。引用键 `alrashdi2024`。

**设计与任务：** 使用合成相关高斯矩阵与稀疏二元信号，并应用于 GSSK/MIMO 信号检测，不是公开表格回归数据。Section 3.2 示例设置 `N=256`、测量比 `M/N=0.85` 或 2、稀疏比例 0.2、Toeplitz 相关参数 0.3，每个参数点 100 次 Monte Carlo；Section 6 分析通信错误率。

**结论边界：** 主模型为 `A=Ψ^(1/2)G/sqrt(N)` 的**左侧/接收端相关**；本项目是特征之间的相关。论文在高维固定比例、相关高斯设计与高斯噪声条件下刻画 square-root Lasso 的渐近系数误差和检测指标。它的平方根损失也不同于本项目平方损失。定位：Section 2.2、Theorem 3.1、Sections 3.2、6。

**本项目如何使用：** 可作为相邻问题的近期发展与未来方向，但应次于前三篇。尤其不能看到“correlated regressors”就写成与本项目同一个特征协方差模型，也不能把论文的系数 RMSE 当作我们的测试预测 RMSE。

## 报告更新时的建议

优先以 Stallrich 等补充“相关结构和效应方向影响恢复”，以 Buhai 补充“限制特征值与相对方向比单一相关性标量更细致”，并明确其预印本身份。若需要第三篇正式期刊研究，用 Minsker 与 Shen 说明分组相关设计下平均情形与污染的后续研究，同时保留目标差异。Square-root Lasso 论文可放入扩展讨论，无需为了增加近年引用而把它写成主实验的理论基础。

这些文章提供的是研究脉络和解释边界。本次代码、数据规模和网格均不因此自动获得其定理保证；本次也没有实现它们的设计搜索、随机旋转、鲁棒增广或平方根损失。
