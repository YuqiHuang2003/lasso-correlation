# Lasso 文献核查与阅读记录

本表保留第一版基础方法记录。当前报告已改用2023–2026年9篇近期工作为核心，见 `docs/recent_research_2023plus.md`；早期论文只用于既有更新和收敛结果的准确归属。

核查日期：2026 年 10 月 28 日（Asia/Shanghai）。本表汇总原始论文、作者公开版本与出版方元数据。论文中的理论结论不是本项目的实验结果；本项目不声称提出新算法。BibTeX 条目见 `report/references.bib`。

## 核心文献

| 引用键 | 文献与核查范围 | 可支持的结论及成立条件 | 对本项目的作用与限制 |
| --- | --- | --- | --- |
| `tibshirani1996` | Tibshirani (1996), *Regression Shrinkage and Selection via the Lasso*, JRSS B 58(1):267–288。核查出版方摘要与元数据。[原文与 DOI](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x) | 以 L1 范数约束最小二乘，兼顾收缩与变量选择，可产生零系数。原文采用约束形式，与适当参数下的惩罚形式对应。 | 引言中说明 Lasso 的来源与建模动机；不能由“稀疏”推出“必然选中真实变量”。本项目的损失含 `1/(2n)`，不得直接搬用其他归一化约定的 λ。 |
| `daubechies2004` | Daubechies, Defrise and De Mol (2004), *An Iterative Thresholding Algorithm for Linear Inverse Problems with a Sparsity Constraint*, CPAM 57(11):1413–1457。核查作者公开全文的摘要、问题与迭代形式。[作者全文](https://sites.math.duke.edu/~ingrid/publications/Comm_Pure_Appl_Math_57.pdf)；[DOI](https://doi.org/10.1002/cpa.20042) | 在线性逆问题中分析阈值化 Landweber 迭代，并在其算子、权重及步长等假设下证明范数收敛。 | ISTA 的方法背景；本项目用有限维近端梯度的标准步长条件，不把这篇广义算子分析中的全部假设视为自动成立。 |
| `beck2009` | Beck and Teboulle (2009), *A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems*, SIAM J. Imaging Sciences 2(1):183–202。核查全文中的算法、Theorems 3.1 与 4.4。[出版方](https://doi.org/10.1137/080716542)；[公开全文](https://www.cs.cmu.edu/~airg/readings/2012_02_21_a_fast_iterative_shrinkage-thresholding.pdf) | 对可达最优值的凸复合目标，光滑项梯度为 Lipschitz 连续且近端步骤精确时，ISTA 与 FISTA 的目标残差分别有 `O(1/k)` 与 `O(1/k²)` 上界；论文包括固定步长与回溯版本。 | 解释三种算法中的两种及理论速率。速率控制目标残差，不直接控制支持集 F1，也不保证每个有限精度或每个实例的墙钟时间排名。标准 FISTA 更新未强制目标单调，本项目直接记录其轨迹。 |
| `tseng2001` | Tseng (2001), *Convergence of a Block Coordinate Descent Method for Nondifferentiable Minimization*, JOTA 109(3):475–494。核查全文的分离结构、Theorem 4.1 及出版方元数据。[出版方](https://doi.org/10.1023/A:1017501703105)；[作者存档全文](https://www.mit.edu/~dimitrib/PTseng/papers/archive/bcr_jota.pdf) | 对满足文中正则性、紧水平集及坐标子问题等条件的可分非光滑问题，给出坐标下降聚点的平稳性结论。一般非光滑凸目标也不能无条件应用坐标下降。 | 对本项目的有限维 Lasso：光滑二次项加可分 L1；λ>0 给出有界水平集，非零列使单坐标子问题严格凸。凸性再将平稳性联系到全局最优性。不引用无条件的统一线性速率。 |
| `friedman2010` | Friedman, Hastie and Tibshirani (2010), *Regularization Paths for Generalized Linear Models via Coordinate Descent*, JSS 33(1):1–22。核查全文的最小二乘坐标更新和路径求解说明。[期刊全文与 DOI](https://doi.org/10.18637/jss.v033.i01)；[开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC2929880/) | 给出循环坐标下降、软阈值更新与路径上的热启动等实现策略，涵盖 Lasso、岭回归和弹性网。 | 作为残差增量式 CD 的权威实现参考。本项目从零初值独立比较单个 λ，不采用完整正则化路径热启动；也不把现代库的筛选或底层语言性能归因于基本 CD 理论。 |
| `zhao2006` | Zhao and Yu (2006), *On Model Selection Consistency of Lasso*, JMLR 7(90):2541–2563。核查期刊摘要与元数据。[期刊原文入口](https://www.jmlr.org/papers/v7/zhao06a.html) | 在论文的固定维数和高维渐近设置及相应信号、噪声、正则化条件下，不可表示条件与 Lasso 模型选择一致性密切相关；条件主要依赖预测变量的协方差结构。 | 说明相关性会影响支持恢复，但不能仅由某个总体相关系数 ρ 推断有限样本是否恢复成功；报告中不将该渐近条件当作本实验已验证的充要判据。 |
| `wainwright2009` | Wainwright (2009), *Sharp Thresholds for High-Dimensional and Noisy Sparsity Recovery Using ℓ1-Constrained Quadratic Programming (Lasso)*, IEEE TIT 55(5):2183–2202。核查作者出版目录和预印本摘要。[作者目录](https://wainwrigwork.github.io/bib/martin_papers_by_year.html)；[预印本](https://arxiv.org/abs/math/0605740)；[DOI](https://doi.org/10.1109/TIT.2009.2016018) | 对满足互不相干等条件的高斯设计，分析维数、样本量和稀疏度与噪声下支持恢复概率的渐近阈值。预印本题名略短于正式版。 | 说明恢复还受样本量、信号与噪声约束。不能从这些渐近阈值直接声称本项目的 `n=2000,p=1000,s=20` 一定恢复，也不能把高精度优化当作统计恢复的保证。 |
| `zou2005` | Zou and Hastie (2005), *Regularization and Variable Selection via the Elastic Net*, JRSS B 67(2):301–320。核查出版方摘要与元数据。[原文与 DOI](https://doi.org/10.1111/j.1467-9868.2005.00503.x) | 弹性网结合 L1 与 L2 惩罚，讨论相关预测变量的分组效应，展示 Lasso 在相关变量选择上的局限。该刊另有 2005 年勘误。 | 用作相关特征讨论的背景，不新增弹性网实验，不宣称论文证明“任意相关性增大都会使 Lasso 恢复更差”。若以后复现弹性网具体公式，须同时查看勘误。 |
| `fercoq2015` | Fercoq, Gramfort and Salmon (2015), *Mind the Duality Gap: Safer Rules for the Lasso*, ICML/PMLR 37:333–342。核查 PMLR 摘要与引用元数据。[会议原文入口](https://proceedings.mlr.press/v37/fercoq15.html) | 将原始对偶间隙用于安全筛选，在收敛求解器等条件下使安全区域收缩；算法适用于多种求解器。 | 支撑以可行对偶点和间隙构造优化证据的思路。本项目只使用间隙诊断，不实现安全筛选；“安全筛选识别求解目标的活跃集”不是“识别数据生成的真实支持集”。 |
| `ndiaye2017` | Ndiaye, Fercoq, Gramfort and Salmon (2017), *Gap Safe Screening Rules for Sparsity Enforcing Penalties*, JMLR 18(128):1–33。核查期刊摘要与元数据。[期刊原文入口](https://jmlr.org/papers/v18/16-577.html) | 推广基于对偶间隙的安全筛选框架，覆盖多类稀疏惩罚及广义线性模型，特别适合块坐标下降。 | 交代后续发展和成熟求解器可能具有的额外机制；参考库只作目标值核查，其性能不作为本项目自实现算法的公平基线。 |
| `bertrand2021` | Bertrand and Massias (2021), *Anderson Acceleration of Coordinate Descent*, AISTATS/PMLR 130:1288–1296。核查 PMLR 元数据、全文算法 3 和 Lasso 实验讨论。[会议入口](https://proceedings.mlr.press/v130/bertrand21a.html)；[全文](https://proceedings.mlr.press/v130/bertrand21a/bertrand21a.pdf) | 将迭代外推用于坐标下降，并在最小二乘、Lasso、弹性网与稀疏逻辑回归上验证。所给理论含特定迭代结构与假设，实际版本以目标下降检查决定是否接受外推；不能概括为所有循环 CD 均有无条件统一加速率。 | 直接展示基础坐标下降后的加速发展，也支持区分理论速率与实际加速收益。本项目不实现 Anderson-CD，引用其方法思路和论文范围，不搬用其速度数字。 |
| `moufad2025` | Moufad, Bannier, Bertrand, Klopfenstein and Massias (2025), *skglm: Improving scikit-learn for Regularized Generalized Linear Models*, JMLR 26(149):1–6。核查期刊元数据和六页全文。[期刊入口](https://www.jmlr.org/papers/v26/24-0008.html)；[全文](https://www.jmlr.org/papers/volume26/24-0008/24-0008.pdf) | 介绍把损失、惩罚与求解器组合的公开框架，并利用稀疏支持的工作集策略减少优化变量，给出包括 Lasso 的基准。论文特别说明主要关注特征多于样本的设置，在相反设置下表现可不同。 | 补足课程要求中的近年发展：从基本更新规则到稀疏结构、加速与模块化实现。它是文献中的现代实现背景，不是本文已测的基线；不能把文中的速度优势归因于本项目基本 CD 内核。 |

核心阅读表共 12 篇研究文献，另有 1 条 UCI 数据集引用。新增 2021 年与 2025 年条目用于说明与本题直接相关的后续发展，不声称穷尽截至 2026 年的全部研究。

## 本项目采用的公式与引用边界

以下是将标准近端、坐标最小化与对偶构造换算到本项目目标归一化后的直接推导。它们不应写成某篇论文采用了完全相同的符号或停止规则。

目标为 `P(w)=||Xw-y||²/(2n)+λ||w||₁`。梯度为 `Xᵀ(Xw-y)/n`，其 Lipschitz 常数为 `L=||X||₂²/n`。固定步长可以取 `1/L`，也可使用满足二次上界条件的回溯；有限次幂迭代通常只给估计值，不能不经校验就称为严格上界。

定义 `Sτ(a)=sign(a)max(|a|−τ,0)`。ISTA 更新为 `w⁺=Sλ/L(w−∇f(w)/L)`。标准 FISTA 在外推点使用同一近端步骤。CD 在 `r=y−Xw` 时更新

`wⱼ⁺ = Sλ(Xⱼᵀr/n + aⱼwⱼ)/aⱼ`，其中 `aⱼ=||Xⱼ||²/n`，随后 `r ← r−Xⱼ(wⱼ⁺−wⱼ)`。

对偶变量采用未除以 λ 的尺度：`max D(u)=yᵀu−(n/2)||u||²`，约束 `||Xᵀu||∞≤λ`。令 `r=y−Xw`，取

`c=max(1, ||Xᵀr||∞/(nλ))`，`u=r/(nc)`。

因此 `u` 可行，弱对偶性给出 `0≤P(w)−P*≤P(w)−D(u)`。主停止量 `gap/max(1,|P|,|D|)` 是本项目预先规定的归一化规则，不是统计误差或模型选择一致性的证书。数值实现应独立检查可行性与微小负间隙的舍入误差。

## 写作时需要保留的区别

- **优化与恢复：** 优化残差相对于同一 λ 的 Lasso 最优值；系数与支持恢复相对于数据生成参数。两者参照物不同。
- **理论与墙钟时间：** ISTA/FISTA 的经典最坏情形迭代上界不等于实测时间排名；实际时间还取决于矩阵乘法、CD 内核、诊断开销和目标精度。
- **相关性与条件数：** 分组等相关总体块的特征值可直接算为 `1−ρ`（重数 `b−1`）与 `1+(b−1)ρ`（重数 1）。这是本项目的代数推导；有限样本标准化后的设计矩阵还需实测，不能把总体块条件数写成样本 Hessian 的精确值。
- **内部实验与论文结论：** 本实验只能为指定数据机制、λ 网格、随机种子和实现提供证据；未观测到预期趋势也应如实记录。

