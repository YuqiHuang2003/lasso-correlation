# 列顺序与 Lasso 支持评价：直接近邻及实现核查

核查日期：2026-10-28，Asia/Shanghai。只读取原始论文、作者全文和官方实现；没有运行新实验，也不修改冻结数据、driver或协议。本文件是针对当前机制问题的窄范围核查，不是系统综述，不能以“本次未检索到同样实验”推导首创性。

## 先区分三个问题

1. **坐标访问顺序影响路径或速度**：这是已有优化研究和实现中的明确事实，不能作为新发现。
2. **固定数学目标的不同数值输出具有不同支持/解释**：Lasso非唯一性、近最优解多样性与有限精度支持计算已有直接研究；两个小gap输出不同不证明数学多解。
3. **本阶段的经验问题**：把一次固定的输入列置换作为处理，逆映回原坐标，在同一X/y/λ、共同证书与预算下，量化同一实现的变量支持、TP/FP/FN和F1变化；同时保留原顺序重复QC、全部计划分母及未决记录。它是具体基准的输出资格审计，不是新的坐标选择算法。

## 更直接的已有实现：knockpy主动打乱Lasso输入列

[作者维护的knockpy 1.3.4 API](https://amspector100.github.io/knockpy/apiref.html#knockpy.knockoff_stats.fit_lasso)及[对应可读源码](https://amspector100.github.io/knockpy/_modules/knockpy/knockoff_stats.html#fit_lasso)是比一般“坐标路径依赖”更直接的先例。`fit_lasso`先拼接原变量与knockoff列，随机排列列，再调用scikit-learn；同时返回排列与逆排列。只有一个`alpha`时直接调用`Lasso`，因此不只涉及交叉验证选择λ。`LassoStatistic.fit`随后按逆排列恢复系数身份。源码与API明确将此安排解释为保持对称性、避免FDR控制违规。

它表明**Lasso拟合中的输入列次序偏好，以及拟合前随机列排列、拟合后恢复身份的做法，已有作者官方实现明确处理**；不能以该效应或该处理流程本身为新发现。但这份实现没有给出本阶段“同一实现、多次固定排列、共同证书与预算、原变量支持F1范围及全目标原序重复QC”的实验结论，也不能据此推断所有数据一定敏感。这里引用软件版本1.3.4，访问日期2026-10-28，不把网页访问时间写成论文年份。其相关方法论文[Spector与Janson，*Powerful Knockoffs via Minimizing Reconstructability*，Annals of Statistics 50(1):252–276，2022](https://lucasjanson.fas.harvard.edu/research.html)的[作者全文Appendix G](https://arxiv.org/html/2011.14625)介绍knockpy；未以该附录替代具体代码功能的历史版本证明。

## 原始论文中的直接与相邻问题

### Li、Osher，2009：L1问题的扫描次序与有限迭代稀疏性

Yingying Li, Stanley Osher. *Coordinate descent optimization for l¹ minimization with application to compressed sensing; a greedy algorithm*. **Inverse Problems and Imaging 3(3):487–503, 2009**。[期刊原始记录](https://www.aimsciences.org/article/doi/10.3934/ipi.2009.3.487)，[作者CAM全文](https://ww3.math.ucla.edu/camreport/cam09-17.pdf)。不能把网页近期抓取时间当成论文年份。

定位：作者PDF §1.1（p.2）区分sequential/cyclic sweep与其他扫描；§2.2–2.3（pp.3–4）、Figure 2比较顺序扫描与贪心扫描的中间系数稀疏性。该文处理L1正则化最小二乘并用于Basis Pursuit，已明确研究访问规则和收敛/稀疏路径的关系。它不是当前固定库实现、达到同一高精度资格后、逆映射原变量F1极差的审计。**因此“顺序可能改变中间支持”本身早已有直接先例。**

### Kim、Lee、Yun，2025：随机排列坐标下降的近期速率理论

Donghwa Kim, Jaewook Lee, Chulhee Yun. *Provable Benefit of Random Permutations over Uniform Sampling in Stochastic Coordinate Descent*. **ICML 2025，PMLR267:30480–30547**。[正式记录](https://proceedings.mlr.press/v267/kim25x.html)，[作者全文](https://arxiv.org/html/2505.23152v1)。

定位：§2.2式(2)–(3)定义RCD与RPCD，RPCD在**每个epoch重新抽取排列**；§3给特定正定二次结构的速率比较；§4.2及Appendix F研究二次、二次加log-sum-exp和L2正则logistic目标，指标是迭代点误差。它是2025年的直接“顺序/排列”优化近邻，但不是普通Lasso支持F1实验。本项目每次拟合固定一个输入列排列，既不等于逐epoch重洗牌，也不检验该文的速率定理；不能借此声称新随机坐标法或RPCD优越性。

### Tibshirani，2013：不同支持与共同拟合的理论背景

Ryan J. Tibshirani. *The lasso problem and uniqueness*. **Electronic Journal of Statistics7:1456–1490, 2013**，DOI10.1214/13-EJS815。[作者正式论文PDF](https://www.stat.cmu.edu/~ryantibs/papers/lassounique-ejs.pdf)，[可检索全文](https://arxiv.org/html/1206.0313)。

定位：§2.1、Lemma1（正式pp.1460–1461）说明精确Lasso解共享拟合，λ>0时共享L1范数；§4.2与§5讨论解集及活跃集。它已经提供“系数/支持不必由相同拟合唯一决定”的基础，但不是输入列顺序实验。这里的结论针对精确最优解；本项目仅观测小gap数值点，不能倒用这些性质证明多解或唯一性。

## 与已核近期近邻的关系

| 原始来源与定位 | 已覆盖内容 | 对本阶段的限制 |
| --- | --- | --- |
| **Bu等，2023**，SPRINTER-CHANOL；[正式DOI](https://doi.org/10.11221/jima.73.223)、[全文](https://www.jstage.jst.go.jp/article/jima/73/4E/73_223/_pdf)，§2.1.2–2.1.3（p.225）、§4（pp.228–231） | 利用多组近最优Lasso表示研究高相关变量选择，实验包括TP/FP/F1；CHANOL思想在文中明确来自更早工作 | 不能把多表示或评分差异当首次发现。它主动构造/使用多样近最优解；没有将其改写成已验证的“固定库输入列置换+原序重复QC”结论 |
| **Bastounis、Cucker、Hansen，2023 Part I**；[所读v1全文](https://arxiv.org/html/2312.11425v1)，§1–2、§3 Example3.1/Table1、§6–7 | 近相关输入下有限精度支持计算失败及条件分析，数值例已经显示输出支持可随病态程度改变 | 所读版本按2023预印本引用；不宣称没有后续发表。它研究近似输入和可信支持计算，本实验不改变输入精度/位宽，也不实现其可信算法 |
| **同作者，2023 Part II**；[所读v1全文](https://arxiv.org/html/2312.11429v1)，§1.1、§2.1–2.3、§4 | 把随机输入、条件、统计支持恢复及计算精度连接起来 | 不能声称首次连接统计恢复与数值精度；其理论不直接给本轮有限置换的F1范围或总体概率 |
| **Moufad等，2025 skglm**；[JMLR正式记录](https://www.jmlr.org/papers/v26/24-0008.html)、[全文](https://www.jmlr.org/papers/volume26/24-0008/24-0008.pdf)，§2（pp.2–3） | 模块化求解器和working-set框架，利用稀疏支持缩小子问题 | 这是实现及算法背景，不是“列顺序改变F1”的实证证据；该软件论文本身不授予本项目新颖性 |
| **Nouraie、Smith、Muller，2026**，*Stability selection via variable decorrelation*，Statistics and Computing36:160（2026-06-10正式发表；[期刊全文/DOI10.1007/s11222-026-10916-7](https://link.springer.com/article/10.1007/s11222-026-10916-7)），§1的n100/p5交换前两列例、§2/Algorithm1 | 明确展示列顺序改变Gram–Schmidt所得Q后，Lasso所选变量身份改变；方法排序、正交化后结合stability selection，强调Q代理不保留原变量的精确解释 | **不是保持原X/y/λ目标不变的求解器顺序审计**：Q随输入顺序改变，原变量与代理变量评价对象不同。不能把此例当本轮同目标证据，也不能省略这一近期“顺序与选择”近邻。其2025预印本[v1](https://arxiv.org/html/2505.20864v1)的玩具例比例为0.8，正式版为0.9；引用以正式版为准 |

## 官方实现：列索引确实进入计算过程

### scikit-learn

[官方Lasso文档](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)的`selection`区分顺序cyclic与逐次随机选择。网页当前stable显示1.9.1；本阶段冻结版本为1.9.0，因此另外核查了[1.9.0官方Cython源码](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.9.0/sklearn/linear_model/_cd_fast.pyx)。在`enet_coordinate_descent`中，非random分支使用当前索引顺序，随后可经`active_set`映射，依次更新残差。由此可判断改变输入索引会改变计算路径；**代码并未保证或断言这一定改变最终支持、恢复评分或精确解。**

### skglm

[官方AndersonCD源码](https://github.com/scikit-learn-contrib/skglm/blob/main/skglm/solvers/anderson_cd.py)可见working set由`np.argpartition`选出，`_cd_epoch`按`for j in ws`更新，且带Anderson外推。因此其路径不只是一个对所有原列简单循环的过程。输入排列也可能改变working-set内部排序、同分处理及后续浮点路径；这些是从代码结构作出的机制推断，不是本轮结果。

这里网页main只用于说明官方实现架构，不将它当成冻结skglm0.5的逐字节证据；实验应以本地锁定文件/依赖哈希和运行清单为准。列置换审计的处理因素是**输入列索引排列的整体影响**，不能仅凭观察就把效应全部归因于某一个内部坐标循环、working-set选择或浮点舍入。

## 可直接用于报告的保守定位

坐标扫描规则影响L1优化路径及速度已有直接研究，近最优Lasso表示和可信支持计算也已有系统分析；knockpy的官方实现还已经在Lasso拟合前随机排列输入列、拟合后恢复变量身份，以处理对称性。本阶段不提出新坐标算法，也不以“输入顺序可能影响选择”作为新发现；经验增量仅在冻结的具体基准上，保持数学目标与原变量评价对象不变，量化固定输入列排列是否改变同一实现通过共同精度检查后的支持与恢复评分。原顺序重复QC用于区分已观测重复波动，完整报告计划覆盖和未决；所获范围只是有限合格输出的观测范围，不是完整解集界。无变化只支持这些排列下的稳定性，有变化也不证明数学多解、新理论或已成立的发表贡献。若效应只出现在本次UJI抽样/预处理臂，应据实限定于该臂，并同时保留news/gene的阴性结果。

## 本次核查未作出的断言

没有把旧论文写成2023+新研究，没有把平滑二次RPCD理论等同普通Lasso恢复，也没有把论坛经验或第三方实现解读作为证据。本次原始论文核查未确认一篇与本阶段完整协议相同的“固定求解器、只置换列、共同合格精度、原变量F1范围与全目标原序重复QC”研究；这是一条有限检索结论，不是不存在直接先例或首创性的证明。已有官方实现已经足以排除“列顺序偏好从未被处理”的说法；后续写稿仍须围绕具体经验增量、阴性结果、QC与评价资格成本重新判断。
