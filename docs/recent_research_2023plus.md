# 2023 年以来的核心研究与第二版实验设计

核查日期：2026 年 10 月 28 日（Asia/Shanghai）。本版相关工作以近期研究为主，经典论文只用于既有更新公式和收敛结果的准确归属。题目保持“特征相关性对 Lasso 求解效率与稀疏恢复的影响”。

| 工作 | 年份与状态 | 实验数据或设计 | 对本项目的具体作用 |
| --- | --- | --- | --- |
| [Tay 等：glmnet 的广义模型路径](https://www.jstatsoft.org/article/view/v106i01) | JSS 2023，正式发表 | 模拟计时，Spambase 分类示例 | 区分完整正则化路径、热启动、模型选择与单个固定目标计时 |
| [Larsson 等：Coordinate Descent for SLOPE](https://proceedings.mlr.press/v206/larsson23a.html) | AISTATS 2023，正式发表 | 相关高斯稀疏信号；bcTCGA、news20、rcv1、Rhee2006 | 显式检查稀疏真值与多指标；SLOPE 的排序惩罚不同于普通 Lasso，未实现其算法 |
| [Żogała-Siudem 与 Jaroszewicz：索引筛选](https://doi.org/10.1007/s10618-023-00950-8) | DMKD 2024；online 2023 | rat-eye 120×31098，约800×800万的Eurostat | 区分计算活跃集与真实支持；索引成本和近似查询的安全性另计 |
| [Yang 与 Hastie：adelie](https://arxiv.org/abs/2405.08631) | 2024 预印本，未核实期刊发表 | ρ=0/0.5/0.95 的合成路径；Leukemia、UJIIndoorLoc、Prostate | 采用其普通 Lasso 实验中的两套真实设计，显式研究相关性；本轮不是整条路径复现 |
| [Stallrich 等：Lasso 符号恢复设计](https://doi.org/10.1093/jrsssb/qkaf026) | JRSS B 2025，正式发表 | 小规模两水平超饱和设计；已知/未知效应方向 | 增加支持布局、混合符号、exact-sign recovery；不将局部最优条件或猜想说成普遍保证 |
| [Buhai：Lasso and Partially-Rotated Designs](https://arxiv.org/abs/2505.11093) | 2025 预印本 | 理论半随机/部分旋转设计，没有真实预测基准 | 区分活跃变量与非活跃变量之间的几何；本轮未进行旋转，不直接套用其保证 |
| [Minsker 与 Shen：污染与相关设计](https://doi.org/10.1016/j.spl.2025.110417) | Statistics & Probability Letters 2025，正式发表 | n1000、p1100，5列相关块，0–15%污染、10次重复 | 借鉴分组与随机支持/符号的控制；本轮无污染、不实现鲁棒增广目标 |
| [Moufad 等：skglm](https://jmlr.org/papers/v26/24-0008.html) | JMLR 2025，正式发表 | MEG/Lasso、Breast-Cancer/Cox、Drug Potency/Group Lasso | 加入安装版本skglm0.5作为现代上下文基线，统一后验对偶精度；并非复现论文软件版本和全部基准 |
| [Zou 等：Lasso 核心集](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1f541e1a674697198cbcf6fd44f63219-Abstract-Conference.html) | ICLR 2026，正式发表 | Synthetic、mediamill、CTs、mnist8m；真实设计矩阵配模拟响应 | 同时评价目标、速度和非零数；区分行抽样近似与完整目标的优化精度，本轮不实现核心集 |

## 新版实验如何承接这些文献

1. **问题结构：** 进入 n500、p1000 的高维设置，同时改变相关性、非零数和信号是否集中在同组；保留配对符号，检查活跃 Gram、样本不可表示分数与信号抵消。借鉴近期恢复研究的细化视角，不以单一ρ概括全部几何。
2. **控制混杂：** 固定单个信号幅度、σ=1和绝对λ三档，不随信号能量或λmax重新缩放；全局SNR另记录。保留第一版作为另一协议的初步结果，两版差异不能只归因于一个因素。
3. **稀疏恢复：** 不只报告非零数量，加入漏选、误选、FDR、精确支持、精确符号和组支持指标；五种子精确恢复以成功数/5呈现，不能当精确概率估计。
4. **现代参照：** 三种基础算法解释机制，skglm提供当代实现背景。其工作集、外推和KKT容差不能混同基本CD；全部尝试计时、独立后验统一证书，不虚构内部轨迹。
5. **公开基准：** UJI连续经度回归与Leukemia高维平方损失优化沿用近期文献中的数据来源。前者train-only处理后451–452列，不硬凑文献全池465；后者无泛化或真实支持结论。

详细逐篇核查、理论前提、数据规模及页码见 `recent_solver_notes.md`、`recent_extensions_notes.md`、`recent_recovery_notes.md` 和 `recent_coreset_notes.md`。主协议与独立评审见 `experiment_protocol_v2.md`、`protocol_v2_review.md`。本项目是受近期工作启发的受控课程研究，不宣称首次发现相关性影响Lasso，也不将文献中的速度或恢复数字作为本项目测量。
