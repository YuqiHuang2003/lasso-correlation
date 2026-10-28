# 2026 年 Lasso 核心集研究核查

Zou, Yuanbin; Huang, Junyu; Wang, Jianxin; Feng, Qilong. *On Coreset for LASSO Regression Problem with Sensitivity Sampling*. **ICLR 2026 正式会议论文**。[会议记录](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1f541e1a674697198cbcf6fd44f63219-Abstract-Conference.html)；[原始全文](https://proceedings.iclr.cc/paper_files/paper/2026/file/1f541e1a674697198cbcf6fd44f63219-Paper-Conference.pdf)。核查日期为2026年10月28日。

研究利用行敏感性抽样构造加权小数据集，以近似完整Lasso目标；理论依赖给定误差/概率参数及尺度条件。它减少样本行数，不是识别零系数的安全变量筛选，也不等于本文对完整目标求解的对偶精度证书。

Section 4（PDF第9–10页）使用如下设计矩阵：

| 数据 | 样本 n | 特征 d |
| --- | ---: | ---: |
| Synthetic | 10000 | 200 |
| mediamill | 30993 | 120 |
| CTs | 53500 | 386 |
| mnist8m | 8100000 | 784 |

重要边界：这些“真实数据”提供设计矩阵，响应均用随机稀疏0/1系数与噪声重新生成；不能把结果描述成这些数据原始任务的真实响应预测。Synthetic还特意构造不均匀杠杆分数。比较完整数据FISTA、敏感性抽样后FISTA、均匀抽样后FISTA；每个实验10次，统计原目标损失、运行时间和非零数，支持阈值1e-6。核心集大小为d的1/2/5/10/15/20倍，λ取0.5/1/5/10。

论文含另一个固定硬件/实现环境，报告的速度数字是论文内结果；本项目没有复现其抽样算法或800万样本实验。本次引用用于把“样本压缩”和“变量筛选/近端更新”区分开，也说明应同时检查目标质量、稀疏度和速度，不能只计时。第二版采用完整目标、真实已知支持的受控合成实验及实际公开设计上的求解验证。
