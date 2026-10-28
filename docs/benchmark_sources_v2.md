# 第二版数据生成与真实基准来源

核查及下载日期：2026 年 10 月 28 日。本文件记录第二版协议，与第一版 Online News Popularity 快照及其结果分开保存。本次仅完成数据准备和设计检查，未在本子任务运行完整求解实验。

## 合成数据的配对控制

入口为 `lasso_study.data_v2.make_synthetic`。默认 n=500、p=1,000、块大小20、独立测试样本1,000、系数幅度1、噪声标准差 σ=1；稀疏度、相关性、布局和种子由第二版实验配置确定。

- 各块由“块公共高斯分量 × √ρ + 特征独立高斯分量 × √(1−ρ)”构造；各块独立。
- 相同种子、相同维度共用底层设计分量和独立高斯噪声向量。相同种子/ρ的 X 和 X_test 在不同稀疏度与布局下完全相同。噪声不按信号强度重新缩放，因此不同条件的总体噪声方差均为 σ²。
- dispersed 每个有效块最多一个信号；clustered 每个有效块恰好五个信号，要求稀疏度为5的倍数。每种布局使用一个完整支持顺序，取前s个，保证不同s的支持嵌套。不同布局使用同一有序正负号向量。
- X 的均值与总体标准差仅由训练数据计算。真实系数定义在该标准化坐标，取值±1。测试集沿用训练变换。训练响应居中，测试响应减去同一个训练均值。
- 元数据保存支持顺序、正负号、设计/噪声哈希、σ、训练信号方差、实测噪声方差，以及信号方差/σ²和信号方差/实测噪声方差。SNR属于观察量，不是固定输入；不能把本协议写成固定SNR=10。

例如默认seed0、ρ=0、s=20的训练信号方差/σ²为18.4889，实测信号方差/噪声方差为18.8412。这只是生成检查，不是求解或恢复结论。

## UJIIndoorLoc 经度回归

[UCI官方页面](https://archive.ics.uci.edu/dataset/310/ujiindoorloc)提供19937条官方训练记录与1111条官方验证记录，520个WiFi接入点强度字段。许可为CC BY 4.0，数据DOI为10.24432/C5MS59。100是未探测到接入点的官方编码。[官方ZIP](https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip)

第二版只把 `trainingData.csv` 的19937条记录作为研究数据池；官方1111条外部验证记录保留原文件但不参与本轮。X只包含WAP001至WAP520，目标为LONGITUDE。LATITUDE、FLOOR、BUILDINGID、SPACEID、RELATIVEPOSITION、USERID、PHONEID、TIMESTAMP均不作为特征。保留官方100编码，再标准化；没有把100解释成实际100dBm。

[Yang与Hastie的2024预印本，§6、Table 1](https://arxiv.org/html/2405.08631v1#S6)使用19937条记录、删除冗余后465个原始特征预测经度。我们实测完整官方训练文件：520个WAP中恰有55个常数列，剩余465列没有完全重复的非恒定列，因而删除常数即可得到与其表格一致的规模。论文未给出具体清洗代码，故这里只称规模一致。

本项目先固定60/20/20划分，再只在训练子集识别常数列和完全重复列，保留首先出现的重复列代表。标准化参数也只由训练子集计算。较少被探测到的WAP可能在训练子集中恒定，因此不强求465列。

| 划分种子 | 训练/验证/测试样本 | 删除训练常数列 | 删除额外非恒定重复列 | 实际特征数 |
| --- | --- | ---: | ---: | ---: |
| 0 | 11962 / 3987 / 3988 | 68 | 0 | 452 |
| 1 | 11962 / 3987 / 3988 | 68 | 0 | 452 |
| 2 | 11962 / 3987 / 3988 | 69 | 0 | 451 |

使用NumPy PCG64的 `default_rng(seed).permutation`，前floor(0.6n)行为训练，接下来的floor(0.2n)行为验证，其余为测试。元数据保存行索引、索引哈希、保留/删除字段及训练均值/尺度。响应只减训练均值，不缩放；MSE单位为原始UCI经度坐标单位的平方。

这里评估的是官方训练数据池内的随机留出预测，不能声称测得跨设备、时间外推或官方外部测试性能。真实数据没有已知真实系数和支持集，不能计算真实支持恢复率。

## Leukemia 高维固定优化任务

[LIBSVM官方数据说明](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#leukemia)提供38条与34条两个文件，均有7129特征。页面说明其源数据经过先按观测、再按特征的标准化，并涉及合并后的数据。[leu.bz2](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/leu.bz2)、[leu.t.bz2](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/leu.t.bz2)

第二版按38条文件在前、34条文件在后的顺序合并，构成72×7129固定最小二乘Lasso优化任务。实测二元标签为−1共25条、+1共47条；本项目再对全72条X按列居中/标准化、对数值标签居中。该快照没有常数列。全部72条只用于对比目标值、对偶间隙、耗时和非零数量，不作泛化评估，也不声称这是一项连续结局回归应用或已知真实支持的恢复实验。

所核LIBSVM页面未明确标注该数据快照的再分发许可。元数据如实记录“许可未明确”，不套用UCI的CC BY 4.0。原始/合并数据保留本地供研究使用，代码提交包应排除这些数据文件，通过下载脚本获取；来源与下载说明可以保留。

## 文件、哈希与复现

下载命令：`.venv/bin/python scripts/download_benchmarks.py`。脚本保存到 `data/benchmarks/ujiindoorloc/` 与 `data/benchmarks/leukemia/`；每处的 `source.json` 包含完整来源、时间、文件大小和SHA256。下载脚本锁定本次内容哈希，未来官方内容变化时会拒绝静默替换。

| 文件 | SHA256 |
| --- | --- |
| ujiindoorloc.zip | 893512b82dfd7a7c345d84195b1c8019fbca0fa0d7820ce491ce5aa45ec3782f |
| trainingData.csv | 45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168 |
| validationData.csv | 5f90c536648cd657b2c516d20c4e0968d4003279ea6bd5d5d5322d3f1e8905c0 |
| leu.bz2 | 2f7dc1c8ea4343b17038e27ff777dad0a87d79a5fa29de54765055cc841cb75b |
| leu.t.bz2 | 582161e3d5112f6c1604d262413acca5e33cad1b3f522c681faeb369e8b4a271 |
| leukemia.libsvm | da6fe1bea490771a47f3913fb44ed42bfda6da5dc9cce8f3531dce6bfb100fc9 |

设计检查命令：`PYTHONPATH=src .venv/bin/python -m pytest tests/test_data_v2.py -q`。5项检查通过，涵盖设计/噪声配对、支持嵌套、布局计数、训练集专属预处理、来源哈希及优化专用Leukemia接口。
