# 第二版实验与交付包复现

第二版主题固定为特征相关性与稀疏恢复。实验采用固定噪声σ=1、固定绝对正则化规则、两种真支持布局、三档稀疏度，以及ISTA、FISTA、循环CD和skglm上下文基线。第一版结果保留作过程记录；第二版分析只能读取 `results/v2/`，不能把第一版、smoke或pilot混入主结果。

## 环境安装

本次环境为Linux、Python 3.13，数值比较使用float64和单线程CPU。基础科学包锁定在 `requirements.txt`，报告依赖锁定在 `requirements-report.txt`，现代基线依赖为skglm 0.5、Numba 0.68.1、llvmlite 0.50.0。循环CD需要GCC编译本项目提供的C99源码；生成的库不放入交付包。

可以使用普通虚拟环境一次安装：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-modern.txt -r requirements-report.txt
```

也可以让三个可选现代基线包独立安装到项目目录，保留基础科学库：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-report.txt
.venv/bin/python -m pip install --no-deps --target .modern_deps -r requirements-modern.txt
```

`modern_solver.py` 自动将存在的 `.modern_deps` 加入导入路径。这里使用 `--no-deps` 是因为NumPy、SciPy、scikit-learn等依赖已经按基础锁定文件安装；不能把它当成省略基础依赖的安装方式。两种安装方式选其一即可，不依赖本次机器上其他项目的环境。

## 获取数据及检查

```bash
.venv/bin/python scripts/download_benchmarks.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python -m pytest -q
.venv/bin/python scripts/run_v2.py smoke
```

数据说明见 `docs/benchmark_sources_v2.md`。UJI只使用官方训练文件的WAP特征预测LONGITUDE，训练子集专属常数/重复列删除导致三个种子实际保留452、452、451列。Leukemia合并38/34条文件为72×7129固定优化任务，不能评估真实支持恢复或泛化。LIBSVM来源页未明确该快照的再分发许可，因此全部输入数据均从交付包排除；通过脚本从原来源下载。包内 `docs/data_manifest_v2_*.json` 保留来源、许可和内容哈希，不含原始数据值。

完整复现第一版需要额外运行 `scripts/download_data.py`，但第二版实验本身不需要新闻数据。若不复现第一版，保留其原始结果记录即可。

## 运行第二版

`configs/study_v2.json` 是冻结协议：5种子×4相关性×3稀疏度×2布局×3正则化×4方法，共1440条合成求解；UJI三种子×三档验证集候选×四方法，共36条；Leukemia三档正则化×三次计时重复×四方法，共36条。Leukemia重复是计时重复，不是独立数据种子。

```bash
.venv/bin/python scripts/run_v2.py pilot
.venv/bin/python scripts/run_v2.py synthetic
.venv/bin/python scripts/run_v2.py uji
.venv/bin/python scripts/run_v2.py leukemia
.venv/bin/python scripts/run_v2.py reproduce
```

每个完整命令单线程顺序运行；不要同时启动另一组计时实验。主协议的单次时限为20秒。输入、配置与数值代码签名不同的缓存会被拒绝；如果需要改变协议，应使用独立版本目录保留旧证据，不能把旧结果当成新结果继续汇总。重新计时应先保留旧结果快照，并在独立副本中运行；原目录内已有且签名匹配的求解会被复用。

代表性复现包含两种条件（ρ=0、s=5、分散；ρ=0.9、s=40、聚集）的四种算法，共8条。它核对系数最大差异小于1e-7、目标值差异小于1e-8且状态一致。硬件、BLAS或依赖版本变化可能改变耗时；不能把本机计时当作跨机器保证。

skglm自身的tol是最大次微分/KKT违反量，**不等于本项目归一化对偶间隙**。包装器至多使用三档库容差、每次从零重新拟合，所有尝试及事后证书都计时，再以共同证书验收。库只保存初始点和各次完整拟合终点，不能据此声称获得与ISTA/FISTA/CD检查频率相当的内部精度轨迹。Numba预热单独记录；库拟合期间无法强行中断，超出20秒会保留 `budget_exceeded` 状态。

## 重建图表、报告与交付包

```bash
.venv/bin/python scripts/analyze_v2.py
.venv/bin/python scripts/build_report_v2.py
.venv/bin/python scripts/package_v2.py
```

报告产物为 `report/report_zh_v2.pdf`、`report/main_v2.tex` 和 `report/validation_v2.json`。姓名与学号按用户要求留空。报告生成器可以从PyMuPDF的内置CJK字体重建字体资产，交付包不需要包含生成字体和预览图。TeX源码在完整XeLaTeX/BibTeX环境下编辑与编译；本次交付PDF的实际生成方式以报告验证文件为准，不把提供TeX源码等同于已在本机通过TeX编译。

打包前需先完成分析、报告和8条复现。`package_v2.py` 不运行优化算法：它重新生成输入、检查1440/36/36网格逐格记录、数值签名、配置哈希、数据哈希、每条已返回结果的系数文件哈希和共同证书，以及参考证书与PDF页数/大小。允许合法超时、未收敛与显式失败记录；它们进入 `results/v2/verification.json` 的状态统计及失败列表，不能当成快解，也不会因打包而被隐藏。参考系数未保存，因此打包只能核查所存参考证书的内部一致性、可行比例和精度，不会假称重新求解参考问题。

输出为 `deliverables/comp6704_research_bundle_v2.zip`，旁边保留 `package_v2.json` 和 `manifest_v2.json`。ZIP内逐文件SHA256回读核对；清单不递归计算自身哈希，ZIP哈希写在外部sidecar。包保留源码、配置、测试、说明、原始结果、图和报告，排除输入数据、依赖环境、编译产物、缓存与预览资产。

资源统计汇总已记录的第一版和第二版各阶段墙钟时间，并另加第二版阶段计时之外的预热；包含smoke、pilot和复现。它不包含安装、研究阅读、编程、绘图、排版和打包，也不能补回未记录或被覆盖的更早尝试。线程数固定1，GPU占用为0。
