# COMP6704 特征相关性与 Lasso 稀疏恢复

当前默认版本为**第二版**：受2023–2026年近期文献启发，研究“特征相关性对Lasso求解效率与稀疏恢复的影响”。正文先用中文，姓名和学号按用户要求保留空位。

最新机制阶段见[`列顺序中文简报`](report/column_permutation_v1/column_order_zh.pdf)与[`完整结果`](docs/column_permutation_findings_v1_zh.md)：同一数学目标与原变量坐标不变，仅改变输入列顺序，UJI在sklearn18/40、skglm26/40目标上出现支持变化，最大F1极差约9.63个百分点；784输出、112重复QC及历史核对均通过独立验收。news/gene对照及旧共识通过的10个UJI目标均稳定。完整记录在`results/column_permutation_v1/`，本阶段增量包为`deliverables/comp6704_column_permutation_v1.zip`；它不覆盖此前外部负结果或课程报告。

研究推进与课程交付分别保留。当前研究任务见[`TODO.md`](TODO.md)，已完成的workshop研究见[`docs/workshop_findings_zh.md`](docs/workshop_findings_zh.md)。最新外部审计结果见[`中文进展简报`](report/reference_audit_v1/external_audit_zh.pdf)及[`完整说明`](docs/reference_audit_findings_v1_zh.md)：两个新来源的72个确认目标均无双参考F1分歧，未达到预定推广门槛；目前收窄为UJI抽样/预处理案例，尚不主张已具备发表贡献。新结果位于`results/reference_audit_v1/`，不混入课程第二版汇总。

UJI有界机制对照已完成，见[`处理对照结果`](docs/uji_preprocessing_control_findings_v1_zh.md)：三个事后选定种子共12个目标，原始RSSI精确去重后完整参考共识仍2/12，近比例留锚后为12/12，但合并/删除原坐标身份，不能称原变量恢复改善。新阶段复现入口见[`研究增量复现说明`](docs/reference_audit_reproduction_v1.md)，增量包为`deliverables/comp6704_reference_audit_v1.zip`；原完整包继续保留。

## 阅读入口

- `report/report_zh_v2.pdf`：当前中文报告，`report/report_zh.pdf`同步指向相同内容。
- `report/main_v2.tex`：当前可编辑LaTeX源；`report/references.bib`为扩展引用库，正文只列实际引用。
- `docs/recent_research_2023plus.md`：9篇近期核心研究的内容、数据和本项目取舍；预印本明确标注。
- `docs/experiment_protocol_v2.md`、`docs/protocol_v2_review.md`：新版设计及独立评审。
- `results/v2/summary.json`、`results/v2/*_runs.csv`：严格核对后的汇总与逐次记录。
- `deliverables/comp6704_research_bundle_v2.zip`：当前可携带研究包。
- `docs/reproduction_v2.md`：完整环境、下载、运行、重建和交付验证步骤。

第一版180次合成及Online News实验保留在`results/main/`、`results/real/`，不混入新版。旧说明见`docs/README_v1_archived.md`，旧包另存`deliverables/comp6704_research_bundle_v1.zip`。

## 新版研究设计

合成数据n500、p1000、组大小20，进入p>n设置。ρ=0/0.3/0.6/0.9，真实非零数s=5/20/40，支持分散或每组集中5个真信号，五个配对种子。真系数幅度1、噪声σ1、有序混合正负号固定配对；λ=cσ√(2log(p)/n)，c=.5/1/2，同一c在全部结构中采用相同绝对λ，不以响应相关的λmax重定义惩罚。

360个目标各由ISTA、FISTA、循环CD、skglm求解，共1440次。恢复评价含precision、recall、F1、FDR、精确支持、完整符号、组支持、系数和预测误差。p>n时全Hessian奇异，几何诊断采用真活跃Gram及样本不可表示分数，不将其当作恢复保证。

真实设计采用近期adelie研究中的UJIIndoorLoc和Leukemia。前者预测连续经度，三个train-only处理后的划分实际保留452/452/451列，共36次；后者72×7129，二元标签只用于固定平方损失目标计时，共36次，不评估真实支持或泛化。详见`docs/benchmark_sources_v2.md`。

## 安装与快速验证

推荐Linux、Python3.13和GCC。依赖分为基础、现代求解器与报告；CD首次自动编译本项目C99源码。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-modern.txt -r requirements-report.txt
.venv/bin/python scripts/download_benchmarks.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python -m pytest -q
.venv/bin/python scripts/run_v2.py smoke
```

本机`.venv`复用了已有科学库，现代包在`.modern_deps`隔离安装；上述命令用于独立复现，不依赖本机其他项目路径。等效隔离安装命令见`docs/reproduction_v2.md`与`configs/skglm_environment.json`。

## 完整运行及重建

数值阶段顺序运行，避免并发影响计时：

```bash
.venv/bin/python scripts/run_v2.py pilot
.venv/bin/python scripts/run_v2.py synthetic
.venv/bin/python scripts/run_v2.py uji
.venv/bin/python scripts/run_v2.py leukemia
.venv/bin/python scripts/run_v2.py reproduce
.venv/bin/python scripts/analyze_v2.py
.venv/bin/python scripts/build_report_v2.py
.venv/bin/python scripts/package_v2.py
```

仅从结果重建图表：`.venv/bin/python scripts/analyze_v2.py`。默认严格验证1440/36/36笛卡尔网格、唯一性、参数与数据配对。`--allow-partial`仅供开发，不能用于报告最终结果。

签名完全匹配的现存运行会复用。重新计时或改协议应先在独立目录保留旧结果，不能把旧日志当作新数据；报告文字变更不改变数值来源，无需重跑已通过的算法。

## 计时、精度与解释

统一float64、单线程、零初始，后验归一化对偶间隙≤1e-6。基本方法每五步/epoch检查，阈值时间是采样上界。求解时间含初始化及证书/恢复诊断；预测与敏感性评估在计时后单列。近端精确SVD、数据准备、参考解、验证选择及C/Numba预热分别记录；cold-start图只为ISTA/FISTA加回SVD，不代表下载到模型的全部成本。

skglm0.5包含工作集和Anderson外推，其KKT容差不等于本项目间隙。包装器最多三次从零拟合，重试累计计时并使用共同证书。库只保存拟合端点，不与基本算法混画密集精度曲线。基本方法在更新间检查20秒；库不能中断正在进行的拟合，超额标`budget_exceeded`。未收敛单列，不能当成快解。

五种子IQR是实例离散程度，不是置信区间。Leukemia三次重复只描述同一目标计时。组F1不等于变量恢复，非零数量不等于真信号数。固定σ、单个幅度和绝对λ后，总体SNR仍随稀疏度及符号抵消变化，已记录。主支持阈值1e-6；数值敏感性1e-8/1e-4，0.01/0.05为预定硬阈值补充，不据真值选最佳阈值。

## 报告与交付

PDF由独立脚本生成并逐页检查。TeX源在完整XeLaTeX/BibTeX环境编译，分页可能不同；当前桌面连接缺少多文件TeX支持。课程PDF限制最多10页、少于20MB，身份保持空位，最终代码位置、文件名和Canvas提交由作者定稿。

打包脚本重建输入并核查保存系数的证书与哈希，不重新求解。ZIP含源码、测试、配置、结果、图和报告，排除输入数据、环境、二进制和预览；数据清单保留来源与哈希。Leukemia再分发许可未明确，按脚本从官方来源下载，不随包转发。

`results/v2/verification.json`保存网格、质量、复现和资源核查；`deliverables/manifest_v2.json`为文件哈希，`package_v2.json`为ZIP信息。研究包是本地交付物，没有自动上传或提交课程平台。日期和时间均按Asia/Shanghai。
