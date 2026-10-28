# Workshop 研究证据与复现

本包只包含“固定优化 gap 下的支持评价充分性”研究。它保留先导、独立确认、真实设计半合成验证、小型复现和数值核验的研究证据；课程作业旧报告及原始输入数据不在包中。姓名、学号保持空白。

主报告为 `report/workshop/research_update_zh.pdf`；先导简报为 `report/workshop/pilot_brief_zh.pdf`。结论从保存分析动态生成，不把数值参考共识称为精确支持，不把追加观测时间称为可保证部署节省。研究轨迹可以超时；完整记录和诚实分母比“全部收敛”更重要。

## 目录与研究设计

- `src/lasso_study/`：固定的合成/真实设计、基础求解器、现代参考包装器和 C 坐标更新源码。
- `configs/workshop_*.json`：原始注册配置；`configs/skglm_environment.json`：现代求解库环境说明。
- `results/workshop/{smoke,pilot,confirm,real}/`：配置、环境、完成记录、全部运行状态、两套数值参考、系数轨迹 NPZ、核验和分析。异常记录即使后来补齐也保留。
- `results/workshop/reproduce/`：两个预定确认目标、四次新 FISTA/CD 轨迹的复现证据；已存数值参考直接复用，不重新拟合参考。
- `figures/workshop/`、`report/workshop/`：研究图和报告，及其内容/验证 JSON。预览图片、字体和渲染中间文件不入包。
- `manifest_workshop.json`：逐载荷 SHA256、CRC32 和字节数；包外 `package_workshop.json` 记录 ZIP 哈希、全部验收结果和资源账目。

合成设计为 n=500、p=1000、真非零数20，三种几何为 block ρ=0、block ρ=0.9、Toeplitz ρ=0.9；两种支持布局、两种幅度设置及 FISTA/CD 配对。ρ=0 时两族设计相同，仅保留一个公共基线。确认种子为200–219，与先导100–102分离。240个条件目标、480条轨迹只有20个独立数据种子；条件、方法、阈值和检查点不能当作独立重复。

真实设计臂从 UJIIndoorLoc 固定 seed-0 训练池抽取1000行，设置真非零数15，使用300–309种子。每个子样本先去常量，再按原 WAP 顺序及完整20列索引块保留特征，p随保存元数据变化。原响应不用于此半合成任务、选变量或调参数；响应由预设稀疏系数与固定噪声生成。40个目标、80条轨迹只有10个独立抽样/噪声种子，且不与合成臂宣称同难度横比。

## 环境

从解压后的 `comp6704_workshop/` 目录执行以下命令。需要 Python 3.13、可用的 GCC，以及可安装锁定依赖的平台。原记录的准确版本保存在各阶段 `environment.json`。版本不同会改变数值签名，不能直接覆盖旧结果或宣称原计时一致。

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-modern.txt -r requirements-report.txt
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
```

也可将现代求解库隔离安装到项目本地目录；基础科学依赖仍安装在虚拟环境中：

```bash
.venv/bin/python -m pip install -r requirements.txt -r requirements-report.txt
.venv/bin/python -m pip install --no-deps --target .modern_deps -r requirements-modern.txt
```

`skglm==0.5`、`numba==0.68.1`、`llvmlite==0.50.0` 是锁定现代依赖。其库内部容差不等于共同归一化对偶 gap；各参考必须通过项目共同证书、预算、系数、目标和阈值支持/符号共识检查。C 动态库和 JIT 缓存会在本地生成，不随包分发。无需 GPU。

## 原始输入数据

包内没有 `data/` 目录或原始/预处理样本矩阵。真实设计核验需先从官方源下载 UJIIndoorLoc：

```bash
.venv/bin/python scripts/download_benchmarks.py --dataset uji
```

来源为 [UCI UJIIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc)，DOI [10.24432/C5MS59](https://doi.org/10.24432/C5MS59)，官方许可 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。归属：Torres-Sospedra、Montoliu、Martinez-Us、Arnau 和 Avariento（2014）。ZIP、CSV 的预期 SHA256、许可链接和下载元数据保存在 `docs/workshop_uji_source.json`；下载器校验冻结快照，源内容变化会显式失败。研究只使用官方 `trainingData.csv` 构建的源训练池。原官方验证集未用于本半合成任务。

下载脚本也保留既有 Leukemia 入口，因共享文件完整性而随包提供；该数据不属于本 workshop 研究，也没有放入包中。不要为本研究运行 `--dataset all`。

## 先检查保存证据

完整数值核验会按固定种子重构 X/y，并重算两参考、最终点及五个 gap 首次观测点的共同证书；它不运行优化器。所有保存检查点还检查支持计数、H/R记账与掩码，Gap Safe 球测试不在每个检查点重新计算。真实臂需上述官方输入。请在没有其他计时实验时执行：

```bash
.venv/bin/python scripts/verify_workshop.py smoke pilot confirm real
```

每个阶段 `verification.json` 必须 `all_passed=true`、`numerical_recomputed=true`。`--structure-only` 适合排查文件结构，不能替代正式数值验收。核验重构、矩阵向量乘和证书检查是新增验证成本，不是原优化计时的重跑。

基础小测试不需要运行整个网格：

```bash
.venv/bin/python -m pytest tests/test_workshop_data.py tests/test_workshop_runner.py tests/test_solvers.py
```

包中保留所有现有测试；部分旧数据测试属于课程研究，可能需要本包未分发的旧输入。不要把缺失旧输入改成伪造数据。

## 重建分析和报告

在核验通过后，按阶段重建分析和图：

```bash
.venv/bin/python scripts/analyze_workshop.py smoke
.venv/bin/python scripts/analyze_workshop.py pilot
.venv/bin/python scripts/analyze_workshop.py confirm
.venv/bin/python scripts/analyze_workshop.py real
.venv/bin/python scripts/diagnose_workshop_references.py
.venv/bin/python scripts/analyze_workshop_reference_impact.py
.venv/bin/python scripts/build_workshop_brief.py
.venv/bin/python scripts/build_workshop_update.py
```

正式分析要求当前全数值核验及输入哈希仍匹配；`--allow-unverified` 仅用于开发，不能进入最终包。先导简报内置六条已核查书目，进展报告增补两条近最优多支持/对偶支持不确定性的近期工作；定位见 `docs/workshop_nearest_work.md` 和 `docs/workshop_reference_ambiguity_literature.md`，无需课程报告的主 BibTeX 文件。共享渲染器会自动生成本地字体、公式中间图和页面预览；报告不需要 TeX 编译。最终交付前仍应逐页视觉检查。

参考近依赖诊断和恢复评价后果分析都只读取保存系数/固定设计，不新增优化。后者在全部40个目标上分别以两套候选参考系数计算真实支持F1，记录评价会否随参考实现而变；它是10个配对种子的事后比较，不能证明哪套更正确或数学多解，也不取消原坐标参考不确定记录。

图中的 observed reference-match yield 定义为“已达指定 gap、参考共识且逐索引匹配的轨迹数 / 全部预注册轨迹数”。未达档或参考不确定不计作已观测匹配，也不据此认定支持错误。正文另外报告已达且参考共识分母。Wilson 区间仅在固定几何/布局/幅度/方法/τ/gap的条件内计算，每个种子一条；不对汇合条件或两种配对方法做独立同分布推断。主τ为1e-6，1e-4为支持阈值敏感性，先匹配后离开的轨迹仅按主τ逐方法计一次。

## 小型新求解复现

小型复现重新求解两个预定确认目标：`block_seed200_rho0_dispersed_strong` 和 `toeplitz_seed200_rho0.9_clustered_mixed_weak`，各运行 FISTA/CD，共四次新轨迹。它校验重构哈希、最终系数、目标和共同证书，不要求机器耗时相同；两个目标的既有双参考系数复用，不新增参考拟合。

```bash
.venv/bin/python scripts/reproduce_workshop.py --check-only
.venv/bin/python scripts/reproduce_workshop.py --output results/workshop/reproduce_local
```

已归档的 `results/workshop/reproduce/` 不应覆盖。新输出必须为空且与原研究目录分离。正式原包的 `reproduction.json` 记录四次新轨迹、通过状态、输入与源签名及实际总秒数；本地新目录用于独立检查，不自动替换它。

如需从零运行完整研究，应先把已归档证据移到独立保留位置，再在新工作副本执行下列顺序。不要在原证据目录直接覆盖，也不要并行计时：

```bash
.venv/bin/python scripts/run_workshop.py smoke
.venv/bin/python scripts/run_workshop.py pilot
.venv/bin/python scripts/run_workshop_followup.py confirm
.venv/bin/python scripts/run_workshop_followup.py real
```

之后按“核验→分析→小型复现→报告→打包”完成。固定源/依赖与配置签名不同会拒绝复用；超时、参考不确定和异常都应保留，不能仅挑已收敛轨迹汇报。

## 最终打包与资源范围

只在所有计时、核验、分析、复现和报告更新结束后执行：

```bash
.venv/bin/python scripts/package_workshop.py --check-only
.venv/bin/python scripts/package_workshop.py
```

输出为 `deliverables/comp6704_workshop_research_bundle.zip`，统一前缀 `comp6704_workshop/`，以及包外 `manifest_workshop.json`、`package_workshop.json`。README 使用本文内容，不改原工作区 README。脚本核验完整数值验收、当前源/配置/输入哈希、四次新轨迹复现通过、报告页数/大小/哈希，再逐载荷读回 SHA256 与 CRC32，最后复查工作区输入未变化；活动 `*_console.log` 从载荷中排除。无需所有研究轨迹都收敛。

资源记录从旧版验证提取只含资源的快照 `docs/workshop_prior_resources.json`，旧研究已记录约0.5148小时（1853.229秒，单线程）。再加本研究四阶段 `completion.wall_sec_this_invocation`、这些阶段计时外预热、四次新轨迹复现的实际总时间。数值核验墙钟单列，并另外给出含核验的扩展总数；复现计时已包含它自身准备、诊断和写出，不另加一次预热。GPU使用为0小时。

这些是已保存墙钟账目，不是逐线程CPU核心小时。安装、阅读、编码、分析、绘图、报告生成及打包未计入；没有记录或被更早覆盖的尝试也不能事后精确还原。原始阶段计时、预热和资源边界都保留在包内，不将不收敛或验证成本藏入“快解”。
