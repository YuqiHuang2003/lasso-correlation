# 第一版说明归档

以下保留第一版协议与入口，现行默认使用第二版。

# COMP6704 特征相关性与 Lasso 求解研究

比较 ISTA、FISTA 和循环坐标下降在分组相关设计中的优化速度及统计恢复。报告先使用中文。所有结果由保存的运行日志生成；试跑结果与主实验分开保存。

## 阅读入口

- `report/report_zh.pdf`：中文研究报告草稿，包含实测结果、图表和参考文献。
- `report/main.tex`：可编辑 LaTeX 源文件；`report/references.bib` 为核查后的引用。
- `docs/execution_plan.md`：课程要求、项目安排与执行顺序。
- `docs/findings_zh.md`：研究发现及局限；`docs/literature.md` 为文献阅读表。
- `results/summary.json`、`results/main_runs.csv`、`results/real_runs.csv`：汇总与逐次结果。
- `deliverables/comp6704_research_bundle.zip`：代码、配置、报告和原始结果快照。

报告首页姓名和学号按用户要求保留空白横线，默认重新构建也保持留空。实验不使用 GPU。公开数据采用 CC BY 4.0，归属及处理方式见 `docs/data_source.md`。

## 环境与安装

推荐 Linux、Python 3.13 和可用的 GCC；实验固定单线程、float64。可从当前文件夹建立独立环境：

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

依赖版本与实际 BLAS、编译器及硬件另保存在每个实验阶段的 `environment.json`。本次运行的 `.venv` 通过项目内 `.pth` 复用了机器已有科学计算环境；上述安装步骤用于独立复现，不依赖该机器路径。CD 在首次运行时把本项目 `src/lasso_study/cd_kernel.c` 编译到 `build/cd_kernel.so`，无需 Numba；编译和动态库首次加载不算单次求解时间。

## 快速检查

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/python scripts/run_study.py smoke
```

smoke 是小规模流程，不能替代主实验。所有脚本路径基于项目位置解析；测试命令从项目根目录执行。保存有不同数值代码版本的阶段会拒绝覆盖，请把旧阶段移动到独立目录后重跑。不要将旧日志混入新的实验网格。

## 从头运行

```bash
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/run_study.py pilot
.venv/bin/python scripts/run_study.py main
.venv/bin/python scripts/run_study.py real
.venv/bin/python scripts/run_study.py reproduce
.venv/bin/python scripts/robustness.py
.venv/bin/python scripts/analyze.py
```

阶段顺序执行，避免并发数值任务影响计时。主实验约数分钟至数十分钟，具体取决于 CPU/BLAS。单次求解上限为 120 秒、100000 步，实际状态以日志为准。首次阈值记录以每 5 步/epoch 检查为基础，不能解读为精确的连续穿越时间。`robustness.py` 另用每 20 步检查验证计时敏感性。

只重建图表的一条命令为：

```bash
.venv/bin/python scripts/analyze.py
```

默认严格检查 180 条主合成结果、27 条真实结果及网格唯一性，缺失时退出。`--allow-partial` 仅用于开发，其产物明确标记 partial，不能用于报告最终结果。

## 报告构建

LaTeX 主文件为 `report/main.tex`，使用 `ctexart` 和 Fandol 中文字体，可在支持 XeLaTeX 的环境运行：

```bash
cd report
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

本项目另提供与相同实测结果对应的中文 PDF 排版脚本，不依赖系统 TeX：

```bash
.venv/bin/python -m pip install -r requirements-report.txt
.venv/bin/python scripts/build_report.py
```

PDF 使用 PyMuPDF 内置 Droid Sans Fallback 字体生成可嵌入的中文字体资源。绘图由 Matplotlib 生成；PDF 草稿采用简洁单栏版式，课程推荐的 NeurIPS 模板不是硬性要求。最终提交前须填写姓名、学号和报告中的代码获取位置；当前本地 ZIP 是可检查的代码交付物，没有自动上传或提交至 Canvas。

当前已验证的 9 页 PDF 由上述排版脚本生成。桌面连接只支持单文件 TeX 编译，不能校验本项目的图片与参考文献组合；LaTeX 源请在完整 XeLaTeX 环境编译，其分页可能与本次 PDF 略有不同。独立下载脚本固定 CSV 校验和，官方文件变化时会明确退出；原始下载清单另存为 `docs/data_download_manifest.json`。

## 结果与计时口径

每个 `results/main/runs/*.json` 保存配置哈希、数值源文件签名、数据哈希、λ、状态、全轨迹、阈值记录、参考目标、可行性、KKT 残差、诊断耗时和恢复指标。系数单独保存在 `coefficients/*.npz`。合成数据可由配置与种子重建；真实 CSV 下载脚本检查 SHA256，分割索引与训练标准化参数保存在阶段的 `datasets/`。

主要时间包含迭代与所有检查点诊断；精确 SVD 单独记录，并在 `cold_start_sec` 中加回 ISTA/FISTA。CD 不需要 SVD。数据生成、标准化、模型选择及参考解另计，不能悄悄并入或排除某个算法的迭代。环境记录会列出可见 CPU 数量；计时固定的是 BLAS/OpenMP 线程数，并未保证整台共享机器完全空闲。

支持集阈值固定为绝对值大于 1e-6，另检查 1e-8 和 1e-4。FISTA 的 `objective_increases` 是相邻记录检查点间观察到的上升次数，不是逐次迭代上升总数。真实数据没有真系数或真支持集，不能从其测试误差宣称恢复率。

## 目录

`src/` 算法与数据；`configs/` 冻结参数；`scripts/` 入口；`tests/` 正确性；`results/` 原始及汇总；`figures/` 图；`report/` 中文报告与引用；`docs/` 协议、阅读与发现；`data/raw/` 官方下载。交付 ZIP 排除运行环境、编译二进制和原始数据，下载脚本保留数据来源。
