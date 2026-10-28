# 外部参考审计 v1 复现入口

日期：2026-10-28，Asia/Shanghai。此阶段与课程v2、原workshop结果分别保存。正文先用中文，身份留空。当前结论见`reference_audit_findings_v1_zh.md`，执行状态见根目录`TODO.md`。

## 环境与数据

沿用`reproduction_v2.md`中的Python及依赖安装方式；本机使用`.venv/bin/python`，现代求解器在`.modern_deps`、报告依赖在`.report_deps`，脚本自行加入相应目录。单线程、float64、CPU；不需要GPU。来源仍由官方脚本下载，原始数据不随研究增量包分发。

```bash
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/download_benchmarks.py
```

新闻数据固定旧seed0训练池，抽样后保留全部非常量列，不作20列分块截断。基因数据固定72×7129矩阵。两来源旧标签均不用于本次模拟响应。新data module、driver、config及protocol的拟合前哈希见`results/reference_audit_v1/protocol_lock.json`。

## 已保存结果的只读核验

以下步骤不调用优化器。前两步再生设计、核对NPZ和证书；分析阶段计算X-only组和评价；最后独立验收分析。分析需先通过数值验收，报告需通过分析验收。

```bash
.venv/bin/python scripts/verify_reference_audit_v1.py pilot
.venv/bin/python scripts/verify_reference_audit_v1.py confirm
.venv/bin/python scripts/analyze_reference_audit_v1.py pilot
.venv/bin/python scripts/analyze_reference_audit_v1.py confirm
.venv/bin/python scripts/verify_reference_analysis_v1.py
.venv/bin/python scripts/build_reference_audit_brief_v1.py
```

验收包含精确有理数F1差异与严格δ比较、精度资格及共识、种子三态事件和区间、oracle-c平局、组净效应真值与含任意真变量的区别。第二轮独立验收没有重新进行完整成对组搜索；其组存在性来自冻结的分块扫描记录，范围在审查中披露。

每次核验/分析都会写入新验收或诊断耗时、文件哈希；随后的依赖结果需按上述顺序重建。不得仅拿旧验收的`all_passed=true`而忽略输入哈希。既有优化结果不会因此重新求解。

## 从头运行优化

首次运行前先执行不生成数据、不拟合的`--check-only`，核对预定网格；参考数值阶段顺序运行，避免计时相互干扰。

```bash
.venv/bin/python scripts/run_reference_audit_v1.py pilot --check-only
.venv/bin/python scripts/run_reference_audit_v1.py pilot
.venv/bin/python scripts/run_reference_audit_v1.py confirm
```

已有一致签名结果由runner复用，不是新的计时。若需真正重新计时，请使用新工作目录或完整复制后再运行，保留本次冻结结果。先导按完整性/成本决定进入确认，决策见`pilot_budget_decision.json`，不是按有无正面事件决定保留来源。

新阶段共80目标、160次公共参考调用；skglm内部重算另计，完整记录见两阶段`completion.json`。库调用是返回后检查软预算，不能声称强制20秒中断。

## UJI机制对照

X-only处理链审计：

```bash
.venv/bin/python scripts/audit_uji_preprocessing_v1.py
```

它不生成响应、不优化；旧300–309用于复现，新600–609只是行探针。其结果不能替代新的响应或求解确认。

后续小型处理对照另用`configs/uji_preprocessing_control_v1.json`、`docs/uji_preprocessing_control_protocol_v1.md`与`results/uji_preprocessing_control_v1/`。只选择三个已看过的旧种子，明确为事后机制检查，最多48次新公共参考调用；保持旧y/λ，但删除/合并列可能改变参数化和目标，降维F1不能当原变量F1改善。

```bash
.venv/bin/python scripts/run_uji_preprocessing_control_v1.py --check-only
.venv/bin/python scripts/run_uji_preprocessing_control_v1.py
.venv/bin/python scripts/verify_uji_preprocessing_control_v1.py
```

输入来源、映射和预处理位置的详细解释见`uji_preprocessing_audit_v1.md`，新对照的实际结果以独立验收及结果说明为准。

## 文献与交付边界

最接近原始研究及贡献限制见`reference_audit_novelty_assessment_v1.md`。2023年预印本与正式期刊分别注明；没有用近期论文数量替代新颖性证据。

外部简报由ReportLab/Matplotlib生成，并非TeX编译。根目录原课程报告、原workshop完整包仍可单独使用；研究增量的结果不回写旧汇总。所有包与报告均为本地交付，没有自动投稿、上传或提交课程平台。
