# 方案

要依次调用这些 skill，按照里面要求完成任务。

用户偏好：
- 排版引擎：LaTeX（XeLaTeX 两遍编译）
- 竞赛类型：全国大学生数学建模竞赛（国赛/CUMCM）
- 论文语言：中文
- 子问题数量：4 个
- 版本策略：原版归档保留；融合重算版在 `hybrid_rebuild/` 独立生成

workflow:

| step | skill | 本阶段重点 | 主要产物 |
| --- | --- | --- | --- |
| 1 | `2analysis-modeling` | 固化两版融合口径、信息集、SOC 跨日和结算解释 | `reports/ANALYSIS_MODELING_REPORT.md` |
| 2 | `3coding-visual` | 从 1 月 1 日连续模拟；重算四问和 8 种预报组合；生成工作簿与数据图 | `code/`、`results/`、`reports/RESULTS_REPORT.md`、`figures/*.pdf` |
| 3 | `4drawio` | 更新总体路线、信息集与滚动决策流程图 | `figures/*.drawio`、`figures/*.pdf`、`reports/DRAWIO_REPORT.md` |
| 4 | `5writing` | 使用国赛 LaTeX 模板撰写融合重算版论文并插入图表 | `paper/` |
| 5 | `6verity` | 文本、数值、代码复现、编译和逐页 PDF 验收 | `reports/VERIFY_REPORT.md` |

## 融合建模方向

- 问题一：统一确定性 LP，显式记录弃电，保留详细表格与效率/容量灵敏度。
- 问题二：历史滚动预测 + 80% 分位安全裕度；A 版完美信息结果仅作 oracle。
- 问题三：冻结已执行决策、继承 SOC、正调整费用；比较 6/12/18 的 8 种启用组合。
- 问题四：正式策略不读取未来实际价格；历史价格滚动预测为主，完美价格仅作 oracle。
- 问题二至四：从 2025-01-01 00:00 的 6000 kWh 开始连续仿真，1 月作为状态和预测热身，正式输出 2 月 1 日至 12 月 31 日的 334 天。

## 风险控制

- 所有策略记录训练截止日和可见信息，禁止未来信息泄漏。
- 费用分量必须非负并可重构总费用；退款解释只作敏感性对照。
- 每时段回代供需、SOC、容量和功率约束，并检查同时充放电。
- 原版仅存放在 `archive/original_before_hybrid_20260912/`，新流程不得覆盖该目录。

