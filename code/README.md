# 复现说明

在项目根目录运行：

```bash
MPLCONFIGDIR=.mplconfig python3 code/run_all.py
```

程序固定随机种子 `20260910`，先执行数据门禁，再依次运行问题 1、2、3、4，最后生成 `code/outputs/`、`figures/*.pdf`、五个结果工作簿和 `reports/RESULTS_REPORT.md`。
