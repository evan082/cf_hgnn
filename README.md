# CF-HGNN — 下一活动预测 + 选择性过程约束（SPC）

**主创新：SPC** — 训练用不确定性加权软合法正则，推理仅在高熵时施加合法掩码。  
属性 Transformer 为强基线；全程 hard mask / 一致性特征拼接为对照。

```bat
cd d:\data111\cf_hgnn
REM SPC 四组消融（Base / HardMask / SoftLegal / SPC）+ 噪声
run_py.bat scripts\run_spc_study.py --config configs\helpdesk_spc.yaml --seeds 42 --force

REM 组合1：Road Traffic Fine（低熵，4k 子采样）
run_py.bat scripts\run_spc_study.py --config configs\traffic_spc.yaml --seeds 42 123 2024

REM 属性 TF Acc 基线
run_py.bat scripts\prepare_data.py --config configs\helpdesk_acc.yaml
run_py.bat scripts\cache_legal_next.py --config configs\helpdesk_acc.yaml
run_py.bat scripts\run_acc_benchmark.py --config configs\helpdesk_acc.yaml --seeds 42 123 2024
```

| 步骤 | 脚本 |
|------|------|
| 数据 | `prepare_data.py` |
| Petri / 合法后继 | `discover_petri.py`, `cache_legal_next.py` |
| SPC 消融 | `run_spc_study.py` |
| Acc 基线 | `run_acc_benchmark.py` |

环境：`run_py.bat` → `d:\data111\pythonProject\.venv`。  
文档：`docs/DIRECTION.md`、`docs/RESULTS.md`、`docs/PAPER_OUTLINE.md`、`docs/PT_DIFF.md`。
