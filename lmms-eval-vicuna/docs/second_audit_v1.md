# SECOND Token Selection Audit v1 使用说明

该审计在同一份 80 条 POPE Popular 清单上顺序运行四个条件：

| 条件 | 含义 |
| --- | --- |
| A | Standard LLaVA，不进入 SECOND 递归路径 |
| B | Recursive-AllToken-Raw，保留全部 token，直接使用最后 stage 原始 logits |
| C | Recursive-AllToken-CD，保留全部 token，开启 SECOND 对比解码 |
| D | Full SECOND，应用原 token selection 并开启对比解码 |

只有 `C 正确、D 错误`才记为当前 CD 配置下的 Selection Harm；`B 正确、C
错误`是 CD Harm。

## 运行前提

在已经能够运行原 SECOND 实验的环境中执行，例如：

```bash
conda activate SECOND_clean
cd /home/yxq/Downloads/SECOND-main/.worktrees/second-audit-v1/lmms-eval-vicuna
```

脚本默认使用 GPU 0、1，沿用双卡稳定设置：

```text
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
```

可以通过环境变量覆盖 GPU 或端口：

```bash
CUDA_VISIBLE_DEVICES=2,3 SECOND_AUDIT_PORT=29882 bash scripts/run_second_audit.sh SOURCE_LOG
```

## 一条命令运行全部审计

```bash
bash scripts/run_second_audit.sh \
  /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/logs/liuhaotian__llava-v1.6-vicuna-7b/20260810_001849_samples_pope_pop.jsonl
```

也可以显式指定一个必须为空或不存在的输出目录：

```bash
bash scripts/run_second_audit.sh \
  /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/logs/liuhaotian__llava-v1.6-vicuna-7b/20260810_001849_samples_pope_pop.jsonl \
  /home/yxq/Downloads/SECOND-main/audit_runs/pope_pop_audit_v1
```

脚本按以下顺序执行：

1. 从旧 D 日志按 seed 42 抽取 FN30、FP10、TP20、TN20；
2. 顺序运行 A、B、C、D，共 320 个 sample-condition evaluation；
3. 要求新 D 的 80 条预测与旧 D 逐样本完全一致；
4. parity 通过后生成 A→B、B→C、C→D 配对分析。

旧实验 3000 条约耗时 2 小时 16 分。线性估算 320 次约 15 分钟，但 AllToken、
attention 输出和 PNG 写入都会增加开销，因此应预留约 30–60 分钟。该范围只是
运行规划估计，不是保证值。

## 输出结构

```text
OUTPUT_ROOT/
├── pope_pop_seed42_80.json
├── A|B|C|D/
│   ├── results/
│   └── artifacts/<condition>/rank_<n>/<question_id>/
│       ├── input_image.png
│       ├── sample.json
│       ├── token_000_stage_<s>_to_<s+1>.npz
│       ├── ..._attention.png
│       ├── ..._proposed_mask.png
│       └── ..._applied_mask.png
├── d_parity.json
└── report/
    ├── report.json
    └── report.md
```

红色 mask 图表示 selector 提议选中的 token；绿色 mask 图表示实际送入下一 stage
的 token。B/C 中绿色应当全 1，而 D 中绿色应与红色一致。数值分析必须使用 `.npz`
中的原始数组，不能从 PNG 反推。

## 结果解释边界

- 80 条清单人为增加了错误样本比例，不能直接报告总体 Accuracy/F1 或总体 Harm Rate。
- 首先看 `report.md` 中 C→D 的 harm/benefit 和 FN Rescue。
- 只有 D parity 完全通过，内部可视化才可解释。
- 如果 C 至少救回 30 个抽样 FN 中的 5 个，再进入 3000 条、E 条件和 COCO bbox
  coverage 的第二阶段。

新评测结果中：

- `pope_pred_yes_ratio` 是模型预测 Yes 比例；
- `pope_gt_yes_ratio` 是 GT Yes 比例；
- `pope_invalid_ratio` 是无法解析为 Yes/No 的比例；
- `pope_yes_ratio` 在 schema 2 中是 `pope_pred_yes_ratio` 的兼容别名。

## 常见失败

- AllToken OOM：脚本会直接停止，不会偷偷降低保留率；记录发生条件和 stage 后再决定
  是否改单卡/显存方案。
- D parity 失败：不要分析 attention，先检查审计接线或非确定性来源。
- 输出目录非空：脚本拒绝复用，换一个新目录，防止覆盖或混合两次实验。
