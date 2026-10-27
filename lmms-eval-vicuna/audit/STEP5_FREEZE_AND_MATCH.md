# 第五步起点：冻结 C/D 变化样本并匹配对照

本脚本只读历史评测、COCO 标注和原图，生成清单，不加载模型、不调用 GPU、不产生新预测。实际清单生成由用户执行。

## 已核实的来源

- SSH：`yxq@10.10.65.31:10003`，远端 `id -un` 和 `hostname` 均返回 `yxq`。
- 工程：`/home/yxq/Downloads/SECOND-main/lmms-eval-vicuna`。
- 完整配对报告：`audit/runs/step4_ablation_v1/bcd_full_paired_report.json`。
- B：`audit/runs/step4_ablation_v1/full_B_20260926T050031Z`。
- C：`audit/runs/step4_ablation_v1/full_C_20260926T082950Z`。
- D：`audit/runs/step2_token_selection_v2/full_20260925T124957Z`。
- 三次记录的 commit：`1a4bdd2a9e4a0bd23f67a31821cdcbb4af962ac4`；三次验证均为 PASS、3000/3000。
- 原图映射：`audit/step1_full/pope_pop_3000_baseline/image_mapping.jsonl`。
- 交叉核对元数据：同目录下 `enhanced_features.jsonl`。
- COCO 标注：`/home/yxq/CAAC64/annotations_trainval2014/annotations/instances_val2014.json`。

注意：原配对报告和评测日志的 `image_id` / `doc.id` 是评测文档编号，不能拿它关联 COCO 标注。新清单把它保存为 `dataset_image_id`；`image_id`、`coco_image_id` 均保存通过原图映射取得的真实 COCO 编号。例如问题 1 的文档编号为 3000，COCO 编号为 310196。

原配对报告 `changed_samples` 包含 B/C 与 C/D 的变化并集（116 条），不能直接当作这 30 条。脚本独立比较全部 C、D 预测，按 `C_prediction != D_prediction` 提取，并对照报告核验。

## 冻结范围与实现

1. 核验三次 run manifest、validation report、样本数、问题 ID 集合、问题、真值、输入提示、记录的 commit、SECOND 参数和关键运行配置；从原始样本重新计算正确数。
2. 固定全部 30 条 C/D 变化问题，预期 13 条改善、17 条退化。随机种子不会重新抽取这 30 条，只决定匹配同分候选的顺序。
3. 通过问题 ID 关联 Step 1 的原图映射；用增强元数据、COCO 图像编号、文件名和尺寸交叉检查。冻结样本及对照的当前图片 SHA256 必须与历史映射相同。
4. 从 COCO 重新提取目标类别的**所有**实例框，记录 crowd 标志、全图对象数、图像长宽比。正样本匹配使用最大目标框的面积比例与中心坐标，同时记录实例数。历史增强特征的单框、预测、错误分层不参与匹配。
5. 目标为阴性的样本不构造目标大小或位置。GT 与 COCO 标注矛盾时保留原始变化样本并标记待核查，排除有此问题的对照；不能靠标注强行改写 POPE 真值。
6. 给每个变化问题找一条在历史运行中 C/D 输出相同的对照。匹配下面的必需条件，按预先声明的几何界限筛选候选，再进行全局匹配。
7. 保存全部输入文件和新清单哈希、历史 run manifest、当前工作区状态、匹配规则、每对距离及候选数。默认拒绝覆盖已存在的输出目录。

## 对照规则

必需条件：问题目标类别、真值、C 的正误状态均相同。例如 C 正确而 D 错误的 car 正样本，匹配 C/D 都正确的 car 正样本；C 错误而 D 改善的样本，匹配 C/D 都错误的同类别同真值样本。

对照候选排除全部 30 条变化样本涉及的 COCO 原图，避免同图不同问题作为独立对照。每条对照问题、每张对照图片最多使用一次。变化样本自身可以来自同一张图；后续统计须保留这种图像相关性。

| 协变量 | 严格匹配界限 | 明示放宽界限 |
| --- | --- | --- |
| 最大目标框面积比例的比值（仅 GT=yes） | 不超过 2 倍 | 不超过 4 倍 |
| 最大目标框中心距离，归一化到单位正方形对角线（仅 GT=yes） | 不超过 0.15 | 不超过 0.30 |
| 全图对象数加 1 的比值 | 不超过 2 倍 | 不超过 3 倍 |
| 图像长宽比的比值 | 不超过 1.5 倍 | 不超过 2 倍 |

全局目标依次为：匹配尽可能多的样本、使用尽可能少的放宽匹配、总距离尽可能小。目标实例数差异也计入距离。面积、位置界限是本次预先声明的经验规则，不是证明两张图具有相同难度的科学定律；应检查每对的差异。

不会放宽目标类别、真值或 C 正误状态，不会自动复用对照图片。不可匹配时记录原因，不凑足 30 对。

## 由用户执行的命令

在服务器上逐行运行，保留反斜杠后的换行：

```bash
cd /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna
conda activate SECOND_clean

python audit/freeze_step5_cases.py \
  --report audit/runs/step4_ablation_v1/bcd_full_paired_report.json \
  --features audit/step1_full/pope_pop_3000_baseline/enhanced_features.jsonl \
  --image-mapping audit/step1_full/pope_pop_3000_baseline/image_mapping.jsonl \
  --coco /home/yxq/CAAC64/annotations_trainval2014/annotations/instances_val2014.json \
  --output-dir audit/manifests/step5_changed30_matched_v1 \
  --seed 42 \
  --expected-count 3000 \
  --expected-changed 30 \
  --expected-corrected 13 \
  --expected-degraded 17
```

默认参数与以上命令相同，亦可直接运行 `python audit/freeze_step5_cases.py`。以上是清单生成，不是实验推理。

成功或部分匹配后检查：

```bash
cat audit/manifests/step5_changed30_matched_v1/freeze_report.json
```

输出：

- `changed30.json`：全部 30 条变化问题，含改善/退化方向、原图哈希和全部目标框。
- `controls.json`：成功匹配的对照，数量最多 30。
- `combined.json`：变化样本与对照的并集，数量最多 60。
- `pairs.json`：逐对 ID、严格/放宽级别、距离、协变量差异、候选数及未匹配原因。
- `freeze_provenance.json`：来源、匹配政策、历史记录与当前工作区信息。
- `freeze_report.json`：生成结果及下一步门槛。
- `sha256.json`：上述六个 JSON 文件的哈希索引。

需要独立核对冻结文件内容是否被修改时运行：

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

folder = Path('audit/manifests/step5_changed30_matched_v1')
hashes = json.loads((folder / 'sha256.json').read_text())
for name, expected in hashes.items():
    actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit('FAIL: hash mismatch: ' + name)
cases = json.loads((folder / 'changed30.json').read_text())
if cases['provenance_sha256'] != hashes['freeze_provenance.json']:
    raise SystemExit('FAIL: manifest provenance hash differs')
print('PASS: frozen files match their SHA256 index')
PY
```

## 如何判断结果

`status=PASS` 表示全部 30 条冻结且按声明的规则取得 30 个不重复对照；不是 SECOND 因果验证通过。核对 30/13/17、control_count=30、combined_count=60、unmatched_count=0、两种 overlap=0。放宽匹配即使 PASS 也必须查看距离，后续可比较只纳入严格对的敏感性结果。

`status=NEEDS_REVIEW`、进程退出码 2 表示冻结的 30 条保留，但对照不足。报告会区分：标注需人工检查、必需分层无候选、几何距离超出界限、唯一对照图片竞争。不要更改 30 条样本或扩大界限反复尝试直到凑满；先分析具体原因。若科学上允许新的匹配方案，应在不同目录建立 v2 并保留 v1。

`FAIL` 表示输入校验失败或原图哈希变更；脚本不提交输出清单。应修正明确的文件路径或查明来源不一致，不要绕过验证。

已有输出目录会报拒绝覆盖。若此前已经完成生成，直接查看文件，不必重复执行；需要修改匹配政策时使用新版本目录。

## 后续边界

先在冻结并集上重复 C/D，确认变化问题和对照是否能复现，再决定目标 patch 补回、等数量背景补回或 padding 移除实验。不应把本次输出相同的对照直接命名为“稳定对照”。在 C/D 输出差异可复现且干预改变输出前，不能把这些样本判定为“筛选漏目标导致错误”。

这 30 条按结果选出的样本不能用于估计整体准确率或缺陷发生率；这些仍以 3000 条为分母。匹配几何信息也无法完全控制遮挡、纹理、显著性、语境、标注遗漏和模型置信度。COCO bbox 不是精确分割 mask，attention 图也不是因果证据。

不要直接用新 manifest 调用原有 `run_step4_ablation.sh` 或 `run_step3_provenance.sh`：它们会覆盖清单路径并使用原诊断集数量。新并集需要后续适配 runner 后才能推理。

同一个历史 commit 不能证明三次运行没有不同未提交修改，也不能证明模型权重、算子与硬件数值行为完全相同。本脚本记录能核实的来源并明确上述边界，不补造缺失的历史环境信息。
