# 48 条冻结样本：独立标签复核与 C/D 重复验证

目的：先判断 30 条历史 C/D 变化能否复现、首个答案 token 的实际筛选是否一致，再选有可靠视觉证据的案例做干预。18 条匹配对照保留；未匹配的 12 条变化样本同样保留。本步骤不改变 POPE 真值，不重新匹配，不运行 3000 条。

## 实验范围

输入 `audit/manifests/step5_changed30_matched_v1/combined.json`（当前 48 条），按问题 ID 关联预测，绝不拿子集 doc_id 对齐全量 doc_id。30 条变化问题来自 28 张图片；图像相关性必须保留。匹配报告的 NEEDS_REVIEW 合法地保留，不强制它变为 PASS。

推理顺序 C1 → D1 → D2 → C2，每次独立进程。C=全部视觉 token+CD，D=SECOND 筛选+CD。每条件新增两次，四次合计 192 条问题推理。仍用 SECOND_clean、GPU 0,1、2 进程、batch_size=1、compact、generation_type=recursion、fix_grid=2x2、stages=[-2,-1,0,1]、attention_threshold=1.0、contrastive_alphas=[0.8,0.8,0.8]。显式固定原默认种子 `0,1234,1234,1234`。这不是跨种子的鲁棒性实验；不额外引入 deterministic 算子开关以免同时改变实验条件。

每次记录实际启动命令、当前 commit、工作区差异、推理源文件哈希、包版本、GPU UUID/driver、模型缓存引用与文件元信息、输入清单哈希、预测、最终 token 序列、三个 stage 转换的实际保留掩码哈希、attention 哈希、Yes/No 分数差。

运行前后比对环境和源文件；发现变化停止。模型大权重只检查缓存 blob 路径、大小、mtime，不反复读取全部权重，因此不能声称已完整校验权重内容。历史运行未保存的 dirty 状态也不能恢复。子集重跑改变样本分配/顺序，历史不一致不应自动解释为代码变化或基线失效。

## 命令 1：生成独立复核包（不加载模型）

```bash
cd /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna
conda activate SECOND_clean
python audit/step5_repeatability.py review --output audit/runs/step5_review_v1
```

生成 `audit/runs/step5_review_v1/blind_review.zip`。48 条问题混排，网页只显示问题、原图和匿名编号，不显示原标签、C/D 预测、改善/退化或对照身份。问目标是否可见、是否可以明确定位以及判断依据，允许 uncertain。

在本地 Windows PowerShell（不是服务器 shell）下载：

```powershell
scp -P 10003 root@10.10.65.31:/home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/step5_review_v1/blind_review.zip .
```

解压后用浏览器打开 index.html，可点击图片查看原图；填写复核者并导出 visual_review.json。网页不自动保存，关闭前必须导出；可导入该 JSON 继续。建议两人独立复核，分歧时保留不确定性。不要发给独立复核者 review_key_private.json 或原始清单；如果已看过模型结果，应记录这一事实，不能宣称完全盲法。复核 48 条全体而非只挑五条，以免选择性审查。

五条当前 COCO 无目标标注的问题是 513、591、945、2355、2497。原始图片与两份 POPE 缓存的 RGB 已核对一致；但目标缺标不是目标不存在的证明。不得自动把 orange 颜色当橙子、cupcake 当饮用杯、遥控器当电视，不能根据画面外语境造框。本轮不生成或应用自动“修正标签”。

## 命令 2：准备重复实验计划（不加载模型）

```bash
python audit/step5_repeatability.py run --study audit/runs/step5_repeatability_v1 --prepare-only
```

应显示 `Preflight passed; NO MODEL RUN.`。检查冻结文件哈希、图片哈希、原历史文件和历史样本记录、任务配置等价、当前 commit/源文件/环境后写 study_plan.json。历史配对报告没有全量当前代码一致性证据，这一步只冻结当前可验证的状态。

如果新计划创建失败，不要盲目跳过检查或更改 commit。错误会显示具体原因；保留错误信息。创建计划后不更换环境、不修改脚本或模型代码。

## 命令 3：正式运行四次推理

```bash
bash audit/run_step5_repeatability.sh audit/runs/step5_repeatability_v1
```

建议在 tmux 中运行；同一组 GPU 不要同时启动其他实验。四次串行，耗时以当前硬件和样本长度为准，不能仅用问题数保证时长。可直接观察 console.log。

每次目录分别是 C1、D1、D2、C2。自动调用现有 token 捕获验证器检查数量、stage、消耗掩码和解码一致性，另外逐条核对评测日志与捕获记录。历史预测差异是研究结果，不因与旧答案不同而中断；结构缺失或来源变化才中断。

成功完成的运行可在再次执行相同命令时校验后跳过。失败留下的目录不会覆盖或自动删除：先查 console.log，保留并重命名失败目录后才能重试。只允许比较最终冻结的一组完整结果，不能重复到出现期望答案再挑结果。

## 命令 4：查看汇总

四次全部成功后自动生成报告，不需要另跑生成器：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('audit/runs/step5_repeatability_v1/repeatability_report.json')
r = json.loads(p.read_text())
print(json.dumps(r['summary'], ensure_ascii=False, indent=2))
PY
```

如需从已完成的四次记录重新汇总（不跑模型）：

```bash
python audit/step5_repeatability.py report --study audit/runs/step5_repeatability_v1
```

## 报告必须分开看

- data_validation_status：文件、数量、ID 和捕获是否完整。PASS 不代表稳定。
- prediction_repeatability_status：C1/C2 和 D1/D2 的 yes/no 是否逐条一致。
- historical_prediction_reproduction_status：四次预测是否分别与原 C/D 答案一致。当前两次稳定但历史不一致单列，不覆盖原基线。
- first_answer_mask_repeatability_status：各条件两次的首个答案 token、三个 stage 转换的 applied/consumed 掩码是否逐条一致。只能说明所记录的生成位置；不是所有答案 token 的稳定性。
- final_token_sequence_repeatability_status：最终全部输出 token 序列是否一致。
- yes_no_margins 和 margin_absolute_repeat_difference：逐条查看 Yes/No 分数差与数值变化。没有校准前不人为设一个“低置信”阈值。
- first_answer_attention_bitwise_equal：记录 attention 位级一致性；微小数值差异不能自动等同于区域变化或预测错误。

机制候选字段 `mechanism_screen_candidate_ids_pending_label_review` 只是自动初筛：历史变化复现、掩码与输出序列一致、无当前标注警告。它不取代人工标签审查；未标注警告也不保证标签正确。

## 依结果决定后续

1. 预测不稳定：先排查同输入、同 prompt、数值执行、rank/样本顺序及随机状态，不做目标补回结论。
2. 答案稳定但掩码不稳定：先判断差异集中在哪个 stage、多少 token，不能把一次掩码视为固定机制证据。
3. 当前稳定但不复现历史：核查全量到子集的调度差异与历史来源；保留两种结果，不重写历史基线。
4. 预测和掩码均复现、目标可见可定位：选择退化与改善案例，设计等数量目标/背景 patch 干预；预先指定对照、token 数和随机种子。
5. 标签有疑点或目标不可定位：原始基准统计照常保留，机制分析单独列出；补充审定标签分析必须标为补充结果，不能替代原始 3000 条统计。

旧 run_step4_ablation.sh / run_step3_provenance.sh 会覆盖诊断集路径，不能用来运行这个 48 条集合。使用上述专用入口。
