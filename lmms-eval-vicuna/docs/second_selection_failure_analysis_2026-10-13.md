SECOND 视觉 token 筛选的初步失败分析、研究问题与 motivation

日期：2026-10-13。范围：LLaVA-v1.6-Vicuna-7B、POPE Popular 3000 条、当前仓库记录的 B/C/D 配置。本文基于已有日志重新计算统计、阅读实现并核实文献，没有新增模型推理。

建议采用的核心表述是：**当前证据显示 SECOND 的筛选存在局部退化案例；需要研究的问题是，在多尺度递归与对比解码中，如何判断被删除的低分 token 是否包含不可替代的任务证据，以及如何控制删减引起的决策变化。** 现有数据尚不能证明筛选整体有害，也不能证明所有退化都由漏选目标造成。

**已经观察到什么**

从历史 B/C/D 的原始样本日志按 question_id 对齐，并核对问题、真值和输入文本后，重新得到：

| 条件 | 配置 | 正确数 / 3000 | Accuracy | FN：有目标却回答 No | FP：无目标却回答 Yes |
| --- | --- | ---: | ---: | ---: | ---: |
| B | 递归多尺度、全部 token、不启用 CD | 2648 | 88.27% | 307 | 45 |
| C | 递归多尺度、全部 token、启用 CD | 2710 | 90.33% | 227 | 63 |
| D | SECOND 筛选、启用 CD | 2706 | 90.20% | 230 | 64 |

来源：[全量配对报告](../audit/runs/step4_ablation_v1/bcd_full_paired_report.json)。B/C/D 不是普通单次前向 baseline 的替代名称；这里没有估计 A→B 的递归路径收益。

B→C 修正 80 条、损害 18 条，净增加 62 条正确答案；其中 80 条改善全部是 FN 被救回，18 条损害全部是新增 FP。因此，不能把当前 CD 的效果概括为“模型更倾向回答 No”。

C→D 仅改变 30/3000=1% 的答案，其中 13 条改善、17 条退化。17 条退化包含 14 条新增 FN、3 条新增 FP；13 条改善包含 11 条 FN 救回、2 条 FP 修正。准确率净变化为 −0.133 个百分点，现有报告的题目级双侧精确 McNemar p=0.5847。该检验没有处理同图多题的聚类依赖，不能作为跨图像、跨运行的最终推断；它也没有提供整体显著退化的支持。

D 的 294 条错误中，277 条在 C 中也错误，只有 17 条是相对 C 新增的错误。**不能把 D 的全部错误或全部 230 个 FN 都归因于筛选。** 同样，C/D 输出相同也不代表内部证据没有变化。

对 D 的 3000 个首答案 token、三个 stage 转换，共 9000 份记录重新汇总，按真正进入下一阶段的 consumed mask 计算：

| 阶段转换 | 全量平均保留率 | 17 条退化平均保留率 | 13 条改善平均保留率 |
| --- | ---: | ---: | ---: |
| −2 → −1 | 96.94% | 95.59% | 95.30% |
| −1 → 0 | 95.69% | 93.87% | 93.59% |
| 0 → 1 | 93.96% | 92.37% | 91.97% |

第一轮在 1260/3000 条上实际全部保留。末阶段累积视觉 token 数平均为 2891.592，全量为 3060，减少约 5.50%。这不是端到端时延或 FLOPs 节省率，也不代表所有生成位置的保留率。

因此，“删得过多”不是当前证据最支持的统一解释：总体删减不大，而且改善组与退化组的保留率接近。更合理的切入点是**删除位置、证据作用和解码边界**。这里的组间均值只是描述性统计，样本是按结果选出的，不能据此推断因果关系。

新增统计及逐样本分数见 [统计 JSON](second_selection_failure_stats_2026-10-13.json)，其中保留了输入路径、关键文件 SHA256 和检查结果。

**筛选为什么可能失败：按证据强弱分开讨论**

1. 注意力评分没有直接保证“删后仍能回答问题”。

SECOND 以多尺度注意力为依据选择后续视觉信息。[SECOND，ICML 2025](https://proceedings.mlr.press/v267/park25c.html) 当前实现的 `get_heatmap` 聚合语言模型和视觉编码器注意力，进行归一化、上采样和跨尺度加权。它是选择信号，不能直接解释成每个 token 对正确答案的因果贡献。[本地实现](../LLaVA-NeXT/llava/mm_utils.py)

低分 patch 可能承载小目标、局部边缘、遮挡后残留的特征或排除错误类别的证据；高分区域可能包含相关背景或 attention sink。这些是合理候选机制，尚未在本项目的退化样本上证实。视觉 attention sink 研究提供了“高注意力可以分配给与文本无关的视觉 token”的直接相关证据，但不能替代对当前案例的检验。[Kang et al., 2025](https://arxiv.org/abs/2503.03321)

注意力可解释性文献也要求把权重与实际干预效果分开；不能据此断言所有注意力都无用。[Jain & Wallace, 2019](https://aclanthology.org/N19-1357/)，[Wiegreffe & Pinter, 2019](https://aclanthology.org/D19-1002/)

具体待验证问题：被删掉的目标相关 patch，是否比等数量背景 patch 更能恢复正确答案的分数与预测？对于 GT=no 的题目，应研究排除证据及错误共现线索，不能虚构不存在目标的 bbox。

2. 熵驱动预算描述注意力分布形状，未校准为任务证据充分性。

当前 `attn_entropy_topk_based_recursion` 接收二维热图，在最后一维逐行归一化，计算各行熵的平均值，再除以 log(宽度)，而不是计算全图归一化的二维熵。`entropy_2d` 只是额外审计指标，没有控制当前筛选。[本地实现](../lmms_eval/recursion_utils.py)

忽略数值 epsilon，其实际统计量为：

\[
q_{rc}=A_{rc}/\sum_j A_{rj},\qquad
H_{row}=\frac{1}{R}\sum_r[-\sum_c q_{rc}\log q_{rc}],
\]

\[
k=\operatorname{clip}\left(\frac{\exp(\lambda H_{row}/\log W)-1}{\exp(\lambda)-1},0.1,1\right).
\]

这给出一个可严格说明的局限：矩阵 `[[100,100],[1,1]]` 与其转置具有完全相同的全局数值分布，但逐行归一化熵分别为 1.0000 和 0.0801；当 λ=1 时，公式保留率分别为 1.0 和截断后的 0.1。这只是公式的合成反例，不是本实验图片旋转的实测结果。它说明该预算受空间排布影响，不能直接等同于模型的不确定性或目标证据量。

此外，代码通过 `score >= 第 k 大分数` 生成 mask；上采样造成的相同分数可能让实际保留数高于请求数。因而预算分析应使用 consumed mask，而不能只看 λ 或 requested_keep_ratio。当前 `attention_threshold=1.0` 对应公式参数，并不表示固定保留全部 token。

具体待验证问题：在可比 token 预算下，目标覆盖、反事实分数变化或跨尺度一致性，能否比当前行熵更可靠地判断“还需要多少视觉证据”？

3. 递归选择可能延续早期偏差，但尚无逐阶段干预证据。

前一阶段的选择决定下一阶段新增的细粒度输入；下一次评分又依赖当前已保留的信息。如果粗尺度遗漏了关键局部，后续可能缺乏足够的重新定位依据。[递归路径](../LLaVA-NeXT/llava/model/language_model/llava_recursion.py)

这里应说“可能延续早期偏差”，不能说“一次漏选后所有信息永久消失”：模型还保留较粗尺度的特征，并在每个生成 token 开始时重置 mask。当前也没有通过单阶段替换证明某个早期决策是最终错误的必要原因。

具体待验证问题：固定其他阶段，只替换一个阶段的选择，是否能恢复后续目标覆盖及答案？若能，是早期漏选限制了后续细粒度证据，还是末阶段直接删掉了关键局部？

4. 筛选改变的分数会与 CD 的决策规则发生交互：这是目前最直接的解码证据。

当前实现采用相邻阶段 logits 差，并限定末阶段 top-20 候选。对于仍在候选集合内的词，三个 α 均为 0.8 时：

\[
z_{CD}=z_3+0.8(z_1-z_0)+0.8(z_2-z_1)+0.8(z_3-z_2)
=1.8z_3-0.8z_0.
\]

这是当前配置的代数性质，不适用于任意不等权重或不同候选过滤规则。中间阶段仍通过 token 选择影响末阶段，不能说它们没有作用。[解码实现](../lmms_eval/second_audit.py)

记 m=z(Yes)−z(No)，则候选范围内 m_CD=1.8m_3−0.8m_0。历史 30 条 C/D 变化问题的首阶段 Yes/No margin 全部一致；其 60 条记录与该公式的最大残差约为 4.58×10⁻⁶。因此，这组配对中末阶段 margin 的差异基本按 1.8 倍传入最终 Yes/No margin。

| question_id | 问题目标 / GT | C 的末阶段 margin | D 的末阶段 margin | C 的 CD margin | D 的 CD margin |
| --- | --- | ---: | ---: | ---: | ---: |
| 2253 | bowl / yes | 3.1250 | −1.3594 | 7.3125 | −0.7594 |
| 1025 | handbag / yes | −0.1406 | −0.2344 | 0.1469 | −0.0219 |
| 2802 | bottle / no | −0.4063 | −0.3906 | −0.0063 | 0.0219 |

2253 是末阶段相对 Yes/No 的偏好大幅反转；1025 是末阶段仍偏 No，但 CD 原本能够把最终答案推到 Yes，筛选后这种修正未能越过边界；2802 则是小变化经过 CD 后带来新增 FP。457 的 D 最终 Yes/No margin 为 0，记录答案为 No，属于需要单列的平分案例。

在 17 条退化中，4 条的末阶段 Yes/No margin 从正变负；其余 13 条的 C/D 末阶段 margin 都为负，却在 CD 后形成不同答案。这种分类描述的是记录到的数值路径，既不等于证明数值噪声，也不等于证明目标区域丢失。

具体待验证问题：如何区分“视觉证据确实受损”与“较小分数变化越过 CD 边界”？如何根据证据可靠性调节对比强度或触发重新检查？VCD 是这一解码问题的重要背景，而不是对当前失败机制的直接证明。[Leng et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html)

5. Padding、AnyRes 几何和标签属于待排查因素。

当前代码中结合 image/padding mask 的调用处于注释状态，但这本身不足以断言 padding 导致错误。应根据运行时 provenance 统计真实进入模型的 padding 比例、全部目标实例的有效覆盖以及跨尺度重复覆盖；单纯叠加热图不构成因果证据。修正几何覆盖之后仍需要等预算干预。

已有第五步文档指出 513、591、945、2355、2497 存在 POPE 正标签与当前 COCO 目标标注不对应的问题。缺少 COCO 框不证明物体不存在，也不能据此修改原始 POPE 标签。机制分析应先独立复核图像，保留不确定项。[重复验证与标签复核说明](../audit/STEP5_REPEATABILITY.md)

**研究问题应如何收敛**

建议主问题写为：

> 在多尺度视觉语言推理中，如何在有限视觉 token 预算下，选择对当前问题具有反事实贡献的视觉证据，并使选择后的对比解码对证据删减保持可靠？

| 子问题 | 可检验的定义 | 最小验证方式 |
| --- | --- | --- |
| 证据选择 | 低分 token 被删除后，是否损失了对正确答案不可替代的证据？ | 目标 patch 与等数量背景 patch 的补回或交换；匹配 stage、token 数、padding 与有效图像面积，保留空间顺序；比较真实答案 margin、答案恢复率 |
| 预算分配 | 当前热图熵是否能预测维持证据所需的 token 数？ | 行熵、全图熵、固定预算与证据相关指标的对照；在独立验证集上选择预算，在相同预算下比较 harm/benefit |
| 递归传播 | 哪一阶段的选择使后续证据与分数发生变化？ | 单阶段替换，其余阶段策略及解码规则固定；记录后续选择变化，不把一次替换解释为纯局部效应 |
| 解码交互 | 错误来自筛选、CD，还是二者交互？ | 补齐“筛选/全量 × CD 开/关”的 2×2；固定已记录 logits 的 α 扫描仅用于首 token 机制分析，不当作完整生成重跑 |
| 可复现性 | 变化是否超过相同输入下的运行差异？ | 已规划的 C1→D1→D2→C2；分别检查答案、首 token 掩码、完整输出序列和 margin；按图像聚类统计 |

优先完成已有 48 条（30 条变化题 + 18 条匹配对照）的标签复核与重复实验，再在可复现且目标可定位的案例上做干预。候选方向可包括问题相关评分、空间覆盖约束、保留补充证据以及证据不足时扩大预算；这些目前只是研究设计，不是已验证有效的方法。

不能仅以恢复 C 的答案作为“grounding 正确”的证明：错误也可能由额外 token 数、位置变化、干扰缓解或模型偏好变化解释。目标补回必须有等预算背景/随机对照；必要时进一步做像素级对象干预，并单独说明图像编辑带来的分布变化。

**可用于开题或论文草稿的 motivation**

视觉语言模型的可靠回答依赖于充分且与问题相关的视觉证据。SECOND 通过选择性融合多尺度信息并进行对比解码，为缓解感知幻觉提供了有效思路，但注意力驱动的筛选仍需要回答一个关键问题：低分视觉 token 是否可以在不损伤任务证据的前提下被删除？我们在 LLaVA-v1.6-Vicuna-7B 的 POPE Popular 配对审计中观察到，筛选在首答案 token 的末阶段仅减少约 5.50% 的累积视觉输入，却同时产生 13 个改善和 17 个退化案例，未呈现明确的总体准确率增益。进一步的解码分析显示，当前等权对比配置会放大末阶段的分数变化，使视觉信息的局部变化能够跨越答案决策边界。这些发现促使我们研究任务证据是否被保留，以及证据变化如何影响解码，而不能只用注意力集中程度或保留比例评价筛选质量。为此，我们拟通过可复现的阶段审计与等预算反事实干预，识别对回答具有实际贡献的视觉信息，并探索结合证据充分性与解码稳定性的自适应选择机制，使模型在控制视觉计算开销的同时减少新增错误。此段提出的是研究动机与待验证方向，而非已建立的因果结论。

上述背景分别由 [SECOND](https://proceedings.mlr.press/v267/park25c.html)、[POPE](https://aclanthology.org/2023.emnlp-main.20/)、[Visual Attention Sink](https://arxiv.org/abs/2503.03321) 和 [VCD](https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html) 支持；实验数字与代数分析来自本地记录及实现。

可用的方向题目：**面向证据充分性与解码稳定性的多尺度视觉 token 选择**。VASparse 已研究视觉感知的稀疏化与幻觉缓解，因此不宜把“视觉相关 token 筛选”本身宣称为全新贡献；后续需证明所提出的证据判据、失败分解或干预机制与已有工作的实质区别。[VASparse，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhuang_VASparse_Towards_Efficient_Visual_Hallucination_Mitigation_via_Visual-Aware_Token_Sparsification_CVPR_2025_paper.html)

**证据边界与记录问题**

- 历史 D 的 validation_report 为 PASS，同时记录了相对 Step 1 的 22 个预测差异。PASS 表示该验证器的捕获检查通过，不表示历史预测完全复现。[验证报告](../audit/runs/step2_token_selection_v2/full_20260925T124957Z/validation_report.json)
- 截至本次检查，step5_repeatability_v1 下有 study_plan.json，没有 C1/D1/D2/C2 完成目录或 repeatability_report.json。因此，本报告不能声称这 30 个变化已经重复复现。
- 20261012_181814 日志重新计数得到 TP=1275、TN=1434、FN=225、FP=66，Accuracy=90.30%；它与历史 D 有 31 条预测差异。20261012_213638 的 Accuracy=90.2333%，与历史 D 有 5 条差异。这里未独立核实两次新日志的完整运行来源，不把它们当作严格同条件重复实验，也不混入历史 C/D 配对。
- 历史 B/C/D 记录相同 commit，并不能补回当时未记录的所有未提交修改、权重内容或数值执行状态。
- 元数据把 ret_attn 描述为 rank-transformed，但当前 get_heatmap 中 rank 变换写入 vis_attn_list，实际返回来自 ret_attn_list。该文字描述与所读代码存在不一致；本文据实际代码只称其为复合选择信号。历史数值语义仍应以运行源码与保存数组共同核验。
- 本次没有直接读取权限受限的冻结 manifest 目录；48 条、18 个对照、标签警告等信息来自可读取的 review_package_report.json 和第五步文档，而不是对 manifest 内容重新验证。
- 小目标、遮挡、attention sink、padding 占位、早期遗漏均未在当前案例上得到反事实证明。现有样本不足以对其他 VLM、其他 POPE 划分或开放式生成进行推广。

**参考文献及其作用**

1. Park, W., Kim, W., Kim, J., & Do, J. (2025). *SECOND: Mitigating Perceptual Hallucination in Vision-Language Models via Selective and Contrastive Decoding*. ICML, PMLR 267:48027–48040. [论文](https://proceedings.mlr.press/v267/park25c.html)。研究对象与多尺度选择框架。
2. Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W. X., & Wen, J.-R. (2023). *Evaluating Object Hallucination in Large Vision-Language Models*. EMNLP, 292–305. [论文](https://aclanthology.org/2023.emnlp-main.20/)。POPE 评估及对象共现偏差背景。
3. Leng, S., et al. (2024). *Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding*. CVPR, 13872–13882. [论文](https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html)。视觉对比解码背景。
4. Kang, S., Kim, J., Kim, J., & Hwang, S. J. (2025). *See What You Are Told: Visual Attention Sink in Large Multimodal Models*. arXiv:2503.03321. [论文](https://arxiv.org/abs/2503.03321)。与任务无关的高注意力视觉 token，直接关联评分可靠性问题。
5. Zhuang, X., Zhu, Z., Xie, Y., Liang, L., & Zou, Y. (2025). *VASparse: Towards Efficient Visual Hallucination Mitigation via Visual-Aware Token Sparsification*. CVPR, 4189–4199. [论文](https://openaccess.thecvf.com/content/CVPR2025/html/Zhuang_VASparse_Towards_Efficient_Visual_Hallucination_Mitigation_via_Visual-Aware_Token_Sparsification_CVPR_2025_paper.html)。视觉稀疏化与幻觉缓解的直接相关工作。
6. Jain, S., & Wallace, B. C. (2019). *Attention is not Explanation*. NAACL-HLT, 3543–3556. [论文](https://aclanthology.org/N19-1357/)。注意力与因果重要性之间的区别；其原始实验对象是 NLP 模型，不能直接当作 SECOND 的实证结论。
7. Wiegreffe, S., & Pinter, Y. (2019). *Attention is not not Explanation*. EMNLP-IJCNLP, 11–20. [论文](https://aclanthology.org/D19-1002/)。补充注意力解释需要模型相关检验与方差校准，避免把前述观点绝对化。
