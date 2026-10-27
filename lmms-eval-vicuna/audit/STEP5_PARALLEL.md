# 两轮交叉 GPU 调度

新增独立入口，不修改原 `step5_repeatability.py` 或原串行计划。使用新目录 `audit/runs/step5_parallel_v2`，不复用已按 GPU 0–3 预检的 `step5_parallel_v1`。

| 轮次 | GPU 1/2 | GPU 3/4 |
| --- | --- | --- |
| 1 | C1，端口 29931 | D1，端口 29932 |
| 2 | D2，端口 29931 | C2，端口 29932 |

每个任务两个进程、完整相同48条输入、batch_size=1、相同种子和compact记录。四次共192条推理。实际子进程以GPU UUID指定设备，避免CUDA编号顺序与nvidia-smi编号不一致。run_manifest.json记录显式GPU组、UUID、端口和命令。

这是跨GPU组与轮次的一致性检查，不是同组GPU重复实验：C1/C2和D1/D2同时改变了轮次、GPU组及并发伙伴，不能单凭这四次分离硬件与时间影响。发生差异后，应补做同一组GPU上的同条件复跑。并行CPU/磁盘竞争也可能影响耗时，不应拿这些运行做严谨性能结论。

## 用户运行

```bash
cd /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna
conda activate SECOND_clean
unset CUDA_VISIBLE_DEVICES
export NUM_PROCESSES=2
nvidia-smi
python audit/step5_parallel_repeatability.py round --round 1 --study audit/runs/step5_parallel_v2 --pair-a 1,2 --pair-b 3,4 --prepare-only
```

prepare-only检查计划及GPU占用，不启动模型。它只检查选定的 GPU 1、2、3、4；GPU 0 即使繁忙也不会参与，GPU 5 保留备用。选中 GPU 任一显存占用超过2048MiB时阻止启动。这是避免已有模型占用的操作门槛，不是保证运行永不OOM。正式运行前仍应确认这些卡空闲。

GPU 0 的既有占用不属于本轮目标设备；脚本不会杀进程、重置 GPU 或把 GPU 0 加入实验。正式运行前仍需重新查看选定的 GPU 1–4，因为空闲状态只是当前时刻的快照。

```bash
bash audit/run_step5_parallel.sh 1 audit/runs/step5_parallel_v2 --pair-a 1,2 --pair-b 3,4
```

该命令自动同时启动C1/D1，等待两者完成。每任务console.log独立，终端显示带任务名前缀的日志。任一任务失败则本轮返回非零，成功的同伴结果保留；不要继续第二轮。

第一轮成功后：

```bash
bash audit/run_step5_parallel.sh 2 audit/runs/step5_parallel_v2 --pair-a 1,2 --pair-b 3,4
```

第二轮在启动前校验C1/D1的完成记录、输出哈希和GPU分配。D2/C2自动并行，全部成功后生成repeatability_report.json。

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('audit/runs/step5_parallel_v2/repeatability_report.json')
r = json.loads(p.read_text())
print(json.dumps(r['summary'], ensure_ascii=False, indent=2))
PY
```

建议在tmux会话内运行，每轮只调用一次入口。不要手动给两个子任务同时传相同端口，不要将NUM_PROCESSES改成4；这会改变实验结构。

同一研究目录有调度锁，防止两个调度器同时写入。两轮开始前检查GPU占用和端口；启动后不会再检查全组空闲，以免把本轮同伴误判为外部占用。避免手动开启额外任务，这个检查无法消除外部程序同时启动的竞争窗口。

已完成的任务会校验后跳过。失败目录拒绝覆盖，先查看console.log并保留失败证据，再重命名后重试；不要反复运行到出现期望答案。原标签复核流程与复核包仍可继续使用。

报告保留原有完整性、预测、历史复现、首个答案token筛选掩码及输出序列的分开判断，并明确标注跨GPU组设计。逐样本增加同组C/D比较：GPU1/2上C1与D2，GPU3/4上C2与D1；这些比较仍跨轮次，不能消除所有时间/执行变化。
