# 09_Full_Run_Plan.md — 50 轮全量实验与分布式冲突验证计划

<!-- 最后更新：2026-10-29 -->
<!-- 级别：执行计划。用户 2026-10-29 指令；实测结果完成后回写 01_Facts / 00_Current_State。 -->

## §0 用户目标（原话要点）

1. 先另开一个空分支，存储当前代码版本，以后封存不动。
2. 先按老办法跑**完整的非分布式全量训练**：大轮数 **50 轮**；为了让训练效果和历史差不多，把每个客户端本地训练轮数调小。
3. 全量跑完后，再测**分布式 10 客户端、全量训练、本地轮数同样调小、50 轮**，重点观察同时写入冲突。
4. 先回答历史参数是多少；计划写进 memory 后提交 `dev`。

## §1 历史参数（单机正式口径）【实测·来自 hparams/脚本】

### CIFAR10

| 参数 | 值 | 出处 |
|---|---|---|
| 客户端 | 10 | `run-test/cifar10/cifar_fed_hparams.py` |
| 全局轮数 | 15 | 同上 |
| 本地 epoch | **1** | 同上 |
| train batch | 64 | 同上 |
| test batch | 1000 | 同上 |
| lr | 0.01 | 同上 |
| 模型 | ResNet18CIFAR | 同上 |
| 数据划分 | IID，全量 50k | `Arguments.dd_type=IID` |
| 恶意率 | 0.30（3/10） | 同上 |
| 攻击 | 标签翻转 `{5→3}` 或服务端 MinMax | `run_dpfla_label_flipping.py` / `run_screen_server_attack.py` |
| 梯度放大 | `FL_GRADIENT_SCALE=1`（不放大） | `apply_common_env` |
| 测试集 | 全量 10k | `FL.test` |

### VisDrone

| 参数 | 值 | 出处 |
|---|---|---|
| 客户端 | 10 | `run-test/visdrone/visdrone_fed_hparams.py` |
| 全局轮数 | 15 | 同上 |
| 本地 epoch | **3** | 同上 |
| train batch | 16 | `run_screen_server_attack.py` 默认 |
| test batch | 64 | 同上 |
| lr | 0.0002 | `visdrone_fed_hparams.py` |
| 模型/分辨率 | yolov8m @1280 | `run_screen_server_attack.py` 默认 |
| AMP | 1 | 同上 |
| 数据划分 | IID，全量 6471 train / 548 val | 同上 |
| 恶意率 | 0.30（3/10） | 同上 |
| 攻击 | MinMax / IPM（服务端注入） | 同上 |
| 梯度放大 | 1（不放大） | `run_screen_server_attack.py` |

### 关键结论：本地轮数只能这样调

- **VisDrone**：3 → **1**。50 轮 × 1 = 50 个本地 epoch，历史 15 × 3 = 45，量级接近。
- **CIFAR10**：历史已经是 **1**，无法再降到 0.5（当前训练器只支持整数 epoch）。50 轮 × 1 = 50，历史 15 × 1 = 15，本地更新量约为历史 3.3 倍；这是“50 轮”必然的结果，不做额外降采样/半 epoch 改造。
- 如果 50 轮结果明显高于历史，只记录，不据此推翻历史 15 轮结论；若明显偏低，先排查参数/数据/OOM，而不是直接改结论。

### 2026-10-31 运行时修订（用户要求：50 轮总时长接近历史 5-10 轮、效果≈收敛）

- **VisDrone 正式 50 轮改为每轮每客户端 `FL_LOCAL_STEPS=12`**（仅当 `GLOBAL_ROUNDS>=50` 时由入口脚本自动设置）。
  - 总本地更新量 = 50 × 12 = 600 step ≈ 历史 **5 轮 × 3 epoch**（615 step，约 15 个本地 epoch）。
  - 实测单轮（10 端 / 1 epoch / 41 step）约 5.4 min；降到 12 step 后单轮约 2.0–2.3 min，单条约 **1.8–2.0 h**（原 4.5 h）。
  - 效果目标：末轮 ≈ 历史 R5（DPFLA MinMax 43.85 / 基线 45.08），即已接近收敛；不追求超过 15 轮平台。
  - 代价：每轮本地更新更小、聚合轮更多，攻击/防御的单轮强度会变弱，但 50 轮累计量相当。
- **CIFAR 不变**：`FL_LOCAL_STEPS=24`，50 × 24 = 1200 step ≈ 历史 15 轮（1170 step），目标仍是 ~70%。
- 该修订只影响 50 轮全量队列；日常 5/15 轮实验仍走 1/3 epoch，不受影响。

### CIFAR 50 轮等效方案（2026-10-29 用户确认）

历史 CIFAR 每客户端约 5000 条，BS64、`drop_last=True` → 每轮约 **78 步**；15 轮共约 **1170 步**。
50 轮若要同等本地优化步数，每轮应约 **23–24 步**。已实现 `FL_LOCAL_STEPS`：

```bash
export FL_LOCAL_STEPS=24   # 50 × 24 = 1200 步，接近历史 1170
```

- `federated_learning/client.py` 已支持每轮本地 step 上限；默认未设置时行为不变。
- 保持全量数据、BS64、lr0.01 不变；每轮 shuffle 后只跑前 24 个 batch，50 轮覆盖全量数据。
- 梯度平均改为按实际 step 数（`steps_done`），避免上限生效后梯度尺度偏小。
- 不建议用调大 batch 硬凑：BS192/224 会改变优化动态，且 lr 未同步调整，效果不可比。
- 不建议减少每轮参与客户端：会破坏 10 客户端 / 30% 恶意 / DPFLA 语义。
- 实施前先跑 5 轮 CIFAR 对照（`FL_LOCAL_STEPS=24` vs 历史 15×1 曲线量级），确认无异常后再进入 50 轮。

## §2 分支与封存

- 已创建并推送只读分支：

```text
archive/2026-10-29-pre-fullrun  @ 043cac5
```

- 该分支保存 50 轮实验开始前的完整代码快照；**不再提交、不合并、不删除**。
- 后续所有改动只进 `dev`；需要对比时 `git diff archive/2026-10-29-pre-fullrun..dev`。

## §3 阶段 A：非分布式全量 50 轮

### A1 实验矩阵（每数据集 9 条，共 18 条）

CIFAR10：
1. 基线：`no_attack + fedavg`
2. `label_flipping` × {fedavg, DPFLA, mkrum, tmean}
3. `minmax` × {fedavg, DPFLA, mkrum, tmean}

VisDrone：
1. 基线：`no_attack + fedavg`
2. `minmax` × {fedavg, DPFLA, mkrum, tmean}
3. `ipm` × {fedavg, DPFLA, mkrum, tmean}

不跑已淘汰的高斯 / SignFlipping，避免把 50 轮算力花在旧候选上。

### A2 运行方式

- 单进程、顺序跑，一次只跑一条；用 tmux，`FL_ATTACHED=1` 前台运行。
- 统一环境变量：

```bash
export FL_GLOBAL_ROUNDS=50
export FL_LOCAL_EPOCHS=1
# FL_LOCAL_STEPS 只给 CIFAR；VisDrone 运行前必须 unset
export FL_NUM_WORKERS=10
export FL_FRAC_WORKERS=1.0
export FL_MALICIOUS_RATE=0.3
export FL_GRADIENT_SCALE=1
export FL_ATTACHED=1
```

- CIFAR 额外：`FL_TRAIN_BATCH_SIZE=64`、`FL_TEST_BATCH_SIZE=1000`、`FL_LOCAL_LR=0.01`、**`FL_LOCAL_STEPS=24`**。
- VisDrone 额外：

```bash
export YOLO_MODEL_SIZE=m
export FL_IMG_SIZE=1280
export FL_AMP=1
export FL_TRAIN_BATCH_SIZE=16
export FL_TEST_BATCH_SIZE=64
export FL_LOCAL_LR=0.0002
export FL_LOCAL_EPOCHS=1
unset FL_LOCAL_STEPS   # VisDrone 不设步数上限
```

- CIFAR 入口：
  - 基线：`run-test/cifar10/run_no_attack_baseline.py`
  - 标签翻转：`run_no_defense_label_flipping.py` / `run_dpfla_label_flipping.py` / `run_mkrum_label_flipping.py` / `run_tmean_label_flipping.py`
  - MinMax：`FL_SERVER_ATTACK=minmax FL_RULE=<fedavg|DPFLA|mkrum|tmean> python run-test/cifar10/run_screen_server_attack.py`
- VisDrone 入口：
  - 基线：`run-test/visdrone/run_no_attack_baseline.py`
  - MinMax/IPM：`FL_SERVER_ATTACK=<minmax|ipm> FL_RULE=<...> python run-test/visdrone/run_screen_server_attack.py`

### A3 验收

- 每条跑满 50 轮，进程自然退出 0，无 OOM/NaN。
- 日志落到 `logs_3/<dataset>/`，每条有完整参数块和 50 轮曲线。
- 记录 best / last / 末 5 轮均值，与历史 15 轮表格并列；不强行要求“提升”，以“量级接近、无异常”为准。
- 结果回写 `01_Facts.md`（带日期/出处）；计划与状态回写 `00_Current_State.md`。

### A4 资源与顺序

- 先 CIFAR 9 条，再 VisDrone 9 条；VisDrone 50 轮 × 9 条成本远高于 CIFAR，按 tmux 排队。
- 每次只跑一个，避免 GPU/CPU 抢占和日志互相干扰。
- 需要 GPU + 大内存；**后续统一按 32GB GPU 配置规划**。CIFAR 无压力；VisDrone 保持 `v8m@1280 + AMP=1 + BS16`，若 32GB 下评估 OOM，先降 test batch，不要同时改分辨率/模型（保持单变量）。当前 2GB 无卡容器不能跑，换有卡/高内存主机。

## §4 阶段 B：分布式 10 客户端 50 轮与写入冲突

### B1 范围（2026-10-29 用户明确）

- **分布式只跑 DPFLA**，不再跑 FedAvg / Multi-Krum / Trimmed-Mean 分布式分支。
- **仅用于验证分布式能否使用**（10 client 同时训练、聚合、退出、文件写入），**不用于比较 accuracy / mAP**；所有准确率/mAP 对比一律走非分布式阶段 A。
- 每数据集 1 条代表性实验：
  - CIFAR：`label_flipping + DPFLA`
  - VisDrone：`minmax + DPFLA`
- 10 个 client 进程、全量分区、`--rounds 50`、`--local-epochs 1`、`--malicious-rate 0.3`、`--full-data`；CIFAR 另加 `FL_LOCAL_STEPS=24`。
- 不扩展成 9 条/数据集的分布式矩阵；若以后确需扩展，需用户另行明确。

### B2 启动前

1. 在 server 机器 `prepare` 全量分区，拷贝到所有 client。
2. 在每个 client 机器先单独运行一次 `prepare` 或单 client 预热，生成 `cache/` 标注缓存，避免 10 个进程同时首次写 cache。
3. 每个 client 使用唯一 `--workdir`；`partitions.json` 只读共享。
4. 检查磁盘空间：DPFLA `orthogonal_matrices`、VisDrone cache、日志。

### B3 冲突观察点

| 路径 | 写入方 | 风险 |
|---|---|---|
| `cache/cache_VisDrone_*.pkl` / `cache_yolo_*.pkl` | 每个 client 首次加载 | 10 进程同时 `pickle.dump` 同一文件 → 截断/损坏 |
| `orthogonal_matrices/*.pkl` | server DPFLA | 单 server 正常；多 server 会冲突 |
| `visdrone_temp.yaml` | server 评估 | 单 server 正常 |
| `tb/` | 单机 FL | 分布式不写 TB；若误开需隔离 |
| `logs_3/` | 单机 FL | 分布式用各自 `workdir`，不应写这里 |
| `dist_run/partitions.json` | 所有 client 读 | 只读，不写 |
| `__pycache__/` | 所有进程 | 可能并发写，但 Python 可容忍 |

### B4 执行顺序

1. **2 轮全量冲突探针**：10 client + 全量数据 + 本地 1 epoch，跑 2 轮；检查 cache 文件哈希、`orthogonal_matrices`、日志无 Traceback。
2. 无冲突 → 正式 **50 轮**；结束后同样检查文件哈希/大小/日志。
3. 若 cache 冲突：改为每个 client 独立 cwd/cache，或预热后只读；不直接改业务代码。
4. 若 `orthogonal_matrices`/`visdrone_temp.yaml` 冲突：确认是否只有一个 server；多 server 需隔离目录。
5. 记录每轮 client 更新数、server 聚合数、退出码、文件哈希，回写 `01_Facts/03_History`。

### B5 验收

- 10 个 client 每轮都有 update；50 轮 `rounds.jsonl` 完整；server/client 退出 0。
- 无 `File exists` / `Permission` / `pickle` 截断 / `KeyError` / `NaN`。
- 两个数据集代表性实验均完成；如发现冲突，记录“路径 + 错误 + 修复方案”；修复后只重跑分布式可用性验证，不扩分布式矩阵。
- 分布式结果只写“是否可用/冲突情况”，**不写 accuracy/mAP 结论**；指标对比仍以阶段 A 非分布式结果为准。

## §5 不做的事

- 不把 CIFAR local epochs 改成 0.5；已改用 `FL_LOCAL_STEPS=24` 控制每轮本地步数。
- 不通过调大 batch 硬凑 CIFAR 总步数；优化动态会变，且未做 lr 同步。
- 不为“分布式”新增 TB 写入（已明确关闭）。
- 不跑 FedAvg / Multi-Krum / Trimmed-Mean 的分布式分支；分布式只做 DPFLA 可用性验证。
- 不把分布式结果用于 accuracy/mAP 对比。
- 不并发跑多个单机实验。
- 不在当前 2GB 无卡容器上跑 50 轮全量。
- 不改攻击强度/恶意率/梯度放大等已定口径；本轮只改全局轮数、VisDrone local epochs，以及已确认的 CIFAR 每轮本地 step 上限。

## §6 提交与状态

- 本计划写入 memory 后提交 `dev`。
- 分支 `archive/2026-10-29-pre-fullrun` 只读封存。
- 完成阶段 A 后更新 `01_Facts` / `00_Current_State`；完成阶段 B 后补冲突结论。
