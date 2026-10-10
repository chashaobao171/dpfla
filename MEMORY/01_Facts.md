# 01_Facts.md — 事实层（仅实测数据）

<!-- 最后更新：2026-10-10 -->
<!-- 规则：每条必须带 日期 + 出处（日志文件名或代码位置）。无出处的推断不得写入本文件。 -->

## §1 VisDrone 联邦基线（好配置口径）

好配置口径：YOLOv8s、BS=64、LR=2e-4、LR_SCHEDULE=constant、10 客户端、Non-IID。

| 规则 | 轮次 | 最佳 mAP@0.5 | 日期 | 出处 |
|------|------|-------------|------|------|
| FedAvg | R6 | 21.93% | 2026-06-08 | `logs_3/visdrone/run_no_attack_baseline_20260608_0755.log` |
| FedAvg | R8 | 22.49% | 2026-06-08 | 同上 |
| FedBN | R7（峰值） | 21.55% | 2026-06-07 | `logs_3/visdrone/run_no_attack_baseline_fedbn_20260607_1625.log` |
| FedLA | R8 | 22.49% | 2026-06-08 | `logs_3/visdrone/run_no_attack_baseline_fedla_20260608_1703.log` |
| FedAvg（100 轮，Bug1-3 修复后） | R26-33 | ≈23.7% | 2026-06-10 | `logs_3/visdrone/run_no_attack_baseline_20260610_1139.log` |

FedAvg vs FedBN 逐轮（同口径）：

| Round | FedAvg | FedBN |
|---|---|---|
| R1 | 7.81% | 6.08% |
| R2 | 16.78% | 15.45% |
| R3 | 19.50% | 17.92% |
| R4 | 21.16% | 20.18% |
| R5 | 21.68% | 20.42% |
| R6 | 21.93% | 20.34% |
| R7 | — | 21.55% |
| R8 | 22.49% | 20.94% |

结论：FedBN 在 VisDrone（标签偏斜型 Non-IID）下**全程不优于 FedAvg**；FedLA 与 FedAvg 接近。

## §2 集中式 oracle（上限参考）

| 配置 | mAP@0.5 | 日期 | 出处 |
|------|---------|------|------|
| YOLOv8l, BS=16, EPOCH=50, LR=0.01, Mosaic+Cosine | 41.4%（Epoch 3） | 2026-06-09 | 历史记载；原始脚本 `run_oracle_yolov8l.py` **已不在仓库** |

【推断·未验证】集中式 vs 联邦差距约 2×。注意 oracle 是 v8l + LR0.01 + 50 epoch，联邦是 v8s + LR2e-4 + 3-10 epoch，**两者不同口径，差距来源未做拆解**，不能简单归因于"联邦本身"。

## §3 反例（差配置）

| 配置 | 结果 | 日期 | 出处 |
|------|------|------|------|
| BS=16, EPOCH=3, LR=5e-4, cosine+warmup | R15 触顶 14.99% → R30 14.68% | 2026-06-09 | `01_Workbench`（旧）记载 |
| FedAvgM 0.9 恒开 + clip 10.0（9/2 demo） | 10 轮仅 2.55% | 2026-09-02 | `logs_3/demo_out.log` / `01_ChangeLog`（旧）记载 |

## §4 2026-09 运行日志

| 日期 | 内容 | 结果 | 出处 |
|------|------|------|------|
| 2026-09-07 | proof_dynamic，1 轮 | **训练前**冷启动 mAP=0.00%；**R1=3.24%**（P=26.45%, R=6.85%） | `logs_3/visdrone/proof_dynamic_20260907_1520.log` |
| 2026-09-15 | 基线 | ❌ val 子进程被杀 `code=-9`（OOM，非代码 bug） | `logs_3/visdrone/run_no_attack_baseline_20260915_2144.log` |
| 2026-09-16 | 50 轮基线 | **R1=3.35%**（日志在 R2 处中断） | `logs_3/visdrone/run_no_attack_baseline_20260916_1437.log` |

**关键事实**：R1≈3.3% 在 2026-06-10 / 09-07 / 09-16 三次独立运行中一致 → 评估链路（含子进程 YOLO val）**工作正常**。
"训练前 mAP=0%" 是冷启动（检测头刚随机初始化 + BN 为 COCO 统计量）的正常现象，**不是 bug**。

## §5 MNIST（已冻结）

| 实验 | 结果 | 日期 | 出处 |
|------|------|------|------|
| FedAvg 无攻击 | ≈94% | 2026-05-15 | `logs_3/mnist/` |
| **DPFLA**（标签翻转 mr=30%） | **95.0%** | 2026-05-15 | `logs_3/mnist/run_dpfla_label_flipping_20260515_1036.log` |
| FedAvg（标签翻转 mr=30%） | 4.57% | 2026-05-15 | `logs_3/mnist/run_no_defense_label_flipping_20260515_1020.log` |

攻击成功率：DPFLA=0% vs FedAvg=94.29%。

## §6 当前生效配置（代码事实，2026-10-10 核对）

出处：`run-test/visdrone/visdrone_fed_hparams.py`、`federated_learning/fl_core.py`、`federated_learning/client.py`

| 参数 | 值 | 位置 |
|------|----|----|
| YOLO_MODEL_SIZE | `s`（默认） | visdrone_fed_hparams.py |
| LOCAL_LR | `2e-4` | 同上 |
| LR_SCHEDULE | `constant` | 同上 |
| GLOBAL_ROUNDS | `50` | 同上 |
| LOCAL_EPOCHS | `3` | 同上 |
| TRAIN_BATCH_SIZE | `64` | 同上 |
| NUM_WORKERS | `10` | 同上 |
| MALICIOUS_RATE | `0.1` | 同上 |
| FL_SERVER_MOMENTUM | `0`（FedAvgM 关） | fl_core.py:760 |
| FL_GRAD_CLIP | `0`（裁剪关） | client.py:479 |
| drop_last | `False` | client.py |

## §7 已知就绪但未验证的能力

| 能力 | 状态 | 说明 |
|------|------|------|
| Mosaic + MixUp | 代码已合入 | `visdrone_dataset.py`（Mosaic 50% / MixUp 10%），效果未单变量验证 |
| FedLA 聚合 | 已实现 | `average_weights_fedla()`，仅 FedLA 一路跑到 R8=22.49% |
| DPFLA 防御（YOLO） | 已实现 | SVD+KMeans 路径，历史曲线与 FedAvg 接近、攻击命中率 0%（特征选取已从 fc2.weight 改为检测头 cv3.weight，**改动后未重跑对比**） |
