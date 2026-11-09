# 10_YOLOv8s_Plan.md — YOLOv8s 迁移计划

<!-- 最后更新：2026-11-09 -->
<!-- 级别：执行计划。目标/验收来自用户 2026-11-09 指令；外部调研标注来源；本项目结论待实测后回写 01_Facts。 -->
<!-- 前置：YOLOv8m 阶段已封存 archive/2026-11-09-yolov8m-complete @ 4f6866c；本计划只在 dev 执行。 -->

## §0 任务定义（用户原话要点）

1. 当前 `dev` 复制出一个新分支作为小结快照，标志 YOLOv8m 任务基本完成。
2. 后续在 `dev` 把 VisDrone 检测模型替换为 **YOLOv8s（确定，必须用这个）**。
3. 在 YOLOv8s 条件下，此前定下的 mAP / accuracy 目标必须继续达标。
4. 背景：课题落地场景是**边缘终端**，YOLOv8m 在边缘设备上做联邦训练基本不可行。

**已执行**：快照分支 `archive/2026-11-09-yolov8m-complete` 已创建并推送（`4f6866c`），只读，不合并、不修改。

## §1 验收目标（沿用原定要求，不降低）

| # | 数据集 | 目标 | 说明 |
|---|---|---|---|
| G1 | VisDrone | 无攻击联邦基线 mAP@0.5 **≥32%** | 原始项目目标；建议争取 ≥35% |
| G2 | VisDrone | MinMax / IPM 两条攻击线下，DPFLA **≥35%**，且比 Multi-Krum / Trimmed-Mean **高 ≥1%** | 与 15 轮/50 轮正式矩阵同口径 |
| G3 | VisDrone | 50 轮全量复核同样满足 G1/G2 | 若 50×12 step 对 v8s 欠拟合，允许改用 15×3 epoch 口径并明确标注 |
| G4 | CIFAR10 | 有攻防 **≥65%**，DPFLA 比 Multi-Krum / Trimmed-Mean **高 ≥2%** | CIFAR 仍用 `ResNet18CIFAR`，YOLO 替换不应影响；做回归确认 |
| G5 | 分布式 | VisDrone 10 客户端 × 50 轮 DPFLA 可用性 PASS | 只验证链路，不作指标结论 |

**统一约束**：10 客户端 / IID / 恶意率 30% / `FL_GRADIENT_SCALE=1`（不放大）/ 单变量；v8m 的历史数值只作参照，**不能直接当作 v8s 的验收依据**。

> **主硬门槛是 G2（攻击 + DPFLA ≥35）**。G1=32% 只是原始最低线；由于 DPFLA 在攻击下通常比无攻击基线低约 1.2 点（见 §2.5），Phase 1 的**预门槛设为无攻击基线 ≥36.2%**，达不到就不进入攻击矩阵。CIFAR10 的 65% 目标当前有 3–5 点余量，不是本轮瓶颈。

## §2 调研结论（2026-11-09，外部资料 + 本项目历史）

### 2.1 模型规模与边缘可行性

| 模型 | 参数量 | GFLOPs | fp32 单份权重 | 相对 v8m |
|---|---:|---:|---:|---:|
| YOLOv8s | **11.17M** | **28.8** | **≈44.7MB** | 参数 −57%、算力 −64% |
| YOLOv8m | 25.90M | 79.3 | ≈103.6MB | — |

- 数据来源：Ultralytics 官方 `yolov8.yaml`（`s: 11,166,560 parameters, 28.8 GFLOPS`；`m: 25,902,640 parameters, 79.3 GFLOPS`）。
- 注：官方数字是 COCO 80 类；本项目替换为 VisDrone 10 类检测头后参数量会略降，Phase 0 以实际打印为准。
- 含义：v8s 每轮通信 payload 约为 v8m 的 43%，训练/推理算力约为 36%，更符合边缘终端约束；但 v8s 仍是 GPU 级模型，不等于 MCU 可训练。
- 边缘联邦可行性有公开先例：TNO 2025 在 VisDrone 上使用 YOLOv8s 做联邦目标检测（FedBN，非 IID）；UltraFlwr 把 Ultralytics YOLO 与 Flower 结合并支持 backbone/neck/head 部分聚合；Ultralytics 的 Scaleout 案例也在边缘 GPU 节点上做 YOLO 联邦微调。

### 2.2 VisDrone + YOLOv8s 精度参照（集中式，非本项目口径）

| 来源 | 配置 | 结果 | 说明 |
|---|---|---|---|
| IJCS 2024（Small Object Detection in Aerial Drone Imagery based on YOLOv8） | YOLOv8s @1280 | **mAP 35.5 / AP50 55.9** | 集中式训练；说明 v8s@1280 有足够精度上限 |
| SL-YOLO（arXiv 2411.11477） | YOLOv8s | AP50 **43.0** / AP50-95 26.0 | 分辨率/配方不同，作量级参考 |
| YOLOv8s-SSB（ACM 2024） | 原版 YOLOv8s | AP50 约 **39–40**（改进版 44.2，原文称 +4.9） | 同上 |
| DGBL-YOLOv8s（MDPI 2025） | 原版 YOLOv8s | mAP **36.3** | 指标定义/口径需以原文为准 |

**判断**：公开集中式 v8s 在 VisDrone 上的 AP50 大致 36–56，本项目 G1=32 / G2=35 属于保守目标，**理论上可达**；但联邦 + 少轮数 + 攻击会显著低于集中式，必须实测。

### 2.3 本项目历史证据（最重要）

- **v8s @1024 / BS32 / AMP / 10 客户端 IID / 2e-4 / 3 local epochs / 15 轮：R15 = 35.92%**（`01_Facts.md` §1b，`logs_3/visdrone/run_no_attack_baseline_20261010_1813.log`）。
  - 这是 v8s 已在本项目管线中达到 35.92 的直接证据；但当时是 1024/BS32，本轮主口径是 1280/BS16，仍需重测。
- v8m @1280 / 12 step / 50 轮：无攻击基线 39.65%、DPFLA MinMax 38.36%、DPFLA IPM 38.32%（`01_Facts.md` §8）。
  - v8s 预计比 v8m 低若干点；若沿用 50×12 step（总 600 step），可能欠拟合，**不能假设 v8s 自动达标**。
- DPFLA 阈值（aux 0.60/0.75/0.48）是在 v8m/CIFAR 上标定的；v8s 的 aux 分布可能偏移，Phase 2 必须先诊断再决定是否重标定。

### 2.4 结论

1. 模型替换本身是**单变量可控**的：代码已支持 `YOLO_MODEL_SIZE=s`，`yolov8s.pt` 已在仓库根。
2. 主要风险不是“能不能加载”，而是**v8s 在 1280/少 step 下能否达到 32/35**，以及 **DPFLA 阈值是否仍能分离 v8s 的恶意/诚实更新**。
3. 推荐先用本项目已证明有效的 **15 轮 × 3 local epochs** 口径做达标验证，再决定是否跑 50 轮全量；不要直接上 50×12 step 赌结果。

### 2.5 可行性判断（2026-11-09，用户强调“攻击 + DPFLA ≥35”）

**已知攻击代价（v8m，DPFLA 相对无攻击基线）**：

| 口径 | 基线 | DPFLA MinMax | DPFLA IPM | 差值 |
|---|---:|---:|---:|---:|
| 15 轮 × 3 local epochs | 45.08 | 43.85 | 43.93 | **−1.23 / −1.15** |
| 50 轮 × 12 step | 39.65 | 38.36 | 38.32 | **−1.29 / −1.33** |

→ 两条攻击线、两种训练量下，DPFLA 相对基线的损失稳定在 **约 1.2 点**。因此：

- 若 v8s 无攻击基线只有 **35.9**（历史 v8s@1024 R15），攻击 + DPFLA 预计 **≈34.7**，**达不到 35**。
- 要保证攻击 + DPFLA ≥35，v8s 无攻击基线至少需要 **≈36.2**（35 + 1.2 余量）。
- **Phase 1 预门槛 = 36.2%**；低于此值不启动 Phase 2，先按 P1c 调整训练配方。

**v8s 可能的提升杠杆（按优先级，单变量）**：

1. **增加本地训练量**：15 轮 × 3 epochs（45 local epochs）优先；若 50 轮全量，改用 50×24 或 50×41 step，而不是 50×12（后者只有约 15 local epochs，可能欠拟合）。
2. **学习率**：2e-4 → 3e-4 / 5e-4（v8s 容量小，2e-4 可能偏保守）。
3. **分辨率**：1280 为主；若边缘时延要求或显存受限，再单独做 1024 口径（不与 1280 混比）。
4. **训练轮数**：15 → 25 轮（若算力允许；v8m 25 轮基线曾到 47.64）。
5. **数据增强/优化器**：Mosaic/MixUp、momentum 等仅在上述无效时再动，且必须单变量。

**若仍达不到 36.2 的处置**：

- 不允许通过削弱攻击强度、调低恶意率或换回 v8m 来“达标”。
- 先向用户报告“v8s 容量/训练量冲突”，在以下选项中决策：
  a. 增加训练量/轮数（推荐，仍保持 v8s）；
  b. 接受 15 轮口径下的 best/last5 报告，但明确标注未达 35；
  c. 调整验收口径（需用户明确同意）。
- CIFAR10 不受影响：DPFLA 在标签/MinMax 线均为 68–70，65 目标有余量。

## §3 代码现状与改造清单（先确认，后动手）

### 3.1 已支持、无需改

- `federated_learning/models/model.py`：`YOLO_MODEL_SIZE` 支持 `n/s/m/l/x`，默认 `s`；`yolov8s.pt` 已存在。
- `federated_learning/fl_core.py`：TB 目录名从 `YOLO_MODEL_SIZE` 读取，s 会自动写成 `yolov8s`。
- `federated_learning/fl_algorithm/DPFLA.py`：YOLO 特征键 `model.22.cv3.*` 对 v8s 同样存在，按类切片数仍为 10。
- `federated_learning/attack_alg/server_attacks.py`：MinMax/IPM 与模型规模无关。
- `run_distributed_smoke.py` / `launch_distributed.py`：已支持 `--model-size s`。

### 3.2 需要参数化/改默认（建议，待用户确认）

| 位置 | 现状 | 拟改 | 风险 |
|---|---|---|---|
| [run_screen_server_attack.py](file:///root/chashaobao/DPFLA-master/run-test/visdrone/run_screen_server_attack.py#L34) | `setdefault("YOLO_MODEL_SIZE","m")` | 改为 `"s"`；显式 `YOLO_MODEL_SIZE=m` 仍可复现 v8m | 旧 v8m 命令需显式带 env |
| [run_screen_label_flipping.py](file:///root/chashaobao/DPFLA-master/run-test/visdrone/run_screen_label_flipping.py#L34) | 同上 | 同上 | 同上 |
| [full50_supervisor.py](file:///root/chashaobao/DPFLA-master/scripts/full50_supervisor.py#L201) | VisDrone 环境写死 `YOLO_MODEL_SIZE=m` | 改为读取 `FULL50_YOLO_SIZE`（默认 `s`），TB 前缀按实际 size 生成（当前 [L219](file:///root/chashaobao/DPFLA-master/scripts/full50_supervisor.py#L219) 写死 `yolov8m`） | 需同步清理旧 TB 前缀逻辑，避免 s/m 混目录 |
| [run_distributed_full.sh](file:///root/chashaobao/DPFLA-master/scripts/run_distributed_full.sh#L25) | VisDrone `MODEL_SIZE=m` | 改为 `${YOLO_MODEL_SIZE:-s}` 或 `MODEL_SIZE=${MODEL_SIZE:-s}` | 分布式 v8m 复现需显式 m |
| VisDrone 入口脚本 | 无“必须 s”的保护 | 新增 `FL_REQUIRE_YOLO_SIZE=s` 检查（或启动断言），实际 size 不符则直接报错 | 低；可防漏设 env 静默跑成 m |
| [visdrone_fed_hparams.py](file:///root/chashaobao/DPFLA-master/run-test/visdrone/visdrone_fed_hparams.py#L18-L27) | 默认已是 `s`，注释仍写“可 export m” | 更新注释：v8s 为强制口径，v8m 仅历史归档 | 低 |

### 3.3 建议的收敛方式

- **不复制一套 v8s 脚本**，只做参数化 + 默认值收紧；保留 `YOLO_MODEL_SIZE=m` 作为历史复现开关。
- 所有 v8s 正式命令显式带 `YOLO_MODEL_SIZE=s`；入口断言二次兜底。
- 代码改动单独提交，先跑 Phase 0 冒烟，确认不破坏 v8m 历史复现路径（用一条 1 轮 v8m 冒烟验证 env 仍可覆盖）。

## §4 实验设计（分阶段 + 硬门槛）

### 固定因子

```
10 客户端 / IID / 恶意率 0.3 / FL_GRADIENT_SCALE=1
YOLO_MODEL_SIZE=s / FL_IMG_SIZE=1280 / FL_AMP=1 / FL_TRAIN_BATCH_SIZE=16 / FL_TEST_BATCH_SIZE=64
LOCAL_LR=2e-4 / constant / seed 对齐 Arguments 默认
```

### Phase 0 — 代码准备与冒烟（不产生指标结论）

1. 完成 §3.2 的参数化/默认值改动；`py_compile` + `bash -n`。
2. 1 轮冒烟：加载 `yolov8s.pt`、替换 10 类检测头、1280/BS16/AMP 前向+反传+val。
3. 记录：实际参数量（应 ≈11.17M）、峰值显存、单轮耗时、TB 目录名是否含 `yolov8s`。
4. **G0**：无 OOM/NaN；参数量正确；日志明确 `yolov8s`；1 轮 mAP 正常（非 0 即可，不做精度结论）。

### Phase 1 — v8s 基线校准（决定后续口径）

| 步骤 | 配置 | 目的 | 门槛 |
|---|---|---|---|
| P1a | 5 轮 × 3 local epochs，无攻击 | 链路/收敛趋势冒烟 | mAP 随轮次上升，无 OOM |
| P1b | **15 轮 × 3 local epochs，无攻击** | 主验收基线（对齐历史 v8s@1024 口径） | **G1：mAP@0.5 ≥32%**；**G1-pre：≥36.2%**（进入 Phase 2 的必要条件） |
| P1c | 若 P1b <36.2%，单变量救援 | 只动一个杠杆，逐次记录 | 见下方顺序 |
| P1d | 50 轮 × 12 step，无攻击（可选） | 判断 50 轮全量口径是否适合 v8s | ≥36.2% 才用于 Phase 3 |

**P1c 单变量顺序（每次只改一项，失败再下一项）**：

1. `FL_LOCAL_LR` 2e-4 → 3e-4（再 5e-4）。
2. `FL_LOCAL_EPOCHS` 3 → 5（计算量允许时）。
3. `FL_IMG_SIZE` 1280 → 1024（仅当 1280 不达标或边缘时延要求；作为独立口径报告）。
4. `FL_TRAIN_BATCH_SIZE` 16 → 8 或 32（最后考虑；会改变优化动态）。

> 不建议先改模型/攻击/恶意率；**G1-pre 未过之前不要进入攻击矩阵**。若 15×3 仍达不到 36.2，再按 P1c 逐项调整。

### Phase 2 — 15 轮攻击/防御筛选（G2）

- **前置**：Phase 1 的 G1-pre（无攻击基线 ≥36.2）已通过。
- 8 条：`{MinMax, IPM} × {FedAvg, DPFLA, MKrum, TMean}`，其余与 P1b 完全一致。
- **G2**：DPFLA ≥35%，且比 MKrum/TMean 都高 ≥1%（末5 为主口径）。
- 若 DPFLA 不达标，按顺序诊断：
  1. 确认服务端攻击已注入且 `local_models` 已同步（日志关键字/权重）。
  2. 打印 v8s 的恶意端/诚实端 aux 分布、KMeans 标签、聚合权重，与 v8m 的 0.707/0.461 对照。
  3. 若分布分离不足，重标定 DPFLA 阈值（0.60/0.75/0.48）——一次只改一个，记录依据；不得改攻击强度或恶意率。
- **不要在 G2 未过时扩到 50 轮或分布式。**

### Phase 3 — 50 轮全量复核（G3，Phase 2 通过后）

- VisDrone 9 条：基线 + `{MinMax, IPM} × {FedAvg, DPFLA, MKrum, TMean}`。
- 口径二选一，必须在报告中写明：
  - **P3-A**：50 轮 × 12 step（与 v8m 50 轮全量一致；若 P1d 证明达标）。
  - **P3-B**：50 轮 × 24 step 或 15 轮 × 3 epoch（若 12 step 对 v8s 欠拟合；与 v8m 50 轮不可直接横比）。
- **G3**：R50 / 末5 满足 G1/G2；无 OOM/NaN；9 条日志归档。

### Phase 4 — CIFAR10 回归（G4）

- 本任务只改 VisDrone 模型/默认值，CIFAR 代码路径不动；做 2–3 条回归即可：
  - 基线、标签翻转 + DPFLA、MinMax + DPFLA（15 轮，沿用 11-01 口径）。
- **G4**：结果与 11-01 归档在 ±0.5 内，且 DPFLA ≥65%；若偏差大，先查代码改动是否误伤 CIFAR。

### Phase 5 — 分布式 v8s 可用性（G5）

- VisDrone 10 客户端 × 50 轮，MinMax + DPFLA，`--model-size s`，超时 18h。
- **G5**：每轮 10 条 update、server/client 退出 0、`rounds.jsonl` 完整、无 cache/文件写冲突；指标不作为结论。
- CIFAR 分布式已 PASS，且本任务不改分布式代码路径，默认不重跑；若改了 `run_distributed_smoke.py` 才补 1 条 CIFAR 冒烟。

### Phase 6 — 归档与记忆回写

- `01_Facts.md` 新增 v8s 章节（基线、四方法、双攻击，R50/末5/末10/best + 日志出处）。
- `00/03/08/09` 更新状态；`log/README.md` 增加 v8s 索引。
- 完成后可再建 `archive/YYYY-MM-DD-yolov8s-complete` 快照。

## §5 资源与风险

### 资源

- 32GB GPU 规划：v8s 参数量/FLOPs 约为 v8m 的 43%/36%，1280/BS16/AMP 预计可跑；Phase 0 实测峰值显存，若 OOM 先降 test batch 或 train batch（单变量）。
- 单条耗时以 Phase 0 实测为准；按 FLOPs 比例预计明显低于 v8m，但不要用估算值替代实测。
- 磁盘：v8s TB/日志体积小于 v8m；仍需预留 `logs_3/`、`tb/`、`logs_dist_full/` 空间。

### 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| v8s 在 50×12 step 下欠拟合 | 基线 <32 | 改用 15×3 epoch 或 50×24/41 step；在报告中注明口径 |
| v8s 基线达不到 36.2 | 攻击 + DPFLA 预计 <35 | 先加本地训练量/学习率/轮数；不要进入 Phase 2，不要削弱攻击 |
| v8s 容量不足，DPFLA <35 | G2 失败 | 先确认攻击可见性，再重标定 DPFLA 阈值；不要改攻击强度 |
| DPFLA 阈值从 v8m 迁移失效 | 误杀/漏杀 | Phase 2 打印 aux 分布，必要时重标定并记录 |
| 代码默认值改动破坏 v8m 复现 | 历史不可复现 | 保留 `YOLO_MODEL_SIZE=m` 显式开关；提交前跑 1 轮 v8m 冒烟 |
| 1280 在边缘设备仍偏重 | 与课题背景冲突 | 主验收保持 1280；另设 1024 边缘口径作为附加实验，不与 1280 混比 |
| TB/日志 s/m 混目录 | 结果误读 | TB 前缀按实际 size 生成；v8s 正式 run 前清理/隔离旧前缀 |

## §6 调研来源（外部，未在本项目复现）

- Ultralytics YOLOv8 官方模型配置（v8s/v8m 参数量与 GFLOPs）：https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/v8/yolov8.yaml
- IJCS 2024, *Small Object Detection in Aerial Drone Imagery based on YOLOv8*（v8s@1280 VisDrone）：https://iaeng.org/IJCS/issues_v51/issue_9/IJCS_51_9_13.pdf
- TNO 2025, *Federated object detection for defence and security applications using realistic unbalanced heterogeneous data distributions*（VisDrone + YOLOv8s + FedBN）：https://publications.tno.nl/publication/34645138/YSsjzRDW/spek-2025-federated.pdf
- SL-YOLO（YOLOv8s VisDrone AP50 43.0）：https://arxiv.org/html/2411.11477
- YOLOv8s-SSB（原版 v8s VisDrone AP50 约 39.3）：https://dl.acm.org/doi/pdf/10.1145/3703935.3704034
- UltraFlwr（Ultralytics YOLO + Flower 联邦，部分聚合）：https://arxiv.org/pdf/2503.15161v2
- Ultralytics Scaleout 边缘联邦案例：https://www.ultralytics.com/customers/scaleout-cuts-model-updates-from-weeks-to-hours-with-ultralytics-yolo
- Frontiers 2026, UAV 联邦目标检测（YOLOv8n + Jetson Nano 边缘推理）：https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2026.1649168/full

> ⚠️ 以上外部数据均为**集中式/不同口径**，只用于判断“v8s 精度上限与边缘可行性”，不能替代本项目 v8s 联邦实测。

## §7 执行顺序总览

```
Phase 0 代码参数化 + 1 轮冒烟
   ↓ G0
Phase 1 v8s 基线（5 轮 → 15×3 轮）→ G1 ≥32%
   ↓
Phase 2 15 轮 MinMax/IPM × 四方法 → G2 DPFLA ≥35 且领先 ≥1
   ↓
Phase 3 50 轮全量（P3-A 12 step 或 P3-B 15×3/24 step）→ G3
   ↓
Phase 4 CIFAR 回归 → G4
   ↓
Phase 5 分布式 v8s 可用性 → G5
   ↓
Phase 6 归档 + 回写 01_Facts / 00 / 03 / 08 / 09
```

**当前状态**：快照分支已建；本计划待用户确认代码改造清单后进入 Phase 0。未确认前不改训练/模型代码。
