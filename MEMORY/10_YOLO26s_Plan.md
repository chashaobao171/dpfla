# 10_YOLO26s_Plan.md — YOLO26s 迁移计划

<!-- 最后更新：2026-11-09 -->
<!-- 级别：执行计划。目标/验收来自用户 2026-11-09 指令；外部调研标注来源；本项目结论待实测后回写 01_Facts。 -->
<!-- 前置：YOLOv8m 阶段已封存 archive/2026-11-09-yolov8m-complete @ 4f6866c；本计划只在 dev 执行。 -->
<!-- 选型：YOLO26s 为首选；YOLOv8s 仅作为 YOLO26s 集成失败时的备选，不再默认使用。 -->

## §0 任务定义（用户 2026-11-09 指令）

1. 当前 `dev` 复制出一个新分支作为小结快照，标志 YOLOv8m 任务基本完成（已完成）。
2. 后续在 `dev` 把 VisDrone 检测模型替换为更轻、更强的 **YOLO26s**（用户明确“可以用 yolo26s，应该比 yolov8s 更好达到目标”）。
3. 在 YOLO26s 条件下，此前定下的 mAP / accuracy 目标必须继续达标。
4. 背景：课题落地场景是**边缘终端**，YOLOv8m 在边缘设备上做联邦训练基本不可行；YOLO26s 比 v8s 更小（10.01M vs 11.17M）、算力更低（22.8 vs 28.8 GFLOPs），且 VisDrone 公开结果更高（mAP50 41.0 vs 40.0）。

**已执行**：快照分支 `archive/2026-11-09-yolov8m-complete` 已创建并推送（`4f6866c`），只读，不合并、不修改。

## §1 验收目标（沿用原定要求，不降低）

| # | 数据集 | 目标 | 说明 |
|---|---|---|---|
| G1 | VisDrone | 无攻击联邦基线 mAP@0.5 **≥32%** | 原始项目目标；G2 需要更高余量，见 §2.3 |
| G2 | VisDrone | MinMax / IPM 两条攻击线下，DPFLA **≥35%**，且比 Multi-Krum / Trimmed-Mean **高 ≥1%** | **主硬门槛** |
| G3 | VisDrone | 50 轮全量复核同样满足 G1/G2 | 口径二选一，需明确标注 |
| G4 | CIFAR10 | 有攻防 **≥65%**，DPFLA 比 Multi-Krum / Trimmed-Mean **高 ≥2%** | CIFAR 仍用 `ResNet18CIFAR`，YOLO 替换不应影响 |
| G5 | 分布式 | VisDrone 10 客户端 × 50 轮 DPFLA 可用性 PASS | 只验证链路，不作指标结论 |

**统一约束**：10 客户端 / IID / 恶意率 30% / `FL_GRADIENT_SCALE=1`（不放大）/ 单变量；v8m 的历史数值只作参照，**不能直接当作 YOLO26s 的验收依据**。

> **主硬门槛是 G2（攻击 + DPFLA ≥35）**。G1=32% 只是原始最低线；v8m 实测 DPFLA 在攻击下比无攻击基线低约 1.2 点（§2.3），因此 Phase 1 的**预门槛设为无攻击基线 ≥36.2%**，达不到就不进入攻击矩阵。CIFAR10 的 65% 当前有 3–5 点余量，不是本轮瓶颈。

## §2 调研结论（2026-11-09）

### 2.1 YOLO26 / YOLO26s 是否开源、能否直接用

| 项 | 结论 |
|---|---|
| 发布 | Ultralytics 于 **2026-01-14** 发布 YOLO26；官方文档 `docs.ultralytics.com/models/yolo26/` |
| 许可 | **AGPL-3.0 + Enterprise**，代码/模型/文档开源；满足本项目学术研究使用 |
| 代码 | 本地 `ultralytics 8.4.50` 已包含 `ultralytics/cfg/models/26/yolo26.yaml`，无需升级 |
| 权重 | `yolo26s.pt` 已通过镜像下载到项目根（**19.48MB**；GitHub/HF 直连超时，用 `ghproxy.net` 成功） |
| 结构【实测】 | 24 层；Detect 在 **layer 23**；`nc=80, reg_max=1, end2end=True`；80 类参数量 **10,009,784** |
| 换头【实测】 | `Detect(nc=10, reg_max=1, end2end=True, ch=(128,256,512))` 可用；10 类换头后参数 **9,955,604** |
| 损失【实测】 | 必须用 `model.init_criterion()` 返回的 `E2ELoss`，不能硬编码 `v8DetectionLoss`；1280/BS1 前向+反传通过，GPU reserved 约 0.66GB |

### 2.2 模型规模与 VisDrone 精度参照

| 模型 | 参数量 | GFLOPs | fp32 单份权重 | 来源 |
|---|---:|---:|---:|---|
| YOLO26s | **10.01M** | **22.8** | ≈38.2MB | Ultralytics 官方 yolo26.yaml / 用户提供的 Sensors 表 |
| YOLOv8s | 11.17M | 28.8 | ≈44.7MB | Ultralytics 官方 yolov8.yaml |
| YOLOv8m | 25.90M | 79.3 | ≈103.6MB | Ultralytics 官方 yolov8.yaml |

**VisDrone2019（Aero-YOLO26, Sensors 2026, 26(19), 6346, Table 4/5；集中式）**：

| 模型 | 参数量 | GFLOPs | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLOv8s | 11.17M | 28.8 | 40.0% | 23.7% |
| YOLO11s | 9.46M | 21.7 | 39.5% | 23.3% |
| **YOLO26s** | **10.01M** | **22.8** | **41.0%** | **24.5%** |

→ YOLO26s 相对 v8s：参数 −10.4%、算力 −20.8%、mAP50 **+1.0**、mAP50-95 +0.8。**这是集中式结果，不能直接当作本项目联邦结果**，但说明精度上限足够，G1=32 / G2=35 属于保守目标。

### 2.3 可行性判断（重点：攻击 + DPFLA ≥35）

**已知攻击代价（v8m，DPFLA 相对无攻击基线）**：

| 口径 | 基线 | DPFLA MinMax | DPFLA IPM | 差值 |
|---|---:|---:|---:|---:|
| 15 轮 × 3 local epochs | 45.08 | 43.85 | 43.93 | **−1.23 / −1.15** |
| 50 轮 × 12 step | 39.65 | 38.36 | 38.32 | **−1.29 / −1.33** |

→ DPFLA 在攻击下的损失稳定在 **约 1.2 点**。因此：

- 要保证攻击 + DPFLA ≥35，YOLO26s 无攻击基线至少需要 **≈36.2%**（35 + 1.2 余量）。
- **Phase 1 预门槛 = 36.2%**；低于此值不启动 Phase 2，先按 P1c 调整训练配方。
- 历史 v8s@1024 R15=35.92，按此推算攻击后约 34.7，**处于临界/不足**；这也是改用 YOLO26s 的原因之一，但仍需实测，不能假定 YOLO26s 自动达标。

**YOLO26s 提升杠杆（按优先级，单变量）**：

1. **增加本地训练量**：15 轮 × 3 epochs 优先；50 轮全量改用 50×24 或 50×41 step，而不是 50×12（后者约 15 local epochs，可能欠拟合）。
2. **学习率**：2e-4 → 3e-4 / 5e-4（小模型 + COCO 预训练，2e-4 可能偏保守）。
3. **分辨率**：1280 为主；若边缘时延/显存受限，再单独做 1024 口径（不与 1280 混比）。
4. **训练轮数**：15 → 25 轮（若算力允许）。
5. **数据增强/优化器**：Mosaic/MixUp/momentum 仅在上述无效时再动，且必须单变量。

**若仍达不到 36.2 的处置**：

- 不允许削弱攻击强度、调低恶意率或换回 v8m 来“达标”。
- 先报告“YOLO26s 容量/训练量冲突”，由用户决定：增加训练量/轮数（推荐）、接受未达 35 的报告、或调整验收口径。
- CIFAR10 不受影响：DPFLA 在标签/MinMax 线均为 68–70，65 目标有余量。
- 备选：若 YOLO26s 集成被 E2E 损失/DPFLA 特征证明不可行，再退回 YOLOv8s（v8s 计划保留在 §7）。

## §3 代码现状与改造清单（先确认，后动手）

### 3.1 已支持、无需改

- `federated_learning/models/model.py`：已有 `YOLO_MODEL_SIZE` 选择 n/s/m/l/x；`yolov8*.pt` 路径。
- `ultralytics 8.4.50`：已支持 YOLO26 配置与 `Detect(end2end=True)`；`model.init_criterion()` 会返回 `E2ELoss`。
- `federated_learning/attack_alg/server_attacks.py`：MinMax/IPM 与模型规模无关。
- 数据、聚合、CIFAR 路径不需要改。

### 3.2 必须改（YOLO26s 集成）

| 位置 | 现状 | 拟改 | 风险/备注 |
|---|---|---|---|
| [yolo_wrapper.py](file:///root/chashaobao/DPFLA-master/federated_learning/models/yolo_wrapper.py#L113-L140) | 硬编码 `YOLO(f'yolov8{size}.pt')` / `.yaml` | 增加 `YOLO_MODEL_FAMILY`（`8`/`26`），权重名 `yolov8{size}.pt` / `yolo26{size}.pt`；默认 `8` 保持历史可复现，YOLO26 实验显式设 `26` | 旧 v8m 复现必须显式 `YOLO_MODEL_FAMILY=8` |
| [yolo_wrapper.py](file:///root/chashaobao/DPFLA-master/federated_learning/models/yolo_wrapper.py#L192-L205) | 换头用 `Detect(nc=..., ch=...)` 默认 `reg_max=16, end2end=False` | 换头时保留 `reg_max=head.reg_max`、`end2end=head.end2end`；YOLO26 为 `reg_max=1, end2end=True` | 不保留会导致 E2E 头结构错误 |
| [yolo_wrapper.py](file:///root/chashaobao/DPFLA-master/federated_learning/models/yolo_wrapper.py#L228) | 硬编码 `v8DetectionLoss` | 改为 `self.model.init_criterion()`；v8 仍返回 v8DetectionLoss，YOLO26 返回 E2ELoss | 已验证 YOLO26 必须 E2ELoss |
| [yolo_wrapper.py](file:///root/chashaobao/DPFLA-master/federated_learning/models/yolo_wrapper.py#L175-L190) | 只补 `model.args.hyp` | 同时补 `model.args.box/cls/dfl`（YOLO26 checkpoint 的 args 是 dict，无 hyp）；`E2ELoss` 依赖 `model.args.box` 等 | 已验证：不补会 `AttributeError: 'dict'/'SimpleNamespace' object has no attribute 'box'` |
| [yolo_wrapper.py](file:///root/chashaobao/DPFLA-master/federated_learning/models/yolo_wrapper.py#L305-L360) | 递归移动 criterion，但跳过 callable 的 `one2many/one2one` | 显式递归 `E2ELoss.one2many/one2one`，并设置其 `.device`；移动 `stride/proj/bbox_loss/assigner` 张量 | 已验证：不处理会 CPU/GPU 混用 |
| [DPFLA.py](file:///root/chashaobao/DPFLA-master/federated_learning/fl_algorithm/DPFLA.py#L226-L250) | `preferred_keys` 写死 `model.22.cv3.*` | 从 state_dict 动态探测 Detect 层索引（YOLO26 为 23），取 `model.{idx}.cv3.{0,1,2}.2.weight`；兜底 `one2one_cv3` | YOLO26 头索引是 23，写死 22 会失败 |
| [fl_core.py](file:///root/chashaobao/DPFLA-master/federated_learning/fl_core.py#L656-L663) | TB `_size_tag` 写死 `yolov8{size}` | 改为 family-aware：`yolo26s` / `yolov8s` | 避免 TB 目录混 |
| `run_distributed_smoke.py` / `launch_distributed.py` | 只有 `--model-size` | 增加 `--model-family`（`8`/`26`），透传到 server/client 环境 | 分布式正式 v26s 需要 |
| `scripts/full50_supervisor.py` | VisDrone 环境写死 `YOLO_MODEL_SIZE=m`；TB 前缀写死 `yolov8m` | 改为 `FULL50_YOLO_FAMILY`/`FULL50_YOLO_SIZE`（默认 `26`/`s`），TB 前缀按实际 family/size 生成 | 旧 v8m 队列需显式设 family=8 |
| `scripts/run_distributed_full.sh` | VisDrone `MODEL_SIZE=m` | 改为 `${YOLO_MODEL_FAMILY:-26}` / `${YOLO_MODEL_SIZE:-s}` | 同上 |
| VisDrone 入口脚本/打印 | 写死 `yolov8{size}.pt` | 改为 family-aware 显示；默认 `YOLO_MODEL_FAMILY=26`、`YOLO_MODEL_SIZE=s` | 仅日志，不影响计算 |
| 新增保护 | 无 | 增加 `FL_REQUIRE_YOLO_FAMILY=26` 启动断言：实际 family/size 不符直接报错 | 防止漏设 env 跑成 v8 |

### 3.3 建议的收敛方式

- **不复制一套 YOLO26 脚本**，只做参数化 + 默认值收紧；保留 `YOLO_MODEL_FAMILY=8` 作为历史复现开关。
- 所有 YOLO26 正式命令显式带 `YOLO_MODEL_FAMILY=26 YOLO_MODEL_SIZE=s`；入口断言二次兜底。
- 先跑 1 轮 smoke，再跑 5/15 轮基线；不要直接上 50 轮。

## §4 实验设计（分阶段 + 硬门槛）

### 固定因子

```
10 客户端 / IID / 恶意率 0.3 / FL_GRADIENT_SCALE=1
YOLO_MODEL_FAMILY=26 / YOLO_MODEL_SIZE=s
FL_IMG_SIZE=1280 / FL_AMP=1 / FL_TRAIN_BATCH_SIZE=16 / FL_TEST_BATCH_SIZE=64
LOCAL_LR=2e-4 / constant / seed 对齐 Arguments 默认
```

### Phase 0 — 代码准备与冒烟（不产生指标结论）

1. 完成 §3.2 改动；`py_compile` + `bash -n`。
2. 1 轮冒烟：加载 `yolo26s.pt`、换 10 类 E2E 头、1280/BS16/AMP 前向+反传+val。
3. 记录：实际参数量（换头后约 9.96M）、峰值显存、单轮耗时、TB 目录名是否含 `yolo26s`。
4. **G0**：无 OOM/NaN；参数量正确；日志明确 `yolo26s`；1 轮 mAP 正常（非 0 即可，不做精度结论）。

**2026-11-09 实测（Phase 0 完成）**：

- 基线 1 轮 × 1 local epoch：初始 mAP 0.01%，**R1 5.06%**，loss 284.06，5.1 min；无 OOM/NaN。
- MinMax + DPFLA 1 轮 × 1 local epoch：**R1 5.03%**，loss 284.68，5.8 min；攻击已注入（3/10），DPFLA Detect=layer 23，恶意端 aux=0.666 → 权重 0.020，诚实端 0.200–0.451 → 权重 1.000。
- 换头后参数量 **9,955,604**；1280/BS16/AMP 冒烟显存充足。
- 日志：`logs_3/visdrone/run_no_attack_baseline_20261109_1228.log`、`logs_3/visdrone/run_screen_minmax_DPFLA_20261109_1234.log`。
- ⚠️ 1 轮/1 epoch 只验证链路，**不是精度结论**；v26s 的 MinMax aux=0.666 低于 v8m（0.844/0.874），Phase 2 需监控 0.75 阈值。

### Phase 1 — YOLO26s 基线校准（决定后续口径）

| 步骤 | 配置 | 目的 | 门槛 |
|---|---|---|---|
| P1a | 5 轮 × 3 local epochs，无攻击 | 链路/收敛趋势冒烟 | mAP 随轮次上升，无 OOM |
| P1b | **15 轮 × 3 local epochs，无攻击** | 主验收基线 | **G1：≥32%**；**G1-pre：≥36.2%**（进入 Phase 2 的必要条件） |
| P1c | 若 P1b <36.2%，单变量救援 | 只动一个杠杆 | 见下方顺序 |
| P1d | 50 轮 × 12 step，无攻击（可选） | 判断 50 轮全量口径是否适合 YOLO26s | **已实测不适合**（R1 0.00 / R2 0.88，12 step 欠拟合）；改用 15×3 |

**P1c 单变量顺序（每次只改一项，失败再下一项）**：

1. `FL_LOCAL_LR` 2e-4 → 3e-4（再 5e-4）。
2. `FL_LOCAL_EPOCHS` 3 → 5（计算量允许时）。
3. `FL_IMG_SIZE` 1280 → 1024（仅当 1280 不达标或边缘时延要求；独立口径报告）。
4. `FL_TRAIN_BATCH_SIZE` 16 → 8 或 32（最后考虑）。

> **G1-pre 未过之前不要进入攻击矩阵**；不要改攻击/恶意率/模型族。

### Phase 2 — 15 轮攻击/防御筛选（G2）

- **前置**：Phase 1 的 G1-pre（无攻击基线 ≥36.2）已通过。
- 8 条：`{MinMax, IPM} × {FedAvg, DPFLA, MKrum, TMean}`，其余与 P1b 完全一致。
- **G2**：DPFLA ≥35%，且比 MKrum/TMean 都高 ≥1%（末5 为主口径）。
- 若 DPFLA 不达标，按顺序诊断：
  1. 确认服务端攻击已注入且 `local_models` 已同步。
  2. 打印 YOLO26s 的恶意端/诚实端 aux 分布、KMeans 标签、聚合权重，与 v8m 的 0.707/0.461 对照。
  3. 若分布分离不足，重标定 DPFLA 阈值（0.60/0.75/0.48）——一次只改一个，记录依据；不得改攻击强度或恶意率。
  4. 检查 DPFLA 特征键是否命中 `model.23.cv3.*.2.weight`（YOLO26 头索引 23）。

### Phase 3 — 50 轮全量复核（G3，Phase 2 通过后）

- VisDrone 9 条：基线 + `{MinMax, IPM} × {FedAvg, DPFLA, MKrum, TMean}`。
- 口径二选一，必须在报告中写明：
  - **P3-A**：50 轮 × 12 step（与 v8m 50 轮全量一致；若 P1d 证明达标）。
  - **P3-B**：50 轮 × 24/41 step 或 15 轮 × 3 epoch（若 12 step 对 YOLO26s 欠拟合；与 v8m 50 轮不可直接横比）。
- **G3**：R50 / 末5 满足 G1/G2；无 OOM/NaN；9 条日志归档。

### Phase 4 — CIFAR10 回归（G4）

- 本任务只改 VisDrone 模型/默认值，CIFAR 代码路径不动；做 2–3 条回归：
  - 基线、标签翻转 + DPFLA、MinMax + DPFLA（15 轮，沿用 11-01 口径）。
- **G4**：结果与 11-01 归档在 ±0.5 内，且 DPFLA ≥65%；若偏差大，先查代码改动是否误伤 CIFAR。

### Phase 5 — 分布式 YOLO26s 可用性（G5）

- VisDrone 10 客户端 × 50 轮，MinMax + DPFLA，`--model-family 26 --model-size s`，超时 18h。
- **G5**：每轮 10 条 update、server/client 退出 0、`rounds.jsonl` 完整、无 cache/文件写冲突；指标不作为结论。
- CIFAR 分布式已 PASS，且本任务不改分布式聚合路径，默认不重跑；若改了 `run_distributed_smoke.py` 才补 1 条 CIFAR 冒烟。

### Phase 6 — 归档与记忆回写

- `01_Facts.md` 新增 YOLO26s 章节（基线、四方法、双攻击，R50/末5/末10/best + 日志出处）。
- `00/03/08/09` 更新状态；`log/README.md` 增加 YOLO26s 索引。
- 完成后可再建 `archive/YYYY-MM-DD-yolo26s-complete` 快照。

## §5 资源与风险

### 资源

- 32GB GPU：YOLO26s 参数量/FLOPs 约为 v8m 的 39%/29%，1280/BS16/AMP 预计可跑；Phase 0 实测峰值显存。
- 单条耗时以 Phase 0 实测为准；按 FLOPs 比例预计明显低于 v8m。
- 磁盘：YOLO26s 权重 19.5MB；TB/日志体积小于 v8m。

### 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| E2E 损失/设备不兼容 | 训练报错/NaN | 已实测：用 `init_criterion()` + 递归移动 `one2many/one2one` + 设 `.device` |
| DPFLA 特征键写死 model.22 | 找不到特征/崩溃 | 动态探测 Detect 层索引；YOLO26 为 23，取 `model.23.cv3.*.2.weight` |
| YOLO26s 在 50×12 step 下欠拟合 | 基线 <32 | 改用 15×3 epoch 或 50×24/41 step；注明口径 |
| YOLO26s 基线达不到 36.2 | 攻击 + DPFLA 预计 <35 | 先加本地训练量/学习率/轮数；不要进入 Phase 2，不要削弱攻击 |
| YOLO26s 容量不足，DPFLA <35 | G2 失败 | 先确认攻击可见性，再重标定 DPFLA 阈值 |
| 权重下载失败 | 无法开始 | GitHub 直连慢；已用 `ghproxy.net` 下载；保留 `yolo26s.pt` 本地副本 |
| 代码默认值改动破坏 v8m 复现 | 历史不可复现 | 保留 `YOLO_MODEL_FAMILY=8 YOLO_MODEL_SIZE=m` 显式开关；提交前跑 1 轮 v8m 冒烟 |
| 1280 在边缘设备仍偏重 | 与课题背景冲突 | 主验收保持 1280；另设 1024 边缘口径作为附加实验，不与 1280 混比 |
| YOLO26 集成被证伪 | 需要备选 | 退回 YOLOv8s（保留 v8s 参数化路径与既有计划），但优先解决 E2E 集成 |

## §6 调研来源（外部，未在本项目复现）

- Ultralytics YOLO26 官方文档（2026-01-14 发布，AGPL-3.0 + Enterprise）：https://docs.ultralytics.com/models/yolo26/
- YOLO26 官方模型配置（s: 10,009,784 params, 22.8 GFLOPs）：https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26.yaml
- Ultralytics HF 仓库（`yolo26s.pt`）：https://huggingface.co/Ultralytics/YOLO26
- Aero-YOLO26（Sensors 2026, 26(19), 6346, Table 4/5；VisDrone YOLO26s vs YOLOv8s）：用户提供截图；论文题 *Aero-YOLO26: A Lightweight Detector for Small-Object Detection in UAV-Acquired Images with Collaborative Feature Optimization*
- YOLO26 边缘部署（Jetson，Ultralytics 8.4）：https://docs.ultralytics.com/guides/nvidia-jetson/
- 备选 YOLOv8s 参考：https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/v8/yolov8.yaml ；TNO 2025 VisDrone+ YOLOv8s 联邦：https://publications.tno.nl/publication/34645138/YSsjzRDW/spek-2025-federated.pdf

> ⚠️ 外部 VisDrone/COCO 数据均为**集中式/不同口径**，只用于判断“YOLO26s 精度上限与边缘可行性”，不能替代本项目联邦实测。

## §7 备选：YOLOv8s（仅在 YOLO26s 集成被证伪时启用）

- v8s 参数化路径与 YOLO26s 相同（`YOLO_MODEL_FAMILY=8`）；v8s 研究数据保留：
  - 官方 11.17M / 28.8 GFLOPs；VisDrone 公开 mAP50 40.0/23.7（Sensors 2026）。
  - 本项目历史 v8s@1024/BS32/AMP/15×3 R15=35.92（`01_Facts.md` §1b）。
- v8s 的 5 轮探针（2026-11-09 10:14–10:40）R1=19.50%、R2=26.18% 后因任务切换 YOLO26s 中止；**不完整，不进入结论**。

## §8 执行顺序总览

```
Phase 0 YOLO26s 代码参数化 + 1 轮冒烟
   ↓ G0
Phase 1 YOLO26s 基线（5 轮 → 15×3 轮）→ G1 ≥32 / G1-pre ≥36.2
   ↓
Phase 2 15 轮 MinMax/IPM × 四方法 → G2 DPFLA ≥35 且领先 ≥1
   ↓
Phase 3 50 轮全量（P3-A 12 step 或 P3-B 15×3/24-41 step）→ G3
   ↓
Phase 4 CIFAR 回归 → G4
   ↓
Phase 5 分布式 YOLO26s 可用性 → G5
   ↓
Phase 6 归档 + 回写 01_Facts / 00 / 03 / 08 / 09
```

**当前状态**：YOLO26s 已确认开源且本地可加载/换头/反传；**Phase 0 代码改造与端到端冒烟已完成（2026-11-09）**；下一步 Phase 1（5 轮 → 15 轮×3 local epochs 基线，预门槛 ≥36.2%）。
