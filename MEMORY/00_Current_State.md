# 00_Current_State.md — 当前状态

<!-- 最后更新：2026-11-09（回写 2026-11-01/02 实验结果） -->
<!-- ⚠️ 本文件会过期。与 01_Facts.md 冲突时，以 01_Facts.md 的实测数据为准 -->

## 一句话

**DPFLA**：联邦学习投毒防御项目，落地场景 **VisDrone + YOLOv8** 目标检测。

- 原目标（无攻击联邦基线 mAP@0.5 ≥ 32%）→ ✅ **已超额**：**v8m @1280 跑满 25 轮 → 峰值 47.64%（R13）、平台 ≈47%**（见 `01_Facts.md` §1c）。
- **当前阶段（2026-11-02）：50 轮全量非分布式矩阵 18/18 完成；分布式 DPFLA 可用性验证 CIFAR 50/50 PASS、VisDrone 42/50（超时中断，已把超时提到 18h）。**

## 高层路线（2026-10-23 用户定稿 → 详见 `08_Roadmap.md`）

1. **P1 CIFAR10 有攻防 ≥65%**：✅ **已达标（2026-10-23 实测）** —— 标签翻转 **70.00%**、高斯 **67.99–69.50%**（ResNet18CIFAR / 15 轮 / 不放大）。
2. **P2 DPFLA 审计**：✅ 已确认 **SVD + KMeans 都在用，k=2（固定）**，详见 `08_Roadmap.md` §3。
3. **P3 对比 Multi-Krum / Trimmed-Mean**：✅ **已达标** —— CIFAR 15 轮（MinMax：DPFLA 69.25%，末5 +37.90/+13.43）与 50 轮全量复核均通过；VisDrone 50 轮全量两条攻击线均满足“领先两防御 ≥1”（MinMax 38.36 / IPM 38.32）。50 轮数据见本页「最新进度（2026-11-02）」表与 `01_Facts.md` §8。
4. **P4 分布式改造**：✅ **可用性验证已完成** —— CIFAR 10 端 × 50 轮 PASS（last 69.98）；VisDrone 10 端 × 50 轮完成 **42/50** 后因 12h 超时中断（R42 server 36.53%，无 checkpoint），最新提交已把超时提到 18h，待复跑。分布式只做可用性验证，不用于 accuracy/mAP 对比。

## 最新进度（2026-11-02，50 轮全量）★

**非分布式 18/18 全部跑满 50 轮**（CIFAR 9 + VisDrone 9），主口径末 5 轮均值；完整数据见 `01_Facts.md` §8，归档索引见 `log/README.md`。

| 数据集 / 攻击 | 基线 | FedAvg | **DPFLA** | Multi-Krum | Trimmed-Mean |
|---|---:|---:|---:|---:|---:|
| CIFAR 标签翻转 M5（48 step） | 73.74 | 55.66 | **70.25** | 67.52 | 66.85 |
| CIFAR MinMax（24 step） | 71.24 | 56.58 | **68.12** | 17.77 | 49.51 |
| VisDrone MinMax（12 step） | 39.65 | 30.23 | **38.36** | 14.09 | 28.12 |
| VisDrone IPM（12 step） | 39.65 | 35.87 | **38.32** | 26.17 | 32.55 |

- 末 5 轮：CIFAR 标签线 DPFLA 领先 MKrum **+2.88**、TMean **+6.08**；MinMax 线 **+47.96 / +18.71**；VisDrone MinMax **+23.99 / +10.15**、IPM **+12.44 / +5.91**。
- ⚠️ **口径**：CIFAR 标签线为 **M5 全类循环 + 48 step/轮**、基线 73.74；MinMax 线为 **24 step/轮**、基线 71.24，两线不可直接横比。VisDrone 为 **12 step/轮**（50×12=600 step ≈ 历史 5 轮），基线 39.65 低于历史 5 轮 45.08 / 25 轮 47.64，**不可与 15/25 轮口径直接比较**。
- 分布式（只验证可用性，不作指标结论）：CIFAR 10 端 × 50 轮 **PASS**（last 69.98）；VisDrone **42/50** 因 12h orchestrator 超时中断（R42 server 36.53%），已把超时提到 18h，待复跑。

## 当前主任务（2026-11-09 起）：YOLO26s 迁移 ★

- **背景**：课题落地场景是**边缘终端**；YOLOv8m 在边缘设备上做联邦训练不现实，必须换成更轻的模型。用户 2026-11-09 明确：**优先 YOLO26s**（比 YOLOv8s 更小、算力更低、VisDrone 公开 mAP50 更高）；YOLOv8s 仅作为 YOLO26s 集成失败时的备选。
- **快照**：YOLOv8m 阶段成果已封存到只读分支 **`archive/2026-11-09-yolov8m-complete`**（`4f6866c`），不再修改。
- **开源核验（2026-11-09）**：YOLO26 于 2026-01-14 由 Ultralytics 发布，**AGPL-3.0 + Enterprise**；本地 `ultralytics 8.4.50` 已含 `yolo26.yaml`；`yolo26s.pt` 已下载到项目根（19.48MB）。【实测】80 类 10,009,784 参数；24 层、Detect 在 layer 23、`reg_max=1, end2end=True`；换 10 类头后 9,955,604 参数；1280/BS1 前向+反传通过（E2ELoss，GPU reserved ≈0.66GB）。
- **硬性要求**：VisDrone 检测实验使用 **YOLO26s**；CIFAR10 仍用 `ResNet18CIFAR`。
- **必须继续达标的目标**（沿用此前定稿）：
  1. VisDrone 无攻击联邦基线 mAP@0.5 **≥32%**。
  2. VisDrone 在 MinMax / IPM 两条攻击线下，DPFLA **≥35%**，且比 Multi-Krum / Trimmed-Mean **高 ≥1%**。
  3. CIFAR10 回归目标不变：有攻防 **≥65%**，DPFLA 比 Multi-Krum / Trimmed-Mean **高 ≥2%**。
- **主硬门槛**：VisDrone **攻击 + DPFLA ≥35%**。因 DPFLA 在攻击下通常比无攻击基线低约 **1.2 点**，Phase 1 的无攻击基线**预门槛设为 ≥36.2%**；达不到先调训练配方，不进入攻击矩阵。CIFAR 65% 当前有 3–5 点余量，不是瓶颈。
- **约束**：恶意率仍固定 30%、IID、不放大梯度、单变量；不复用 v8m/v8s 的数值结论，必须重跑 YOLO26s 自己的基线/攻击/防御矩阵。
- **计划**：见 `MEMORY/10_YOLO26s_Plan.md`；**Phase 0 已完成（2026-11-09）**——代码参数化 + 1280 端到端冒烟通过：基线 R1 5.06%、MinMax+DPFLA R1 5.03%（1 轮×1 epoch，仅链路验证），攻击/DPFLA/评估均无 KeyError/NaN/OOM。
- **Phase 2 MinMax 线（2026-11-09）**：**YOLO26s 11×3 + LR1e-3 + MinMax + DPFLA：R10=35.23%、R11=35.69% → 攻击 + DPFLA ≥35% 已达成**（基线 R10=36.02，仅低 0.79；恶意端权重 0.020，无 NaN/OOM）。日志/TB 已入库。
- **攻击规则变更（2026-11-13 用户定稿）**：CIFAR10 与 VisDrone **都必须支持两类攻击，标签翻转必须包含**；**VisDrone 舍弃 IPM**，两类攻击改为 **标签翻转 + MinMax**。20:33 用户再次更新：**停止基线，直接做部分标签翻转筛选**，每条上限 50 轮；先 5 轮筛策略，候选 10–11 轮，必要时 50 轮；确认达不到“DPFLA 比 MKrum/TMean 高 ≥1%”就停止换策略。22:42 再次更新：筛选顺序 **DPFLA → TMean → MKrum**；若 DPFLA 对 TMean 不达标（<TMean+1 或 <30）**跳过 MKrum 直接换下一个类数**；`FL_FLIP_RATIO` 固定 1.0，翻转类数 **1→2→3→4**；四种都不行才考虑替换 TMean。**one_class / two_class 均判定不通过（DPFLA 30.63 / 30.58 vs TMean 30.84 / 30.78，未领先 ≥1）→ 均跳过 MKrum。**
- **2026-11-14 06:25 配方变更（用户要求，便于展示）**：后续改用**小步多轮**：**20 轮 × 1 epoch、LR 8e-4**（训练量≈旧 5×3），目标 **R20 ≈ 旧 R5（≈30.6）**；判定点改为 R20，门槛不变。checkpoint 带 `_r20x1` 后缀；通过者续到 50 轮展示 ≥35 曲线。
- **2026-11-14 10:48 标签翻转筛选终止（用户指令）**：1→4 类全部试完，DPFLA R20 = 30.63 / 30.58 / 30.68 / 30.58；TMean = 30.84 / 30.78 / 30.74（four_class 仅 R1–R2 即中止）。**全部差距 −0.2 ~ −0.06，无 pattern 通过**。根因：TMean 的 `trim_ratio` 默认=恶意率 0.30，10 端每端裁 3，正好裁掉 3 个攻击者，结构上近似最优；调参/换类数无解。
- **下一步（用户定稿）**：**替换 TMean 基线**。候选：**FedAvg（无防御，已有 `run_no_defense_label_flipping.py`，推荐优先）**、simple median（`rule=median` 已支持，需补 VisDrone 标签翻转入口）。替换后同口径 20×1 测 DPFLA 领先是否 ≥1，再补 MKrum 凑齐三臂；通过后 DPFLA 续 50 轮冲 ≥35。
- **代码改造清单（已完成）**：`yolo_wrapper`（family/换头/E2ELoss/device）、`model.py`、`DPFLA`（动态 layer22/23）、`fl_core`（评估子进程/TB family/每轮 checkpoint）、`server`（全实验自动 checkpoint）、分布式 CLI、50 轮队列、VisDrone 入口默认值、标签翻转 pattern 参数化 + MKrum/TMean 入口、CIFAR flip_ratio 修复。历史 v8m/v8s 复现需显式 `YOLO_MODEL_FAMILY=8`。

## 历史进度（2026-10-23，15/5 轮口径）

| 项 | 状态 |
|---|---|
| CIFAR10 P1 | ✅ 标签 70.00 / 高斯 67.99–69.50（≥65）|
| CIFAR10 P3 | ✅ **MinMax 15 轮：DPFLA 69.25 vs MKrum 29.19 / TMean 57.01**（末5 +37.90/+13.43）；标签线 +3.00/+2.1~2.2；高斯线已不作为 +2 依据（04_Retracted R13）|
| DPFLA 算法 | ✅ 四处增强 + **协同重复检测**（相对距离 <0.01 → 权重 0.02）|
| 攻击实现 | ✅ `attack_alg/server_attacks.py`：MinMax / IPM / ALIE / SignFlipping（服务端注入），CIFAR + VisDrone 共用；`FL_IPM_SCALE` / `FL_ALIE_ZMAX` 可调 |
| VisDrone（MinMax） | ✅ DPFLA **43.85** / FedAvg **41.17** / TMean **40.17** / MKrum **29.30**（均 5 轮）→ DPFLA 领先 TMean +3.68、MKrum +14.55 |
| VisDrone（SignFlipping） | ⚠️ 四臂完成：DPFLA **43.37** / MKrum **42.91** / TMean 39.46 / FedAvg 35.05 → **仅领先 MKrum +0.46（<1，未达标）**，不能作为第二攻击 |
| VisDrone（IPM） | ✅ 四臂完成：DPFLA **43.93** / TMean **41.46** / MKrum **34.88** / FedAvg 42.85 → 领先 **+2.47 / +9.05**，达标 |
| ALIE（备选） | DPFLA 5 轮 43.87；探针 TMean 18.59 / MKrum 18.10（伤害仅约 2 点，力度偏低）|
| VisDrone 验收 | ✅ **MinMax 与 IPM 两条攻击线均满足：DPFLA 领先 TMean/MKrum ≥1%，且 DPFLA ≥35%** |
| 攻击组合 | CIFAR = 标签反转 + MinMax；**VisDrone = MinMax + IPM**；高斯、SignFlipping 已排除 |
| TB 对照 | VisDrone 共 10 个目录：9 条最终 + 基线前 5 轮切片 `..._20261021_1421_first5`（21.21/37.05/41.44/43.82/45.08）|
| 存档分支（只读） | ✅ `archive/2026-10-27-full`（`dba6888`，含 yolo 权重/cache/runs）/ `archive/2026-10-27-lean`（`61119a0`，去掉 FedAIoT-main、refs、死代码）—— 均已推 GitHub，**不再修改** |
| 本地 zip 备份 | ✅ `/root/chashaobao/data.zip`（2.2G，原始数据）/ `/root/chashaobao/DPFLA-master.zip`（366M，含 .git）|
| 下一篇 | ✅ **50 轮全量已完成（2026-11-02）**：非分布式 18/18、分布式 CIFAR 50/50 PASS、VisDrone 42/50（超时，待 18h 复跑）；见上方新表与 `01_Facts.md` §8/§9。本表其余行为 2026-10-23 历史快照。 |
| dev 推送 | ✅ 见 `git log origin/dev -1`；存档与 zip 流程已入 `06_Git_Workflow.md` |

## 进度快照（2026-10-17）

| 项 | 状态 |
|---|---|
| VisDrone 基线 | ✅ **47.64%**（R13；R25 45.73%；v8m @1280 / BS=16 / AMP，**25 轮已跑满**；R1–R5 与旧 5 轮逐位一致）|
| VisDrone 攻击+无防御 | 🟢 **② 已达标**：高斯 **σ=0.02** 跑满 10 轮，**R10 32.37%**（基线同期 **47.22%**，**−14.85**），R4–R10 稳定在 27–32% → **可用攻击档**（已归档 `log/visdrone/02_...`）；标签翻转线仍为 −6 点稳态偏移 |
| VisDrone 攻击+DPFLA | ✅ **5 轮探针已完成**（2026-10-23，5 轮：R1–R5 = 21.16/35.97/41.11/43.04/43.79，基线同期 21.21/37.05/41.44/43.82/45.08 → 平均 −0.71）；后续 50 轮全量见上方新表 |
| MNIST / CIFAR10 | ✅ MNIST 三份齐全；**CIFAR10 已换 ResNet18CIFAR 新口径**（旧 CNN 产物已删，数值见 `01_Facts` §5b；新结果见 §5c）|
| 日志归档 | ✅ 新建 `log/`（只放合格日志；原始日志在 `logs_3/`）|

## 本阶段任务：VisDrone 攻击/防御三实验

> 以下为 2026-10-23 的 5/15 轮口径命令，保留作复现参考；50 轮全量入口与结果见 `09_Full_Run_Plan.md` 与上方新表。

**统一口径（必须与基线一致，否则不可比）**：`v8m` + `imgsz=1280` + `BS=16` + `AMP` + 10 客户端 + **攻击率 30%（=3 个恶意客户端）** + IID。

```bash
cd /root/chashaobao/DPFLA-master      # 确认在 dev 分支

COMMON="YOLO_MODEL_SIZE=m FL_IMG_SIZE=1280 FL_TRAIN_BATCH_SIZE=16 FL_AMP=1 FL_TEST_BATCH_SIZE=64 FL_DATALOADER_WORKERS=4"

# ① 基线（已完成，如需复现）
./scripts/run_fl_tmux.sh run_no_attack_baseline       $COMMON

# ② 有攻击 + 无防御（FedAvg）—— 预期 mAP 明显下降
./scripts/run_fl_tmux.sh run_no_defense_label_flipping $COMMON FL_MALICIOUS_RATE=0.3

# ③ 有攻击 + DPFLA 防御 —— 预期接近基线
./scripts/run_fl_tmux.sh run_dpfla_label_flipping      $COMMON FL_MALICIOUS_RATE=0.3
```

- 观测：`tmux attach -t fl`（Ctrl-b d 脱离）／`tail -f logs_3/visdrone/<新日志>`
- 攻击设置：**多标签反转**（映射随实验递进，②③ 同用同一映射；当前为全类循环 `{i:(i+1)%10}`）；**行为率 100%**；30% 恶意客户端（3 个）
- ⚠️ **攻击强度主杠杆 = `FL_GRADIENT_SCALE`**（[client.py:522-543](file:///root/chashaobao/DPFLA-master/federated_learning/client.py#L522-L543)，**代码默认 15**：上传前把恶意客户端的权重 delta 放大 15×；**代码默认值不要改**，用环境变量覆盖）
- `FL_FLIP_RATIO` 只控制投毒**数量**（默认 1.0 = 全翻源类；<1 只翻一部分）
  - 2026-10-14 实测：默认 ×15 → **R1 即崩**（mAP 0.00%、测试损失 9412/10866，远超未训练的 902）；且把 `flip_ratio` 从 1.0 降到 0.1 **完全无效** → 见 `04_Retracted.md` R10 / R11
  - **实验②已跑结果**（详见 `01_Facts.md` §1d）：×15 → R1 崩到 0%（损失 9412/10866）；×1 单标签 → R1 19.88%、R4 43.24%（基线 43.82%）几乎无效；**多标签递进（覆盖 34.3%→83.5%→100%）→ R1 17.53→16.47→15.09，且全类循环仅 R2 下探 27.23 后 R3 回弹 35.45 → 纯映射已饱和**
  - ✅ **高斯线已定档（2026-10-21）**：σ=0.005 无效、σ=0.05 崩、**σ=0.02 可用**（10 轮 R10 32.37%，基线同期 47.22%，−14.85）。→ **下一步：用该口径跑 ③ DPFLA 防御**（`run_dpfla_*` 同映射/同 σ，验证防御能否拉回接近基线）。口径固定：v8m@1280 / **恶意率 30%（任务硬要求不可调）** / 行为率 100% / 不放大。
  - 标签翻转线若继续，「仅改映射」已满（覆盖面 100% / 行为率 1.0 / `FL_FLIP_RATIO` 1.0），需换 `FL_GRADIENT_SCALE=2~3`（×1 降 6 点 / ×15 崩，跨度大）。
- **产出后**：达标即复制进 `log/visdrone/02_...` 与 `03_...`，并更新 `log/README.md` 索引

## 关键上下文（新会话必读）

1. **环境**：两台 autodl 主机交替用（新机 4090 48G / 20 核 / **90G 内存**；老机 4080S 32G / 12 核 / **62G 内存**）。**跑 GPU 必须非沙箱**。详见 `05_Environment.md` §1。
2. **路径**：VisDrone 路径**唯一来源是 `server.py:98-100`**（恒为 `/root/autodl-tmp/data/visdrone`），与 `data/images|labels` 无关。
3. **代码基线口径**：FedAvgM / 梯度裁剪均 env 门控、**默认关**；`FL_IMG_SIZE` 默认 640。
4. **已完成里程碑**：32% → 35.92%（1024/v8s，R15）→ **47.64%**（1280/v8m，R13 峰值；25 轮平台 ≈47%）。两次都**非单变量**，若要写进论文需补归因臂。
5. **"换聚合模型"已证伪**（`04_Retracted.md` R1）；**轮数增加已排除**（曲线已收敛）。

## 已知未决 / 风险（2026-11-02）

- ⚠️ **VisDrone 分布式 50 轮待复跑**：42/50 因 12h orchestrator 超时中断，R42 server mAP=36.53%，无 `result.json` / checkpoint，不能续跑；最新提交 `0a78a7e` 已把超时提到 18h，需从头重跑。
- ⚠️ **VisDrone 50 轮口径不可与历史直接横比**：12 step/轮（50×12=600 step ≈ 历史 5 轮），基线 39.65 低于历史 5 轮 45.08 / 25 轮 47.64；若要写进论文，需补“同训练量”对照臂或明确注明口径。
- ⚠️ **CIFAR 两条攻击线基线不同**：标签线为 M5 + 48 step/轮（基线 73.74），MinMax 线为 24 step/轮（基线 71.24），跨线比较必须先对齐口径。
- ⚠️ **47.64% 归因仍未做**：历史 25 轮基线同时改了模型(s→m)+分辨率(1024→1280)+batch(32→16)，无法单独归因；且无集中式 oracle @1280 外部参照。
- ⚠️ [fl_core.py](file:///root/chashaobao/DPFLA-master/federated_learning/fl_core.py#L199-L202) 评估权重的类别名第 5/6 位写反（truck/tricycle 对调）——不影响 mAP 数值，仅影响按类读数，待修。
- 训练循环仍为手写版：缺 EMA / warmup / 原生增强（AMP 已有，默认关）。

## 铁律

1. **单变量**：一次只改一个变量（`03_History.md` / `04_Retracted.md` 教训）。
2. **不手搓**：优先用原生/现成实现；大改前先调研并出清单。
3. **落库**：实验结论先写 `01_Facts.md` 再推 dev；`logs_3/` 原始日志**也入库**（2026-10-16 用户要求，见 `06_Git_Workflow.md`）。
4. 硬约束见 `05_Environment.md` §冻结红线（DPFLA=SVD+KMeans；指标=YOLO 原生 val mAP@0.5；聚合 float32）。
5. **开关审计**：做防御测试时若同时开了 **DP 与高斯攻击**开关，必须先核实两者**不是同一段代码路径**（2026-10-20 用户要求）。
6. **恶意客户端比例固定 30%**（项目任务要求，2026-10-21 用户明确）：**不要再建议/询问调高恶意占比**（如 30%→50%），攻击强度只能从映射 / 放大 / 加噪 / 攻击范式等方向调。
7. **训练总轮数上限 = 15 轮**（2026-10-23 用户明确，原 25）；已归档的 25 轮基线仅作历史参照。
