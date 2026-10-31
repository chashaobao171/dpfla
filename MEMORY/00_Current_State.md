# 00_Current_State.md — 当前状态

<!-- 最后更新：2026-10-29 -->
<!-- ⚠️ 本文件会过期。与 01_Facts.md 冲突时，以 01_Facts.md 的实测数据为准 -->

## 一句话

**DPFLA**：联邦学习投毒防御项目，落地场景 **VisDrone + YOLOv8** 目标检测。

- 原目标（无攻击联邦基线 mAP@0.5 ≥ 32%）→ ✅ **已超额**：**v8m @1280 跑满 25 轮 → 峰值 47.64%（R13）、平台 ≈47%**（见 `01_Facts.md` §1c）。
- **当前阶段：进入研究本体——VisDrone 上的「攻击 vs 防御」实验。**

## 高层路线（2026-10-23 用户定稿 → 详见 `08_Roadmap.md`）

1. **P1 CIFAR10 有攻防 ≥65%**：✅ **已达标（2026-10-23 实测）** —— 标签翻转 **70.00%**、高斯 **67.99–69.50%**（ResNet18CIFAR / 15 轮 / 不放大）。
2. **P2 DPFLA 审计**：✅ 已确认 **SVD + KMeans 都在用，k=2（固定）**，详见 `08_Roadmap.md` §3。
3. **P3 对比 Multi-Krum / Trimmed-Mean**：✅ **CIFAR 已达标（2026-10-23，换 MinMax 攻击）** —— DPFLA 69.25%，末5 领先 MKrum **+37.90** / TMean **+13.43**，三口径 ≥2%；标签线 DPFLA 70.00%（末5 +2.11~2.20）。🟡 **VisDrone 进行中**：MinMax + DPFLA 5 轮 R5 **43.85%**（≥35 目标），待 TMean/MKrum/FedAvg 与 15 轮正式对比。
4. **P4 分布式改造**（未来，不在当前范围）：多终端各跑客户端；当前改动须遵守 `08_Roadmap.md` §5 的"反向约束"。

## 最新进度（2026-10-23）

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
| 下一篇 | 🚧 **50 轮全量实验（进行中，2026-10-31 09:48 起）**：CIFAR 9 条 `FL_LOCAL_STEPS=24`；VisDrone 9 条 `FL_LOCAL_STEPS=12`（50×12=600 step ≈ 历史 5 轮 15 epoch，单条约 1.8–2.0h）。自动队列 `scripts/full50_supervisor.py`，体检已通过（CIFAR R5 43.09 / VisDrone R5 33.01），测试产物已删除；结束后自动覆盖同类型 TB 并推 dev。 |
| dev 推送 | ✅ 见 `git log origin/dev -1`；存档与 zip 流程已入 `06_Git_Workflow.md` |

## 进度快照（2026-10-17）

| 项 | 状态 |
|---|---|
| VisDrone 基线 | ✅ **47.64%**（R13；R25 45.73%；v8m @1280 / BS=16 / AMP，**25 轮已跑满**；R1–R5 与旧 5 轮逐位一致）|
| VisDrone 攻击+无防御 | 🟢 **② 已达标**：高斯 **σ=0.02** 跑满 10 轮，**R10 32.37%**（基线同期 **47.22%**，**−14.85**），R4–R10 稳定在 27–32% → **可用攻击档**（已归档 `log/visdrone/02_...`）；标签翻转线仍为 −6 点稳态偏移 |
| VisDrone 攻击+DPFLA | 🟡 **探针已跑**（2026-10-23，5 轮：R1–R5 = 21.16/35.97/41.11/43.04/43.79，基线同期 21.21/37.05/41.44/43.82/45.08 → 平均 −0.71）；**15 轮完整 + MKrum/TMean 对比待跑** |
| MNIST / CIFAR10 | ✅ MNIST 三份齐全；**CIFAR10 已换 ResNet18CIFAR 新口径**（旧 CNN 产物已删，数值见 `01_Facts` §5b；新结果见 §5c）|
| 日志归档 | ✅ 新建 `log/`（只放合格日志；原始日志在 `logs_3/`）|

## 本阶段任务：VisDrone 攻击/防御三实验

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

## 已知未决 / 风险

- ⚠️ **攻击强度未定（当前最大未决项）**：标签翻转「×1 打不动、×15 崩」、高斯「σ=0.005 无效、σ=0.05 崩」——**两条线都还没找到「可控但有感」的档位** → 下一步按上方 ⏭ 待办先试**更强的高斯（σ=0.01/0.02）**。
- ⚠️ **归因未做**：47.64% 同时改了 模型(s→m)+分辨率(1024→1280)+batch(32→16)，无法单独归因。
- ⚠️ **47.64% 尚无外部参照**（未跑集中式 oracle @1280）。多项内部核验已通过（无泄漏、初始 0%、类别映射一致），详见 `01_Facts.md` §1c。
- ⚠️ `fl_core.py:199-202` 评估权重的类别名**第 5/6 位写反**（truck/tricycle 对调）——**不影响 mAP 数值**，仅影响按类读数，待修。
- 训练循环仍为手写版：缺 **EMA / warmup / 原生增强**（AMP 已有）。

## 铁律

1. **单变量**：一次只改一个变量（`03_History.md` / `04_Retracted.md` 教训）。
2. **不手搓**：优先用原生/现成实现；大改前先调研并出清单。
3. **落库**：实验结论先写 `01_Facts.md` 再推 dev；`logs_3/` 原始日志**也入库**（2026-10-16 用户要求，见 `06_Git_Workflow.md`）。
4. 硬约束见 `05_Environment.md` §冻结红线（DPFLA=SVD+KMeans；指标=YOLO 原生 val mAP@0.5；聚合 float32）。
5. **开关审计**：做防御测试时若同时开了 **DP 与高斯攻击**开关，必须先核实两者**不是同一段代码路径**（2026-10-20 用户要求）。
6. **恶意客户端比例固定 30%**（项目任务要求，2026-10-21 用户明确）：**不要再建议/询问调高恶意占比**（如 30%→50%），攻击强度只能从映射 / 放大 / 加噪 / 攻击范式等方向调。
7. **训练总轮数上限 = 15 轮**（2026-10-23 用户明确，原 25）；已归档的 25 轮基线仅作历史参照。
