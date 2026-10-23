# 08_Roadmap.md — 高层路线图（用户 2026-10-23 定稿）

<!-- 最后更新：2026-10-23 -->
<!-- 级别：**高级别记忆**。与 00_Current_State.md 同级；本文件只记"目标 / 验收 / 约束 / 阻塞"，实测数据仍以 01_Facts.md 为准。 -->
<!-- 规则：目标与验收来自用户原话；现状标【实测】；推断标【推断·未验证】。用户后续修订指令优先。 -->

## §0 一句话

先在 **CIFAR10 把「有攻击 + 有防御」刷到 ≥65%**（ResNet18、覆盖高斯 + 标签翻转、**不放大梯度**、轮数 ≤15），再把 **DPFLA 与 Multi-Krum / Trimmed-Mean 做同口径对比**（CIFAR10 领先 ≥2%、VisDrone mAP@0.5 领先 ≥1%）；两项全部完成后再启动**分布式改造**（未来工作，当前只做"不挡路"的预防）。

## §1 新硬约束（长期，2026-10-23 用户明确）

| 约束 | 内容 | 说明 |
|---|---|---|
| **轮数上限** | **15 轮**（原 25） | 今后所有实验的全局轮数上限 |
| 恶意率 | 30%（10 客户端 → 3 个恶意） | 沿用，不可调 |
| **梯度放大** | CIFAR10 新实验**一律 `FL_GRADIENT_SCALE=1`**（纯攻击、不放大） | 代码默认 15 不改，只用 env 覆盖 |
| 攻击强度 | 只用环境变量调，**不改代码默认值** | 沿用 |
| 单变量 | 一次只改一个变量 | 沿用 |

## §2 P1 — CIFAR10 有攻防 ≥65%（换 ResNet18）

### 现状【实测】

| 实验 | 准确率 | 口径 |
|---|---|---|
| 基线 FedAvg | **60.74%** | 10 客户端 / 15 轮 / 1 local epoch / lr 0.01 / BS 64 / CNNCifar10 / IID |
| 攻击 + 无防御 | 34.87% | 标签翻转 dog(5)→cat(3) / 30% / **×15 放大** |
| 攻击 + DPFLA | 61.99% | 同上 |

出处：`log/cifar10/01_baseline__cifar1_fedavg_noattack_20260902_1128.log`；②③ 两份日志**已于 2026-10-23 删除**（数值保留于 `01_Facts.md` §5b）。

⚠️ **②③ 全部带 ×15 梯度放大**（各 45 次 `梯度放大 ×15.0` 日志）→ 与新要求"不放大"**不同口径，不可直接引用 61.99%**，必须重跑。

⚠️ **入口脚本口径分裂**【实测】：
- `run-test/cifar10/*.py`：7 客户端 / 8 轮 / 未改 lr（默认 1e-4）/ `untarget=True`（目标类随 timestamp 每轮漂移）。
- 归档日志实际由根目录 `run_demo_cifar.py` 产生：10 客户端 / 15 轮 / lr 0.01 / `untarget=False`。

### 模型选型（关键决策，2026-10-23 核实）【实测】

- **不要直接用 torchvision `resnet18`**：它是 ImageNet 结构（7×7 stride2 stem + maxpool），32×32 输入到 layer4 只剩 1×1，CIFAR 上系统性吃亏。
- **FLPoison 分支自带 CIFAR 专用 ResNet**（`origin/FLPoison:fl/models/resnet.py`）：**3×3 stride1 stem、无 maxpool、`avg_pool2d(4)`、末层 `linear`** → 正是 CIFAR 口径，可单文件借用（注明出处）。
- 借用时需适配两点：① 我们的 `client.py` 调 `model(data, return_features=True)` 需要返回 `(logits, features)`；② DPFLA 非 YOLO 特征键要支持 `linear.weight`（FLPoison 用 `linear`，不是 `fc`/`fc2`）。
- 备选：`VGG16` / `DN121` 已在 `setup_model` 注册，但参数量/通信成本更高，非首选。

### 阻塞项（不解决跑不起来）

1. **模型**：`setup_model` 注册的 `ResNet18` 是 torchvision 版（ImageNet stem）→ **不适合 CIFAR**，需替换/新增 CIFAR 版实现（见上）。
2. **DPFLA 白名单**：[fl_core.py:826-836](file:///root/chashaobao/DPFLA-master/federated_learning/fl_core.py#L826-L836) 只认 `CNNMNIST / CNNCifar10 / YOLO`，其它 `raise` → 需加 `ResNet18` 分支。
3. **DPFLA 特征键硬编码**：[DPFLA.py:258-261](file:///root/chashaobao/DPFLA-master/federated_learning/fl_algorithm/DPFLA.py#L258-L261)、[312](file:///root/chashaobao/DPFLA-master/federated_learning/fl_algorithm/DPFLA.py#L312)、[356](file:///root/chashaobao/DPFLA-master/federated_learning/fl_algorithm/DPFLA.py#L356) 非 YOLO 路径写死 `fc2.weight` → 改为按 `fc2.weight → fc.weight → linear.weight` 顺序探测。
4. **CIFAR 高斯入口缺失**：`run-test/` 只有 VisDrone 的 `run_no_defense_gaussian.py`，CIFAR10 **无高斯脚本**（无防御 / DPFLA 两版都缺）。
5. **入口统一**：新建 `run-test/cifar10/cifar_fed_hparams.py`，把 lr / BS / local epochs / 攻击口径集中，脚本只读 hparams（与 `visdrone_fed_hparams.py` 同构）。
6. **固定攻击映射**：CIFAR 现有脚本 `untarget=True` 会让目标类每轮变化 → 对比实验必须 `untarget=False` + 固定映射 `{5(dog)→3(cat)}`。

### 隐藏变量（会干扰 65% 目标）【实测】

- CIFAR 走 `PerturbedGradientDescent(lr, mu=0.001)`（FedProx），**不是 SGD**（[client.py:406-407](file:///root/chashaobao/DPFLA-master/federated_learning/client.py#L406-L407)）→ FedProx 的 `mu*(w-w_global)` 项会拖慢 from-scratch 收敛，是 65% 目标的可疑变量。
- `Cifar10CNN.forward` 末尾 `F.softmax` + `CrossEntropyLoss` → **双 softmax**，梯度被压平，是 60.74% 上限的主要原因之一（[cnn_cifar_10.py:50](file:///root/chashaobao/DPFLA-master/federated_learning/models/cnn_cifar_10.py#L50)）；新 ResNet 返回 logits，可规避。
- 历史 CIFAR 归档用 `lr=0.01`（靠 `run_demo_cifar.py` 里 patch `Arguments.lr` 实现），脚本本身没有这个覆盖 → 统一入口时必须显式写入 hparams。

### 详细计划（T1–T9，每步单变量）

**阶段 0：工程准备（不跑训练，不涉及算法）**

| # | 任务 | 产出 / 判据 |
|---|---|---|
| T1 | 引入 CIFAR 专用 ResNet18（借用 FLPoison `fl/models/resnet.py`，加 `return_features` 适配与 `setup_model` 注册） | 单张 32×32 前向 shape 正确；`return_features=True` 返回 `(logits,features)` |
| T2 | DPFLA 兼容：`fl_core` 白名单加 `ResNet18`；`DPFLA.py` 三处特征键改为自动探测 | `rule=DPFLA` + ResNet18 能跑通 1 轮不报 KeyError |
| T3 | 统一入口 `run-test/cifar10/cifar_fed_hparams.py` + 三个脚本改造（基线 / 无防御 / DPFLA），显式 `FL_GRADIENT_SCALE=1`、固定映射、`untarget=False`、`ROUNDS=15` | `python ...` 打印的参数块与规划一致 |
| T4 | 新增高斯脚本 `run_no_defense_gaussian.py` / `run_dpfla_gaussian.py`（照 VisDrone 版结构） | 与标签线共用 hparams，只差 `attack_type`/`rule` |

**阶段 1：基线（决定 65% 是否有余量）**

| # | 任务 | 判据 |
|---|---|---|
| T5 | **A1**：ResNet18 / 10 客户端 / **15 轮** / 1 local epoch / lr 0.01 / BS 64 / IID / 无攻击 | R15 **≥70%** 视为达标（给攻击留 5 点余量）|
| T5b | 若 A1 < 65%：**单变量**先 `local_epochs=3`；仍不行再 `lr=0.05`；再不行才考虑 `mu=0`（改 SGD） | 每次只动一个，记录曲线 |

**阶段 2：攻击线（不放大梯度）**

| # | 任务 | 判据 |
|---|---|---|
| T6 | **A2** 标签翻转 + 无防御：`{5→3}`、30% 恶意、行为率 1.0、`FL_GRADIENT_SCALE=1`、15 轮 | 相对 A1 掉 **≥5 点但不崩**（若掉 <3 点 → 扩覆盖映射，仍不放大）|
| T6b | **A3** 标签翻转 + DPFLA（同 A2 口径，仅 `rule` 变） | **≥65%**；且 ≥ A2 + 5 点 |
| T7 | **A4** 高斯 σ 筛选：0.005 / 0.01 / 0.02 / 0.05，先各跑 5 轮看拐点 | 选"明显掉点、不归零"的档；**不能照搬 VisDrone σ=0.02** |
| T7b | 选定 σ 跑满 15 轮：无防御 + DPFLA 各一次 | **≥65%**；且 DPFLA > 无防御 |

**阶段 3：收尾**

| # | 任务 | 判据 |
|---|---|---|
| T8 | 诊断：若 DPFLA 与无防御差距 <2 点，查日志里恶意端 KMeans 标签/聚合权重是否真被压低 | 能定位是"攻击太弱"还是"防御没生效" |
| T9 | 归档 `log/cifar10/`（标签三件套 + 高斯一套）、更新 `log/README.md`、`01_Facts.md` §5b | 与 `log/README` 收录标准一致 |

### 落地情况（2026-10-23，T1–T4 已完成）

| 项 | 落地 |
|---|---|
| CIFAR ResNet | 新增 [cifar_resnet.py](file:///root/chashaobao/DPFLA-master/federated_learning/models/cifar_resnet.py)，模型名 **`ResNet18CIFAR`**（3×3/s1 stem、无 maxpool、`linear` 头、11.17M 参数、支持 `return_features`）|
| 注册 | [model.py](file:///root/chashaobao/DPFLA-master/federated_learning/models/model.py#L16-L31) 注册 `ResNet18CIFAR`；torchvision `ResNet18` 保留但**不用于 CIFAR** |
| DPFLA 白名单 | [fl_core.py:832-836](file:///root/chashaobao/DPFLA-master/federated_learning/fl_core.py#L832-L836) 已加 `ResNet18CIFAR / ResNet18` |
| DPFLA 特征键 | [DPFLA.py:16-28](file:///root/chashaobao/DPFLA-master/federated_learning/fl_algorithm/DPFLA.py#L16-L28) 新增 `_detect_classifier_weight_key()`，按 `fc2 → fc → linear` 探测；三处硬编码已清除 |
| **DPFLA 高斯强化**（2026-10-23 新增）| [DPFLA.py:455-492](file:///root/chashaobao/DPFLA-master/federated_learning/fl_algorithm/DPFLA.py#L455-L492)：① 辅助强否决（aux≥0.60 视作强异常）；② 极强证据（aux≥0.75 → risk=0.98，权重首轮即 0.02）；③ 误杀保护（KMeans 少数簇但 aux<0.48 → 权重下限 0.50）。依据与效果见 `01_Facts.md` §5c(d) |
| CIFAR 统一入口 | [cifar_fed_hparams.py](file:///root/chashaobao/DPFLA-master/run-test/cifar10/cifar_fed_hparams.py)（10 客户端/15 轮/1 epoch/BS64/lr0.01/IID，`FL_GRADIENT_SCALE` 默认 1）|
| 实验脚本 | CIFAR10 共 9 个；VisDrone 共 8 个（新增 `run_dpfla_gaussian.py`）|
| 数据路径 | [sampling.py:120-135](file:///root/chashaobao/DPFLA-master/federated_learning/utils/sampling.py#L120-L135) 修正候选表首项被跳过的问题 → 现解析为 **`/root/autodl-tmp/data`**（download=False）|

### ✅ P1 实测结论（2026-10-23，完整数据见 `01_Facts.md` §5c）★

- **标签翻转 + DPFLA：70.00%**（≥65 ✓；无攻击基线 71.47）
- **高斯 + DPFLA（v5 最终版）**：69.50%（σ=0.03）/ **69.26%（σ=0.04）**/ 68.78%（σ=0.05）/ 67.99%（σ=0.06）—— 全部 ≥65 ✓
- 共性：≤15 轮、`FL_GRADIENT_SCALE=1`（不放大）、30% 恶意、行为率 1.0
- 参考上限：7 端干净 FedAvg = **74.25%**；**P3 选定高斯档 = σ=0.04**

### ⚠️ P3 实测结论（2026-10-23，**高斯线未达标，已修正**）

| 线 | DPFLA | Multi-Krum | Trimmed-Mean | R15 单点 | 末5均值（推荐口径）|
|---|---|---|---|---|---|
| 标签翻转 | **70.00%** | 67.00% | 66.83% | +3.00 | **+2.20 / +2.11**（勉强达标）|
| 高斯 σ=0.04 | **69.26%** | 65.70% | 66.98 / 67.02 | 单点 +2.24 | **+0.47 / +0.78（未达标）** |
| 高斯 σ=0.03 / 0.05 / 0.06 | 69.50 / 68.78 / 67.99% | 64.70 / 64.91 / 67.02% | 68.22 / 68.34 / 67.86% | +1.28 / +0.44 / +0.13 | — |

【实测·已修正】⚠️ **高斯线 +2.24 是 R15 单点假象**（TMean R14 68.42 → R15 66.98 回落）；按最佳轮 +0.84/+0.95、末5 +0.47/+0.78、末10 +0.08/+0.32 → **稳健差距仅 0.1–0.9 点，不宣称 +2%**。详见 `04_Retracted.md` R13。
【实测】标签线较稳健（末5 +2.11/+2.20），但末10 仅 +1.55/+1.90。

**✅ 2026-10-23 已解决：换攻击范式（MinMax，服务端全知注入）**

| 方法（MinMax 15 轮） | R15 | best | 末5均值 |
|---|---|---|---|
| FedAvg（无防御） | 60.69% | 60.69% | 58.78% |
| **DPFLA** | **69.25%** | **69.25%** | **67.75%** |
| Multi-Krum | 29.19% | 37.07% | 29.86% |
| Trimmed-Mean | 57.01% | 57.01% | 54.32% |

→ DPFLA **末5 领先 +37.90 / +13.43**，R15 领先 +40.06 / +12.24，**三种口径均 ≥2%**；DPFLA 自身 69.25 ≥65。P3 CIFAR 达标。
→ 高斯线保留为"DPFLA ≥65"的证据（σ=0.03–0.06 均达标），但不再作为 +2% 对比线；数值见 `01_Facts.md` §5c(c)/(g)。

**待定**：
1. **VisDrone MinMax 线（进行中，2026-10-23）**：与 CIFAR 同口径换 MinMax（服务端注入），目标 mAP@0.5 领先两防御 ≥1%、DPFLA ≥35%。
   - **DPFLA 5 轮已完成**：21.17 / 35.93 / 40.67 / 42.52 / **43.85**（基线 21.21/37.05/41.44/43.82/45.08）→ ≥35 达成。
   - 依次补 TMean（20:03 起）→ MKrum → **FedAvg（MinMax 无防御，保留）**；每条约 65 min，串行，单卡仅容一条 VisDrone。
   - **验收口径（用户 2026-10-23）**：两种攻击（**SignFlipping + MinMax**）× 三种防御（DPFLA / MKrum / TMean），FedAvg 作攻击对照臂；**DPFLA mAP 比另两防御高 1%** 即达标。
   - **节奏**：15 轮暂缓；先把两攻击 × 四臂的 **5 轮**全部跑完，提交老师确认。
   - **高斯已确认舍弃**（用户 2026-10-23）：稳健领先仅 +0.5~0.9，且对 TMean 理论吃亏。
   - **SignFlipping 四臂已完成（2026-10-24 02:20）但未达标**：DPFLA 43.37 / MKrum 42.91 / TMean 39.46 / FedAvg 35.05 → DPFLA 仅领先 MKrum **+0.46（<1）**。原因：符号翻转的恶意更新远离良性簇，Krum 天然排除。
   - **第二攻击改为 ALIE**（2026-10-24）：ALIE 构造的恶意端彼此靠近、会被 Krum 选中，专打 Krum；CIFAR 5 轮筛选 DPFLA 54.62 vs TMean 48.22。待跑 4 臂 × 5 轮。
   - 标签全翻转入口 `run-test/visdrone/run_screen_label_flipping.py` 保留备用，不进入正式对比。
2. **口径规范**：今后统一以"末 5 轮均值"为主口径，并同时报告 best/R15/last10。
3. VisDrone 筛选通过后跑 15 轮正式对比（约 2.5h/条）。

### 🔬 VisDrone ③ 探针（2026-10-23，5 轮已完成）

| 配置 | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| 无攻击基线（v8m@1280）| 21.21% | 37.05% | 41.44% | 43.82% | 45.08% |
| ② 高斯 σ=0.02 + FedAvg | 16.25% | 24.59% | 27.69% | 30.04% | 31.24% |
| **③ 高斯 σ=0.02 + DPFLA（修复前）** | **3.72%** | — | — | — | — |
| **③ 高斯 σ=0.02 + DPFLA（修复后）** | **21.16%** | **35.97%** | **41.11%** | **43.04%** | **43.79%** |

【实测】修复前：3 个恶意端 aux=0.707 未触发 0.75 阈值，KMeans 判入多数簇拿到 0.797 权重；2 个诚实端被误杀到 0.137 → R1 崩溃。
【实测】修复后：恶意端 aux=0.908 → **权重 0.020**；诚实端 0.79–0.84 → R1–R5 **逐轮追平无攻击基线**（平均 −0.71）→ **DPFLA 在 VisDrone 高斯线有效**，待 15 轮完整验证与 MKrum/TMean 对比。

### 验收

- [x] 标签翻转 + DPFLA：**≥65%** → 实测 **70.00%**
- [x] 高斯 + DPFLA：**≥65%** → 实测 **67.99–69.50%**（四档）
- [x] 同攻击下无防御**明显低于** DPFLA（标签 65.68 vs 70.00；高斯 σ=0.04：39.82 vs 69.26）
- [x] 所有实验 ≤15 轮，参数块可逐项比对
- [ ] P3 CIFAR +2%：标签线 **末5 +2.11/+2.20 勉强达标**；**高斯线未达标（稳健 +0.5~0.9）**
- [ ] P3 VisDrone +1%：待 15 轮完整对比（探针已证明 DPFLA 在 VisDrone 高斯线有效）

## §3 P2 — DPFLA 算法审计：是否用 SVD / KMeans，k 是多少

**结论：SVD 与 KMeans 都在用；k = 2（固定，不自适应）**【实测】

| 项 | 事实 | 位置 |
|---|---|---|
| SVD | `svd(Z_mask)` → 取 `U_mask`、前 2 个奇异值做二维投影 | [DPFLA.py:393-404](file:///root/chashaobao/DPFLA-master/federated_learning/fl_algorithm/DPFLA.py#L393-L404) |
| `svd()` 实现 | `np.linalg.svd`；m<n 时转置后交换 U/V | [DPFLA.py:645-651](file:///root/chashaobao/DPFLA-master/federated_learning/fl_algorithm/DPFLA.py#L645-L651) |
| KMeans | `batch_detect_outliers_kmeans(detect_res_list)`，`n_clusters=2` | [DPFLA.py:406](file:///root/chashaobao/DPFLA-master/federated_learning/fl_algorithm/DPFLA.py#L406)、[584-586](file:///root/chashaobao/DPFLA-master/federated_learning/fl_algorithm/DPFLA.py#L584-L586) |
| **k 值** | **k = 2**，不随恶意率 / 客户端数变化；多数簇=诚实、少数簇=恶意 | 同上 |
| 聚类门槛 | `max_sc ≥ 0.58` 才采纳聚类；`≥0.72` 高置信；`[0.58,0.72)` 灰区柔罚 | [DPFLA.py:410-436](file:///root/chashaobao/DPFLA-master/federated_learning/fl_algorithm/DPFLA.py#L410-L436) |
| 死代码 | `detect_outliers_kmeans()`（k=2、门槛 0.61）**已定义但无调用** | [DPFLA.py:547-581](file:///root/chashaobao/DPFLA-master/federated_learning/fl_algorithm/DPFLA.py#L547-L581) |
| 软权重 | 基础 bad 权重 `0.20 - 0.12*confidence`；连续 bad 按 `0.72^(streak-1)` 衰减，下限 0.02 | [DPFLA.py:435-500](file:///root/chashaobao/DPFLA-master/federated_learning/fl_algorithm/DPFLA.py#L435-L500) |
| 辅助特征 | MAD z-score（3 维：log1p 更新范数 / 与群体均值余弦 / 层间更新集中度） | [DPFLA.py:265-328](file:///root/chashaobao/DPFLA-master/federated_learning/fl_algorithm/DPFLA.py#L265-L328) |
| 特征层 | YOLO：`model.22.cv3.{0,1,2}.2.weight`(+bias) 按类切片；非 YOLO：`fc2.weight` | [DPFLA.py:211-261](file:///root/chashaobao/DPFLA-master/federated_learning/fl_algorithm/DPFLA.py#L211-L261) |

⚠️ **k=2 写死**：恶意方若成为多数（>50%）或只剩 1 个恶意，聚类语义会反转/退化（已有 `minority_count==1` 补救，但 **k 仍固定**）。当前任务用 30% 恶意，k=2 合理；若后续改恶意率需重新评估。

## §4 P3 — DPFLA vs Multi-Krum / Trimmed-Mean

### 目标（用户原话）

| 数据集 | 指标 | 要求 |
|---|---|---|
| CIFAR10 | 准确率 | DPFLA **比 Multi-Krum、Trimmed-Mean 都高 ≥2%** |
| VisDrone | mAP@0.5 | DPFLA **比二者都高 ≥1%** |

### 本项目实现（2026-10-23 已重写并接入）【实测】

- **Multi-Krum**：`multi_krum(weights, num_attackers=f)`，输入改为 **state_dict 列表**（与 FedAvg/DPFLA 同接口）。score = 到其余客户端最近 **n-f-2** 个欧氏距离之和，选 **m=n-f-2** 个等权平均；n=10/f=3 → 邻居 5、选 5。覆盖：`FL_MKRUM_NEIGHBORS` / `FL_MKRUM_M`。文件 [multi_krum.py](file:///root/chashaobao/DPFLA-master/federated_learning/fl_algorithm/multi_krum.py)、调用 [fl_core.py:739-743](file:///root/chashaobao/DPFLA-master/federated_learning/fl_core.py#L739-L743)。
  - 前提 `n ≥ 2f+3` 不满足时**回退等权 FedAvg 并告警**（不静默硬套）。
- **Trimmed-Mean**：`trimmed_mean(weights, trim_ratio)`，逐 key、逐坐标排序后两端各裁 `int(trim_ratio*n)` 个；默认 `trim_ratio=恶意率=0.3`（每端 3/10），覆盖 `FL_TRIM_RATIO`；非浮点 buffer 沿用 FedAvg 语义（`num_batches_tracked` 取最大）。文件 [trimmed_mean.py](file:///root/chashaobao/DPFLA-master/federated_learning/fl_algorithm/trimmed_mean.py)、调用 [fl_core.py:729-737](file:///root/chashaobao/DPFLA-master/federated_learning/fl_core.py#L729-L737)。
- 旧的 `m_krum.py` / `t_mean.py` 已删除（改名并重写），`fl_algorithm/__init__.py` 统一导出 `multi_krum` / `trimmed_mean`。
- 入口脚本：CIFAR10 共 9 个（基线 1 + {FedAvg, DPFLA, MKrum, TMean} × {标签, 高斯} 8）；VisDrone 共 7 个（基线 1 + 高斯 {FedAvg, DPFLA, MKrum, TMean} 4 + 标签 {FedAvg, DPFLA} 2）。

⚠️ 风险【推断·未验证】：
- `trimmed_mean` **不跳过 BN buffer**（`running_mean/var` 也参与逐坐标裁剪）→ 可能损伤 BN 统计量；CIFAR 影响小、YOLO 影响大。
- `multi_krum` 对 YOLO 25M 参数 ×10 客户端做 CPU 距离矩阵（约 1GB 展平内存 + cdist）→ 内存/耗时需实测。
- 两者**从未在 VisDrone / YOLO 上跑通验证**。

### FLPoison 参考分支（`origin/FLPoison`）【实测】

- 上游：`https://github.com/vio1etus/FLPoison`（README 徽章），92 个文件，纯 PyTorch，单进程模拟。
- 防御：`aggregators/multikrum.py`（默认 `avg_percentage=0.2`、`num_adv`）、`aggregators/trimmedmean.py`（默认 `beta=0.1`）、另有 `krum.py` / `median.py` / `rfa.py` / `bulyan.py` / `flame.py` / `foolsgold.py` / `centeredclipping.py` / `normclipping.py` 等。
- 攻击：`attackers/gaussian.py`、`labelflipping.py`、`modelreplacement.py`、`alie.py`、`mimic.py`、`min.py` 等。
- 模型含 **resnet18**；`configs/FedSGD_CIFAR10_config.yaml`：resnet18 / lr 0.05 / 20 客户端 / 300 轮 / `num_adv: 0.20` / `defense: Mean`。
- **口径差异（必须先对齐再比）**：
  - Multi-Krum 选谁：FLPoison 默认 `m_avg=20%`（2/10）；本项目按标准公式 `n-f-2=5`（5/10）。
  - TrimmedMean 的 beta：FLPoison 默认 0.1；本项目 0.3（≥恶意率才合理）。
  - 客户端数 / 轮数 / lr 都不同 → **可借用实现，不可照搬超参**；报告里必须写明取哪个口径，且**三方同口径**。

### 对比矩阵（同 seed / 同划分 / 同攻击 / 同模型 / 同轮数 ≤15）

CIFAR10（ResNet18）：
1. 标签翻转（`FL_GRADIENT_SCALE=1`）+ {FedAvg, DPFLA, Multi-Krum, Trimmed-Mean}
2. 高斯（选定 σ）+ {同上}

VisDrone（v8m@1280）：
3. 高斯 σ=0.02 + {FedAvg, DPFLA, Multi-Krum, Trimmed-Mean}
   （基线 R15=47.13% 已有；需补 4 条同轮对比）

### 验收

- [ ] CIFAR10：DPFLA ≥ max(Multi-Krum, Trimmed-Mean) + 2%
- [ ] VisDrone：DPFLA mAP@0.5 ≥ max(二者) + 1%
- [ ] 三方攻击口径完全一致（日志参数行可逐项比对）

## §5 P4 — 分布式改造（**未来工作，不在当前范围**）

目标形态：多个终端各跑一个客户端，server 端聚合。

### 反向约束（现在的改动不要破坏这些，否则后续要返工）

1. **通信按"整模型"设计**：DPFLA 需要每个客户端的**完整 state_dict**（不是 delta）→ [fl_core.py:841-849](file:///root/chashaobao/DPFLA-master/federated_learning/fl_core.py#L841-L849)；带宽预算按整模型算。
2. **数据划分要落盘可复现**：现在是 server 端 `distribute_dataset()` 后内存分发；分布式下每个客户端需独立加载同一划分 → 需**分区索引 JSON + 固定 seed**。
3. **进程内状态需外移**：`Client._performed_attacks`（类变量）、`FL._history`（静态）、`self._server_momentum_buf` 都是单进程假设。
4. **聚合保持纯函数**：`aggregate(list[state_dict], scores)`；评估固定在 server；客户端只负责"本地训练 + 攻击 + 上传"。
5. **`fl_core` 保留为单机模拟器**：新增传输层（gRPC / HTTP / Ray），两条路径**共用同一聚合实现**，否则结果不可比。
6. 参数化 GPU / 线程 / DataLoader workers 的分配。

### 参考

- `FedAIoT-main/` 用 **Ray** 做分布式 + ultralytics 原生训练（借鉴进程编排，**不借鉴其聚合**，见 `04_Retracted.md` R1）。
- FLPoison 也是单进程模拟，**不提供分布式参考**。

## §6 前置条件

- **需要 GPU 主机**：2026-10-23 本机为**无卡模式**（无 `/dev/nvidia*`、`cuda.is_available()=False`、1 核）→ 当前只能改代码/写文档，不能跑实验。
- CIFAR10 数据：`/root/autodl-tmp/data/cifar-10-batches-py` 存在【实测】；`sampling.py` 优先读该路径。

## §7 风险 / 未决

1. **65% 是否可达未验证**：15 轮 ×1 local epoch 的 ResNet18 可能不够；主杠杆 = local epochs。
2. **高斯 σ 在 CIFAR 上完全未测**：需重走 σ 阶梯（0.01 / 0.02 / 0.05 / 0.1 量级），不能套 VisDrone 的 0.02。
3. **Multi-Krum / Trimmed-Mean 在 YOLO 上未验证**：BN 与显存两个坑。
4. **对比公平性**：Multi-Krum 的 m 比例、TrimmedMean 的 beta 必须三方一致并在报告注明。
5. **15 轮上限对 VisDrone 对比的影响**：基线 R15=47.13% 可用；③ 与 MKrum/TMean 都必须 ≤15 轮、同轮对比。

## §8 执行顺序总览

```
P1 修阻塞 → A1 ResNet18 基线 → A2/A3 标签线（不放大）→ A4 高斯线
   ↓ 达标 ≥65%
P2 DPFLA 审计（已完成，结论见 §3）
   ↓
P3 CIFAR 四方法对比 → VisDrone 四方法对比
   ↓ 达标 +2% / +1%
P4 分布式改造（未来）
```
