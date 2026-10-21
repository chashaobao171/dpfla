# 03_History.md — 时间线（改动与回退）

<!-- 最后更新：2026-10-20 -->
<!-- 追加式：新事件加在最上方。回退/证伪的事件保留，不删除。 -->
<!-- 结论性判断请以 01_Facts.md 为准；本文件只记"做了什么、结果如何"。 -->

## 2026-10-21

- **TensorBoard 真正 logdir 排查**（用户仍能看到未清理项）：确认 AutoPanel 的 TB 进程为 `--logdir /root/tf-logs`，而此前只清了仓库内 `tb/`。已清理 `/root/tf-logs`（29 → **6**，只留刚跑的 20261021_1140 + `log/` 对应的 5 个）、删除 `/root/tf-logs-archive`；全盘 tfevents 现为 6(tf-logs) + 7(runs) + 1(tb 镜像) = 14。
- **高斯 σ=0.05 复验**（用户要求）：R1 0.00%（损失 408.8596，与 10-20 完全一致 → 可复现），R2 0.00%（损失 366.6）→ 判定为"权重破坏"型崩档，按判据 R2 后停止；日志留在 `logs_3/visdrone/run_no_defense_gaussian_20261021_1140.log`，tb 按新策略删除。

- **日志/张量板清理**（用户要求）：`tb/` 11 个目录**全部删除**——按规则"只保留与 `log/` 归档实验对应的 tensorboard"，而现存 11 个 tb 无一条对应 `log/`（VisDrone 基线早于 `tb/` 启用）。
- `logs_3/` 从 42 个文件删到 **13 个**（9.7M → 3.2M）：删除旧基线（20260608~20261010 共 16 条，含量为 0 或已被 45.08% 取代）、5 条 MNIST 重复运行、`demo_out.log`、`visdrone_all.log`、`oracle_1ep`、`proof_dynamic`、以及崩/废弃高斯 4 条（σ=0.5 / 0.05 / 0.05修复后 / 0.01 无完整轮次）。
- **保留**：`log/` 对应三数据集的 7 份、VisDrone 35.92% 里程碑（`..._20261010_1813.log`）、标签翻转 4 条（1949/2022/2057/0959）、高斯 σ=0.005（2021）。
- 被删的高斯崩溃日志**数值已固化在 `01_Facts.md` §1d(d)**，不再依赖原始文件。
- **`runs/` 旧 TensorBoard 也清理**（用户指出"tb 没清干净"）：`runs/` 37 个 tfevents 删掉 30 个（含 2026-06-10 ~ 2026-10-10 的 16 次旧 VisDrone 基线、MNIST/CIFAR 的重复运行），**仅保留 7 个**——每个恰好对应 `log/` 中一份最终归档（VisDrone 20261013、MNIST 05-15 两份 + 09-02 基线、CIFAR10 09-02 三份）。

## 2026-10-20（后续·高斯攻击线）

- 本机有卡（RTX 4090 48G / 208 核 / 1007G）；跑 `run_no_defense_gaussian`（新增脚本）共 5 次，口径 = 30% 恶意 / 行为率 100% / v8m@1280 / 不放大。
- **诊断并修复真实 bug**：`gaussian_attack` 给 BN `running_var` 加噪 → 变负 → eval 模式 sqrt(负数) → NaN → mAP 0.00%（测试损失却正常，因它在 `train()` 后用 batch 统计量算）。修复 = 跳过 BN 统计量。详见 `04_Retracted.md` R12。
- **结果阶梯**（详见 `01_Facts.md` §1d(d)）：σ=0.5 崩、σ=0.05 崩（修复后仍崩，真实权重破坏）、σ=0.005 **无效**（≈基线 −0.45）。
- 用户判定：**σ=0.005 的 R2=36.61% 太高 → 打不动，停止高斯线**。
- **新增长期规则**（用户 2026-10-20）：做防御测试时若同时开了 **DP 与高斯攻击**开关，必须先核实两者**不是同一段代码路径**。

## 2026-10-20（本会话）

- 从 GitHub `dev` 拉取（快进 3 个提交 → `d7d6658`）。**当前主机为无卡模式**（无 `/dev/nvidia*`、`cuda.is_available()=False`、1 CPU 核）→ 未跑训练。
- **删除两条崩溃日志**（用户要求）：`run_no_defense_label_flipping_20261014_1908.log`、`..._1929.log` —— 均为 `FL_GRADIENT_SCALE=15`、R1=0.00% 的退化运行。**数值保留在 `01_Facts.md` §1d(b)**，原始日志不再保留。
- **记忆整理（全量校验 80 条"出处"引用）**：
  - 修正错文件名：`02_Methods.md` 的 `fl_avg.py` → **`fl_algorithm/fed_avg.py`**。
  - `04_Retracted.md` 顶部补记**"出处文件可核验性"**说明：旧记忆文件（`00_QuickStart.md` 等）与本次删除的崩溃日志均已不存在，引用仅用于指认来历。
  - 其余引用（`logs_3/`、`log/`、代码文件）全部可解析；`visdrone_temp.yaml` 为运行时生成且被 gitignore，非缺失。

## 2026-10-16 ~ 10-17（上一会话）

- 从 GitHub **`dev` 分支**拉取覆盖本地工作区；本机 **4090 48G / torch 2.5.1+cu124**。
- 代码改动（②③ 脚本同步，保持口径一致）：
  - `run_no_defense_label_flipping.py` / `run_dpfla_label_flipping.py`：把单一 `source_class/target_class` 改为 **`FLIP_MAPPING` 多标签映射**；默认 `FL_FLIP_RATIO=1.0`。
  - 映射三次递进：`{1:9}` → `{0,1,2}→9` → `{0,1,2}→9,{3,4}→8` → **`{i:(i+1)%10}` 全类循环**。
- **实验②（攻击+无防御）追加 3 次运行**（均 `FL_GRADIENT_SCALE=1` 不放大 / 恶意率 30% / 行为率 100%）：

| 映射 | 覆盖 | R1 | R2 | R3 | 出处 |
|---|---|---|---|---|---|
| `{0,1,2}→9` | 34.3% | 17.53% | 34.82% | 39.08% | `..._20261016_2022.log` |
| `{0,1,2}→9,{3,4}→8` | 83.5% | 16.47% | 31.75% | — | `..._20261016_2057.log` |
| `{i:(i+1)%10}` | **100%** | **15.09%** | **27.23%** | 35.45%（R4 37.51 / R5 39.11）| `..._20261017_0959.log` |

- **结论**：纯数据投毒**仅靠加强标签映射已饱和**（覆盖 ×2.9，R1 仅 −2.44 点；全类循环 R2 单轮 <30% 后 R3 回弹）→ 用户据此**停止映射实验**，转"方案清单"评估。数据见 `01_Facts.md` §1d(c)，清单见 `chatlog/2026-10-16_多标签翻转攻击强度迭代.md`。
- **Git 规则**：`.gitignore` 放开 `logs_3/` 及 `*.log`（用户要求**原始日志一并推送**），`06_Git_Workflow.md` 登记该例外。

## 2026-10-14（上一会话）

- 从 GitHub **`dev` 分支**拉取覆盖本地工作区（切到分支 `dev`；另留安全分支 `backup-before-dev` 指向原 `9c5e458`）。本机 **4090 48G / 20 vCPU / 96G**。
- 代码改动：
  - 新增 **`FL_FLIP_RATIO`**：标签翻转的**实例比例**（改动 `poisoned_dataset.py`、`label_flip_attack.py`、`client.py` 与两个 VisDrone 实验脚本）
  - TensorBoard 目录由 `runs/` 改为 **`tb/`**（**纳入 git → 跨主机留存**），目录名加入 **模型大小 + 分辨率 + 运行时间戳**
- 启动 **实验②（攻击 + 无防御）**，三次尝试：

| # | 配置 | R1 | 结论 |
|---|---|---|---|
| 1 | ×15 + flip_ratio=1.0 | 0.00%（损失 9412.6）| 崩 |
| 2 | ×15 + flip_ratio=0.1 | 0.00%（损失 10866.9）| 崩；**投毒量降 90% 完全无效** |
| 3 | **×1 + flip_ratio=1.0** | **19.88%**（损失 297.8）| **攻击几乎无效**（R4 43.24% vs 基线 43.82%）|

- **根因确认**：`client.py:522-543` 的内联**更新外推**（`FL_GRADIENT_SCALE`，**默认 15**）才是崩溃来源。agent 曾误判"本项目没有梯度放大"（只查了未被调用的 `EnhancedLabelFlipAttack`）→ 见 `04_Retracted.md` **R10 / R11**。
- 结论：**纯数据投毒（×1）在此数据/模型上打不动模型**。"可控但有感"的攻击需另找杠杆（源类 / 翻转类别数 / 放大倍数取中间值）—— **待办**。

## 2026-10-10（上一会话）

- 盘点项目文件结构；**保守清理**：删除全部 `__pycache__`、`.ipynb_checkpoints`×3、`.idea/`、`.vscode/`、`cache/`(32M)、`visdrone_temp*.yaml`、空 `data/`。
- **重构 MEMORY** 为分层事实记忆包（本文件即新版之一）；修正 FedAIoT / 子进程 val 的错误结论（见 `04_Retracted.md`）。
- 环境：autodl 无卡模式（无 GPU、1 CPU 核）→ 未跑训练。

## 2026-09 系列

| 日期 | 事件 | 结果 | 证据 |
|------|------|------|------|
| 2026-09-16 | 50 轮基线运行 | R1=3.35%，日志在 R2 中断 | `logs_3/visdrone/run_no_attack_baseline_20260916_1437.log` |
| 2026-09-15 | 基线运行 | ❌ val 子进程被 OOM 杀（`code=-9`）| `..._20260915_2144.log` |
| 2026-09-07 | proof_dynamic 单轮验证 | 训练前冷启动 mAP=0%；R1=3.24% | `logs_3/visdrone/proof_dynamic_20260907_1520.log` |
| 2026-09-04 | **回退训练管线到纯 FedAvg**（9b3169d 口径）| FedAvgM/梯度裁剪改环境变量门控、默认关 | commit 见下 |
| 2026-09-04 | 全入口统一默认 YOLOv8s | `model.py` / `yolo_wrapper.py` 默认 `n`→`s` | ChangeLog |
| 2026-09-02 | 9/2 demo（FedAvgM 0.9 恒开 + clip 10.0）| 10 轮仅 2.55%，loss 705→1010 | `logs_3/demo_out.log` |

09-04 回退内容（`fl_core.py` / `client.py`）：
- FedAvgM 门控：`FL_SERVER_MOMENTUM` 默认 `0`（纯 FedAvg）
- 梯度裁剪门控：`FL_GRAD_CLIP` 默认 `0`（关闭）
- `drop_last` True→False（对齐 9b3169d）

## 2026-06-10（FedAvgM 引入 → 崩塌 → 修复 → 回退）

| 时间 | 事件 | 结果 |
|------|------|------|
| 上午 | YOLOv8s→l + conf=0.001 尝试 | 训练前冷启动 mAP=0%，被**误判**为"val bug"（见 04_Retracted） |
| 下午 | **Bug1+2+3 修复**（commit 9b3169d） | val 子进程检测头 / 梯度平均 / BN `num_batches_tracked` → 100 轮基线 R26-33≈23.7% ✅ |
| 傍晚 | 引入 **FedAvgM（服务端动量 0.9）+ 梯度裁剪**（commit 87e3f7a，同一提交还改了 LR 2e-4→0.01、EPOCH 10→1）| Round 2-3 mAP 崩塌 |
| 晚 | 修 FedAvgM：只对可学习参数做动量（8b196f0）、设备统一（d93c551）| mAP 仍 0（LR=0.01 未回退）|
| 深夜 | 追加两份分析（`round3_collapse_*.md`，观点互相矛盾）| — |
| 深夜 | f2741ff：BN 统计量重置 + 超参回退（LR=2e-4, EPOCH=3）| — |
| 深夜 | 8554d4d：**撤销 BN reset（实测有害）** + 修 `drop_last=False` | — |

> ⚠️ 87e3f7a 一次提交改了 4 个变量（聚合 + 裁剪 + LR + EPOCH），**导致无法归因**——这是本项目最重要的流程教训。

## 2026-06-09 及之前

| 日期 | 改动 | 文件 | 结果 |
|------|------|------|------|
| 2026-06-09 | Mosaic + MixUp 实现（50%/10%）| `visdrone_dataset.py` | ✅ 合入，效果未单变量验证 |
| 2026-06-09 | 分层 LR backbone×0.5 / head×2.0 | `client.py` | ✅ 合入 |
| 2026-06-09 | AMP 混合精度（旧记"已完成"）| `client.py` | ⚠️ **与实现不符**：2026-10-10 全仓核查无 `autocast`/`GradScaler`（见 04_Retracted R9）|
| 2026-06-09 | val conf 0.001 → 0.25（子进程 eval 仍用 0.001）| `fl_core.py` | ✅ 合入 |
| 2026-06-09 | 差配置诊断：BS16+EPOCH3+cosine → 14.68% | `visdrone_fed_hparams.py` | 已回退 |
| 2026-06-08 | FedLA 实现完成（`average_weights_fedla()`）| `fl_core.py` | R8=22.49% |
| 2026-06-07 | FedBN 实现完成 | `fl_core.py` | R7 峰值 21.55%，全程 ≤ FedAvg |
| 2026-06-06 | local epochs = 10 | hparams | ✅ |
| 2026-05-15 | MNIST：DPFLA 95% vs FedAvg 4.57% | — | ✅ 冻结 |
| 2026-05-14 | 环境迁移到 autodl，路径统一 | 多处 | ✅ |

## 关键 Git 提交

```
b9159b1 feat: 新增一键备份脚本 backup.sh
98c8e72 merge: 合并本地同步分支 fc7b29c
45bd3c9 sync: 云平台工作区同步
fc7b29c feat: 本地工作区同步 - run_demo + fl_core/client/model 修复 + MEMORY 整理
8554d4d revert: 撤销 BN reset（有害）+ 修复 drop_last=False
f2741ff fix: BN 统计量重置 + 超参回退（LR=2e-4, EPOCHS=3）
51bd31b docs: 追加 R3 mAP 仍为 0 的深度分析报告
d93c551 fix: FedAvgM 设备统一
8b196f0 fix: FedAvgM 只对可学习参数做动量平滑
87e3f7a feat: FedAvgM 服务端动量平滑 + 梯度裁剪
255e4c2 docs: 新增 FedAIoT 参考项目 + 深度借鉴报告
9b3169d fix: val 子进程检测头 + 梯度平均 + BN 聚合 (Bug 1+2+3)
8ef2a47 Initial commit: DPFLA federated learning project
```
