# 03_History.md — 时间线（改动与回退）

<!-- 最后更新：2026-10-10 -->
<!-- 追加式：新事件加在最上方。回退/证伪的事件保留，不删除。 -->
<!-- 结论性判断请以 01_Facts.md 为准；本文件只记"做了什么、结果如何"。 -->

## 2026-10-10（本会话）

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
