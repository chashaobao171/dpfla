# 00_Current_State.md — 当前状态

<!-- 最后更新：2026-10-10 -->
<!-- ⚠️ 本文件会过期。与 01_Facts.md 冲突时，以 01_Facts.md 的实测数据为准 -->

## 一句话

**DPFLA**：联邦学习投毒防御项目，落地场景 **VisDrone + YOLOv8** 目标检测。
核心目标：**无攻击联邦基线 mAP@0.5 ≥ 32%** —— ✅ **已于 2026-10-10 达成**：1024 分辨率下 **R15 = 35.92%**（640 基线仅 23.7%，集中式 oracle ≈41%）。详见 `01_Facts.md` §1b。

## 进度快照（2026-10-10）

- ✅ **里程碑：32% 目标达成。** 1024 + BS=32 + AMP → **R7 破 32%，R15 = 35.92%**（R11–R15 已收敛）。
- 代码仍处**纯 FedAvg 基线口径**（FedAvgM / 梯度裁剪均 env 门控、默认关）。
- 本会话新增 env 门控：`FL_IMG_SIZE`（分辨率）、`FL_AMP`（混合精度）、`FL_TRAIN_BATCH_SIZE` / `FL_TEST_BATCH_SIZE` / `FL_DATALOADER_WORKERS`。
- 工作区有**未提交改动**（client.py 的 AMP、fl_core/sampling 的分辨率、scripts/、MEMORY/、refs/）。
- 环境：autodl **有卡模式** RTX 4080S 32GB（跑 GPU 必须**非沙箱**）。

## 当前阻塞 / 未决

1. ~~mAP 距目标差 ~8–10 个点~~ → **已解决**（1024 达成 35.92%）。
2. **归因未做**：1024 那次同时改了 分辨率 + batch + AMP，需补 **`640 + BS=32 + AMP`** 对照臂才能定论。
3. **"换聚合模型"是错误方向**，已回退（见 `04_Retracted.md`）。
4. 下一步主线：① 补对照臂做归因；② 扫分辨率档位；③ 训练循环原生化（AMP 已有，仍缺 EMA/warmup/原生增强）；④ **回到研究本体**：无攻击 / 无防御+标签翻转 / **DPFLA**+标签翻转 三条曲线。

## 下一步（按优先级）

1. **补对照臂归因**：`640 + BS=32 + AMP`，确认 +12 点是否主要来自分辨率。
2. **分辨率档位扫描**：800 / 768，找精度/速度平衡点。
3. **训练循环原生化**：AMP 已有，仍缺 **EMA / warmup / 原生增强**。
4. **研究本体三条曲线**：无攻击基线 / 无防御+标签翻转 / **DPFLA**+标签翻转（`01_Facts.md` §7 记载 DPFLA 特征已改检测头，**改动后未重跑**）。

> **轮数已排除**（R26-33 已收敛）；**"换聚合"已证伪**（见 `04_Retracted.md`）。
> 铁律：一次只改一个变量（教训见 `03_History.md`）。

## 实验命令（已备好，有卡可直接跑）

```bash
# ✅ 已验证配方（R15=35.92%，已超 32% 目标）
# 1024 必须开 AMP；测试 batch 不能大（否则容器内存 OOM 被 SIGKILL 137）
./scripts/run_fl_tmux.sh run_no_attack_baseline FL_IMG_SIZE=1024 FL_TRAIN_BATCH_SIZE=32 FL_AMP=1 \
    FL_TEST_BATCH_SIZE=64 FL_DATALOADER_WORKERS=2

# 归因对照臂（待跑）：同 batch + 同 AMP，仅分辨率不同
./scripts/run_fl_tmux.sh run_no_attack_baseline FL_IMG_SIZE=640 FL_TRAIN_BATCH_SIZE=32 FL_AMP=1 \
    FL_TEST_BATCH_SIZE=64 FL_DATALOADER_WORKERS=2

# 显存/内存实测（RTX 4080S 32GB，2026-10-10）：
#   1024 + BS=64 + fp32 → ❌ OOM（第 1 个 batch）
#   1024 + BS=32 + fp32 → ❌ OOM（backward 阶段 31.3GB/32GB）
#   1024 + BS=32 + AMP  → ✅ 24GB 稳定（这是最终成功的配方）
#   1024 下 TEST_BATCH_SIZE=256 + workers=6 → ❌ 容器内存打爆，SIGKILL(137) ⇒ 降到 64/2

# tmux 保证断网不影响：脱离 Ctrl-b d，重连 tmux attach -t fl
# 日志：logs_3/visdrone/<tag>_<北京时间>.log
```

> 注：改用 BS=32/16 会**偏离历史 BS=64 基线**；但两个臂同 batch，分辨率对比仍成立。若要坚持 BS=64，需先给训练循环加 AMP 或梯度累积（属原生化改造）。

> `FL_IMG_SIZE` 同时作用于**训练**（`sampling.py`）与**评估**（`fl_core.py`），默认 640 行为不变。

## 硬约束（不得违反）

见 `05_Environment.md` §冻结红线。核心 3 条：
1. DPFLA 主路径 = SVD + KMeans（`use_validation=False`）；
2. 指标 = YOLO 原生 `val()` 的 mAP@0.5，**禁用 fallback**；
3. FedAvg 聚合统一 **float32**。
