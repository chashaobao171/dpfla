# FedAvgM Bug 4 修复后 mAP 仍持续为 0：深度根因分析

> **日期**: 2026-06-10 晚
> **现象**: FedAvgM 设备错误已修复，learnable_keys 过滤已生效，但 mAP 仍 R1=0%, R2=0.53%, R3=0%
> **预期**: R1 > 5%, R2 > 12%, R3 > 18%

---

## 一、证据清单（按优先级）

### 证据 1：BN running_var 在 FedAvgM 执行前就已经很小

```
Round 1 FedAvgM 执行后 debug 日志（fl_core.py:793）:
BN running_var sample [model.0.bn.running_var]: range=[0.0005, 0.0284], mean=0.0087, std=0.0078

Round 2 FedAvgM 执行后:
BN running_var sample [model.0.bn.running_var]: range=[0.0004, 0.0271], mean=0.0077, std=0.0070

Round 3 FedAvgM 执行后:
BN running_var sample [model.0.bn.running_var]: range=[0.0003, 0.0263], mean=0.0071, std=0.0066
```

**关键观察**：BN running_var 在 Round 1 FedAvgM 之前就处于极小值（mean=0.0087，正常应为 ~1.0），并且在每次 FedAvgM 后持续缓慢下降。说明 BN 统计量在模型加载时就错了，FedAvgM 的 learnable_keys 过滤只阻止了"加速恶化"，但不能修复"已有的错误"。

### 证据 2：当前超参 vs 历史好配置

| 参数 | 当前（差，mAP=0） | 历史好配置（Bug修复前基线） |
|------|-----------------|------------------------|
| LR | **0.01** | **2e-4** |
| LOCAL_EPOCHS | **1** | **3** |
| GLOBAL_ROUNDS | 50 | 100 |
| Total 训练量 | 50×1=50 epoch | 100×3=300 epoch |
| FedAvgM | ✅ 已修复（learnable_keys） | ❌ 失败（设备错误） |
| R1 mAP | 0% | 3.36% |
| R2 mAP | 0.53% | ~8% |
| 最终 mAP | 持续 0% | R26-33≈23.7% |

### 证据 3：Loss 爆炸

```
R1: 损失=1426.6611  ← 正常（COCO 预训练换头后）
R2: 损失=2169.4446  ← ⚠️ 爆炸！Loss 反而上升 50%
R3: 损失=1288.1367  ← 下降但 mAP=0
```

R2 loss 爆炸 + mAP=0.53% 强烈暗示：**模型在 LR=0.01 下剧烈震荡，检测头完全失效**。

### 证据 4：FedAvgM 执行无报错，但 mAP 未恢复

- ✅ learnable_keys 过滤正常工作（BN buffer 被跳过）
- ✅ 设备统一修复生效（old_weights CPU → GPU）
- ✅ BN running_var 不再被 FedAvgM 加速压缩（但已在错误值）
- ❌ BN 统计量在模型加载时就已经错了
- ❌ LR=0.01 太高，检测头训练震荡

---

## 二、根因分析

### 根本原因 A（主要）：BN 统计量从加载时就错了

COCO 预训练的 YOLOv8s 的 BN running_var/running_mean 是针对 COCO 数据集统计的。当替换检测头并迁移到 VisDrone 数据集时：

1. **模型加载**：`YOLO('yolov8s.pt')` → running_var 约 1.0（正常）
2. **换头后**：`new_head.bias_init()` 只初始化检测头，**不更新 BN 统计量**
3. **VisDrone 数据分布与 COCO 完全不同** → COCO 的 BN 统计量对 VisDrone 完全不适用
4. **训练阶段**（`model.train()`）：BN 用 batch 统计量，掩盖了问题
5. **评估阶段**（`model.eval()`）：BN 用 COCO 的 running_mean/var → 归一化完全错误 → mAP=0

**为什么 FedAvgM 的 learnable_keys 修复没有解决问题**：BN buffer 确实被跳过了，但 running_var 已经在极小值（约 0.008），这在模型加载时就已经发生了。FedAvgM 修复阻止了"继续恶化"，但不能"回退到正确值"。

### 根本原因 B（次要）：超参配置灾难性

- **LR=0.01**：YOLOv8 官方 fine-tune LR 通常 1e-4 ~ 5e-4，0.01 是 10-50 倍。YOLOv8 官方微调 recipe 用 `lr0=0.01` 是针对 SGD + momentum=0.937 + warmup 的，本项目 SGD LR=0.01 无法收敛。
- **LOCAL_EPOCHS=1**：只有 1 个 epoch 的本地训练，检测头参数还没来得及适应数据分布就被聚合了。3 个 epoch 才能让检测头充分适应。
- **总训练量**：50轮×1epoch=50 epoch vs 原来 100轮×3epoch=300 epoch，差 6 倍。

---

## 三、解决方案

### 方案 1：重置 BN 统计量（立即可试）

在模型换头后、训练开始前，对所有 BN 层调用 `.reset_running_stats()` 并切换为 train 模式让 BN 自适应：

```python
# 在 yolo_wrapper.py 换头后添加（约第 204 行之后）：
for module in self.model.modules():
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.SyncBatchNorm):
        module.reset_running_stats()
        module.train()  # 强制 BN 用 batch 统计量
```

或者更简单：在第一轮客户端训练开始前，对 `simulation_model` 和 `global_model` 做一次前向传播（train 模式）来"预热" BN 统计量。

### 方案 2：超参回退到历史好配置（推荐）

| 参数 | 回退到 |
|------|--------|
| LR | **2e-4**（不是 0.01） |
| LOCAL_EPOCHS | **3**（不是 1） |

这样：
1. LR=2e-4 足够小，检测头能稳定适应 VisDrone 数据
2. 3 个 epoch 让 BN 统计量在本地训练中充分更新
3. 仍然使用 FedAvgM（已修复），服务带动量平滑加速收敛

### 方案 3：同时做 BN 统计量重置 + 好超参（最稳妥）

组合方案 1 和 2，既重置 BN 统计量，又用历史好超参，确保万无一失。

---

## 四、验证计划

### 最小验证（方案 2，最快）

只回退超参，不改代码：

```bash
# 临时覆盖 visdrone_fed_hparams.py 的值
export FL_LOCAL_LR=0.0002
export FL_LOCAL_EPOCHS=3
python run-test/visdrone/run_no_attack_baseline.py
```

预期结果（FedAvgM 已修复的前提下）：
- R1: 3-5% mAP
- R3: 10-15% mAP
- R10: 18-22% mAP
- R50: 28%+ mAP

### 最稳验证（方案 3）

同时回退超参 + 添加 BN 统计量重置。

---

## 五、问题汇总

| # | 问题 | 根因 | 优先级 |
|---|------|------|--------|
| 1 | BN running_var 极小（0.008） | COCO 预训练统计量不适配 VisDrone，训练前未重置 | **P0** |
| 2 | LR=0.01 太高 | visdrone_fed_hparams.py 默认值激进 | **P0** |
| 3 | LOCAL_EPOCHS=1 太少 | 同上，训练量不足 | **P1** |
| 4 | FedAvgM 设备错误 | 已修复 ✅ | **已解决** |
| 5 | FedAvgM 影响 BN buffer | 已修复（learnable_keys）✅ | **已解决** |
