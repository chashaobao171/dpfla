# FedAvgM Bug 4 修复后 mAP 仍持续为 0：深度根因分析

> **日期**: 2026-06-10
> **现象**: FedAvgM learnable_keys 过滤已生效，但 mAP 仍 R1=0%, R2=0.53%, R3=0%
> **正常基线**: LR=2e-4, E=3, constant, 无 FedAvgM → R1=3.36%, R26-33≈23.7%

---

## 一、核心结论

| 假设 | 评估 | 优先级 |
|------|------|--------|
| **BN 统计量加载时就错了（COCO→VisDrone）** | ❌ **不是根因**。COCO→VisDrone 是正常的迁移学习场景，LR=2e-4 基线（同样 COCO BN）能到 23.7% 证明 BN 统计量不是问题 | P2 |
| **LR=0.01 太高** | ✅ **是主要根因**。50 倍 LR 提升 + 无 warmup + constant 调度 = 训练在第一轮就崩溃 | **P0** |
| **E=1 太少** | ✅ **是次要根因**。BN 统计量来不及稳定 + 检测头来不及适应 | **P1** |

---

## 二、详细分析

### 2.1 为什么 BN 统计量"加载时就错了"不是根因

文档中的证据：
```
Round 1 FedAvgM 执行后: running_var mean=0.0087
```

但这个值是**客户端训练 1 个 epoch 之后**的 BN 状态，不是"加载时"的状态。

**COCO 预训练 BN 的真实初始值**：
- YOLOv8s COCO 预训练的 `running_var` 初始值在 **0.5~2.0** 范围
- 这是正常的、健康的统计量

**LR=2e-4 基线证明了 BN 没有问题**：
- 同样的 COCO→VisDrone 转换
- 同样的换头流程
- LR=2e-4, E=3 → R1=3.36%, R26-33≈23.7%
- 如果 BN"加载时就错了"，基线也不可能到 23.7%

**running_var=0.0087 是怎么来的？**

在 `model.train()` 模式下，PyTorch BN 每次 forward 都更新：
```python
running_var = (1 - momentum) * batch_var + momentum * running_var
# PyTorch 默认 momentum=0.1
```

LR=0.01 下，卷积 weight 在第一个 step 就被巨大梯度冲击，输出分布剧烈变化：
1. 卷积输出 variance 变得异常（可能极小）
2. batch_var ≈ 0.001（异常小）
3. running_var = 0.9 * 1.0 + 0.1 * 0.001 ≈ 0.9（第一次 batch 后）
4. 经过 33 个 batch（1 个 epoch），如果 batch_var 持续异常 → running_var 被压缩到 0.008

**这是 LR=0.01 的结果，不是原因。**

### 2.2 为什么 LR=0.01 是主要根因

**对比实验**：

| 配置 | LR | EPOCH | 调度 | R1 mAP | R3 mAP |
|------|-----|-------|------|--------|--------|
| 正常基线 | **2e-4** | **3** | constant | 3.36% | ~19% |
| 当前（崩） | **0.01** | **1** | constant | **0%** | **0%** |

**LR=0.01 的问题**：

1. **50 倍提升**：从 2e-4 到 0.01，没有过渡
2. **无 warmup**：Ultralytics 官方 YOLOv8 用 `warmup_epochs=3`，LR 从 0 线性升到 0.01。DPFLA 直接 0.01 起步
3. **constant 调度**：不衰减。官方用 cosine 调度，LR 从 0.01 衰减到 0.0001
4. **检测头随机初始化**：新检测头 bias_init() 后，LR=0.01 的第一步就把 bias/weight 推到极值

**R2 loss=2169（爆炸）的直接解释**：

LR=0.01 × 梯度(正常) = 更新幅度巨大 → 模型参数跳到错误区域 → loss 爆炸

### 2.3 为什么 E=1 是次要根因

E=1 的问题不是"训练量不足"，而是**BN 统计量来不及稳定**：

1. 每个客户端只训 1 个 epoch（~33 batch）
2. BN running_var 只更新了 33 次
3. 然后 10 个客户端的 BN stats 被平均
4. 下一轮又从"被平均后"的 BN stats 开始

在 LR=2e-4 下，BN 参数变化平缓，33 次更新足够稳定。
在 LR=0.01 下，BN 参数剧烈震荡，33 次更新后 running_var 已被破坏。

---

## 三、文档中两个假设的准确评估

### 假设 ①：BN 统计量加载时就错了

**部分正确的事实**：
- COCO 预训练 BN 确实不完全匹配 VisDrone 数据分布
- 但这在**所有迁移学习场景**中都存在，不是 bug

**错误的归因**：
- running_var=0.008 不是"加载时"的值，是**LR=0.01 训练后的值**
- 如果 BN"加载时就错了"，LR=2e-4 基线也不可能工作

**正确的理解**：
- BN 统计量需要在 VisDrone 数据上**渐进自适应**
- LR=2e-4 提供了平缓的自适应过程
- LR=0.01 导致自适应过程崩溃

### 假设 ②：LR=0.01 太高

**完全正确。**

但这只是问题的一部分。更精确的说法是：

> **LR=0.01 + 无 warmup + constant 调度 + E=1** 的组合是灾难性的

单独 LR=0.01 不一定崩溃（如果加 warmup + cosine + E=3）。

---

## 四、我还发现了什么问题

### 问题 3：`train_loader` 使用了 `drop_last=False`（已有，但交互作用加剧）

```python
# client.py
train_loader = DataLoader(self.local_data, self.local_bs,
                          shuffle=True, drop_last=False, ...)  # ← 不丢弃最后一个 batch
```

当 `BS=64` 时，最后一个 batch 可能只有 **1~3 个样本**。

在 `model.train()` 模式下，BN 用这个极小 batch 计算统计量：
- batch_var(3 samples) 极不稳定（可能为 0 或极大）
- running_var 被错误更新

**但在 LR=2e-4 下这不是问题**，因为梯度小，模型参数变化平缓，batch 统计量即使有不稳定也有限。

**在 LR=0.01 下这是致命的**，因为每个 batch 的梯度都很大，最后一个不完整 batch 的异常统计量直接破坏 BN。

### 问题 4：`num_workers=4` + `persistent_workers=True` 可能导致 DataLoader 状态污染

在 client.py 中：
```python
num_workers = _visdrone_dataloader_workers()  # 默认 4
persistent_workers=(num_workers > 0),
```

当 E=1 时，每个 round 都新建 DataLoader，但 `persistent_workers=True` 可能保留之前的状态。

这不是主要问题，但值得注意。

### 问题 5：FedAvgM 动量缓冲的初始化时机

当前代码：
```python
if key not in self._server_momentum_buf:
    self._server_momentum_buf[key] = torch.zeros_like(global_weights[key])
buf = SERVER_MOMENTUM * buf + delta  # 第一次：buf = 0 + delta = delta
```

FedAvgM 在第一次生效时，动量缓冲初始化为 0，然后 `buf = delta`。这意味着**第一轮 FedAvgM 的修正幅度是正常的一半**（没有动量累积）。

这不是 mAP=0 的原因，但可能影响收敛速度。

---

## 五、修复方案

### 方案 A：保守回退（推荐，今晚验证）

直接回退到已知好配置 + FedAvgM：

```python
# visdrone_fed_hparams.py
LOCAL_LR = float(os.environ.get("FL_LOCAL_LR", "0.0002"))   # 0.01 → 2e-4
LOCAL_EPOCHS = int(os.environ.get("FL_LOCAL_EPOCHS", "3"))   # 1 → 3
GLOBAL_ROUNDS = int(os.environ.get("FL_GLOBAL_ROUNDS", "100")) # 50 → 100
LR_SCHEDULE = "constant"  # 保持（先不加 cosine，一次只改一个变量）
```

FedAvgM（已修复）保持启用。

**验证标准**：
- R1 > 3% → 基线恢复
- R3 > 15% → 正常收敛
- R10 > 22% → 超过历史好配置
- R30 > 28% → FedAvgM 生效确认

### 方案 B：渐进提升 LR（如果方案 A 有效）

在方案 A 验证成功后，逐步提升 LR：

| Phase | LR | EPOCH | 调度 | 目标 |
|-------|-----|-------|------|------|
| 1 | 2e-4 | 3 | constant | R30 > 28% |
| 2 | 5e-4 | 3 | constant | R30 > 30% |
| 3 | 1e-3 | 3 | cosine+warmup | R50 > 32% |
| 4 | 5e-3 | 1 | cosine+warmup | R100 > 35% |

### 方案 C：BN 统计量重置（不推荐单独使用）

```python
# yolo_wrapper.py 换头后
for module in self.model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.reset_running_stats()
```

**不推荐原因**：
- reset_running_stats() 把 running_mean/var 清零
- eval 时 BN 输出完全错误（需要大量 batch 重新估计）
- LR=2e-4 基线证明了 COCO BN 可以正常工作，不需要 reset

---

## 六、验证计划

### 今晚就做（方案 A）：

```bash
# 1. 回退参数
export FL_LOCAL_LR=0.0002
export FL_LOCAL_EPOCHS=3
export FL_GLOBAL_ROUNDS=100

# 2. 清理
rm -rf runs/VisDrone_YOLO_fedavg*

# 3. 运行
python run-test/visdrone/run_no_attack_baseline.py
```

### 判断标准：

| 时间 | 指标 | 判断 |
|------|------|------|
| R1 | mAP > 3% | ✅ 基线恢复 |
| R1 | mAP = 0% | ❌ 还有其他问题 |
| R3 | mAP > 15% | ✅ 收敛正常 |
| R10 | mAP > 22% | ✅ 超过历史好配置 |
| R30 | mAP > 28% | ✅ FedAvgM 生效 |

---

## 七、总结

| # | 问题 | 是否是根因 | 修复方式 |
|---|------|----------|---------|
| 1 | LR=0.01 太高 | ✅ **主要根因** | 回退到 2e-4，逐步提升 |
| 2 | E=1 太少 | ✅ **次要根因** | 回退到 3，逐步降低 |
| 3 | BN 统计量"加载时就错了" | ❌ 不是根因 | 不需要修复 |
| 4 | 无 warmup | ⚠️ 加重因素 | Phase 2+ 添加 |
| 5 | constant 调度 | ⚠️ 加重因素 | Phase 2+ 改 cosine |
| 6 | drop_last=False | ⚠️ 交互作用 | 可改为 True |

**一句话结论**：

> **不是 BN 的问题，是 LR 的问题。COCO BN 在 LR=2e-4 下能正常工作到 23.7%。LR=0.01 破坏了 BN 的自适应过程。先回退到 2e-4 + E=3 验证基线，再逐步提升 LR。**
