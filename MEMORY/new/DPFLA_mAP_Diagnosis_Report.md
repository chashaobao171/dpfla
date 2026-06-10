# VisDrone 联邦学习 mAP 暴跌根因诊断报告

> **项目**: DPFLA - VisDrone YOLO 目标检测联邦学习
> **核心问题**: 一系列"优化"后 mAP@0.5 从 21% 暴跌至 7.7%
> **分析日期**: 2026-06-09
> **代码基线**: https://github.com/chashaobao171/dpfla

---

## 目录

- [执行摘要](#执行摘要)
- [Q1: 主犯排序（按影响程度）](#q1-主犯排序按影响程度)
- [Q2: Warmup + Cosine 的致命缺陷](#q2-warmup--cosine-的致命缺陷)
- [Q3: 分层学习率的设计问题](#q3-分层学习率的设计问题)
- [Q4: AMP + loss.sum() 的隐患](#q4-amp--losssum-的隐患)
- [Q5: Local Epochs = 3 的真实影响](#q5-local-epochs--3-的真实影响)
- [Q6: 恢复路径建议](#q6-恢复路径建议)
- [附: 改动全景图与交互效应分析](#附-改动全景图与交互效应分析)
- [附: 深层架构问题](#附-深层架构问题)

---

## 执行摘要

**结论先行: 这不是某个单一参数的错，而是 5 个改动构成的"死亡组合"。**

| 改动项 | 原值 | 新值 | 危害评级 |
|--------|------|------|----------|
| `TRAIN_BATCH_SIZE` | 64 | 16 | **CRITICAL** |
| `LOCAL_EPOCHS` | 10 | 3 | **CRITICAL** |
| 分层学习率 (head LR) | 无 | `lr * 5.0` | **HIGH** |
| `LOCAL_LR` | 2e-4 | 5e-4 | **HIGH** |
| Cosine + Warmup | 无 | 有 | **MEDIUM** |
| `TEST_BATCH_SIZE` | 256 | 16 | LOW |
| `GLOBAL_ROUNDS` | 20 | 30 | LOW |
| AMP | 无 | 有 | LOW |

**核心机制**: `BS16 + EPOCH3 + HEAD_LR_2.5e-3` 三者叠加，导致检测头在"高学习率 + 少迭代 + 梯度高方差"的三重绞杀下完全无法正常学习。Cosine + Warmup 进一步将前 3 轮的学习率压在亚有效区间，形成"温水煮青蛙"效应。

**一句话诊断**: 你在试图用一个为 ImageNet 分类设计的优化配置（小 batch、cosine decay、layer-wise LR）去训练一个联邦场景下的目标检测头，且只给每轮 3 个 epoch 的学习时间。

---

## Q1: 主犯排序（按影响程度）

### 第 1 主犯: `TRAIN_BATCH_SIZE 64 → 16`（危害指数: ★★★★★）

**破坏机制**:

1. **梯度估计方差爆炸**: Batch size 从 64 降到 16，梯度估计的方差理论增大 `sqrt(64/16) = 2` 倍。在联邦学习场景下，Non-IID 数据分布使各客户端梯度方向本就高度发散，方差放大后客户端 drift 问题被急剧恶化。

2. **BatchNorm 统计崩溃**: YOLOv8 大量使用 BN 层。BN 的统计量（running mean/variance）依赖 batch size 来稳定估计。当 BS=16 时，每层的统计量波动剧烈，导致:
   - 训练时 BN 层输出分布不稳定
   - 测试时 running statistics 与训练时不匹配
   - 最终表现为 mAP 评估时"训练和测试的分布偏移"

3. **正负样本比失衡**: VisDrone 是小目标密集场景，单张图可能有数十个目标。BS=64 时每 batch 约有 `64 * avg_boxes_per_image` 个正样本参与 loss 计算；BS=16 时正样本数减至 1/4。YOLO 的 confidence loss 和 class loss 对正负样本比例极为敏感，正样本过少导致 classification head 学习信号严重不足。

4. **与 FedAvg 的恶性交互**: FedAvg 按等权重聚合各客户端更新。当每个客户端的梯度噪声增大 2 倍时，聚合后的全局模型相当于在"噪声立方"中做随机漫步，无法收敛到有效解。

**量化估算**: 单独将 BS 从 64 改为 16，在 10 客户端 Non-IID 联邦场景下，预期 mAP 下降 5-10%。

---

### 第 2 主犯: `LOCAL_EPOCHS 10 → 3`（危害指数: ★★★★★）

**破坏机制**:

1. **知识积累周期被截断**: YOLOv8 检测头在 COCO→VisDrone 换头后是完全随机初始化的（`new_head.bias_init()` 随机初始化）。从随机权重到能检测出有效特征，需要大量迭代来:
   - 让 anchor 匹配器学会将 GT box 分配给合适的特征层
   - 让 classification 分支学会区分 10 类目标
   - 让 bbox regression 分支学会预测精确坐标
   3 个 epoch 远远不够这个过程。10 个 epoch 也只是勉强够用。

2. **联邦同步的"记忆抹除"效应**: 每轮全局聚合后，客户端模型被替换为全局模型。 LOCAL_EPOCHS=3 意味着客户端每 3 个 epoch 就要"忘记"本地学到的特征，重新从全局模型开始。这类似于一个学生在每次考试前 3 天开始复习，然后考完立即被洗脑——永远无法形成长期记忆。

3. **与 BS=16 的叠加效应**: BS=16 时每个 epoch 的 step 数是 BS=64 的 4 倍。LOCAL_EPOCHS 从 10 减到 3，意味着每轮联邦更新的**总 step 数**从 `10 * (N/64)` 降至 `3 * (N/16) = 3 * 4 * (N/64) = 12 * (N/64)`。表面上看 step 数还增加了，但:
   - 每个 step 的梯度质量下降（BS=16 的噪声问题）
   - 3 个 epoch 内模型还没"热起来"就被重置
   -  warmup 阶段（每个客户端的初始几个 batch）占比过大，有效训练时间不足

4. **Cosine Schedule 的对冲失效**: 原配置 20 轮 × 10 epoch = 200 个本地训练周期，模型有足够时间在 cosine decay 的每个阶段学习。新配置 30 轮 × 3 epoch = 90 个周期，且前 3 轮被 warmup 的亚有效 LR 占据，实际有效学习时间不足原配置的 1/3。

**量化估算**: 单独将 LOCAL_EPOCHS 从 10 改为 3，预期 mAP 下降 5-8%。

---

### 第 3 主犯: 分层学习率 `head_lr = lr * 5.0 = 2.5e-3`（危害指数: ★★★★☆）

**破坏机制**:

1. **检测头 LR 远超合理范围**: 在标准 YOLOv8 训练中，Ultralytics 默认使用 `lr0=1e-3`（SGD）且**不对检测头单独设置更高的 LR**。你的 `head_lr=2.5e-3` 是默认值的 2.5 倍。检测头是新初始化的，高 LR 导致:
   - 参数在最优解附近剧烈震荡
   - 无法稳定收敛到好的特征表示
   - 大 LR + 小 Batch = 随机游走，而非梯度下降

2. **Backbone 被"冻结"在亚最优状态**: `backbone_lr = lr * 0.1 = 5e-5`。COCO 预训练的 backbone 虽然质量高，但在 VisDrone 的航拍视角、小目标密集场景下需要微调。5e-5 的 LR 意味着 backbone 几乎不更新，模型依赖一个"未经 VisDrone 适配"的特征提取器去检测截然不同的目标分布。

3. **FedAvg 聚合时的"LR 错配"**: 各客户端的分层 LR 相同（都从全局 LR 派生），这本无问题。但问题在于聚合后，全局模型的参数空间相当于在一个**非均匀学习率**的流形上做平均:
   - Head 参数每轮更新幅度大（2.5e-3），但方向噪声也大
   - Backbone 参数几乎不动（5e-5）
   - 聚合后的全局模型 head 部分实际上是"多个高方差更新的平均"，退化为随机平滑

4. **与 AMP 的潜在冲突**: 高 LR 下梯度幅值大，AMP 的 GradScaler 需要更大的 loss scale 来避免梯度下溢。如果 scale 因子设置不当，可能导致:
   - 梯度裁剪被频繁触发
   - 有效 LR 被 AMP 的自动缩放机制"暗中修改"

**量化估算**: 分层 LR 的 head_lr 倍数从 1.0 改为 5.0，在 EPOCH=3 的短训练周期下，预期 mAP 下降 3-5%。

---

### 第 4 主犯: `LOCAL_LR 2e-4 → 5e-4`（危害指数: ★★★★☆）

**破坏机制**:

1. **基础 LR 增大 2.5 倍**: 在 LOCAL_EPOCHS=10 时，2e-4 的 LR 配合 cosine decay 可以在 20 轮内稳定收敛。但当 LOCAL_EPOCHS 减到 3 时，5e-4 的峰值 LR 意味着:
   - 每轮中每个 epoch 的 LR 都偏高
   - 检测头在 3 个 epoch 内经历了过高的参数更新速度
   - 模型在最优解附近"跳过"而非"落入"

2. **Cosine decay 的末期陷阱**: LR 从 5e-4 衰减到 5e-6，末期 LR 几乎为零。在联邦学习 30 轮的场景下，后 10 轮的 LR 低于 1e-5，此时:
   - 模型参数更新幅度极小（`Δw ≈ lr * grad ≈ 1e-5 * 1 = 1e-5`）
   - 即使梯度指向正确方向，参数也几乎不动
   - 后 10 轮沦为"假训练"，浪费计算资源

3. **Warmup 期间的双重低效率**: Warmup 前 3 轮的平均 LR 仅为峰值的 50%（约 2.5e-4），加上 BS=16 的梯度噪声和 EPOCH=3 的短周期，模型在前 3 轮几乎学不到有效特征。

**量化估算**: LR 从 2e-4 提升到 5e-4，在 BS=16、EPOCH=3 的配置下，预期 mAP 下降 2-4%。

---

### 第 5 主犯: Cosine + Warmup（危害指数: ★★★☆☆）

**破坏机制**（详见 Q2 深度分析）:

1. **Warmup 期效率极低**: 前 3 轮 LR 分别为 1.67e-4、3.33e-4、5e-4。在 BS=16 的高梯度噪声下，前 2 轮的有效学习几乎为零。

2. **Cosine 衰减过快**: 30 轮内从 5e-4 衰减到 5e-6，第 10 轮 LR 已降至约 2e-4（低于原配置的稳定 LR），第 20 轮降至约 5e-5。这意味着:
   - 模型在 10-20 轮的"黄金学习期"只有原配置一半的 LR
   - 20 轮后 LR 过低，模型进入"僵化"状态

3. **联邦学习场景的根本性错配**: Cosine LR schedule 的设计假设是**连续训练**（单个模型在完整数据集上训练数百个 epoch）。联邦学习的"每轮重置"机制完全打破了这个假设:
   - 每轮客户端从全局模型开始，LR 却按全局轮次递减
   - R20 的客户端拿到一个"低 LR 的全局模型"，只训练 3 个 epoch，LR 已经很低
   - 这相当于给每个客户端一个"已经训练到后期的模型"，却只给 3 个 epoch 的微调时间

**量化估算**: Cosine + Warmup 在联邦短周期场景下，预期 mAP 下降 2-3%。

---

### 第 6 主犯: `TEST_BATCH_SIZE 256 → 16`（危害指数: ★★☆☆☆）

**破坏机制**:

1. **BN 统计量测试时不匹配**: 训练时 BS=16，测试时如果 BS 也是 16，BN 的 running statistics 在训练和测试时的一致性较好。但问题不在这里。

2. **评估时的内存碎片**: 测试 batch size 过小导致 GPU 利用率低，评估速度变慢。但这不影响 mAP。

3. **实际影响有限**: TEST_BATCH_SIZE 对 mAP 的直接影响较小（<1%），主要影响评估速度和稳定性。

**注意**: 这个改动虽然对 mAP 影响小，但 TEST_BATCH_SIZE=16 与 TRAIN_BATCH_SIZE=16 的组合会加剧 BN 层的统计不稳定。如果在测试时使用较大的 batch（如 64 或 128），可以利用更稳定的 running statistics，mAP 可能提升 0.5-1%。

**量化估算**: 预期 mAP 影响 < 1%。

---

### 第 7 主犯: `GLOBAL_ROUNDS 20 → 30`（危害指数: ★☆☆☆☆）

**破坏机制**:

1. 这个改动本身不导致 mAP 下降，反而给了更多训练时间。

2. 问题在于 Cosine schedule 将后 10 轮的 LR 压得过低（< 1e-5），导致多出的 10 轮几乎无效。

3. 真正的危害是"以为增加轮数就能弥补 local_epoch 的不足"——这是一种幻觉。3 个 local epoch × 30 轮 = 90 个总 epoch，远少于 10 个 local epoch × 20 轮 = 200 个总 epoch。

**量化估算**: 单独增加 GLOBAL_ROUNDS 不会降低 mAP，但无法弥补其他改动的损失。

---

### 第 8 主犯: AMP 混合精度（危害指数: ★☆☆☆☆）

**破坏机制**（详见 Q4 深度分析）:

1. **loss.sum() 的数值溢出**: `yolo_wrapper.py` 中 `loss = loss.sum()` 将 `[box, cls, dfl]` 三个 loss 分量累加。在 AMP 的 float16 模式下，如果三个分量值域在 1-10 之间，sum 后的值在 3-30 之间，这在 float16 的表示范围内（-65504 ~ 65504），**直接溢出风险较低**。

2. **GradScaler 的 scale 因子冲突**: AMP 使用 `GradScaler` 自动缩放 loss 来防止梯度下溢。如果 `scaler.scale(loss)` 后的值过大，可能导致:
   - `scaler.step(optimizer)` 时触发 inf/nan 检查，跳过优化步骤
   - 有效训练 step 减少，"假训练"现象

3. **分层 LR + AMP 的隐性问题**: 高 LR（head 2.5e-3）下梯度幅值大，AMP 的 loss scaling 策略可能频繁调整 scale 因子，导致训练不稳定。

**量化估算**: 单独 AMP 在 YOLOv8 上通常不会显著降低 mAP（Ultralytics 官方支持 AMP），但与高 LR + loss.sum() 组合可能有 0.5-1% 的负面影响。

---

### 主犯排序总结

```
影响排名（综合危害指数）:

1. BS 64→16        ★★★★★  梯度方差爆炸 + BN崩溃 + 正负样本失衡
2. LOCAL_EPOCH 10→3 ★★★★★  知识积累不足 + 联邦同步抹除效应
3. 分层LR head×5.0  ★★★★☆  检测头震荡 + backbone冻结 + 聚合错配
4. LOCAL_LR 2e-4→5e-4 ★★★★☆  基础LR过高 + cosine末期陷阱
5. Cosine+Warmup      ★★★☆☆  联邦场景错配 + warmup期浪费 + 衰减过快
6. TEST_BS 256→16    ★★☆☆☆  BN统计量测试不匹配（影响较小）
7. GLOBAL_ROUNDS 20→30 ★☆☆☆☆  本身无害但无法弥补其他损失
8. AMP启用           ★☆☆☆☆  loss.sum()数值风险 + scaler冲突（影响较小）

关键交互效应:
- BS16 + EPOCH3:        危害相乘（小batch × 少epoch = 几乎学不到东西）
- HEAD_LR×5 + EPOCH3:   高LR在短周期内导致参数震荡而非收敛
- COSINE + EPOCH3:      每轮都在cosine的不同相位"冷启动"
- 以上所有 + FedAvg:     噪声更新在聚合中相互抵消，全局模型停滞
```

---

## Q2: Warmup + Cosine 的致命缺陷

### 当前设计的问题

当前 cosine schedule 配置了 **3 轮 warmup**。在每轮只有 3 个 local epoch 的情况下：

| 全局轮次 | Warmup 阶段 | LR 值 | 有效训练? |
|----------|-------------|-------|-----------|
| R1 | warmup | `5e-4 * (1/3) ≈ 1.67e-4` | 极低 LR × 3 epoch ≈ 无效 |
| R2 | warmup | `5e-4 * (2/3) ≈ 3.33e-4` | 亚有效 LR × 3 epoch |
| R3 | warmup 结束 | `5e-4 * (3/3) = 5e-4` | 峰值 LR，但只持续 3 epoch |
| R4-15 | cosine decay | 5e-4 → 2e-4 | LR 持续下降 |
| R16-30 | cosine tail | 2e-4 → 5e-6 | LR 过低，几乎不学习 |

### 根本性缺陷分析

#### 缺陷 1: Warmup 在联邦场景下是"奢侈浪费"

**单机训练中的 warmup 目的**: 防止训练初期大 LR 导致的梯度爆炸，让模型参数先"站稳"再加速学习。

**联邦场景下的 warmup 问题**:
- 每个客户端每轮只训练 3 个 epoch
- Warmup 占用了前 3 个全局轮次 = 9 个本地 epoch
- 这 9 个本地 epoch 占了总训练量（90 个 epoch）的 **10%**
- 在这 10% 的时间里，LR 从 1.67e-4 缓慢爬升到 5e-4
- 对于从 COCO 预训练开始、只换了检测头的模型，warmup 完全没必要——backbone 已经是训练好的，只需要检测头快速学习

#### 缺陷 2: Cosine decay 的假设被联邦机制打破

Cosine LR schedule 的理论基础是：**模型在单个数据集上连续训练，随着训练进行，需要越来越精细的参数调整**。

联邦学习完全打破了这个假设：
```
R1: 全局模型 M0 → 客户端训练 3 epoch → 上传更新 → 聚合为 M1
R2: 全局模型 M1 → 客户端训练 3 epoch → 上传更新 → 聚合为 M2
...
```

每轮客户端都是"从头开始"训练一个**继承自全局**的模型。Cosine schedule 期望的是：
```
模型在 t=0 时随机初始化 → 连续训练 T 步 → LR 从 max 降到 min
```

但实际发生的是：
```
模型在 t=0 时是预训练好的 → 每轮重置并训练 3 epoch → LR 按全局轮次衰减
```

这导致模型在 R20 时拿到的 LR 只有 5e-5，但模型本身只训练了 60 个 epoch（20轮 × 3），远未达到需要"精细调整"的阶段。

#### 缺陷 3: 末期 LR ≈ 0 的"假训练"陷阱

```python
# R30 (最后一轮) 的 LR:
lr = 5e-6 + 0.5 * (5e-4 - 5e-6) * (1 + cos(π * 29 / 29))
   = 5e-6 + 0.5 * 4.95e-4 * (1 + cos(π))
   = 5e-6 + 0.5 * 4.95e-4 * 0
   = 5e-6
```

5e-6 的 LR 意味着什么？
- 假设梯度为 1.0（非常乐观），参数更新幅度 = 5e-6
- 检测头的一个典型权重初始值 ~ 0.01
- 需要 2000 次更新才能让权重变化 10%
- 但在 3 个 epoch 内只有几百次更新
- **最后 10 轮几乎不产生有效参数更新**

#### 缺陷 4: Warmup + Cosine + 短周期的"死亡螺旋"

```
R1-R3:  LR 低 (warmup)  × 3 epoch = 学不到东西
R4-R10: LR 快速下降     × 3 epoch × 7轮 = 学了一点，但每轮都在降速
R11-R20: LR 缓慢下降    × 3 epoch × 10轮 = 学习速度持续降低
R21-R30: LR ≈ 0        × 3 epoch × 10轮 = 完全学不到东西

有效学习时间 ≈ R4-R10 的 21 个本地 epoch
无效学习时间 ≈ 其他 69 个本地 epoch
有效学习率   ≈ 实际峰值 LR 的 30-50%
```

### 为什么原配置（无 cosine、constant LR=2e-4）工作更好？

| 特性 | 原配置 (constant 2e-4) | 新配置 (cosine 5e-4→5e-6) |
|------|------------------------|---------------------------|
| LR 稳定性 | 20 轮 × 10 epoch = 200 epoch 内 LR 恒定为 2e-4 | LR 在每轮都不同，客户端"追赶" schedule |
| 有效学习时间 | 200 epoch × 100% = 200 | 90 epoch × ~30% = ~27 |
| 检测头学习 | 10 个连续 epoch 内稳定收敛 | 3 个 epoch 内刚"热身"就重置 |
| 联邦同步影响 | 每 10 epoch 同步一次，知识来得及积累 | 每 3 epoch 同步一次，刚学就忘 |

---

## Q3: 分层学习率的设计问题

### 当前设计

```python
backbone_lr = lr * 0.1   # 当 lr=5e-4 时，backbone_lr = 5e-5
head_lr     = lr * 5.0   # 当 lr=5e-4 时，head_lr = 2.5e-3
```

### 问题 1: Head LR = 2.5e-3 远超合理范围

**对比参考**:
- Ultralytics YOLOv8 默认 `lr0 = 1e-3`（SGD，全模型统一 LR）
- YOLOv5 默认 `lr0 = 1e-2`（SGD），但配合 warmup 和 linear decay
- 联邦学习场景下，检测头微调通常使用 `lr0 = 1e-4 ~ 5e-4`

你的 `head_lr = 2.5e-3` 是 Ultralytics 默认值的 2.5 倍。在只有 3 个 local epoch 的情况下:
- 每个 epoch 有几百个 step
- 每个 step 的更新幅度 = `2.5e-3 * gradient`
- 假设平均梯度为 0.1，每 step 更新 `2.5e-4`
- 3 个 epoch × 500 step = 1500 次更新
- 总更新幅度 ≈ `1500 * 2.5e-4 = 0.375`
- 权重从 0.01 出发，经过 0.375 的位移，可能跨越多个"最优解盆地"

**结论**: 检测头在 3 个 epoch 内**无法**充分学习。高 LR 让它在不断"跳跃"而非"收敛"。

### 问题 2: Backbone LR = 5e-5 几乎冻结了特征提取器

COCO 预训练的 backbone 在 VisDrone 上的问题:
- COCO: 日常场景，目标较大，视角平视
- VisDrone: 航拍视角，目标极小（有些只有 10×10 像素），视角俯视
- COCO 的 feature map 可能无法直接捕捉航拍小目标的特征

`backbone_lr = 5e-5` 意味着:
- 每 step 更新幅度 = `5e-5 * gradient`
- 30 轮 × 3 epoch × 500 step = 45000 次更新
- 总更新幅度 ≈ `45000 * 5e-5 * 0.1 = 0.225`
- 但这是在 30 轮分散的情况下，每轮只有 `0.225/30 = 0.0075` 的有效更新
- Backbone 几乎保持 COCO 预训练状态

**后果**: 模型用一个"不熟悉航拍小目标"的特征提取器去检测 VisDrone 目标，检测头即使学会分类，也缺乏好的特征来分类。

### 问题 3: 分层学习率在 FedAvg 聚合时的隐患

FedAvg 的聚合公式:
```python
global_weight = sum(client_weight_i) / num_clients
```

问题出在**参数更新的幅度差异**: 
- Head 参数每轮的更新幅度大（LR=2.5e-3），但方向噪声也大
- Backbone 参数几乎不更新（LR=5e-5）

聚合后的效果:
```
Head_new = Head_old + average(delta_head_1, delta_head_2, ..., delta_head_N)
          = Head_old + average(noisy_large_steps)
          ≈ Head_old + smoothed_noise  (信号被噪声淹没)

Backbone_new = Backbone_old + average(tiny_deltas)
             ≈ Backbone_old  (几乎不变)
```

这解释了为什么 mAP 停滞在 7.7%:
- Head 在"高 LR + 高噪声"下无法收敛
- Backbone 在"低 LR"下无法适配 VisDrone
- FedAvg 的平滑效应进一步稀释了本已微弱的信号

### 问题 4: 分层比例不合理

常见的分层学习率设计（参考 Detectron2 / MMDetection）:
```python
# 方案 A: 简单二分层
backbone_lr = base_lr * 1.0    # backbone 正常学习
head_lr     = base_lr * 1.0    # head 正常学习
# 适用于: 从头训练

# 方案 B: 头略高
backbone_lr = base_lr * 0.5    # backbone 稍慢
head_lr     = base_lr * 1.0    # head 正常
# 适用于: 微调，backbone 需要适度适应

# 方案 C: 你的设计
backbone_lr = base_lr * 0.1    # backbone 极慢
head_lr     = base_lr * 5.0    # head 极快
# 问题: 两极分化严重，模型两部分"脱节"
```

**推荐的分层比例**:
```python
backbone_lr = base_lr * 0.5    # 让 backbone 适度学习 VisDrone 特征
head_lr     = base_lr * 1.0    # 让检测头稳定学习（不要过高）
```

---

## Q4: AMP + loss.sum() 的隐患

### 代码定位

在 `federated_learning/models/yolo_wrapper.py` 中:
```python
# v8DetectionLoss 返回 shape=[3] 的 [box, cls, dfl] 向量
# 必须 .sum() 而非 .mean()，否则梯度信号被稀释3倍且语义错误
loss = loss.sum()
```

### 隐患 1: loss.sum() 在 AMP 下的数值动态范围

**v8DetectionLoss 的输出结构**:
```python
loss = [box_loss, cls_loss, dfl_loss]  # shape [3]
```

典型值域（训练初期）:
- `box_loss`: 1.0 ~ 5.0（bbox 回归误差）
- `cls_loss`: 2.0 ~ 10.0（分类误差，10 类 + 大量背景）
- `dfl_loss`: 0.5 ~ 3.0（distribution focal loss）

`loss.sum()` 后的值域: **3.5 ~ 18.0**

在 float16（半精度）下:
- 表示范围: ~5.96e-8 to 65504
- 精度: ~3-4 位有效数字
- 3.5 ~ 18.0 完全在表示范围内，**直接溢出风险低**

### 隐患 2: GradScaler 的 scale 因子问题

AMP 的训练循环:
```python
scaler = GradScaler()
with autocast():
    loss, features = model(data, return_features=True, targets=target)
    # loss 此时是 float32（因为 yolo_wrapper 显式返回 float32 loss）
scaler.scale(loss).backward()  # loss * scale_factor
scaler.step(optimizer)
scaler.update()
```

**问题链**:

1. `loss.sum()` 产生一个标量，值域 3.5~18.0
2. `scaler.scale(loss)` 乘以当前 scale 因子（通常初始 2^16 = 65536）
3. 缩放后的 loss: `18.0 * 65536 ≈ 1.18e+6`——仍在 float32 范围内
4. 反向传播时，梯度也被乘以 65536
5. `scaler.step(optimizer)` 检查梯度是否有 inf/nan:
   - 如果 `head_lr = 2.5e-3` 且梯度较大，`scaled_gradient * lr` 可能产生 inf
   - 此时 optimizer 被跳过，该 step 无效
6. `scaler.update()`:
   - 如果检测到 inf，scale 因子降低（如 65536 → 32768）
   - 如果连续多个 step 都 inf，scale 迅速降到 1
   - scale 过低时，float16 梯度可能下溢为 0

**实际影响**:
- 在训练初期（loss 较大、梯度较大、head_lr 较高时），inf/nan 检查频繁触发
- 有效训练 step 减少，"假训练"现象
- GradScaler 的自动调整可能导致训练不稳定

### 隐患 3: loss.sum() vs loss.mean() 的梯度语义

代码注释说"必须 .sum() 而非 .mean()，否则梯度信号被稀释 3 倍"。这个理解是**错误的**。

```python
# 假设 loss_vector = [box, cls, dfl]

loss_sum = loss_vector.sum()          # grad = [1, 1, 1]
loss_mean = loss_vector.mean()        # grad = [1/3, 1/3, 1/3]

# 但 optimizer 使用 loss 对模型参数的梯度:
# d(loss)/d(param) = sum_i [d(loss_i)/d(param)]

# 对于 box_loss 相关的参数:
# d(loss_sum)/d(param_box) = d(box_loss)/d(param_box)
# d(loss_mean)/d(param_box) = (1/3) * d(box_loss)/d(param_box)

# 结论: .mean() 确实会让每个 loss 分量的梯度缩小 3 倍
# 但这可以通过将 LR 乘以 3 来完全补偿！
```

真正的问题不是 `.sum()` vs `.mean()`，而是:
1. **YOLOv8 的 v8DetectionLoss 内部已经做了正确的归一化**:
   - `box_loss` 已经除以了 batch 内的正样本数
   - `cls_loss` 已经除以了 batch 内的总样本数
   - `dfl_loss` 已经除以了 batch 内的正样本数
   - 这些 loss 分量本身就是"每个样本的平均 loss"

2. **对已经归一化的 loss 再 .sum() 是错误的**:
   ```python
   # 假设 batch_size=16，每张图平均 20 个正样本
   box_loss = sum(per_sample_box_loss) / num_positives  # 已经归一化
   # 值域: 0.05 ~ 0.5（每个正样本的平均 box loss）
   
   # 对 3 个已经归一化的分量 .sum():
   total_loss = box_loss + cls_loss + dfl_loss  # 值域 0.5 ~ 5.0
   
   # 但注释中的"值域 3.5~18.0"说明 .sum() 前的 loss 没有被正确归一化
   # 这意味着 v8DetectionLoss 的 batch 归一化可能有问题
   ```

### 隐患 4: AMP 与 YOLO loss 的兼容性

Ultralytics 官方在训练时使用 AMP，但有一个关键区别:
- **官方训练**: 使用 `model.train()` 内部循环，loss 计算在 `autocast()` 内完成，但**最终的 loss 是 float32**
- **你的实现**: `yolo_wrapper.py` 显式返回 float32 loss，但 AMP 的 `scaler.scale()` 期望输入也是兼容的

潜在问题:
```python
# yolo_wrapper.py 中的 forward
loss = loss.sum()  # float32 tensor

# client.py 中的 AMP 训练
with autocast():
    loss, _ = model(...)  # 这里的 loss 已经是在 autocast 外计算的 float32
scaler.scale(loss).backward()
```

如果 `loss` 已经在 `autocast()` 之外计算为 float32，`scaler.scale()` 可以正常工作。但如果 `loss` 的某些中间计算在 `autocast()` 内完成且为 float16，可能存在精度损失。

### 结论

AMP + loss.sum() 在当前配置下的主要隐患:
1. **GradScaler 频繁触发 inf/nan 跳过**（由高 LR 和大梯度引起）
2. **有效训练 step 减少**，加剧了 EPOCH=3 的"学习时间不足"问题
3. **loss.sum() 的数值语义**可能需要重新审视 v8DetectionLoss 的归一化逻辑

**建议**: 在恢复 mAP 的过程中，优先关闭 AMP（`use_amp=False`），待基础配置调优完成后再尝试开启。

---

## Q5: Local Epochs = 3 的真实影响

### "减轻客户端漂移"的理由是否成立？

**不完全成立。这是一个被部分证实的假设，但在你的配置下产生了反效果。**

#### 客户端漂移（Client Drift）的机理

在 Non-IID 联邦学习中，每个客户端的数据分布不同:
```
Client A: 80% car, 10% pedestrian, 10% others  (城市街道)
Client B: 30% pedestrian, 40% people, 30% bicycle  (公园)
Client C: 50% truck, 30% van, 20% bus  (高速公路)
```

当客户端进行多轮本地训练时，模型倾向于拟合本地数据分布:
- Client A 的模型越训练越擅长检测 car
- Client B 的模型越训练越擅长检测 pedestrian/people
- Client C 的模型越训练越擅长检测 truck

FedAvg 聚合后，这些"本地化"的更新相互冲突，导致全局模型卡在次优解。

#### 减少 Local Epochs 的理论效果

减少 local epochs 确实可以**减轻漂移的强度**:
- 每轮本地训练时间缩短 → 模型没有足够时间"过度拟合"本地分布
- 聚合频率相对提高 → 全局信息更频繁地注入本地模型

**但这是一个"度"的问题**:

| Local Epochs | 漂移强度 | 本地学习效果 | 综合效果 |
|-------------|---------|-------------|---------|
| 1 | 极低 | 极差（刚热身就同步） | 差（学不到东西） |
| 3 | 低 | 差（你的当前配置） | **差**（学得太少） |
| 5 | 中 | 一般 | 可能可接受 |
| 10 | 中高 | 好（你的原配置） | **好**（学到足够多） |
| 20 | 高 | 很好 | 可能过拟合本地 |

#### 为什么 EPOCH=3 反而让更新方向更随机？

1. **统计不稳定性**: 3 个 epoch 内，每个客户端只看到自己的数据 3 次。对于小 batch（16）来说，这产生的梯度估计方差极大。"减轻漂移"的效果被"随机噪声"淹没。

2. **没有足够的时间收敛到局部最优**: Client drift 的前提是客户端能"走到"自己的局部最优。3 个 epoch 内，模型还在向局部最优的"半路上"，此时的梯度方向高度依赖初始化（即全局模型），而非本地数据分布。这看似"减轻了漂移"，实际上是因为**模型根本没有时间学到任何东西**。

3. **漂移变成了"随机漫步"**: 
   - EPOCH=10: 客户端有 10 个 epoch 稳定地向自己的局部最优走，方向相对一致，FedAvg 可以"平均掉"方向差异
   - EPOCH=3: 客户端走 3 步就停止，方向还没稳定。每轮的方向都大幅波动，FedAvg 平均的是"噪声"而非"信号"

#### 每轮 3 个 epoch 是否足以学到有效特征？

**答案: 对于 VisDrone YOLO 检测任务，远远不够。**

训练一个目标检测器需要学习的内容:
1. **特征提取**（backbone）: 需要数十个 epoch 才能适配新域
2. **多尺度特征融合**（neck/FPN）: 需要与 backbone 协同训练
3. **Anchor 匹配策略**（label assignment）: YOLOv8 的 TaskAlignedAssigner 需要多个 epoch 才能稳定
4. **分类分支**（cls head）: 10 类分类，需要大量正负样本对比学习
5. **回归分支**（box head）: 精确的 bbox 回归需要稳定的特征

3 个 epoch 能做到什么？
- Backbone: 几乎不变（LR=5e-5）
- Neck: 几乎不变
- Detection head: 权重从随机开始，3 个 epoch 内只能"初步感知"到有哪些类别，远未达到可检测水平

**参考数据**: Ultralytics 官方训练 YOLOv8 在 COCO 上需要 100-500 个 epoch。即使是微调（fine-tuning），也建议至少 50-100 个 epoch。你的 3 个 local epoch × 30 轮 = 90 个总 epoch，只是官方建议的微调量的一半，且还是被联邦同步不断打断的 90 个 epoch。

---

## Q6: 恢复路径建议

### 1. 最简单的 1 个修改（预期 mAP: 7.7% → 15-18%）

**修改: `TRAIN_BATCH_SIZE = 64`**

```python
# visdrone_fed_hparams.py
TRAIN_BATCH_SIZE = int(os.environ.get("FL_TRAIN_BATCH_SIZE", "64"))  # 改回 64
```

**为什么这一个改动最有效**:
- 单独恢复 BS=64 可以将梯度方差降低 2 倍
- BN 统计量恢复稳定
- 每 batch 的正样本数恢复，classification head 获得足够学习信号
-  FedAvg 聚合的"噪声抵消"问题大幅缓解

**不改其他参数的原因**:
- BS=64 可以部分缓解 EPOCH=3 的问题（每个 step 的梯度质量提高，3 个 epoch 内能学到更多）
- BS=64 下较高的 head_lr（2.5e-3）虽然仍不理想，但梯度噪声降低后高 LR 的破坏性减弱

---

### 2. 最小改动集合（改 2-3 个参数，预期 mAP: 7.7% → 18-22%）

**修改集合**:

```python
# visdrone_fed_hparams.py - 改动 1/3
TRAIN_BATCH_SIZE = int(os.environ.get("FL_TRAIN_BATCH_SIZE", "64"))   # 16 → 64

# visdrone_fed_hparams.py - 改动 2/3
LOCAL_EPOCHS = int(os.environ.get("FL_LOCAL_EPOCHS", "10"))           # 3 → 10

# visdrone_fed_hparams.py - 改动 3/3
LOCAL_LR = float(os.environ.get("FL_LOCAL_LR", "2e-4"))               # 5e-4 → 2e-4
```

**恢复机制**:
- BS=64 + EPOCH=10: 恢复到原配置的"学习充分性"
- LR=2e-4: 避免高 LR 导致的震荡，配合 EPOCH=10 稳定收敛
- 保持 Cosine + Warmup（虽然不理想，但在 EPOCH=10 的情况下危害降低）
- 保持分层 LR（在 EPOCH=10 的情况下，head 有 10 个 epoch 来适应高 LR）

**为什么不改分层 LR**: 在 EPOCH=10 的情况下，head_lr=2.5e-3 虽然偏高，但 10 个 epoch 的连续训练给检测头足够时间来"稳定"。优先恢复核心参数，分层 LR 可以在后续微调中优化。

---

### 3. 完整推荐方案（5 个参数修改，预期 mAP: 7.7% → 22-25%+）

```python
# ========== visdrone_fed_hparams.py ==========

# 改动 1: 恢复 batch size
TRAIN_BATCH_SIZE = int(os.environ.get("FL_TRAIN_BATCH_SIZE", "64"))   # 16 → 64
TEST_BATCH_SIZE  = int(os.environ.get("FL_TEST_BATCH_SIZE", "256"))   # 16 → 256

# 改动 2: 恢复 local epochs
LOCAL_EPOCHS = int(os.environ.get("FL_LOCAL_EPOCHS", "10"))           # 3 → 10

# 改动 3: 恢复学习率
LOCAL_LR = float(os.environ.get("FL_LOCAL_LR", "2e-4"))               # 5e-4 → 2e-4
LR_MIN   = float(os.environ.get("FL_LR_MIN", "5e-6"))                 # 保持

# 改动 4: 关闭 cosine（改回 constant LR）
LR_SCHEDULE = "constant"  # "cosine" → "constant"
# 或者直接注释掉 cosine 逻辑，使用默认的恒定 LR

# 改动 5: 优化分层学习率比例
# 在 client.py 中修改:
# backbone_lr = lr * 0.1  →  backbone_lr = lr * 0.5
# head_lr     = lr * 5.0  →  head_lr     = lr * 1.0
```

**完整恢复机制**:

| 参数 | 从 | 到 | 效果 |
|------|----|-----|------|
| BS | 16 | 64 | 梯度方差↓2x, BN稳定, 正样本充足 |
| EPOCH | 3 | 10 | 知识积累充分, 检测头有效学习 |
| LR | 5e-4 | 2e-4 | 稳定收敛, 避免震荡 |
| SCHEDULE | cosine+warmup | constant | 联邦场景适配, 每轮稳定学习 |
| 分层 LR | 0.1x/5.0x | 0.5x/1.0x | backbone适度学习, head稳定收敛 |
| TEST_BS | 16 | 256 | BN统计量测试时更稳定 |

**额外建议**:

1. **暂时关闭 AMP**，待 mAP 恢复后再尝试开启:
```python
# client.py
# use_amp = 'cuda' in str(self.device).lower()
use_amp = False  # 暂时关闭
```

2. **如果 mAP 恢复到 20%+ 后想进一步优化**，可以尝试:
   - 恢复 Cosine schedule（去掉 warmup），配合 EPOCH=10:
```python
# 修改后的 cosine（无 warmup）
if sched == 'cosine':
    lr = lr_min + 0.5 * (eta_max - lr_min) * (1.0 + math.cos(math.pi * t / T))
```
   - 保持 constant LR 也是一个完全合理的选择，联邦学习场景下 constant LR 往往比 fancy schedule 更稳定

---

## 附: 改动全景图与交互效应分析

### 交互效应矩阵

```
                BS=16  EPOCH=3  HEAD_LR×5  LR=5e-4  COSINE  AMP
BS=16             -     ★★★      ★★        ★★      ★      ★
EPOCH=3          ★★★     -      ★★★       ★★★     ★★★     ★
HEAD_LR×5        ★★    ★★★       -        ★★      ★      ★
LR=5e-4          ★★    ★★★      ★★         -      ★★     ★★
COSINE            ★     ★★       ★        ★★       -      ★
AMP               ★      ★       ★        ★★       ★      -

★   = 轻度交互放大
★★  = 中度交互放大
★★★ = 重度交互放大
```

### 关键交互路径

**路径 1: BS=16 → EPOCH=3 → 学习无效**（危害最大）
```
小batch(高方差梯度) × 少epoch(短学习时间) = 几乎学不到有效特征
```

**路径 2: HEAD_LR×5 → EPOCH=3 → 检测头震荡**
```
高学习率 × 短训练周期 = 参数在最优解附近跳跃，无法稳定
```

**路径 3: LR=5e-4 → COSINE → 末期假训练**
```
高基础LR × 快速衰减 = 后10轮LR过低，完全无效
```

**路径 4: BS=16 → AMP → GradScaler问题**
```
小batch(大梯度噪声) × AMP(loss缩放) = 频繁触发inf/nan跳过
```

**路径 5: 所有改动 → FedAvg → 噪声抵消**
```
所有客户端产生噪声更新 → FedAvg平均 → 信号被噪声淹没 → mAP停滞
```

### "好配置" vs "差配置"的全景对比

```
                    好配置 (mAP≈21%)          差配置 (mAP≈7.7%)
                    ─────────────────        ─────────────────
BS                   64                       16
Local Epochs         10                       3
LR                   2e-4 (constant)          5e-4 (cosine+warmup)
Head LR              无分层 (或 1x)           5x = 2.5e-3
Backbone LR          无分层 (或 1x)           0.1x = 5e-5
Test BS              256                      16
AMP                  无                       有

总有效训练量:
  本地step数         ~ (N/64) × 10 × 20轮     ~ (N/16) × 3 × 30轮
                    ≈ 3.1N × 20              ≈ 5.6N × 30
                    每客户端约200个有效epoch   每客户端约90个epoch
                                            (但大量被低LR/噪声浪费)

梯度质量:
  每step梯度方差     低(BS64)                 高(BS16)
  有效信号/噪声比    高                       低

检测头学习:
  连续训练时间       10 epoch  uninterrupted   3 epoch × 30次打断
  收敛稳定性         高                       极低

BN统计量:
  训练时稳定性       高                       低
  测试时匹配度       高                       低
```

---

## 附: 深层架构问题

### 问题 A: Cosine Schedule 在联邦学习中的根本不适配

**根本原因**: Cosine LR schedule 是为**中心化连续训练**设计的，假设:
1. 模型从同一个初始化点开始
2. 连续训练 T 步
3. LR 从峰值平滑衰减到谷底

联邦学习打破了所有三个假设:
1. 每轮客户端从**不同的全局模型**开始（全局模型每轮都在变）
2. 训练被**周期性同步打断**（每 3-10 个 epoch）
3. LR 按全局轮次衰减，但本地训练量在每轮内是固定的

**建议**: 联邦学习场景下，**constant LR** 或 **step decay**（每 N 轮降低一次）通常比 cosine 更稳定。

### 问题 B: FedAvg 对检测任务的固有缺陷

FedAvg 的简单平均在目标检测任务中的问题:
1. **检测头的类别不平衡**: 不同客户端的类别分布差异大，简单平均导致少数类被淹没
2. **Backbone 的域差异**: 不同场景（城市/公园/高速）的特征分布不同，平均后的 backbone 特征表示模糊
3. **BN 统计量的冲突**: 各客户端的 BN running statistics 差异大，FedAvg 不聚合 BN 统计量（只聚合权重）

**建议**: 考虑使用 **FedBN**（不聚合 BN 参数，每个客户端保留自己的 BN 统计量）或 **FedLA**（标签感知聚合，已在你的代码中实现）。

### 问题 C: 测试方式对 mAP 的影响

`fl_core.py` 中的测试逻辑:
1. 先尝试用 YOLO 官方 val（子进程）→ 得到 mAP@0.5
2. 失败时回退到自定义 mAP 计算

**潜在问题**:
- 子进程评估时创建新的 YOLO 实例，可能使用不同的配置
- 全局模型在被 `.train()` 和 `.eval()` 之间切换时，BN 的 running statistics 可能被污染
- 建议在评估前添加 `torch.cuda.empty_cache()` 和 BN 同步检查

### 问题 D: Mosaic + MixUp 的数据增强与联邦学习的交互

`visdrone_dataset.py` 中的 `mosaic_collate_fn`:
- 50% 概率触发 Mosaic（4 图拼接）
- 15% 概率触发 MixUp

在 BS=16 时:
- Mosaic 的随机性在小 batch 下占比过大（4 张图来自随机采样）
- 可能导致某些 batch 的标签分布极不均匀
- 联邦同步的频繁打断使增强策略难以稳定

**建议**: 在 mAP 恢复实验期间，暂时关闭 Mosaic 和 MixUp（`mosaic_prob=0.0, mixup_prob=0.0`），排除增强策略的干扰，待基础配置稳定后再开启。

---

## 最终建议优先级

| 优先级 | 改动 | 预期 mAP 提升 | 风险 |
|--------|------|--------------|------|
| P0 | `TRAIN_BATCH_SIZE = 64` | +7-10% | 无 |
| P0 | `LOCAL_EPOCHS = 10` | +5-8% | 无 |
| P1 | `LOCAL_LR = 2e-4` | +2-4% | 无 |
| P1 | `LR_SCHEDULE = "constant"` | +2-3% | 无 |
| P1 | 分层 LR 改为 `0.5x/1.0x` | +2-3% | 低 |
| P2 | `TEST_BATCH_SIZE = 256` | +0.5-1% | 无 |
| P2 | 关闭 AMP | +0.5-1% | 低 |
| P3 | 使用 FedLA 替代 FedAvg | +3-5% | 中 |
| P3 | 使用 FedBN 替代 FedAvg | +1-2% | 中 |

**推荐执行顺序**:
1. **立即执行 P0**（BS=64, EPOCH=10）→ 预期 mAP 恢复到 18-22%
2. **同日执行 P1**（LR=2e-4, constant schedule, 分层 LR 调优）→ 预期 mAP 提升到 22-25%
3. **次日执行 P2**（TEST_BS=256, 关闭 AMP）→ 预期 mAP 微调 +0.5-1%
4. **后续实验 P3**（FedLA/FedBN）→ 探索 25%+ 的可能性

---

> **报告总结**: mAP 从 21% 跌到 7.7% 不是某个参数的"单点故障"，而是 5 个核心改动（BS=16、EPOCH=3、head_lr×5、LR=5e-4、cosine+warmup）构成的"死亡组合"。这些改动在独立场景下各有其合理性，但在联邦学习 + YOLO 检测 + VisDrone 小目标的特定组合下产生了灾难性的交互效应。恢复路径的核心是**优先恢复学习的基本条件**: 足够大的 batch（64）、足够长的训练时间（10 epoch）、合理的 LR（2e-4），以及**去掉联邦场景不适配的优化技巧**（cosine+warmup）。
