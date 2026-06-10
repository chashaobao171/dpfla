# 给 Kimi Agent 的提示词：彻查 Round 2-3 mAP 崩塌问题

## 任务背景

在 `run_no_attack_baseline_20260610_1901.log`（50 轮，BS=64，EPOCH=1，FedAvgM=0.9）中，出现了诡异的 mAP 崩塌现象：

| Round | mAP@0.5 | Loss   | Avg Client Loss |
|-------|---------|--------|-----------------|
| R0 (初始) | 0%     | 2981   | -               |
| R1         | 0%     | 1427   | 706             |
| R2         | **3.22%** | 1103  | 416             |
| R3         | **0%** | 1264   | 347             |
| R4         | **0%** | 1111   | 347             |

而正常的 FedAvg 基线（`run_no_attack_baseline_20260608_0755.log`，**无 FedAvgM**）走势：
- R1: 7.81%, R2: 16.78%, R3: 19.50%, R4: 21.16%（持续收敛）

**关键线索**：问题在引入 FedAvgM 后出现，Round 1 时 FedAvgM 因设备错误（`cuda:0 vs cpu`）在 except 中跳过。

---

## 关键现象

### 1. Round 1 末尾的 FedAvgM 错误
```
2026-06-10 19:03:25.189 | ERROR | ... - FedAvgM 动量平滑失败: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```
- 发生在 `fl_core.py:777`：`delta = gw - ow` → `gw` 在 cuda，`ow` 在 cpu
- 错误被 except 捕获后，`global_weights` 保持为无动量的普通 FedAvg 结果
- 但 `self._server_momentum_buf` 在此轮**未被更新**（except 块内赋值未执行）

### 2. Round 2 时 FedAvgM 首次"似乎"生效
```
2026-06-10 19:05:18.900 | INFO | ... - 聚合完成，开始 FedAvgM 动量平滑...
2026-06-10 19:05:18.901 | WARNING | ... - FedAvgM: 以下 key 在旧权重中不存在，跳过: ['fc2.weight', 'fc2.bias']...
2026-06-10 19:05:18.916 | INFO | ... - FedAvgM 动量平滑完成
```
- `momentum_buf` 初始化（delta 直接作为初始动量）
- **但 `num_batches_tracked` 等 BN buffer 被跳过**（因为它们不是 floating_point tensor）

### 3. Round 2 测试后 mAP=3.22%（很低的正值）
说明模型还在学习，BN 状态部分有效。

### 4. Round 3 开始后 mAP 直接归零
- Precision=0%, Recall=0%
- 之后 Round 3 和 Round 4 都是 mAP=0%
- 但 client loss 仍在下降（347 左右），说明训练本身没问题
- 问题出在**测试/评估环节**

---

## 核心假设（需验证或推翻）

### 假设 A：BN 状态灾难性恶化

正常流程（无 FedAvgM）：
1. 初始模型：COCO 预训练 BN running_mean/var → **非常好**
2. R1 客户端训练：每个客户端本地 BN 做 forward 累积，训练结束后 BN 的 running stats 被客户端本地数据更新（`momentum=0.9` 风格的 EMA）
3. R1 聚合：`average_weights` 对 BN running_mean/var 做**等权平均** → running_mean 被稀释，但 running_var 仍有效
4. R2 评估：BN 使用聚合后的 running stats → **稍差但可用**

FedAvgM 流程：
1. 同上
2. 同上
3. R1 聚合：FedAvgM 失败 → `momentum_buf` 未初始化
4. R2 聚合前：`old_weights` = R1 全局（BN stats = 稀释后的均值）
5. R2 聚合：`global_weights` = FedAvg(R2 客户端) → **BN stats 再次被稀释**
6. R2 FedAvgM：`momentum_buf[key]` 对 BN running_mean/var 做了 `(ow + buf).to(gw_dtype)` → 对均值做了动量更新
7. R2 测试：`load_state_dict` 后 `model.train()` → BN 使用**经过 FedAvgM 动量平滑后的 running_mean**
8. **问题**：如果 `momentum_buf` 对 BN stats 做了错误的缩放或偏移，可能导致 BN 输出恒为 0

**需要验证**：`momentum_buf` 对非权重类 tensor（如 BN running_mean/running_var）做动量平滑是否合理？

### 假设 B：子进程 val 时 BN 状态被 `strict=False load_state_dict` 破坏

每次 val 时：
1. `YOLO(pretrained_yolov8s.pt)` → 加载 COCO 预训练 BN（**非常好**的 running stats）
2. 替换检测头 nc=10
3. `m.model.load_state_dict(ckpt['state_dict'], strict=False)` → COCO BN 被 FL 聚合的 BN 覆盖
4. `model.train()` → 使用覆盖后的 running stats

如果 FL 聚合的 BN running_var 已经变得非常小（趋近 0），那么 `BN(x) = (x - running_mean) / sqrt(running_var + eps) * weight + bias` 会导致输出趋近 0。

**需要验证**：经过多轮 FL 聚合后，BN 的 running_var 是否趋近 0？

### 假设 C：`fc2.weight` 和 `fc2.bias`（DPFLA 兼容层）被跳过导致一致性错乱

Round 2 的 WARNING：
```
FedAvgM: 以下 key 在旧权重中不存在，跳过: ['fc2.weight', 'fc2.bias']...
```
- 这两个 key 在 `old_weights` 中不存在但在 `global_weights` 中存在
- 这意味着它们是 Round 1 末新创建的（DLLayer 首次出现）
- 但关键是：**是否还有其他 key 也存在同样问题？**

### 假设 D：评估时 `model.train()` 导致 BN 使用错误的 momentum

在 `fl_core.py:291`：
```python
model.train()  # ← 切换到训练模式
for param in model.parameters():
    param.requires_grad = True
```

PyTorch 的 BatchNorm：
- `.train()` → 使用 `running_mean/var` 做 EMA 更新（通过当前 batch 的 stats）
- 如果 `num_batches_tracked` 很大但 running_var 很小，每次 batch 的 EMA 更新可能进一步压缩 variance

---

## 待查代码位置

1. **`fl_core.py:747-786`**：`rule == 'fedavg'` 聚合块
   - 检查 `momentum_buf` 对所有 floating_point tensor 都做了平滑
   - **关键**：BN 的 `running_mean` 和 `running_var` 也是 floating_point，**也会被 FedAvgM 平滑**
   - 这可能是问题根源

2. **`fl_core.py:178-291`**：子进程 val 评估
   - 检查 val 前后模型 BN 的 running stats 是否被污染
   - 检查 `model.train()` 后的 BN 行为

3. **`fl_algorithm/fed_avg.py`**：聚合逻辑
   - 检查 `num_batches_tracked = max()` 是否在每次聚合时正确生效
   - 检查 FL 聚合后的 running_var 是否有上界（不能太小）

4. **`models/yolo_wrapper.py:192-204`**：检测头替换
   - 检查新检测头 bias 随机初始化是否在 FL 过程中被破坏

---

## 排查任务清单

- [ ] 对比 Round 1/2/3 的 BN running_mean/running_var 的 L2 norm，量化退化程度
- [ ] 检查 FedAvgM 是否对 BN running_mean/running_var 做了不应该做的动量平滑
- [ ] 检查 `num_batches_tracked = max()` 是否导致 BN 的 running_var 快速收缩
- [ ] 检查子进程 val 时 `strict=False load_state_dict` 是否覆盖了 COCO 预训练的 BN stats
- [ ] 验证正常基线（无 FedAvgM）运行同样 EPOCH=1 时 BN 是否也出现退化（对照实验）
- [ ] 检查 client 训练过程中 BN 是否在 `model.train()` 模式下运行（会更新 running stats）

---

## 参考日志

- **问题日志**：`logs_3/visdrone/run_no_attack_baseline_20260610_1901.log`
- **正常日志**：`logs_3/visdrone/run_no_attack_baseline_20260608_0755.log`（无 FedAvgM）
- **100 轮正常基线**：`logs_3/visdrone/run_no_attack_baseline_20260610_1139.log`（EPOCH=3，有 FedAvgM，**正常**）

---

## 本轮已完成的修改（供参考，勿撤销）

| 文件 | 修改 |
|------|------|
| `fl_core.py:747-786` | FedAvgM 动量平滑（服务端） |
| `fl_algorithm/fed_avg.py:36-41` | `num_batches_tracked = max()` |
| `client.py` | 梯度除以 total_batches |
| `generate_visdrone_yaml.py` | 动态生成 visdrone_temp.yaml |

---

## 输出要求

1. **逐条验证或推翻**上述 4 个假设
2. **给出根因**：到底是什么代码导致 Round 3 mAP 归零
3. **给出修复方案**：需要改哪个文件的哪几行，如何改
4. **给出验证方法**：修复后如何确认问题解决
5. **不要修改代码**：只分析，给出方案
