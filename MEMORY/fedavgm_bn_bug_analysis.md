# FedAvgM 导致 mAP 崩塌根因分析报告

> **日期**: 2026-06-10
> **现象**: R1=0%, R2=3.22%, R3=0%, R4=0%（持续崩塌）
> **正常基线**: R1=7.81%, R2=16.78%, R3=19.50%（无 FedAvgM）

---

## 一、根因结论

**Cursor 的诊断方向正确，但根因定位不够精确。**

真正的原因是：

> **FedAvgM 代码对 BN 的 running_mean/running_var（buffer，非可学习参数）做了动量平滑，导致 BN 统计量持续恶化，running_var 趋近 0，BN 输出恒为 0。**

Cursor 的"Round 1 失败 + BN 持续恶化"是现象描述，不是根因。根因是**代码逻辑错误**：应该只对可学习参数（parameter）做动量平滑，但当前代码对所有 floating_point tensor（包括 BN buffer）都做了。

---

## 二、代码级证据

### 2.1 当前 FedAvgM 代码 (fl_core.py:757-784)

```python
for key in keys_in_both:
    if not (torch.is_tensor(global_weights[key]) and global_weights[key].is_floating_point()):
        continue
    # ... 动量平滑逻辑（对所有 floating_point tensor 执行）
    delta = gw - ow
    buf = SERVER_MOMENTUM * buf + delta
    global_weights[key] = (ow + buf).to(gw_dtype)
```

**问题**：这个循环对所有 `is_floating_point()` 的 tensor 都做动量平滑，**没有区分可学习参数和 BN 统计量**。

### 2.2 BN 层在 state_dict 中的结构

YOLOv8s 的每个卷积层后都有 BN，例如：

```
model.model.0.conv.weight          ← 可学习参数（Conv weight）✅ 应该做动量
model.model.0.conv.bn.weight       ← 可学习参数（gamma）✅ 应该做动量
model.model.0.conv.bn.bias         ← 可学习参数（beta）✅ 应该做动量
model.model.0.conv.bn.running_mean ← **buffer（统计量）❌ 不应该做动量！**
model.model.0.conv.bn.running_var  ← **buffer（统计量）❌ 不应该做动量！**
model.model.0.conv.bn.num_batches_tracked ← **buffer（计数器）**
```

当前代码对 `running_mean` 和 `running_var` 都做了 `buf = 0.9 * buf + delta`，这是**致命的**。

### 2.3 FedAIoT 为什么没有这个问题

FedAIoT (`aggregators/base.py:34-45`)：

```python
named_params = dict(self.global_model.cpu().named_parameters())
for parameter_name, parameter_n_plus_1 in params_n_plus_1.items():
    if parameter_name in named_params.keys():
        parameter_n = named_params[parameter_name]
        parameter_n.grad = parameter_n.data - parameter_n_plus_1.data
```

**关键**：`named_parameters()` **不包含** `running_mean`/`running_var`。这些是 `buffers`（通过 `named_buffers()` 访问），不是 `parameters`。所以 FedAIoT **自动跳过**了 BN 统计量。

### 2.4 DPFLA 的问题

DPFLA 用 `state_dict()` 遍历，`state_dict()` **包含** parameters + buffers。所以 BN 的 `running_mean`/`running_var` 被错误地进入了动量平滑逻辑。

---

## 三、动量平滑如何破坏 BN 统计量

### 数值推演示例

假设某 BN 层的 `running_var`（真实值约 ~1.0）：

| Round | 事件 | running_var | momentum_buf |
|-------|------|-------------|--------------|
| 初始 | COCO 预训练 | **1.0** | — |
| R1 | 客户端训练 → 聚合 | 0.8 | — |
| R1 | FedAvgM 失败（设备错误） | 0.8 | **未初始化** |
| R2 | 客户端训练 → 聚合 | 0.75 | — |
| R2 | FedAvgM 生效：`delta=0.75-0.8=-0.05` | **0.75** | **-0.05** |
| R3 | 客户端训练 → 聚合 | 0.70 | — |
| R3 | FedAvgM：`delta=0.70-0.75=-0.05`, `buf=0.9*(-0.05)+(-0.05)=-0.095` | **0.655** | **-0.095** |
| R4 | 客户端训练 → 聚合 | 0.65 | — |
| R4 | FedAvgM：`buf=0.9*(-0.095)+(-0.05)=-0.1355` | **0.614** | **-0.1355** |
| ... | ... | ... | ... |
| R10 | 持续累积 | **~0.3** | **~0.4** |

**结果**：running_var 从 1.0 被持续压缩到 ~0.3。

BN 前向公式：`y = (x - running_mean) / sqrt(running_var + eps) * gamma + beta`

当 `running_var → 0`：`sqrt(0 + 1e-5) ≈ 0.003`，除以这个数导致输出**爆炸到极大值或 NaN**，模型完全失效。

### 为什么 R2 还有 3.22%

R2 是 FedAvgM **首次生效**的轮次：
- R1: FedAvgM 失败 → BN 只被普通 FedAvg 平均了一次，还部分有效
- R2: `momentum_buf` 初始化为 `delta`（第一轮动量 = 0 + delta），修正幅度还不大
- R3: `buf = 0.9 * 前一轮buf + 新delta`，动量累积效应显现，BN 彻底崩溃

### 为什么 loss 还在下降但 mAP=0

- **训练阶段**：`model.train()` → BN 使用当前 batch 的统计量（不是 running_mean/var），所以训练还能进行，loss 还能下降
- **评估阶段**：`model.eval()` → BN 使用 running_mean/var（已经被破坏），输出全为 0/NaN → mAP=0

这就是"loss 下降但 mAP=0"的核心原因。

---

## 四、Cursor 诊断评估

| Cursor 的说法 | 评估 | 实际 |
|-------------|------|------|
| "Round 1 FedAvgM 失败 + BN 状态持续恶化" | 部分正确 | 是现象，不是根因 |
| "R2 FedAvgM 对 BN running_mean/var 做了不应该做的动量平滑" | ✅ **正确** | 这是核心问题 |
| "R3 BN running_var 趋近 0 → mAP=0" | ✅ **正确** | 结果描述准确 |
| "Round 1 的 BN 状态混乱是根因" | ❌ **错误** | 只是触发条件 |

**Cursor 的 4 个假设评估**：

| 假设 | 评估 |
|------|------|
| A: BN 状态灾难性恶化 | ✅ 正确，但需补充"动量平滑作用于 buffer" |
| B: 子进程 val 时 strict=False 破坏 BN | ❌ 无关，子进程 val 与训练轮次无关 |
| C: fc2.weight/fc2.bias 被跳过 | ❌ 无关，这两个 key 不影响 BN |
| D: model.train() 导致 BN 使用错误 momentum | ❌ 相反，model.train() 用的是 batch 统计量，是"救命"的 |

---

## 五、修复方案

### 核心修复：只对有 grad 的可学习参数做动量平滑

**推荐方案**：用 `named_parameters()` 过滤（与 FedAIoT 逻辑一致）

```python
elif rule == 'fedavg':
    cur_time = time.time()
    old_weights = copy.deepcopy(global_weights)
    global_weights = average_weights(
        local_weights,
        [1 for i in range(len(local_weights))],
        float16_floats=False,
    )
    try:
        SERVER_MOMENTUM = 0.9
        
        # === 新增：获取可学习参数名集合 ===
        learnable_keys = {name for name, p in simulation_model.named_parameters() if p.requires_grad}
        
        keys_in_both = [k for k in global_weights if k in old_weights]
        keys_missing_from_old = [k for k in global_weights if k not in old_weights]
        if keys_missing_from_old:
            logger.warning(f'FedAvgM: 以下 key 在旧权重中不存在，跳过: {keys_missing_from_old[:5]}...')
        
        for key in keys_in_both:
            # === 新增：跳过 BN buffer 等非可学习参数 ===
            if key not in learnable_keys:
                continue
            
            if not (torch.is_tensor(global_weights[key]) and global_weights[key].is_floating_point()):
                continue
            gw_dtype = global_weights[key].dtype
            if key not in self._server_momentum_buf:
                self._server_momentum_buf[key] = torch.zeros_like(global_weights[key])
            gw = global_weights[key].float()
            ow = old_weights[key].float()
            buf = self._server_momentum_buf[key].float()
            delta = gw - ow
            buf = SERVER_MOMENTUM * buf + delta
            self._server_momentum_buf[key] = buf.to(gw_dtype)
            global_weights[key] = (ow + buf).to(gw_dtype)
        logger.info('FedAvgM 动量平滑完成')
    except Exception as e:
        logger.error(f'FedAvgM 动量平滑失败: {e}')
    cpu_runtimes.append(time.time() - cur_time)
```

### 备选方案：显式跳过 BN 统计量

```python
for key in keys_in_both:
    # 显式跳过 BN buffer
    if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
        continue
    # ... 动量平滑
```

**不推荐**，因为：
- 可能有其他类型的 buffer 也需要跳过
- `named_parameters()` 过滤更通用、更安全

---

## 六、验证方法

### 修复后验证步骤

1. **代码修改**：按上述方案修改 `fl_core.py`
2. **清理 momentum_buf**：删除或重置 `_server_momentum_buf`（因为之前已经被 BN 统计量污染了）
3. **运行验证**：

```bash
python run-test/visdrone/run_no_attack_baseline.py
```

4. **判断标准**：

| 轮次 | 预期 mAP | 说明 |
|------|---------|------|
| R1 | > 5% | 正常冷启动范围 |
| R2 | > 12% | 持续上升 |
| R3 | > 18% | 超过修复前基线 |
| R5 | > 25% | 接近或超过正常基线 |
| R10 | > 28% | 验证 FedAvgM 有效 |

5. **对照实验**（可选）：同时跑一个**完全关闭 FedAvgM** 的基线，确保修复后的 FedAvgM 确实比纯 FedAvg 好：

```python
# 临时关闭 FedAvgM 的方法：注释掉动量平滑循环
# for key in keys_in_both:
#     if key not in learnable_keys:
#         continue
#     ...
```

---

## 七、额外建议

### 7.1 清理已污染的 momentum_buf

当前的 `self._server_momentum_buf` 已经被 BN 统计量污染了，修复后需要清理：

```python
# 在 __init__ 中或修复后首次运行时
self._server_momentum_buf = {}  # 清空，重新初始化
```

### 7.2 添加 BN 统计量监控日志

为了未来排查类似问题，建议添加 BN 统计量监控：

```python
# 在 FedAvgM 循环中（调试用，确认 running_var 正常）
if 'running_var' in key:
    rv = global_weights[key]
    logger.debug(f'BN running_var range: [{rv.min():.4f}, {rv.max():.4f}, mean={rv.mean():.4f}]')
```

### 7.3 修复后如果仍有问题

如果修复 BN 过滤后 mAP 仍不正常，按优先级检查：

1. **E=1 是否太小**：当前 `LOCAL_EPOCHS=1`，总训练量 = 50轮 × 1 = 50 epoch。原好配置是 20轮 × 10 = 200 epoch。如果 50 轮后 mAP < 20%，尝试 `E=3, R=100`。
2. **LR=0.01 是否过高**：如果 R3 出现 loss NaN，回退到 `LR=0.005`。
3. **constant 调度是否合适**：如果 R20 后 mAP 停滞，加 cosine 调度。

---

## 八、总结

| 问题 | 答案 |
|------|------|
| 根因是什么？ | FedAvgM 对 BN running_mean/running_var 做了动量平滑 |
| Cursor 的诊断对吗？ | 方向正确（BN 恶化），但根因应精确为"动量作用于 buffer" |
| 修复要改几行？ | **3 行**：加 `learnable_keys` 定义 + `if key not in learnable_keys: continue` |
| 修复后预期？ | R10 > 28%，最终 32%+ |
| 是否需要回退全部改动？ | **不需要**，只修复 FedAvgM 的过滤逻辑，梯度裁剪和参数调整保留 |

---

## 九、修复状态（2026-06-10 傍晚）

**✅ 已实施**（commit `87e3f7a` 之后）：

| 文件 | 修改 |
|------|------|
| `fl_core.py:764` | `learnable_keys = {name for name, _ in simulation_model.named_parameters() if _.requires_grad}` |
| `fl_core.py:770-772` | `if key not in learnable_keys: continue` |
| `fl_core.py:786-793` | BN running_var debug 日志监控 |
| `fl_core.py:764` | 删除未使用的 `dev = self.device` |
| `MEMORY/01_ChangeLog.md` | 追加修复记录 |
