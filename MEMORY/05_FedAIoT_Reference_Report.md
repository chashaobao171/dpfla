# FedAIoT 深度借鉴报告：FedAIoT vs DPFLA 逐项对比

> **调研日期**: 2026-06-10
> **基于**: `MEMORY/fl_yolo_visdrone_iid_optimization_v2.md`（KimI调查）+ `MEMORY/fl_yolo_visdrone_project_survey.md`（KimI调查）+ `FedAIoT-main/` 源码深度分析
> **目标**: 为 DPFLA 的 VisDrone 基线从 ~24% 提升到 32%+ 提供具体可执行方案

---

## 一、FedAIoT 核心发现总结

### 1.1 FedAIoT 为什么能达到 34.85% mAP@0.5

FedAIoT 能达到 34.85% 的原因**不在于某个单一 tricks，而是一套组合配置**：

| 配置项 | FedAIoT | DPFLA（当前） | 差距 |
|--------|---------|-------------|------|
| 客户端 LR | **0.1** | **2e-4** | **500x** |
| 客户端 Batch Size | **12** | **64** | 5x |
| 服务端优化器 | **SGD, lr=1.0** | **无** | 关键缺失 |
| 服务端动量 | **0.9** | 无 | 关键缺失 |
| CosineAnnealingWarmRestarts | **T_0=10, T_mult=2** | constant | 调度差距 |
| Warmup | **3 轮 warmup** | 无 | 冷启动差距 |
| 全局轮数 | **600** | 30 | 20x |
| 梯度裁剪 | **max_norm=10** | 无 | 稳定性差距 |
| 数据增强 | HSV+几何+MixUp | Mosaic(50%)+MixUp(10%) | 增强组合差距 |
| Momentum (客户端) | **0.937** | 0.9 | 微差 |
| Weight Decay | **5e-4** | 5e-4 | 一致 |

### 1.2 最重要的发现：服务端优化（Server Momentum）

FedAIoT 的 FedAvg 不是简单的参数平均，它在聚合后还有一步：

```python
# aggregators/base.py - FederatedAveraging.step()
# 步骤1: 计算加权平均参数
params_n_plus_1 = self._average_updates(updated_parameter_list, weights)

# 步骤2: 计算"伪梯度" = 旧参数 - 新参数
for parameter_name, parameter_n_plus_1 in params_n_plus_1.items():
    parameter_n = named_params[parameter_name]
    parameter_n.grad = parameter_n.data - parameter_n_plus_1.data

# 步骤3: 服务端 SGD 再走一步（lr=1.0, momentum=0.9）
self.optimizer.step()
```

这相当于 **FedAvgM**（服务端动量），公式上等价于：

```
w_{t+1} = w_t + 1.0 * (avg(w_i_t) - w_t) + momentum * (w_t - w_{t-1})
```

**等价于**：FedAIoT 实际上在做 FedAvgM（服务端版本），这是它收敛更快的核心原因。

DPFLA 目前的 `fed_avg.py` 只做了简单平均，没有这一步服务端优化。

---

## 二、逐项差距详解与借鉴方案

### 2.1 服务端优化（Server Momentum）— 🔴 最高优先级

**现状**：DPFLA 的 `fed_avg.py` 只做 `Σ(w_i * n_i)`，没有服务端 SGD 步骤。

**FedAIoT 做法**：聚合后计算伪梯度 `Δ = w_avg - w_old`，再走 SGD `w_new = w_old + lr * Δ + momentum * prev_grad`。

**借鉴方案**（两种路径）：

#### 路径 A：完整复刻 FedAIoT 服务端优化
在 `fed_avg.py` 的 `FedAvg.average_weights()` 之后，增加一个服务端 SGD step：

```python
# fed_avg.py 修改
def average_weights(self, local_weights, marks=None):
    avg_weights = self._average_weights_impl(local_weights, marks)

    # 新增：服务端优化步骤（FedAIoT 核心）
    if self.server_lr > 0:
        self.server_optimizer.zero_grad()
        named_params = dict(self.global_model.named_parameters())
        with torch.no_grad():
            for name, avg_w in avg_weights.items():
                if name in named_params:
                    named_params[name].grad = named_params[name].float() - avg_w.float()
        self.server_optimizer.step()

    self.global_model.load_state_dict(avg_weights, strict=False)
    return self.global_model.state_dict()
```

同时在 hparams 中新增：
```python
SERVER_LR = float(os.environ.get("FL_SERVER_LR", "1.0"))  # FedAIoT 默认 1.0
SERVER_MOMENTUM = float(os.environ.get("FL_SERVER_MOMENTUM", "0.9"))
```

#### 路径 B：简化版——直接参数更新加动量平滑
不做服务端梯度下降，只对聚合结果做 EMA 平滑（不需要改优化器结构）：

```python
# 简化版：服务端动量平滑
self.server_momentum_buffer = {}
for name, avg_w in avg_weights.items():
    if name in self.server_momentum_buffer:
        smooth = SERVER_MOMENTUM * self.server_momentum_buffer[name] + (1 - SERVER_MOMENTUM) * avg_w
    else:
        smooth = avg_w
    self.server_momentum_buffer[name] = smooth
    avg_weights[name] = smooth
```

**预期收益**：+3~5% mAP（服务端动量是 FedAvg 收敛加速的最有效手段之一）

---

### 2.2 学习率大幅提升 — 🔴 最高优先级

**现状**：DPFLA `LOCAL_LR = 2e-4`

**FedAIoT 做法**：`lr = 0.1`（客户端）

**分析**：在 IID 场景下，client drift 极小，LR 可以大幅提升。FedAIoT 用 0.1 是因为它每轮只有 1 个 local epoch（极端情况），DPFLA 每轮 3-10 个 epoch，LR 应该比 0.1 低一些。

**借鉴方案**：

```python
# visdrone_fed_hparams.py
LOCAL_LR = float(os.environ.get("FL_LOCAL_LR", "0.01"))  # 从 2e-4 → 1e-2（提升50x）
```

建议梯度：
- 保守：0.01（FedAIoT 的 1/10）
- 激进：0.05（FedAIoT 的 1/2）

**预期收益**：+2~4% mAP

---

### 2.3 学习率调度器 — 🟡 高优先级

**现状**：DPFLA 支持 cosine（基于全局轮数），但默认 constant

**FedAIoT 做法**：
- Warmup: 3 轮（lr 从 0 线性上升到目标 lr）
- 主调度：`CosineAnnealingWarmRestarts(T_0=10, T_mult=2)`：每 10 轮重启一次 cos annealing

**借鉴方案**：

```python
# client.py 中 VisDrone 训练时
if use_warmup:
    # Warmup 阶段（FedAIoT: 3 epochs）
    warmup_epochs = 3
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        # CosineAnnealingWarmRestarts: T_0=10, T_mult=2
        T_0 = 10
        T_mult = 2
        current_T = T_0 * (T_mult ** restart_count)
        t = epoch - warmup_epochs - accumulated_T
        lr = lr_min + 0.5 * (base_lr - lr_min) * (1 + math.cos(math.pi * t / current_T))
```

**预期收益**：+1~2% mAP（更稳定的收敛曲线 + 避免平台期）

---

### 2.4 梯度裁剪 — 🟡 高优先级

**现状**：DPFLA 训练循环中没有梯度裁剪

**FedAIoT 做法**：
```python
# trainers/ultralytics_distributed.py
self.scaler.unscale_(self.optimizer)
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
self.scaler.step(self.optimizer)
```

**借鉴方案**：在 `client.py` 的 `optimizer.step()` 之前加梯度裁剪：

```python
# client.py 中
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
optimizer.step()
```

**预期收益**：稳定性提升，配合高 LR 时防止梯度爆炸

---

### 2.5 数据增强对齐 — 🟡 中优先级

**现状**：DPFLA 有 Mosaic(50%) + MixUp(10%)

**FedAIoT 做法**：
```python
YOLO_HYPERPARAMETERS = {
    'mosaic': 1.0,    # 100% (FedAIoT)
    'mixup': 0.0,     # FedAIoT 默认不用 MixUp
    'hsv_h': 0.015,  # HSV 增强
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'translate': 0.1,  # 随机平移
    'scale': 0.5,      # 随机缩放
    'shear': 0.0,
    'fliplr': 0.5,    # 水平翻转
}
```

**分析**：FedAIoT 用 **Mosaic 100%**，但**不用 MixUp**。它的增强更多依赖 HSV + 几何变换 + mosaic 组合。DPFLA 目前 Mosaic 只有 50%，且没有 translate/scale/shear。

**借鉴方案**：
```python
# visdrone_dataset.py
MOSAIC_RATIO = 1.0    # 从 0.5 → 1.0
MIXUP_RATIO = 0.0     # FedAIoT 不依赖 MixUp
HSV_H = 0.015          # 启用 HSV
HSV_S = 0.7
HSV_V = 0.4
TRANSLATE = 0.1        # 新增
SCALE = 0.5            # 新增
```

**预期收益**：+1~2% mAP

---

### 2.6 Momentum 同步 — 🟢 低优先级

**现状**：DPFLA `local_momentum = 0.9`

**FedAIoT 做法**：`momentum = 0.937`

**分析**：0.9 vs 0.937 差距极小，可以忽略或微调。

---

## 三、DPFLA Bug 4-10 修复（次优先级）

来自 KimI 报告的 Bug 4-10 分析：

| Bug | 文件 | 问题 | 对 mAP 影响 | 优先级 |
|-----|------|------|-----------|--------|
| Bug 4 | client.py:408,469 | client_grad 初始化冗余 + `if name in` 静默丢弃 | DPFLA 场景+1% | P1 |
| Bug 5 | fl_core.py:518-520 | return 后死代码 | 0% | P4 |
| Bug 6 | yolo_wrapper.py:592 | 异常兜底 loss=0.5 注入假梯度 | 偶发异常时+0.5~2% | P1 |
| Bug 7 | fed_avg.py | marks=0 静默 fallback | 0% | P3 |
| Bug 8 | DPFLA.py | 正交矩阵重复生成 | 0%（纯性能） | P2 |
| Bug 9 | client.py | global_epoch 命名歧义 | 0% | P4 |
| Bug 10 | sampling.py | 路径验证缺失 | 0% | P3 |

---

## 四、综合行动计划（基于 FedAIoT 核心发现 + KimI Bug 报告）

### 阶段 1：快速验证（1-2天）

不改动代码结构，只调参数。验证 FedAIoT 经验在 DPFLA 上是否有效。

#### 实验 1A：服务端动量 + 高 LR（最关键）
```bash
FL_LOCAL_LR=0.01 \
FL_SERVER_LR=1.0 \
FL_SERVER_MOMENTUM=0.9 \
FL_GLOBAL_ROUNDS=50 \
FL_LOCAL_EPOCHS=3 \
python run-test/visdrone/run_no_attack_baseline.py
```
**判断**：R10 mAP > 28% → 服务端动量是瓶颈，继续

#### 实验 1B：Mosaic 100% + 高 LR
```bash
FL_LOCAL_LR=0.01 \
FL_SERVER_LR=1.0 \
FL_MOSAIC_RATIO=1.0 \
FL_MIXUP_RATIO=0.0 \
python run-test/visdrone/run_no_attack_baseline.py
```
**判断**：与 1A 对比，量化 Mosaic 的贡献

### 阶段 2：代码改造（穿插在阶段1实验间隙）

| 优先级 | 任务 | 文件 | 改动量 |
|--------|------|------|--------|
| P0 | Bug 4：client_grad 修复 | client.py | 小 |
| P0 | Bug 6：loss 异常分类 | yolo_wrapper.py | 小 |
| P0 | 服务端优化（FedAvgM） | fed_avg.py | 中 |
| P1 | 梯度裁剪 | client.py | 小 |
| P1 | WarmupScheduler | client.py | 中 |
| P2 | Bug 8：正交矩阵缓存 | DPFLA.py | 小 |
| P3 | Bug 5：死代码删除 | fl_core.py | trivial |
| P3 | Bug 7：marks fallback 日志 | fed_avg.py | trivial |
| P3 | Bug 10：路径验证 | sampling.py | trivial |

### 阶段 3：完整实验（确认最优配置后）

跑满 100-200 轮，记录完整收敛曲线。

---

## 五、预期收益汇总

| 改动项 | 预期 mAP 提升 | 风险 |
|--------|-------------|------|
| 服务端优化（FedAvgM） | +3~5% | 低 |
| 学习率 2e-4 → 1e-2 | +2~4% | 中（需梯度裁剪） |
| Warmup + CosineRestarts | +1~2% | 低 |
| Mosaic 50% → 100% | +1~2% | 低 |
| Bug 4+6 修复 | +0~2% | 低 |
| **合计** | **+8~14%** | — |

**目标可行性**：FedAIoT 34.85% 基线，DPFLA 当前 ~24%，差距 ~11%。通过上述组合优化，+8% 是保守估计，+11% 是合理目标。

---

## 六、关键风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| LR=0.01 导致 loss NaN | 中 | 先试 0.001，再逐级提升 |
| 服务端优化与现有 DPFLA 冲突 | 低 | 先用简化版 EMA 平滑 |
| Mosaic 100% 显存增加 | 低 | 当前 50% 已可跑，不变 |
| Bug 修复引人新 bug | 低 | 每个修复后快速跑 5 轮验证 |

---

## 七、FedAIoT 代码关键片段参考

### A. 服务端优化（aggregators/base.py）
```python
class FederatedAveraging:
    def __init__(self, global_model, server_optimizer='sgd',
                 server_lr=1e-2, server_momentum=0.9):
        self.optimizer = SGD(
            filter(lambda p: p.requires_grad, global_model.parameters()),
            lr=server_lr, momentum=server_momentum
        )

    def step(self, updated_parameter_list, weights, round_idx):
        self.optimizer.zero_grad()
        params_n_plus_1 = self._average_updates(updated_parameter_list, weights)
        named_params = dict(self.global_model.cpu().named_parameters())
        state_n_plus_1 = self.global_model.cpu().state_dict()
        with torch.no_grad():
            for parameter_name, parameter_n_plus_1 in params_n_plus_1.items():
                if parameter_name in named_params:
                    parameter_n = named_params[parameter_name]
                    parameter_n.grad = parameter_n.data - parameter_n_plus_1.data
                else:
                    state_n_plus_1[parameter_name] = params_n_plus_1[parameter_name]
        self.global_model.load_state_dict(state_n_plus_1)
        self.optimizer.step()
        return self.global_model.cpu().state_dict()
```

### B. WarmupScheduler（utils.py）
```python
class WarmupScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_epochs, scheduler):
        self.warmup_epochs = warmup_epochs
        self.scheduler = scheduler
        super().__init__(optimizer, -1)
        self._last_lr = [0.0] * len(optimizer.param_groups)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = self.last_epoch / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        return self.scheduler.get_last_lr()
```

### C. 梯度裁剪（trainers/ultralytics_distributed.py）
```python
def optimizer_step(self):
    self.scaler.unscale_(self.optimizer)
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    self.optimizer.zero_grad()
```

### D. VisDrone 超参（loaders/visdrone.py）
```python
YOLO_HYPERPARAMETERS = {
    'lr0': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_bias_lr': 0.1,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
    'translate': 0.1, 'scale': 0.5,
    'mosaic': 1.0,
    'mixup': 0.0,
}
```
