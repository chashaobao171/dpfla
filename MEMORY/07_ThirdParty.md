# 07_ThirdParty.md — 第三方参考工程（客观对照）

<!-- 最后更新：2026-10-10 -->
<!-- 本文件只做客观对照，不承载本项目结论。所有事实均标注出处（file:line 或命令文件）。 -->

## §现存参考工程：FedAIoT-main

- 位置：项目根 `FedAIoT-main/`（**当前仍在 git 跟踪中**，116 文件约 35M）
- 性质：参考借鉴，**非本项目主工程**
- 定位差异：面向 IoT 多模态数据集（CASAS / Ego4D / Energy / UWB / WiDar / WISDM / Epic-Sounds / VisDrone），使用 **Ray 分布式 + ultralytics 原生训练**

### 已核实的客观事实

**1. 聚合器 `aggregators/base.py`（关键）**

```python
def __init__(self, global_model, server_optimizer='sgd', server_lr=1e-2, server_momentum=0.9, eps=1e-3):
    self.optimizer = ...(filter(lambda p: p.requires_grad, global_model.parameters()),
                         lr=server_lr)      # ← server_momentum 未传入，实际动量=0

def step(self, updated_parameter_list, weights, round_idx=0):
    params_n_plus_1 = self._average_updates(...)
    for name, p_np1 in params_n_plus_1.items():
        if name in named_params: named_params[name].grad = named_params[name].data - p_np1.data
    self.global_model.load_state_dict(state_n_plus_1)
    self.optimizer.step()                    # param -= lr*(param-avg)
```

- 【实测】`server_momentum` 参数**被接收但从未使用**（`base.py:10-23`）。
- 【实测】`step()` 等价于 `param ← (1-lr)*param + lr*avg`；`lr=1` 时 = `avg`，**即纯 FedAvg**。
- 【实测】`distributed_main.py:207-210` 虽传 `server_momentum=0.9`，因上条无效。
- 【实测】`commands/overview/fedavg/*.sh`、`commands/float16/*.sh`、`commands/label_error/*.sh` 均为 `--server_optimizer sgd --server_lr 1`。
- 【实测】`commands/grid_search/fedopt/*.sh` 用 `--server_optimizer adam --server_lr $server_lr`（这才是"服务端优化 FedOpt"变体）。
- 【实测】`commands/` 下**没有** VisDrone 的运行命令。

> ⇒ **旧报告断言"FedAIoT = FedAvgM(0.9)"不成立**（详见 `04_Retracted.md` R1）。

**2. VisDrone 加载与超参 `loaders/visdrone.py`**

```python
YOLO_HYPERPARAMETERS = {
  'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005,
  'warmup_epochs': 3.0, 'warmup_bias_lr': 0.1,
  'mosaic': 1.0, 'mixup': 0.0,
  'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
  'translate': 0.1, 'scale': 0.5, 'fliplr': 0.5,
}
NAMES = 12 类（含 block / car_group），与主项目 10 类口径不同
```

**3. 训练器 `trainers/ultralytics_distributed.py`**

- `epochs=1`（每轮每客户端）（`line 40`）
- 优化器 `CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)`（`line 61`）
- `WarmupScheduler(warmup_epochs=3)`（`line 65`）
- AMP：`GradScaler(enabled=True)`（`line 66`）
- 梯度裁剪：`clip_grad_norm_(max_norm=10.0)`（`line 190, 197`）
- 使用 ulralytics 原生 `DetectionModel` + `Loss`（**非本项目的手写训练循环**）

### 与主项目的口径差异

| 维度 | FedAIoT | 本项目 DPFLA |
|------|---------|-------------|
| 聚合 | 单服务器优化器步（示例配置下 = FedAvg）| 可插拔（DPFLA/FedAvg/FedLA/…）|
| 训练内核 | ultralytics 原生 | 手写 forward/backward + YOLOWrapper |
| 客户端 LR | lr0=0.01 + warmup + CosineRestarts | 2e-4，constant |
| Mosaic | 1.0 | 0.5（+ MixUp 0.1）|
| 类别 | 12 类 | VisDrone 10 类 |
| 通信轮数 | 数百~数千 | 30–100 |

### 可借鉴点（【推断·未验证】，需有卡单变量验证）

1. **轮数**：本项目 R26-33 才到 23.7%，轮数可能是主要瓶颈。
2. **Mosaic 比例**：0.5 → 1.0。
3. **优化调度**：warmup + CosineWarmRestarts（需先解决 LR 量级）。
4. **AMP + grad clip**：本项目已实现但 clip 默认关闭（历史教训：与高 LR 组合有害）。

> ⚠️ 报告声称的 "FedAIoT 34.85% mAP@0.5" **未经一手核验**（见 `04_Retracted.md` R8），不得作为决策依据。

## §已失效的参考工程

- `YOLOv8_Federated_Learning-main/`：旧 `03_ThirdParty_Skeleton.md` 引用的工程，**已不在仓库**（见 `04_Retracted.md` R7）。
  其要点（仅存于旧记录）：直接调 `YOLO.train()` / `YOLO.val()` + 对 `state_dict()` 平均；每客户端独立 YAML；单类假设。与主工程差异较大，不可照搬。
