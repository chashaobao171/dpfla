# 00_QuickStart.md — 新会话第一页（必读）

<!-- 最后更新：2026-06-07 -->
<!-- 每次新会话先读这个文件 -->
<!-- 本文件是唯一的事实来源，新会话时以此为准 -->

---

## §当前状态（每次续工前确认）

**项目阶段**：DPFLA VisDrone mAP 优化（进行中，2026-06-07 更新）
**当前任务**：FedBN 实施计划已制定，等待实施。具体：在 `fed_avg.py` 新增 `average_weights_fedbn()`，在 `fl_core.py` 新增 `rule='fedbn'` 分支，新增对照实验脚本。
**下一步**：实施 FedBN → 运行实验对比 FedAvg vs FedBN → 根据结果决定下一步（FedLA 或继续调优）
**最新日志**：在 `logs_3/visdrone/` 下按脚本 tag + 北京时间戳查找

---

## §VisDrone mAP 优化任务（核心优先级，2026-06-06 更新：目标为收敛到更高 mAP）

> 问题：YOLOv8 在 VisDrone 上的 mAP@0.5 基线过低（~30%），目标是让 mAP 收敛到 ≥40%，且有攻击时 DPFLA 防御组也达到 ≥40%。

### 已知实验数据（白名单，永久保留）

```
/root/chashaobao/DPFLA-master/logs_3/mnist/run_no_defense_label_flipping_20260515_1020.log   (FedAvg, 370KB)
/root/chashaobao/DPFLA-master/logs_3/mnist/run_dpfla_label_flipping_20260515_1036.log         (DPFLA,  425KB)
/root/chashaobao/DPFLA-master/runs/MNIST_CNNMNIST_DPFLA_attack-label_flipping_mr-0.3/
/root/chashaobao/DPFLA-master/runs/MNIST_CNNMNIST_fedavg_attack-label_flipping_mr-0.3/
```

### 三层根因诊断（2026-06-01 初诊，2026-06-06 修正）

> ⚠️ **2026-06-06 修正**：用户已验证提高学习率无效。Cosine 调度在 LR 本身就无效的前提下无意义。当前训练走**手写 SGD loop**，`augment=True` 无效。Mosaic/MixUp 需手动实现。

#### 架构层面（换模型，推荐优先）

| 策略 | 说明 | 预期收益 |
|------|------|----------|
| 换模型 | YOLOv8s → YOLOv8m（或 YOLOv9m） | mAP +3~5% |
| 换模型 | YOLOv8s → YOLOv8l | mAP +5~8% |

#### 聚合算法层面（解决 Non-IID，必须做）

| 策略 | 原理 | 预期收益 |
|------|------|----------|
| **FedBN** | 聚合时排除 BN 参数，让各客户端保留自己的 BN 统计量 | mAP +2~5% |
| **FedLA** | 聚合时按每类标签分布加权，而非按样本数加权 | mAP +5~6% |
| **FedBN + FedLA 组合** | 同时解决 BN 层失效 + 标签淹没 | mAP +7~11% |

#### 配置微调（辅助）

| 策略 | 说明 | 预期收益 |
|------|------|----------|
| 减少 local epochs | 10 → 3~5（Non-IID 下本地训练太长加剧 drift） | 配合聚合算法使用 |
| 增大 weight_decay | 5e-4 → 1e-3 | 防止过拟合，提升泛化 |
| 增大 momentum | 当前 → 0.9 | 稳定梯度更新 |
| 增大 batch size | 64 → 128 | 梯度更稳定 |

### 推荐实施顺序（2026-06-06 更新）

```
阶段一：基线提升
  YOLOv8s → YOLOv8m（换模型）
      +
  FedBN（聚合排除 BN 参数，改动极小）
  → 预期：mAP 从 ~30% → 35~40%

阶段二：进一步提升
  FedLA（按标签分布加权，项目已有 _build_local_label_hist()）
      +
  DPFLA（叠加防御）
  → 预期：mAP ≥ 40%，有攻击时也 ≥ 40%

最终对比实验：
  FedAvg（当前基线）→ ~30%
  FedBN + FedAvg → ~35%
  FedBN + FedLA + FedAvg → ~40%
  FedBN + FedLA + DPFLA → ~40%（有攻击）
```

**目标**：无攻击基线 mAP@0.5 ≥ 40%，有攻击时 DPFLA 防御组也 ≥ 40%。

### FedBN 实施计划（2026-06-07，制定中）

#### 原理

FedBN（ICLR 2021）的核心：在聚合时**排除 Batch Normalization 参数**（`running_mean` / `running_var` / `gamma` / `beta`），让各客户端保留自己的 BN 统计量。

在 Non-IID 联邦学习中，每个客户端的 BN 统计量会漂移到各自的数据分布。FedAvg 简单平均后，聚合的 BN 参数不再代表任何客户端的真实分布，导致推理时 BN 层使用与实际特征严重不匹配的统计量。

**现有代码已有部分处理**：`average_weights()` 已有逻辑：非浮点 Tensor（`num_batches_tracked`）直接取第一个客户端的值。但这只是防御性地避免了 `num_batches_tracked` 被平均导致 dtype 错误——它并没有系统性地处理所有 BN 参数。

#### 涉及文件

| 文件 | 改动内容 |
|------|---------|
| `federated_learning/fl_algorithm/fed_avg.py` | 新增 `average_weights_fedbn()` 或修改 `average_weights()` 加参数 `exclude_bn` |
| `federated_learning/fl_algorithm/__init__.py` | 导出新函数 |
| `federated_learning/fl_core.py` | 新增 `rule='fedbn'` 分支调用 |
| `federated_learning/client.py` | **无需改动**（BN 统计量会随 state_dict 自动保存/加载） |
| `run-test/visdrone/run_no_attack_baseline.py` | 新增一行 `rule='fedbn'` 实验 |

#### 具体改动

**Step 1：在 `fed_avg.py` 中新增函数**

```python
def average_weights_fedbn(w, marks, float16_floats: bool = False):
    """
    FedBN 聚合：BN 参数取第一个客户端的值，其他参数加权平均。
    适用于 YOLO 这类含大量 BN 层的模型。
    """
    import torch

    if len(w) == 0:
        return {}

    BN_KEY_SUBSTRINGS = ['bn', '.norm']

    def _is_bn_param(key: str) -> bool:
        key_lower = key.lower()
        return any(sub in key_lower for sub in BN_KEY_SUBSTRINGS)

    marks_sum = float(sum(marks)) if sum(marks) != 0 else 1.0
    w_avg = copy.deepcopy(w[0])

    for key, v0 in w_avg.items():
        if not torch.is_tensor(v0):
            w_avg[key] = v0
            continue
        if not v0.is_floating_point():
            w_avg[key] = v0
            continue

        # FedBN 核心：BN 参数取第一个客户端的值（不做平均）
        if _is_bn_param(key):
            w_avg[key] = v0  # 已在 deep copy 中，直接用 w[0] 的值
            continue

        # 其他参数：加权平均（与原 average_weights 逻辑相同）
        if float16_floats:
            acc = (v0.to(torch.float16) * marks[0]).to(torch.float16)
            for i in range(1, len(w)):
                acc = acc + (w[i][key].to(torch.float16) * marks[i])
            w_avg[key] = (acc * (torch.tensor(1.0 / marks_sum, device=acc.device, dtype=acc.dtype)))
        else:
            acc = v0 * marks[0]
            for i in range(1, len(w)):
                acc = acc + (w[i][key] * marks[i])
            w_avg[key] = acc * (1.0 / marks_sum)

    return w_avg
```

**Step 2：在 `__init__.py` 中导出**

```python
from .fed_avg import average_weights, average_weights_fedbn
```

**Step 3：在 `fl_core.py` 中新增分支**

在 `run_experiment()` 的聚合选择逻辑中（约 line 698），新增：

```python
elif rule == 'fedbn':
    cur_time = time.time()
    global_weights = average_weights_fedbn(
        local_weights,
        [1 for i in range(len(local_weights))],
        float16_floats=False,
    )
    cpu_runtimes.append(time.time() - cur_time)
```

**Step 4：新增对照实验脚本**

复制 `run_no_attack_baseline.py`，将 `rule='fedavg'` 改为 `rule='fedbn'`，其他配置不变。

#### 验证步骤

```
1. 先跑当前 FedAvg 基线（run_no_attack_baseline.py），确认 Round 10 / Round 20 的 mAP@0.5
2. 再跑 FedBN 实验（新增脚本），对比 Round 10 / Round 20 的 mAP@0.5
3. 确认 BN 参数（running_mean/running_var/gamma/beta）在各轮聚合后确实是"取第一个客户端的值"，而非被平均
```

#### 预期效果

- mAP@0.5 提升 +2~5%（预期）
- 所有后续聚合算法（FedLA、SCAFFOLD、DPFLA）都可以叠加在 FedBN 之上

**文献来源**（已联网核查）：
- FedLA/FedProx+LA：IEEE IV 2024，`arXiv:2405.01108`，GitHub: `TixXx1337/...`
- FL-JSDDC：Frontiers Neurorobotics 2026，mAP +3%，收敛 2.2x，无开源代码
- BN 层破坏 Non-IID FL：`arXiv:2301.02982`
- FedBN（BN 层解决）：ICLR 2021，`github.com/med-air/FedBN`
- BN-SCAFFOLD：`arXiv:2410.03281`，解决 BN 对 SCAFFOLD 的破坏

### 评估指标

- 主指标：`mAP@0.5`（YOLO `val()` 的 `metrics.box.map50`）
- 次指标：`mAP@0.5:0.95`（COCO style）
- 对照目标：无攻击基线 ≥ 40%，有攻击 FedAvg < 40%，有攻击 DPFLA ≥ 40%

---

## §冻结红线（禁止擅改）

1. **主实验 DPFLA 必须走 SVD+KMeans 路径**：`fl_core.py` 中 `rule=='DPFLA'` 时 `use_validation=False`，不用 loss 打分版作为主结论
2. **评估指标唯一**：主指标统一为 YOLO 原生 `val()` 的 `mAP@0.5`（`metrics.box.map50`），不用 fallback 自定义 mAP
3. **数据路径**：
   - 最高优先：`/root/autodl-tmp/data/visdrone`（raw VisDrone，train/val 均有 10 类标注）
   - YOLO 标签：`labels_yolo_visdrone10/`（用 `convert_visdrone_to_yolo.py --mode visdrone10 --deploy` 生成）
4. **FedAvg 聚合精度**：YOLO 场景统一 `float16_floats=False`（float32），防止 BN 状态累积误差
5. **YOLO 评估禁用 fallback**：评估失败直接抛错，不降级到自定义 mAP
6. **数据白名单**：`logs_3/mnist/`、`runs/MNIST_*/` 为实验数据保留目录，删除任何数据前需确认不在此列
7. **核心问题诊断**：VisDrone mAP 卡在 30% 的根因是**三层叠加**（Non-IID Client Drift + BN 层统计量失效 + 少数类标签淹没），详见 2026-06-01 更新

---

## §运行命令（3 分钟上手）

### 环境验证

```bash
cd /root/chashaobao/DPFLA-master

# 确认 GPU + CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 确认 ultralytics
python -c "from ultralytics import YOLO; print('YOLO OK', YOLO('yolov8n.pt', verbose=False) and True)"

# 确认数据目录
ls /root/autodl-tmp/data/visdrone/
ls /root/autodl-tmp/data/visdrone/images/train/   # YOLO 格式训练图
```

### 依赖安装（仅首次或依赖异常时）

```bash
pip install -r requirements.txt
pip uninstall opencv-python -y
pip install --force-reinstall --no-deps numpy==1.26.4 opencv-python-headless==4.8.1.78
```

### 运行实验

```bash
# 实验1：无防御基线（FedAvg + 标签翻转攻击）
python run-test/visdrone/run_no_defense_label_flipping.py

# 实验2：DPFLA 防御（DPFLA + 标签翻转攻击）
python run-test/visdrone/run_dpfla_label_flipping.py

# 实验3：无攻击基线
python run-test/visdrone/run_no_attack_baseline.py

# 交互式配置（非 VisDrone）
python main.py
```

### TensorBoard 监控

```bash
tensorboard --logdir runs/ --port 6006
# 曲线目录：runs/VisDrone_YOLO_<rule>_attack-<attack_type>_mr-<malicious_rate>/
```

### 日志位置

```
logs_3/visdrone/<experiment_tag>_<北京时间>.log
# 例如：logs_3/visdrone/run_dpfla_label_flipping_20260515_1430.log
```

---

## §当前实验配置（VisDrone）

来自 `run-test/visdrone/visdrone_fed_hparams.py`：

| 参数 | 默认值 | 可通过环境变量覆盖 |
|------|--------|-----------------|
| 模型大小 | YOLOv8s | `YOLO_MODEL_SIZE=n\|s\|m\|l\|x` |
| 客户端数 N | 10 | `FL_NUM_WORKERS` |
| 恶意率 | 10% | `FL_MALICIOUS_RATE` |
| 全局轮次 | 20 | `FL_GLOBAL_ROUNDS` |
| 本地轮次 | 10 | `FL_LOCAL_EPOCHS` |
| 训练 batch | 64 | `FL_TRAIN_BATCH_SIZE` |
| 测试 batch | 256 | `FL_TEST_BATCH_SIZE` |
| 本地学习率 | 2e-4 | `FL_LOCAL_LR` |
| LR 调度 | constant | `FL_VISDRONE_LR_SCHEDULE=cosine` |
| CPU 线程 | 16 | `FL_CPU_THREADS` |
| DataLoader workers | 6 | `FL_DATALOADER_WORKERS` |

**恶意客户端数** = `int(MALICIOUS_RATE × NUM_WORKERS)`，至少 1 个才算有攻击。
**弱攻击配置**（当前）：`MALICIOUS_BEHAVIOR_RATE=0.26`，`HIGH_FREQ_POOL=[0,1,2]`，`PICK_FROM_HIGH=1`，`LOW_FREQ_TARGET_POOL=[9]`

---

## §核心实验对照表

| 实验 | 脚本 | 攻击 | 防御 | 用途 |
|------|------|------|------|------|
| 无攻击基线 | `run_no_attack_baseline.py` | 无 | 任意 | 健康检查，目标 mAP ≥ 40% |
| 无防御对照 | `run_no_defense_label_flipping.py` | 标签翻转 | FedAvg | 攻击效果基线，目标 mAP ≥ 40% |
| **DPFLA 防御** | `run_dpfla_label_flipping.py` | 标签翻转 | DPFLA(SVD+KMeans) | **核心实验，目标 mAP ≥ 40%** |

A/B 对照目标：同攻击同超参，DPFLA mAP50 ≥ 无攻击基线 mAP50。

---

## §新会话续工检查清单

```bash
# 1. 确认实验状态
ls -lt logs_3/visdrone/*.log | head -5

# 2. 确认 GPU 可用
nvidia-smi

# 3. 确认数据
ls /root/autodl-tmp/data/visdrone/images/train/ | head -5

# 4. 确认 TensorBoard 可启动（如需）
tensorboard --logdir runs/ --port 6006 &
```

---

## §当前任务的精确描述（供新会话直接使用）

> **核心目标（2026-06-06）**：让 mAP@0.5 收敛到更高位置（≥ 40%），而不是加速收敛。用户已验证提高学习率无效。

> **当前方案方向**：换模型（YOLOv8s → YOLOv8m）+ 换聚合算法（FedBN + FedLA），DPFLA 叠加在其上。

> **训练路径确认**：VisDrone 走手写 SGD loop（`client.py:participant_update()`），不是 Ultralytics 原生 `model.train()`，`augment=True` 在当前架构下无效。

> 运行 VisDrone + 标签翻转攻击 + DPFLA（SVD+KMeans）防御实验。
> 配置：10 客户端、恶意率 10%、20 全局轮 × 10 本地轮。
> 目标：无攻击基线 ≥ 40%，有攻击 DPFLA ≥ 40%。
> 指标：YOLO 原生 mAP@0.5（子进程 val），主日志在 `logs_3/visdrone/`。
> 冻结：DPFLA 主路径 = SVD+KMeans（不用 loss 验证打分），评估禁用 fallback。

---

### 🔍 搜索提示词（给更好的搜索 AI）

复制以下内容到你的搜索工具：

```
我正在做一个联邦学习项目：
- 数据集：VisDrone（无人机目标检测，10类）
- 模型：YOLOv8（ultralytics）
- 数据划分：10个客户端，Non-IID（非独立同分布）
- 当前问题：mAP@0.5 卡在 ~30%，不是"收敛慢"，而是三层叠加问题：
  1. Non-IID Client Drift（梯度方向冲突）
  2. BN 层统计量失效（YOLO 含大量 BN 层，Non-IID 下本地统计与全局不一致）
  3. 少数类标签被淹没（FedAvg 按样本数加权）

请帮我找：
1. FedBN（Batch Normalization 参数不参与聚合）在 YOLO 目标检测上的效果
2. FedLA 标签感知聚合 + FedProx+LA 在目标检测上的最新实验结果
3. BN-SCAFFOLD 对 YOLO 这类 BN 密集模型的提升效果
4. 最好有 PyTorch 实现
```
