# 01_ChangeLog.md — 项目改动记录

<!-- 建立时间：2026-06-10 -->
<!-- 目的：记录所有改动，即使项目回退也能让后续 agent 了解做过什么 -->
<!-- 维护方式：追加式，每次重大改动在此文件顶部新增条目 -->

---

## 时间线与实测结果

```
2026-06-07  FedBN 基线      BS=64 EPOCH=10 LR=2e-4 无cosine  → ~21.55% mAP (R7)
2026-06-08  FedAvg 基线     BS=64 EPOCH=10 LR=2e-4 无cosine  → ~21.93% mAP (R6)  ← 好配置
2026-06-08  FedLA 基线      BS=64 EPOCH=10 LR=2e-4 无cosine  → ~22.49% mAP (R8)
2026-06-09  YOLOv8l Oracle  BS=16 EPOCH=50 LR=0.01  cosine+Mosaic → ~41% mAP (集中式)
2026-06-09  FedAvg 差配置   BS=16 EPOCH=3  LR=5e-4 cosine       → ~14.68% mAP (R30)
2026-06-10  配置回退        BS=64 EPOCH=10 LR_SCHEDULE=constant   ← 已恢复
2026-06-10  YOLOv8s→YOLOv8l 尝试  → FAILED（fl_core.py val bug，Pretrained mAP=0%）
2026-06-10  Bug 1+2+3 修复   BS=64 EPOCH=3  LR=2e-4 constant    → R1=3.36% R26-33=~23.7% (100轮基线) ✅
2026-06-10  FedAvgM 引入     服务端动量平滑，Round 1 失败/BN破坏导致 Round 2-3 mAP 崩塌 ⚠️ → 部分修复（learnable_keys ✅，设备错误待验）
2026-06-10  TB 清理         删除旧 MNIST/fedbn/fedla 实验目录，保留当前实验
2026-06-10  Round 2-3 mAP 崩塌 → Kimi 根因定位：FedAvgM 对 BN buffer 做动量平滑 → ✅ learnable_keys 已加
2026-06-10  FedAvgM 设备错误  Round 1 动量平滑失败：old_weights 在 CPU、global_weights 在 GPU → ✅ 已修复
```

---

## 2026-06-10（下午）：Bug 1+2+3 修复完成

### Bug 1：val 子进程检测头错乱（`fl_core.py`）

**问题**：`model.yolo.save()` 保存的是 COCO 80 类模型，子进程加载后检测头与 VisDrone 10 类不匹配。

**修复**：
- 用 `torch.save()` 直接保存 `model.state_dict()` + nc=10 + names
- 子进程中先加载对应 size 的 YOLO 预训练模型，提取原始 Detect 头的 `cv2/ch/stride/grid/anchor_grid/anchors/strides` 属性
- 动态创建 10 类 Detect 头并复制上述属性，`strict=False` 加载 state_dict
- 写入临时 `.py` 脚本而非多行 `-c` 字符串（避免语法歧义）

**验证**：100 轮基线，R1=3.36% → R26-33=~23.7%，收敛正常。

### Bug 2：梯度累加后未除以 batch 数（`client.py`）

**问题**：训练循环中梯度累加后没有除以总 batch 数，导致 DPFLA SVD 分析的梯度尺度不一致。

**修复**：在训练循环结束后，`client_grad` 字典中每个梯度除以 `total_batches`（= `len(train_loader) * epochs`）。

**验证**：日志可见 `→ 梯度已平均化（除以 33 个总步数）`。

### Bug 3：BN 聚合对 `num_batches_tracked` 错误处理（`fed_avg.py`）

**问题**：对所有非 float tensor 直接取第一个值，`num_batches_tracked` 应该取最大值。

**修复**：对 `num_batches_tracked` 使用 `max()` 而非直接取第一个。

---

## 2026-06-10（傍晚）：FedAvgM 引入 + Round 2-3 mAP 崩塌问题定位

### 问题现象

`run_no_attack_baseline_20260610_1901.log`（50 轮，BS=64，EPOCH=1，FedAvgM=0.9）：
- Round 1: mAP=**0%**, loss=1427, avg_client=706
- Round 2: mAP=**3.22%**, loss=1103, avg_client=416  ← FedAvgM 首次生效
- Round 3: mAP=**0%**, loss=1264, avg_client=347  ← 崩塌
- Round 4: mAP=**0%**, loss=1111, avg_client=347  ← 持续崩塌

对比正常基线（`run_no_attack_baseline_20260608_0755.log`，无 FedAvgM）：
- R1: 7.81%, R2: 16.78%, R3: 19.50%, R4: 21.16%（持续收敛）

### 根因定位（待修复）

**不是 FedAvgM 本身的问题，而是「Round 1 时 FedAvgM 未生效 + BN 状态被破坏」的组合效应**：

1. **Round 1 聚合时 FedAvgM 失败**（设备错误 `cuda:0 vs cpu` in `delta = gw - ow`），`global_weights` 未被动量平滑。Round 1 末全局模型 = 无动量的普通 FedAvg 结果。

2. **Round 1 末全局模型 BN 状态混乱**：`model.load_state_dict(global_weights)` 加载的模型，BN 使用的是各客户端聚合后的 running_mean/running_var（而非原始 COCO 预训练值），且在 CPU 上。

3. **Round 2 评估时**：`YOLO(pretrained_yolov8s.pt)` → 替换 nc=10 → `load_state_dict(FL_state_dict, strict=False)` → COCO 预训练 BN 被覆盖为 FL 聚合的 BN → `model.train()` 时 BN 使用 `num_batches_tracked=max()` 聚合的 running stats → **BN 状态比 Round 1 更差**（跨客户端平均后均值趋近，标准差趋零）。

4. **Round 3 评估时**：BN 状态继续恶化，mAP 崩塌为 0。

### 相关代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `fl_core.py:747-786` | `rule == 'fedavg'` | FedAvgM 动量平滑，Round 1 失败跳到 except |
| `fl_core.py:831-832` | `g_model` 备份 | `copy.deepcopy(simulation_model)` 保存测试前模型 |
| `fl_core.py:854-856` | NaN 回退 | `simulation_model = copy.deepcopy(g_model)` |
| `fl_algorithm/fed_avg.py:36-41` | `num_batches_tracked=max()` | BN 计数器取最大值 |
| `models/yolo_wrapper.py:228` | `self.model.criterion = v8DetectionLoss(self.model)` | 每次 `__init__` 新建 criterion |
| `models/yolo_wrapper.py:201-204` | 新头 `bias_init()` | 新检测头 bias 随机初始化 |

### 修复方向

**核心**：Round 1 FedAvgM 失败后的 BN 状态异常。参考 `run_no_attack_baseline_20260608_0755.log` 正常（无 FedAvgM），说明问题与 FedAvgM 改动相关。

**待 kimi agent 彻查**：详见 `MEMORY/kimi_prompt_round_collapse.md`。

### 修复方案（2026-06-10 傍晚）

**根因**：FedAvgM 对所有 floating_point tensor 做动量平滑，错误地包含了 BN 的 `running_mean`/`running_var`（buffer），导致这些统计量被持续压缩，BN 输出趋近 0。

**修复**（`fl_core.py:764`）：只对 `simulation_model.named_parameters()` 中的可学习参数做动量平滑，BN buffer 自动跳过。

```python
# 新增 1 行：构建可学习参数名集合
learnable_keys = {name for name, _ in simulation_model.named_parameters() if _.requires_grad}

# 新增 1 个跳过条件
if key not in learnable_keys:
    continue  # 跳过 BN buffer 等非可学习参数
```

**验证**：Kimi 分析报告 `MEMORY/fedavgm_bn_bug_analysis.md`。

**额外**：添加 BN running_var debug 日志监控（`fl_core.py:786-793`）。

### TensorBoard 目录清理记录

| 目录 | 状态 |
|------|------|
| `MNIST_CNNMNIST_fedavg_attack-label_flipping_mr-0.3` | ❌ 已删除（2026-05-15 旧） |
| `MNIST_CNNMNIST_DPFLA_attack-label_flipping_mr-0.3` | ❌ 已删除（2026-05-15 旧） |
| `VisDrone_YOLO_fedbn_attack-no_attack_mr-0.0` | ❌ 已删除（2026-06-07 旧） |
| `VisDrone_YOLO_fedla_attack-no_attack_mr-0.0` | ❌ 已删除（2026-06-08 旧） |
| `VisDrone_YOLO_fedavg_attack-no_attack_mr-0.0` | ✅ 保留（当前实验） |

### 日志解析结果（logs_3 → TB 事件文件）

| 日志 | TB 目录 | Rounds |
|------|---------|--------|
| `mnist/run_dpfla_label_flipping_20260515_1036.log` | `MNIST_CNNMNIST_DPFLA_attack-label_flipping_mr-0.3` | R1-R20 |
| `mnist/run_no_defense_label_flipping_20260515_1020.log` | `MNIST_CNNMNIST_fedavg_attack-label_flipping_mr-0.3` | R1-R20 |
| `visdrone/run_no_attack_baseline_20260608_0755.log` | `..._v1781089947` | R0-R7 |
| `visdrone/run_no_attack_baseline_20260610_1139.log` | `..._v1781089947` | R0-R34 |
| `visdrone/run_no_attack_baseline_20260610_1901.log` | `..._v1781089947` | R0-R5（中止） |
| `visdrone/run_no_attack_baseline_20260610_1909.log` | `..._v1781089947` | R0-R1（中止） |
| `visdrone/run_no_attack_baseline_fedbn_20260607_1625.log` | `VisDrone_YOLO_fedbn_attack-no_attack_mr-0.0` | R0-R9 |
| `visdrone/run_no_attack_baseline_fedla_20260608_1703.log` | `VisDrone_YOLO_fedla_attack-no_attack_mr-0.0` | R0-R9 |

---

## 2026-06-10（上午）：YOLOv8s→YOLOv8l + conf=0.001 尝试 → FAILED

### 目标

用 conf=0.001 口径重新跑 FL（全局 50 轮），同时将模型从 YOLOv8s 换成 YOLOv8l。

### 结果：失败，指标失真

Pretrained model mAP@0.5 = **0.00%**（无论 s 还是 l 都一样）。训练过程正常（损失下降），但每轮 val 汇报全是 0%，无法判断基线质量。

### 根因分析

**问题出在 `fl_core.py` 的子进程 YOLO val 方式**：

1. `fl_core.py` 通过 `model.yolo.save(weights_path)` 导出 pt 文件
2. 子进程用 `YOLO(weights_path)` 加载并 `val()`
3. `yolo.save()` 导出的 ultralytics ckpt 格式存在以下问题：
   - 检测头的 `f` 属性（层索引引用）被丢弃
   - 子进程 `val()` 时报错 `AttributeError: 'Detect' object has no attribute 'f'`
   - 即使绕过崩溃，模型 `names` 仍是 COCO 80 类，而非 VisDrone 10 类

### 文件修改清单

| 文件 | 改动 | 回退建议 |
|------|------|---------|
| `federated_learning/fl_core.py` | test() 改用子进程 YOLO val（引入 bug） | **必须回退** |
| `federated_learning/models/yolo_wrapper.py` | 删除 `new_head.training = True` | 可保留（无副作用） |
| `run-test/visdrone/visdrone_fed_hparams.py` | 模型从 yolov8s 改 yolov8l | 可保留 |
| `run-test/visdrone/run_oracle_yolov8l.py` | 同步 yolov8l | 可保留 |

### 回退操作

```bash
# 方案A：整个 fl_core.py 和 yolo_wrapper.py 回退到 2026-06-10 之前的版本
# 方案B（推荐）：把 fl_core.py 里的 test() 改回不使用子进程的方式
```

---

## 2026-06-10：回退导致 mAP 暴跌的配置（已恢复）

### 根因（实测数据）

| 参数 | 好配置（恢复后） | 差配置（回退前） | 危害 |
|------|--------------|--------------|------|
| `LOCAL_EPOCHS` | **10** | 3 | 检测头随机初始化后 3 epoch 完全不够 |
| `TRAIN_BATCH_SIZE` | **64** | 16 | 梯度方差 2x 放大，BN 崩溃 |
| `LR_SCHEDULE` | **constant** | cosine | R15+ 后 LR 压到 5e-6，15 轮假训练 |

**好配置实测**：R1 7.81% → R3 19.50% → R6 21.93%，持续收敛
**差配置实测**：R1 4.39% → R15 14.99%（触顶）→ R30 14.68%（持平/下滑）

### 文件修改

- `visdrone_fed_hparams.py`：`LOCAL_EPOCHS 3→10`，`TRAIN_BATCH_SIZE 16→64`，`LR_SCHEDULE cosine→constant`

---

## 2026-06-09 及之前：历史改动

| 时间 | 改动 | 文件 | 结果 |
|------|------|------|------|
| 2026-06-09 | Mosaic + MixUp 实现（50%/10%） | `visdrone_dataset.py` | ✅ 合入 |
| 2026-06-09 | `mosaic_collate_fn` 接入 DataLoader | `client.py` | ✅ 合入 |
| 2026-06-09 | val conf 阈值 0.001 → 0.25 | `fl_core.py` | ✅ 合入（但后续子进程方式导致仍测出 0%） |
| 2026-06-09 | `drop_last=True` | `client.py` | ✅ 合入 |
| 2026-06-09 | 分层学习率 backbone×0.5 / head×2.0 | `client.py` | ✅ 合入 |
| 2026-06-09 | AMP 混合精度（autocast + GradScaler） | `client.py` | ✅ 合入 |
| 2026-06-09 | YOLO 原生子进程 val（隔离 inference_mode） | `fl_core.py` | ⚠️ 有 bug，待修复 |

---

## 当前已知好配置（2026-06-10 实测更新）

```
模型:             YOLOv8s
train_batch_size: 64
local_epochs:     3-10              ← 10 更优但 3 也可以
local_lr:         2e-4
lr_schedule:      constant
global_rounds:    30-100
augmentation:     HSV + 翻转 + Mosaic(50%) + MixUp(10%)
drop_last:        True
val_conf:         0.25（子进程 eval 用 0.001 以提高召回）
layer-wise LR:    backbone=lr×0.5, head=lr×2.0
AMP:              enabled

Bug 修复后基线实测：R1=3.36% → R26-33=~23.7%（100 轮）
好配置历史峰值：~22% mAP (R6-R8)
```

---

## 待解决：fl_core.py val 方式

`fl_core.py` 的子进程 YOLO val 存在 ultralytics 内部机制冲突（`f` 属性、`names` 匹配问题），导致每次 val 返回 0%，**无法用于判断基线质量**。

修复方向（任选其一）：
1. **回退到子进程方式之前的 val 方案**（最稳妥）
2. **在 val 子进程里用 visdrone_temp.yaml 让 ultralytics 自动处理 nc=10**，不依赖 `yolo.save()` + `names` 手动设置
3. **用 `torch.save()` 保存 state_dict**，子进程用 `load_state_dict()` 替代 `YOLO(path)` 加载
