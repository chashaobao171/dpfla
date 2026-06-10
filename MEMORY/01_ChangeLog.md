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
```

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

## 当前已知好配置（最后一次可信测量）

```
模型:             YOLOv8s
train_batch_size: 64
local_epochs:     10
local_lr:         2e-4             ← 好配置 LR
lr_schedule:      constant          ← cosine 已禁用
global_rounds:    30
augmentation:     HSV + 翻转 + Mosaic(50%) + MixUp(10%)
drop_last:        True
val_conf:         0.25
layer-wise LR:    backbone=lr×0.5, head=lr×2.0
AMP:              enabled

好配置实测基线：~22% mAP (R6)
```

---

## 待解决：fl_core.py val 方式

`fl_core.py` 的子进程 YOLO val 存在 ultralytics 内部机制冲突（`f` 属性、`names` 匹配问题），导致每次 val 返回 0%，**无法用于判断基线质量**。

修复方向（任选其一）：
1. **回退到子进程方式之前的 val 方案**（最稳妥）
2. **在 val 子进程里用 visdrone_temp.yaml 让 ultralytics 自动处理 nc=10**，不依赖 `yolo.save()` + `names` 手动设置
3. **用 `torch.save()` 保存 state_dict**，子进程用 `load_state_dict()` 替代 `YOLO(path)` 加载
