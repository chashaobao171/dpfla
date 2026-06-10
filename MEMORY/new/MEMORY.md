# DPFLA 项目记忆

> 建立时间：2026-06-09
> 背景：联邦学习 + YOLOv8 + VisDrone，目标研究对抗攻击
> 重要：所有改动记录详见 `MEMORY/01_ChangeLog.md`

---

## ⭐ 核心目标（每次会话必须记住）

**mAP@0.5 >= 32%**，这是最高优先级。围绕这个目标做的所有决策都应服务于它。

---

## 1. 当前基线结果

|| 配置 | mAP@0.5 | 备注 |
||------|---------|------|
|| **集中式 YOLOv8l oracle** | **~41%** | img=1024, batch=16, lr0=0.01 |
|| **FedAvg YOLOv8s（差配置，已回退）** | **~14.7%** | BS=16, EPOCH=3, LR=5e-4, cosine |
|| **FedAvg YOLOv8s（好配置，已恢复）** | **~22%** | BS=64, EPOCH=10, LR=2e-4, 无 cosine |
|| FedBN (non-IID) | ~20% | YOLOv8s |
|| FedLA (non-IID) | ~20% | YOLOv8s |

**目标**：mAP >= 32%，好配置基线 ~22%，差距 ~10%。

**注意**：当前 `fl_core.py` 的 val 方式（子进程 YOLO val）存在 bug，Pretrained model mAP 汇报为 0%。好配置 ~22% 是 2026-06-08 用旧 val 方式测得的，需回退/修复 val 流程后才能重新验证。详见 `01_ChangeLog.md`。

**注**：好配置不含 Mosaic；Mosaic 在 BS=64 时显存可能紧张，待调优。

---

## 2. 已完成的改动

### ✅ 回退配置修改（2026-06-10 实施，`01_ChangeLog.md`）
- `visdrone_fed_hparams.py`：LOCAL_EPOCHS 3→10，TRAIN_BATCH_SIZE 16→64，LR_SCHEDULE cosine→constant
- 详情见 `01_ChangeLog.md`

### ✅ Mosaic + MixUp 数据增强（2026-06-09 实施）
- **`visdrone_dataset.py`**：`mosaic_collate_fn`（L120-242），50% batch 触发 Mosaic，10% 再触发 MixUp
- **`client.py`**：`DataLoader` 的 `collate_fn` 替换为 `mosaic_collate_fn`（L254-257）

### ✅ conf 阈值修复
- `fl_core.py:204`：val conf=0.25（原 0.001 偏低）

### ✅ drop_last=True
- `client.py:255, 344, 352`

### ✅ 分层学习率
- `client.py`：backbone_lr=lr*0.5, head_lr=lr*2.0

### ✅ AMP 混合精度
- `client.py`：autocast + GradScaler

### ✅ YOLO 原生子进程 val
- `fl_core.py`：子进程调用 yolo val，隔离 inference_mode

---

## 3. 提升 mAP 的可行路径（按预期收益排序）

### 路径 A：Mosaic + MixUp（已实现，效果待验证）
- **预期收益**：+5~10% mAP
- **状态**：代码已写，待跑完验证

### 路径 B：增大 LR（配合 Mosaic）
- **预期收益**：+2~5% mAP
- **原理**：Oracle lr0=0.01，当前 FL lr=2e-4（差 50x）。Mosaic 增强后模型需要更大 LR 才能充分训练
- **操作**：SGD lr 0.01 或 cosine 从 0.01 开始

### 路径 C：训练更多轮（延长 FL 轮次）
- **预期收益**：+1~3% mAP
- **操作**：20轮→50轮，观察收敛

### 路径 D：增大 batch size（配合 LR 缩放）
- **当前**：batch=16，LR=2e-4
- **操作**：batch=32，LR 相应增大（线性缩放）

### 路径 E：SWA / EMA
- **预期收益**：+1~2% mAP
- ultralytics 默认 EMA，FL 框架是否启用待查

---

## 4. 当前代码配置快照

> ⚠️ 以下为尝试 yolov8l 前的配置状态（好配置），fl_core.py val bug 待修复后重新验证

```
模型:             YOLOv8s          (YOLO_MODEL_SIZE 可覆盖)
img_size:         800              (FL_IMG_SIZE 可覆盖)
train_batch_size: 64               ← 2026-06-10 回退
test_batch_size:  16
local_epochs:     10               ← 2026-06-10 回退
local_lr:         2e-4             ← 好配置（之前的测量基于此）
lr_schedule:      constant          ← 2026-06-10 回退（禁用 cosine）
warmup_epochs:    3.0             (cosine 禁用后不生效)
momentum:         0.9
weight_decay:     5e-4
global_rounds:    30
num_clients:      10
augmentation:     HSV + 水平翻转 + Mosaic(50%) + MixUp(10%)
drop_last:        True
val_conf:         0.25             ← ⚠️ 当前 val 方式有 bug，需修复
optimizer:        SGD
layer-wise LR:    backbone=lr*0.5, head=lr*2.0
AMP:              enabled
```

**好配置实测（2026-06-08，旧 val 方式）**：R6 约 21.93%，R8 约 22.49%

---

## 5. 关键代码位置

|| 文件 | 关键位置 | 内容 |
||------|----------|------|
|| `fl_core.py` | L204 | val conf/iou 阈值 |
|| `client.py` | L255, L344, L352 | drop_last 设置 |
|| `client.py` | L254-257 | mosaic_collate_fn 接入 DataLoader |
|| `visdrone_dataset.py` | L120-242 | _mosaic4 / _mixup / mosaic_collate_fn |
|| `visdrone_dataset.py` | L475-480 | Mosaic + HSV + 翻转 增强链 |
|| `visdrone_dataset.py` | L347, L484 | 数据增强配置 |
|| `client.py` | L397-412 | optimizer / layer-wise LR |
|| `yolo_wrapper.py` | L572 | loss.sum() |
|| `visdrone_fed_hparams.py` | L32-46 | 超参数默认值 |

---

## 6. 参考资源

- YOLOv8l.pt: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt
- 本地路径: /root/chashaobao/DPFLA-master/yolov8l.pt
- VisDrone: 训练 6471 张，验证 548 张，10 个类别
- 日志目录: logs_oracle/

## 7. 运行记录

- 2026-06-09 16:40: YOLOv8l + Mosaic run_mosaic_yolov8l（被 kill，未完成）
- 2026-06-09 16:43: 清理 logs_3/visdrone，只保留 3 个历史日志
- 2026-06-10: 回退配置修改（EPOCH 10, BS 64, cosine→constant），见 `01_ChangeLog.md`
- 2026-06-10 上午: YOLOv8s→YOLOv8l + conf=0.001 尝试 → **失败**，fl_core.py val bug，见 `01_ChangeLog.md`
- 2026-06-10 上午: 将本次失败尝试写入 MEMORY，准备回退
