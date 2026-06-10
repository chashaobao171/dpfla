# 00_QuickStart.md — 新会话第一页（必读）

<!-- 最后更新：2026-06-10 -->
<!-- 每次新会话先读这个文件 -->
<!-- 本文件是唯一的事实来源，新会话时以此为准 -->

---

## §当前状态（每次续工前确认）

**项目阶段**：DPFLA VisDrone mAP 优化（进行中，2026-06-10 更新）
**核心目标**：mAP@0.5 >= 32%
**当前任务**：fl_core.py val 方式有 bug（子进程 YOLO val 存在 ultralytics 内部机制冲突），需修复后才能重新验证基线。具体见 `01_ChangeLog.md`。

---

## §当前基线结果（可信测量）

|| 配置 | mAP@0.5 | 日期 | 备注 |
|--|------|---------|------|------|
| **集中式 YOLOv8l oracle** | BS=16, EPOCH=50, LR=0.01, Mosaic | **~41%** | 2026-06-09 | img=1024, 集中式训练 |
| **FedAvg YOLOv8s（好配置）** | BS=64, EPOCH=10, LR=2e-4, constant | **~22%** | 2026-06-08 | R6=21.93%, R8=22.49% |
| **FedBN（non-IID）** | 同上配置 | **~21.5%** | 2026-06-07 | R7 峰值，FedBN 对 VisDrone 无效 |
| **FedLA（non-IID）** | 同上配置 | **~22.5%** | 2026-06-08 | R8=22.49%，FedLA 实现完成待验证 |
| **FedAvg YOLOv8s（差配置）** | BS=16, EPOCH=3, LR=5e-4, cosine | **~14.7%** | 2026-06-09 | 已回退恢复好配置 |

**目标**：mAP >= 32%，好配置基线 ~22%，差距 ~10%。

---

## §冻结红线（禁止擅改）

1. **主实验 DPFLA 必须走 SVD+KMeans 路径**：`fl_core.py` 中 `rule=='DPFLA'` 时 `use_validation=False`，不用 loss 打分版作为主结论
2. **评估指标唯一**：主指标统一为 YOLO 原生 `val()` 的 `mAP@0.5`（`metrics.box.map50`），不用 fallback 自定义 mAP
3. **数据路径**：
   - 最高优先：`/root/autodl-tmp/data/visdrone`（raw VisDrone，train/val 均有 10 类标注）
   - YOLO 标签：`labels_yolo_visdrone10/`（用 `convert_visdrone_to_yolo.py --mode visdrone10 --deploy` 生成）
4. **FedAvg 聚合精度**：YOLO 场景统一 `float16_floats=False`（float32），防止 BN 状态累积误差
5. **数据白名单**：`logs_3/mnist/`、`runs/MNIST_*/` 为实验数据保留目录，删除任何数据前需确认不在此列
6. **好配置参数**（已回退确认）：`LOCAL_EPOCHS=10`、`TRAIN_BATCH_SIZE=64`、`LR_SCHEDULE=constant`（cosine 已禁用）
7. **fl_core.py val bug**：`fl_core.py` 的子进程 YOLO val 存在 ultralytics 内部机制冲突（`f` 属性、names 匹配问题），**每次 val 返回 0%**，必须修复才能继续验证。详见 `01_ChangeLog.md`。

---

## §Git 日常更新流程（必读）

> 详见 `04_GitGuide.md`

```bash
# 1. 查看改动
git status

# 2. 提交（替换 "your message"）
git add -A
git commit -m "your message"

# 3. 推送到 GitHub
git push
```

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
ls /root/autodl-tmp/data/visdrone/images/train/
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

## §当前实验配置（VisDrone，好配置）

来自 `run-test/visdrone/visdrone_fed_hparams.py`：

|| 参数 | 好配置值 | 说明 |
|--|------|---------|------|
| 模型大小 | YOLOv8s | `YOLO_MODEL_SIZE=n\|s\|m\|l\|x` |
| train_batch_size | **64** | ← 2026-06-10 回退确认 |
| local_epochs | **10** | ← 2026-06-10 回退确认 |
| local_lr | 2e-4 | 好配置 LR |
| lr_schedule | **constant** | ← 2026-06-10 回退确认（cosine 已禁用） |
| num_clients | 10 | `FL_NUM_WORKERS` |
| 恶意率 | 10% | `FL_MALICIOUS_RATE` |
| global_rounds | 30 | `FL_GLOBAL_ROUNDS` |
| augmentation | HSV + 水平翻转 + Mosaic(50%) + MixUp(10%) | |
| drop_last | True | |
| optimizer | SGD | |
| layer-wise LR | backbone=lr×0.5, head=lr×2.0 | |
| AMP | enabled | |

**好配置实测（2026-06-08，旧 val 方式）**：R6 约 21.93%，R8 约 22.49%

---

## §核心实验对照表

|| 实验 | 脚本 | 攻击 | 防御 | 状态 |
|--|------|------|------|------|------|
| 无攻击基线 | `run_no_attack_baseline.py` | 无 | 任意 | 目标 mAP >= 32% |
| 无防御对照 | `run_no_defense_label_flipping.py` | 标签翻转 | FedAvg | 攻击效果基线 |
| **DPFLA 防御** | `run_dpfla_label_flipping.py` | 标签翻转 | DPFLA(SVD+KMeans) | **核心实验** |

---

## §新会话续工检查清单

```bash
# 1. 确认实验状态
ls -lt logs_3/visdrone/*.log | head -5

# 2. 确认 GPU 可用
nvidia-smi

# 3. 确认数据
ls /root/autodl-tmp/data/visdrone/images/train/ | head -5

# 4. 确认 TensorBoard（如需）
tensorboard --logdir runs/ --port 6006 &
```

---

## §提升 mAP 的可行路径（按预期收益排序）

> 前提：必须先修复 `fl_core.py` val bug 才能验证以下路径的效果

|| 路径 | 预期收益 | 状态 |
|--|------|---------|------|
| A | 修复 val bug → 重新跑好配置基线验证 | 🔄 待实施 |
| B | 换 YOLOv8m 或 YOLOv8l | 🔄 待实施 |
| C | Mosaic + MixUp（代码已实现）| 🔄 待验证 |
| D | 增大 LR 配合 Mosaic（lr=0.01 量级）| 🔄 待实施 |
| E | 训练更多轮（50+）| 🔄 待实施 |
| F | FedLA（已实现，~22.5%）| 🔄 待验证 |

---

## §冻结决策时间线

|| 日期 | 决策 | 状态 |
|--|------|------|------|
| 2026-06-10 | 配置回退：EPOCH 3→10，BS 16→64，cosine→constant | ✅ 冻结 |
| 2026-06-10 | YOLOv8s→YOLOv8l + conf=0.001 失败，fl_core.py val bug | 🔄 进行中 |
| 2026-06-10 | 回退操作写入 `01_ChangeLog.md`，MEMORY 目录合并整理 | ✅ 完成 |
| 2026-06-09 | 集中式 oracle ~41% vs 联邦 ~22%，差距 2x | 🔄 进行中 |
| 2026-06-09 | Mosaic + MixUp 实现完成（50%/10%）| ✅ 合入 |
| 2026-06-08 | FedLA 实现完成（`average_weights_fedla()` + `rule='fedla'`）| ✅ 完成，待验证 |
| 2026-06-08 | FedBN 对 VisDrone 标签偏斜无效，FedAvg 全程领先 | ✅ 冻结 |
| 2026-06-07 | FedBN 实施完成，R7 峰值 21.55% | ✅ 冻结 |
| 2026-05-15 | DPFLA MNIST 验证成功（95% vs 4.57%）| ✅ 冻结 |
| 2026-04-11 | YOLO 评估禁用 fallback | ✅ 冻结 |
| 2026-04-11 | YOLO 聚合统一 float32 | ✅ 冻结 |

---

## §快速定位提示

|| 需要找什么 | 去哪个文件 |
|--|-----------|-----------|
| 上次做到哪了 | `00_QuickStart.md` → §当前状态 |
| 核心算法原理 | `02_Methods_Reference.md` → §DPFLA / §YOLO集成 |
| 实验脚本怎么跑 | `00_QuickStart.md` → §运行命令 |
| 遇到报错/故障 | `01_Workbench_Memory.md` → §故障字典 |
| 冻结决策（不能改什么） | `00_QuickStart.md` → §冻结红线 |
| 路径/数据/环境配置 | `01_Workbench_Memory.md` → §环境配置 |
| 改动历史 | `01_ChangeLog.md` | |
| Git 日常更新流程 | `04_GitGuide.md` | |
| mAP 暴跌诊断（7.7% 根因） | `01_Workbench_Memory.md` → §mAP暴跌诊断 | |

---

## §当前任务的精确描述（供新会话直接使用）

> **核心目标（2026-06-10）**：mAP@0.5 >= 32%。

> **当前卡点**：fl_core.py 的子进程 YOLO val 存在 ultralytics 内部机制冲突（Pretrained model mAP=0%），必须修复 val 流程才能重新验证基线和后续实验。

> **好配置实测**：BS=64, EPOCH=10, LR=2e-4, constant，mAP R6=21.93%, R8=22.49%（旧 val 方式）

> **修复方向**（任选其一）：
> 1. **回退到子进程方式之前的 val 方案**（最稳妥）
> 2. **在 val 子进程里用 visdrone_temp.yaml**，不依赖 `yolo.save()` + `names` 手动设置
> 3. **用 `torch.save()` 保存 state_dict**，子进程用 `load_state_dict()` 替代 `YOLO(path)` 加载

> 配置：10 客户端、恶意率 10%、30 全局轮 × 10 本地轮。
> 目标：无攻击基线 >= 32%，有攻击 DPFLA >= 32%。
> 指标：YOLO 原生 mAP@0.5（子进程 val），主日志在 `logs_3/visdrone/`。
> 冻结：DPFLA 主路径 = SVD+KMeans（不用 loss 验证打分），评估禁用 fallback。

---

## §搜索提示词（给更好的搜索 AI）

```
我正在做一个联邦学习项目：
- 数据集：VisDrone（无人机目标检测，10类）
- 模型：YOLOv8（ultralytics）
- 数据划分：10个客户端，Non-IID（非独立同分布）
- 当前问题：联邦学习 mAP@0.5 约 22%，集中式 oracle 约 41%，差距 2x
- fl_core.py 子进程 YOLO val 有 bug（ultralytics 内部机制冲突）

请帮我找：
1. fl_core.py 中 YOLO 评估的可靠实现方案（避免 inference_mode 污染，又能正确加载 VisDrone nc=10 模型）
2. YOLOv8 + 联邦学习 + 子进程 val 的正确实现方式
3. 集中式 vs 联邦差距缩小的可行方向（Mosaic/MixUp、LR、模型大小）
```
