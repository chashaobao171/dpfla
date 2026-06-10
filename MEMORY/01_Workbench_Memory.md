# 01_Workbench_Memory.md — 深度工作台记忆

<!-- 最后更新：2026-06-10 -->
<!-- 实验记录 / 故障字典 / 冻结决策的历史演化 -->
<!-- 维护方式：追加式，不覆盖历史 -->
<!-- new/ 版本为最新，合并后写入此文件 -->

---

## §当前进度（2026-06-10）

### 核心问题：fl_core.py val bug

`fl_core.py` 的子进程 YOLO val 存在 ultralytics 内部机制冲突，导致每次 val 返回 0%，**无法用于判断基线质量**。

**根因**：
1. `fl_core.py` 通过 `model.yolo.save(weights_path)` 导出 pt 文件
2. 子进程用 `YOLO(weights_path)` 加载并 `val()`
3. `yolo.save()` 导出的 ultralytics ckpt 格式丢失检测头的 `f` 属性（层索引引用）
4. 子进程 `val()` 时报错 `AttributeError: 'Detect' object has no attribute 'f'`
5. 即使绕过崩溃，`names` 仍是 COCO 80 类而非 VisDrone 10 类

**修复方向（任选其一）**：
1. **回退到子进程方式之前的 val 方案**（最稳妥）
2. **在 val 子进程里用 visdrone_temp.yaml**，让 ultralytics 自动处理 nc=10，不依赖 `yolo.save()` + `names` 手动设置
3. **用 `torch.save()` 保存 state_dict**，子进程用 `load_state_dict()` 替代 `YOLO(path)` 加载

**相关文件改动**：
- `federated_learning/fl_core.py`：`test()` 改用子进程 YOLO val（引入 bug），**必须回退或修复**
- `federated_learning/models/yolo_wrapper.py`：删除了 `new_head.training = True`（无副作用，可保留）

### 好配置 vs 差配置对比

|| 参数 | 好配置 | 差配置 | 危害 |
|--|------|--------|--------|------|
| `TRAIN_BATCH_SIZE` | **64** | 16 | 梯度方差 2x 放大 + BN 崩溃 |
| `LOCAL_EPOCHS` | **10** | 3 | 检测头随机初始化后 3 epoch 完全不够 |
| `LR_SCHEDULE` | **constant** | cosine | R15+ 后 LR 压到 5e-6，后 10 轮假训练 |

**好配置实测**：R1 7.81% → R3 19.50% → R6 21.93%，持续收敛
**差配置实测**：R1 4.39% → R15 14.99%（触顶）→ R30 14.68%（持平/下滑）

---

## §VisDrone mAP 暴跌根因诊断（2026-06-09）

### 死亡组合（5 个改动叠加）

|| 改动项 | 原值 | 新值 | 危害 |
|--|--------|------|------|------|
| 1 | `TRAIN_BATCH_SIZE` | 64 | 16 | CRITICAL：梯度方差爆炸 + BN 崩溃 + 正负样本失衡 |
| 2 | `LOCAL_EPOCHS` | 10 | 3 | CRITICAL：知识积累不足 + 联邦同步记忆抹除 |
| 3 | 分层学习率 head LR | 无 | lr×5.0 | HIGH：检测头震荡 + backbone 冻结 |
| 4 | `LOCAL_LR` | 2e-4 | 5e-4 | HIGH：基础 LR 过高 + cosine 末期陷阱 |
| 5 | `LR_SCHEDULE` | constant | cosine+warmup | MEDIUM：联邦场景错配 |

详见 `01_Workbench_Memory.md` → §mAP暴跌诊断（完整报告在 MEMORY/new/DPFLA_mAP_Diagnosis_Report.md）。

### 恢复路径

| 优先级 | 改动 | 预期收益 |
|--------|------|---------|
| P0 | `TRAIN_BATCH_SIZE=64` | +7-10% |
| P0 | `LOCAL_EPOCHS=10` | +5-8% |
| P1 | `LOCAL_LR=2e-4` | +2-4% |
| P1 | `LR_SCHEDULE=constant` | +2-3% |
| P1 | 分层 LR 改为 `0.5x/1.0x` | +2-3% |

---

## §实验结果汇总

### MNIST（已冻结）

- 无攻击基线：FedAvg，6 客户端，10 轮 → 94%
- DPFLA 防御：标签翻转，恶意率 30% → 95%（vs FedAvg 4.57%）
- 日志：`logs_3/mnist/run_no_defense_label_flipping_20260515_1020.log`
- 日志：`logs_3/mnist/run_dpfla_label_flipping_20260515_1036.log`

### VisDrone 无攻击基线

|| Round | FedAvg | FedBN | FedLA |
|-------|--------|--------|-------|-------|
| R1 | 7.81% | 6.08% | — |
| R2 | 16.78% | 15.45% | — |
| R3 | 19.50% | 17.92% | — |
| R4 | 21.16% | 20.18% | — |
| R5 | 21.68% | 20.42% | — |
| R6 | **21.93%** | 20.34% | — |
| R7 | — | 21.55%（峰值）| — |
| R8 | 22.49% | 20.94% | **22.49%** |

**关键结论**：
- FedBN 对 VisDrone **标签偏斜**无效（适用于特征偏移），FedAvg 全程领先
- FedLA 与 FedAvg 接近（~22.5%），FedLA 实现完成待完整验证
- 集中式 oracle ~41%，联邦基线 ~22%，差距 2x

### VisDrone 标签翻转 + DPFLA（SVD+KMeans）

- 历史结果：mAP 曲线与 FedAvg 高度接近，SVD+KMeans 单簇频繁，恶意命中率 0%
- 原因：YOLO 特征选取（fc2.weight）效果差，后改为检测头 cv3.weight
- 当前状态：待重新运行对比实验（先修复 val bug）

### VisDrone 高斯投毒

- FedAvg：高恶意率下 mAP=0%，无法抵御
- DPFLA（验证集打分）：能评分攻击客户端为 0，恢复 mAP

### 集中式 oracle

- YOLOv8l，BS=16，EPOCH=50，LR=0.01，Mosaic + Cosine
- Epoch 3: mAP50 = **41.4%**
- 证明：YOLO + VisDrone 链路正常，差距来自联邦训练流程本身

---

## §环境配置（autodl 云平台）

### 当前路径映射

|| 资源 | 路径 |
|------|------|
| 项目根 | `/root/chashaobao/DPFLA-master` |
| VisDrone 原始数据 | `/root/autodl-tmp/data/visdrone/` |
| YOLO 训练图 | `/root/autodl-tmp/data/visdrone/images/train/` |
| YOLO 验证图 | `/root/autodl-tmp/data/visdrone/images/val/` |
| YOLO 训练标签 | `/root/autodl-tmp/data/visdrone/labels_yolo_visdrone10/train/` |
| YOLO 验证标签 | `/root/autodl-tmp/data/visdrone/labels_yolo_visdrone10/val/` |
| MNIST | `/root/autodl-tmp/data/MNIST/` |
| CIFAR10 | `/root/chashaobao/data/` |

### 硬编码优先级（读取顺序）

```
VisDrone:  /root/autodl-tmp/data/visdrone  >  /home/featurize/data/visdrone  >  D:/Pycharmworkplace/visdrone
MNIST:     /root/autodl-tmp/data/MNIST      >  /home/featurize/data/MNIST   >  ./data/MNIST
CIFAR10:   /root/autodl-tmp/data/           >  /root/chashaobao/data/       >  data/cifar/
```

### 涉及路径的文件（共 7 个）

`fl_core.py`、`sampling.py`、`visdrone_dataset.py`、`server.py`、`generate_visdrone_yaml.py`、`convert_visdrone_to_yolo.py`、`requirements.txt`

### 依赖安装顺序

```bash
pip install -r requirements.txt
pip uninstall opencv-python -y
pip install --force-reinstall --no-deps numpy==1.26.4 opencv-python-headless==4.8.1.78
# 注意：ultralytics 会自动升级 numpy，需手动回退
```

### 环境验证命令

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望：2.5.1+cu124  True  RTX 4080 SUPER
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

---

## §VisDrone 数据准备

### 生成 YOLO 标签（首次）

```bash
python convert_visdrone_to_yolo.py --mode visdrone10 --root /root/autodl-tmp/data/visdrone --deploy
```

效果：
- 生成 `labels_yolo_visdrone10/train/` 和 `labels_yolo_visdrone10/val/`（10 类，ID 不变）
- 在各 split 目录建立 `labels → labels_yolo_visdrone10/train` 的 symlink

### 数据质量检查

- raw `visdrone` 下 train/val 标签类别均为 `[0..9]`（10 类齐全）
- 不存在"单类标签导致 mAP 异常偏高"的问题

---

## §VisDrone 标签翻转攻击配置

```python
HIGH_FREQ_POOL = [0, 1, 2]          # 高频源类
LOW_FREQ_TARGET_POOL = [9]          # 目标类（class 9 = motor）
PICK_FROM_HIGH = 1
PICK_FROM_OTHERS = 0
MALICIOUS_BEHAVIOR_RATE = 0.26      # 每轮发动攻击的概率
DYNAMIC_SEED = 42                    # 随机种子
```

实现：`client.py:participant_update()` → `label_flipping()`，只翻转 labels，不改 boxes。

---

## §故障字典

### 1. YOLO 加载失败（libGL.so.1）

**现象**：`Failed to load YOLO from ultralytics: libGL.so.1: cannot open shared object file`
**原因**：ultralytics 依赖链安装了 `opencv-python`（GUI 版），无头服务器缺少 libGL
**修复**：
```bash
pip uninstall opencv-python -y
pip install --force-reinstall --no-deps numpy==1.26.4 opencv-python-headless==4.8.1.78
```

### 2. YOLO 评估子进程失败（CPU device 字面量错误）

**现象**：`❌ YOLO原生评估出错`，子进程语法错误
**原因**：`fl_core.py` 生成子进程代码时写了 `device=cpu`（未加引号），Python 字面量解析失败
**修复**：CPU 时改为 `"'cpu'"` 字面量，CUDA 时用 `device=0`

### 3. YOLO 评估后训练状态异常（inference_mode 污染）

**现象**：评估后 loss 无梯度或 NaN，无法继续训练
**原因**：ultralytics `BaseValidator/__call__` 使用 `inference_mode` 装饰器，同进程内可能污染训练状态
**修复**：评估统一在子进程中执行（`subprocess`），主进程只解析 JSON 结果

### 4. pickle 缓存损坏

**现象**：`pickle data was truncated`
**修复**：
```bash
rm cache/cache_yolo_val_*.pkl
```

### 5. VisDrone mAP 全 0（初始或某轮）

**可能原因**：
- YOLO 标签未部署（`images/` 同级无 `labels/`）
- 换头后新检测头随机初始化，Round 0/1 mAP=0 是正常的（冷启动）
- **`fl_core.py` val bug**（当前最可能原因）：子进程 YOLO val 每次返回 0%

**检查**：
```bash
# 确认标签存在
ls /root/autodl-tmp/data/visdrone/images/train/ | head -3
ls /root/autodl-tmp/data/visdrone/labels/train/ | head -3
```

**判断标准**：若 Round 1 起 mAP > 0%，说明链路正常

### 6. 训练看起来很慢（GPU 未利用）

**现象**：nvidia-smi 显示利用率低，日志在 `Creating annotation cache...`
**原因**：训练前期主要是 CPU 在做标注缓存解析 + Dirichlet 数据分配，GPU 计算尚未开始
**处理**：等缓存生成完再观察（建议 `watch -n 1 nvidia-smi`）

### 7. fl_core.py 子进程 val 崩溃（当前卡点）

**现象**：Pretrained model mAP@0.5 = 0.00%，训练过程正常（损失下降），但每轮 val 汇报全是 0%
**根因**：`yolo.save()` 导出的 ultralytics ckpt 丢失检测头 `f` 属性，子进程 val 崩溃
**修复方向**：见 §当前进度 → §核心问题

---

## §TensorBoard 监控

```bash
tensorboard --logdir runs/ --port 6006
```

曲线目录格式：
```
runs/VisDrone_YOLO_<rule>_attack-<attack_type>_mr-<malicious_rate>/
```

写入的标量：
- `accuracy/global`：主指标（分类=accuracy，VisDrone=YOLO mAP@0.5）
- `mAP50/global`：VisDrone 专用（与上同值）
- `loss/test`：测试损失
- `accuracy/attacked_class`：分类任务中被攻击类的准确率

---

## §冻结决策时间线

|| 日期 | 决策 | 状态 |
|------|------|------|
| 2026-06-10 | 配置回退：EPOCH 3→10，BS 16→64，cosine→constant | ✅ 冻结 |
| 2026-06-10 | fl_core.py val bug，val 返回全 0%，必须修复 | 🔄 进行中 |
| 2026-06-09 | 集中式 oracle ~41% vs 联邦 ~22%，差距 2x | 🔄 进行中 |
| 2026-06-09 | Mosaic + MixUp 实现完成 | ✅ 合入 |
| 2026-06-08 | FedBN 对 VisDrone 标签偏斜无效，FedAvg 全程领先 | ✅ 冻结 |
| 2026-06-08 | FedLA 实施完成，待完整验证 | 🔄 进行中 |
| 2026-06-07 | FedBN 实施完成，R7 峰值 21.55% | ✅ 冻结 |
| 2026-06-06 | Local epochs = 10（FedLA 论文推荐值） | ✅ 冻结 |
| 2026-05-15 | DPFLA MNIST 验证成功（95% vs 4.57%）| ✅ 冻结 |
| 2026-05-14 | 环境迁移到 autodl，路径统一 | ✅ 冻结 |
| 2026-04-11 | YOLO 评估禁用 fallback | ✅ 冻结 |
| 2026-04-11 | YOLO 聚合统一 float32 | ✅ 冻结 |

---

## §历史关键实验结果

### MNIST 无攻击基线（FedAvg）

- 配置：6 客户端、无攻击、10 轮、local_epoch=1
- 全局准确率：约 `60% → 94%`（末轮）

### MNIST + label_flipping + DPFLA（SVD+KMeans）

- 配置：6 客户端、恶意率 30%、10 轮、local_epoch=1
- **最终结果**（2026-05-15）：最终准确率 **DPFLA=95.0%** vs **FedAvg=4.57%**
- 攻击成功率 **DPFLA=0%** vs **FedAvg=94.29%**

### VisDrone IID 基线（FedAvg）

- 配置：2 客户端、IID、1 轮、local_epoch=1
- 结果：mAP50≈86–87%，证明 YOLO+FL 链路正常

### VisDrone Gaussian 攻击 + FedAvg

- 配置：2 客户端、恶意率 50%；或 4 客户端、恶意率 25%
- 结果：mAP50=0%，说明简单 FedAvg 无法抵御高斯投毒

### 时间线汇总

```
2026-06-07  FedBN 基线      BS=64 EPOCH=10 LR=2e-4 无cosine  → ~21.55% mAP (R7)
2026-06-08  FedAvg 基线     BS=64 EPOCH=10 LR=2e-4 无cosine  → ~21.93% mAP (R6)  ← 好配置
2026-06-08  FedLA 基线      BS=64 EPOCH=10 LR=2e-4 无cosine  → ~22.49% mAP (R8)
2026-06-09  YOLOv8l Oracle  BS=16 EPOCH=50 LR=0.01  cosine+Mosaic → ~41% mAP (集中式)
2026-06-09  FedAvg 差配置   BS=16 EPOCH=3  LR=5e-4 cosine       → ~14.68% mAP (R30)
2026-06-10  配置回退        BS=64 EPOCH=10 LR_SCHEDULE=constant   ← 已恢复
2026-06-10  YOLOv8s→YOLOv8l 尝试  → FAILED（fl_core.py val bug，mAP=0%）
```

---

## §文献调研与联网核查（2026-06-01）

### 核查结论

|| 算法 | 核查状态 | 来源 |
|------|---------|------|
| FedLA | ✅ 真实有效 | IEEE IV 2024，`arXiv:2405.01108`，GitHub: `TixXx1337/...` |
| FedProx+LA | ✅ 真实有效 | 同上论文，mAP +6%，30% 收敛加速 |
| FL-JSDDC | ✅ 真实有效 | Frontiers Neurorobotics 2026，mAP +3%，收敛 2.2x，**无开源代码** |
| SCAFFOLD | ⚠️ 对 YOLO 效果有限 | 原版对 BN 层不友好；需用 BN-SCAFFOLD（`arXiv:2410.03281`） |
| FedProx | ⚠️ 对 YOLO 效果边际 | 前提是解决 BN 问题 |
| BN-SCAFFOLD | ✅ 解决 BN 问题 | `arXiv:2410.03281`，对 BN 层加 control variate |
| **FedBN** | ⚠️ **对 VisDrone 无效** | 标签偏斜 vs 特征偏移，详见 §冻结决策 |

### 参考开源仓库

- `TixXx1337/Federated-Learning-with-Heterogeneous-Data-Handling`：FedLA/FedProx+LA 官方实现
- `KarhouTam/SCAFFOLD-PyTorch`：SCAFFOLD 基准实现
- `CyprienQuemeneur/fedpylot`：YOLOv7 + 多种聚合算法基准
- `ffyyytt/FLYOLO`：Ultralytics + Flower，FedProx/FedNova 等
