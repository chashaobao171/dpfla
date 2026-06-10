# 01_Workbench_Memory.md — 深度工作台记忆
<!-- 最后更新：2026-06-07 -->
<!-- 实验记录 / 故障字典 / 冻结决策的历史演化 -->
<!-- 维护方式：追加式，不覆盖历史 -->

---

## §VisDrone mAP 优化任务（2026-05-15 启动）

### 问题描述（2026-05-15 更新诊断）

当前 YOLOv8 在 VisDrone 数据集上 mAP@0.5 卡在约 30%，经过诊断确认**不是"收敛慢"，而是"聚合失效"**。

**核心问题：Non-IID Client Drift**

10 客户端 Non-IID 划分下，每个客户端的数据分布完全不同：
- 客户端A 只有 `[人、车]` → 学到的权重方向偏"检测人/车"
- 客户端B 只有 `[自行车、摩托车]` → 方向偏"检测两轮"
- 客户端C 只有 `[船、帆船]` → 方向又不同

FedAvg 简单平均三个不同方向的更新，相当于三个不等大小的力往不同方向拉 → 物体基本不动。模型快速停在 loss 的"死区"，增加轮数无效。

### 真实问题 vs 错误诊断

| 错误诊断 | 正确诊断 | 解决方向 |
|---------|---------|---------|
| 收敛慢（还在爬坡） | 聚合失效（停在差解） | 不能靠增加轮数解决 |
| 学习率太低 | Non-IID client drift | 必须在聚合环节介入 |
| 模型太小 | FedAvg 对 Non-IID 天生不友好 | 换聚合算法 |

### 可能的根因分析

1. **FedAvg 对 Non-IID 天生不友好**：简单平均在不同方向更新间互相抵消
2. **学习率过低**：`LOCAL_LR=2e-4` 对 YOLO 偏保守
3. **local epochs 过少**：10 epochs 对每客户端少量数据不够充分
4. **YOLO 预训练权重未充分微调**
5. **数据增强不足**：当前是否开启 Mosaic/MixUp？

### 实施日志

| 日期 | 尝试策略 | 结果 | 备注 |
|------|---------|------|------|
| 2026-05-15 | 诊断 + 制定优化计划 | 待实施 | 尚未运行实验 |

---

## §环境配置（autodl 云平台）

### 当前路径映射

| 资源 | 路径 |
|------|------|
| 项目根 | `/root/chashaobao/DPFLA-master` |
| VisDrone 原始数据 | `/root/autodl-tmp/data/visdrone/` |
| YOLO 训练图 | `/root/autodl-tmp/data/visdrone/images/train/` |
| YOLO 验证图 | `/root/autodl-tmp/data/visdrone/images/val/` |
| YOLO 训练标签 | `/root/autodl-tmp/data/visdrone/labels_yolo_visdrone10/train/` |
| YOLO 验证标签 | `/root/autodl-tmp/data/visdrone/labels_yolo_visdrone10/val/` |
| MNIST | `/root/autodl-tmp/data/MNIST/` |
| CIFAR10 | `/root/chashaobao/data/`（`cifar-10-python.tar.gz` 解压路径） |

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

恶意客户端在训练时将部分类别的标签翻转为目标类。

### 动态映射策略

```python
HIGH_FREQ_POOL = [0, 1, 2]          # 高频源类（每轮按本地频次排序后抽样）
LOW_FREQ_TARGET_POOL = [9]          # 目标类（当前固定为 class 9 = motor）
PICK_FROM_HIGH = 1                   # 从高频池取几个
PICK_FROM_OTHERS = 0                 # 从其他类取几个
MALICIOUS_BEHAVIOR_RATE = 0.26      # 每轮发动攻击的概率
DYNAMIC_SEED = 42                    # 随机种子（轮次轮换可控）
```

实际行为：恶意客户端每轮按 `seed + 1009*epoch + 9173*client_id` 确定性地选择源类，固定翻转到 `class 9`。

### 与训练的关系

- 标签翻转在 `client.py:participant_update()` 的 `poisoned_data = label_flipping()` 中实现
- 只翻转 labels，不修改 boxes
- `drop_last=False`（含毒化分支也保持）

---

## §故障字典

### 1. YOLO 加载失败（libGL.so.1）

**现象**：`Failed to load YOLO from ultralytics: libGL.so.1: cannot open shared object file`
**原因**：ultralytics 依赖链安装了 `opencv-python`（GUI 版），无头服务器缺少 libGL
**后果**：YOLOWrapper fallback 到 simple CNN，实验口径失真
**修复**：
```bash
pip uninstall opencv-python -y
pip install --force-reinstall --no-deps numpy==1.26.4 opencv-python-headless==4.8.1.78
```
**防复发**：`requirements.txt` 已说明无头环境约束

---

### 2. YOLO 评估子进程失败（CPU device 字面量错误）

**现象**：`❌ YOLO原生评估出错`，子进程语法错误
**原因**：`fl_core.py` 生成子进程代码时写了 `device=cpu`（未加引号），Python 字面量解析失败
**修复**：CPU 时改为 `"'cpu'"` 字面量，CUDA 时用 `device=0`

---

### 3. YOLO 评估后训练状态异常（inference_mode 污染）

**现象**：评估后 loss 无梯度或 NaN，无法继续训练
**原因**：ultralytics `BaseValidator/__call__` 使用 `inference_mode` 装饰器，同进程内可能污染训练状态
**修复**：评估统一在子进程中执行（`subprocess`），主进程只解析 JSON 结果

---

### 4. pickle 缓存损坏

**现象**：`pickle data was truncated`
**修复**：
```bash
# 清理对应缓存文件让其重新完整生成
rm cache/cache_yolo_val_*.pkl
```

---

### 5. VisDrone mAP 全 0（初始或某轮）

**可能原因**：
- YOLO 标签未部署（`images/` 同级无 `labels/`）
- 换头后新检测头随机初始化，Round 0/1 mAP=0 是正常的（冷启动）
- `conf=0.001` + NMS 下新头几乎无有效 TP

**检查**：
```bash
# 确认标签存在
ls /root/autodl-tmp/data/visdrone/images/train/ | head -3
ls /root/autodl-tmp/data/visdrone/labels/train/ | head -3   # symlink 指向 labels_yolo_visdrone10
```

**判断标准**：若 Round 1 起 mAP > 0%，说明链路正常

---

### 6. 训练看起来很慢（GPU 未利用）

**现象**：nvidia-smi 显示利用率低，日志在 `Creating annotation cache...`
**原因**：训练前期主要是 CPU 在做标注缓存解析 + Dirichlet 数据分配，GPU 计算尚未开始
**处理**：等缓存生成完再观察（建议 `watch -n 1 nvidia-smi`）

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

## §历史关键实验结果

### MNIST 无攻击基线（FedAvg）

- 配置：6 客户端、无攻击、10 轮、local_epoch=1
- 全局准确率：约 `60% → 94%`（末轮）
- 类 1 准确率：始终保持 `93%+`（无攻击对照上限）

### MNIST + label_flipping + DPFLA（SVD+KMeans）

- 配置：6 客户端、恶意率 30%、10 轮、local_epoch=1
- 早期版本：Bad update 权重命中 0%，说明当时 KMeans 标签映射有缺陷
- 已修复：`batch_detect_outliers_kmeans()` 统一按"多数派簇=好，少数派=坏"赋义
- **最终结果**（2026-05-15）：最终准确率 **DPFLA=95.0%** vs **FedAvg=4.57%**，攻击成功率 **DPFLA=0%** vs **FedAvg=94.29%**，DPFLA 完全防御住 All-to-All + 梯度放大 ×15 强化攻击

### MNIST 实验数据（白名单，已核实）

```
/root/chashaobao/DPFLA-master/logs_3/mnist/run_no_defense_label_flipping_20260515_1020.log   (FedAvg, 370KB)
/root/chashaobao/DPFLA-master/logs_3/mnist/run_dpfla_label_flipping_20260515_1036.log         (DPFLA,  425KB)
/root/chashaobao/DPFLA-master/runs/MNIST_CNNMNIST_DPFLA_attack-label_flipping_mr-0.3/
/root/chashaobao/DPFLA-master/runs/MNIST_CNNMNIST_fedavg_attack-label_flipping_mr-0.3/
```

### VisDrone 无攻击基线（FedAvg）

- 配置：2 客户端、IID、1 轮、local_epoch=1
- 结果：mAP50≈86–87%，证明 YOLO+FL 链路正常

### VisDrone Gaussian 攻击 + FedAvg

- 配置：2 客户端、恶意率 50%；或 4 客户端、恶意率 25%
- 结果：mAP50=0%，说明简单 FedAvg 无法抵御高斯投毒

### VisDrone Gaussian 攻击 + DPFLA（验证集打分版，非主路径）

- 配置：同 Gaussian 攻击
- 结果：DPFLA 能把攻击客户端评分为 0，诚实客户端评分为 1，聚合后 mAP50 恢复
- 注：这是 `use_validation=True` 历史对照，不作为主方法结论

### VisDrone 标签翻转 + FedAvg vs DPFLA（SVD+KMeans）

- 配置：10 客户端、恶意率 30%、10 轮
- 结果：两方案 mAP50 曲线高度接近
- 原因：SVD+KMeans 在 YOLO 特征下单簇频繁（100 次），恶意权重命中率 0%，退化为 FedAvg
- **冻结决策**：只改 DPFLA 的 YOLO 特征选取，不动训练/评估层
- **当前问题（2026-05-15）**：VisDrone mAP 基线本身过低（远低于 40%），需先提升基线再展示防御效果差距

---

## §文献调研与联网核查（2026-06-01）

### 调研报告来源

- `visdrone_fl_research_report.md`：由其他 AI Agent 调研 20+ 篇文献后总结的替换方案

### 联网核查结论

| 算法 | 核查状态 | 来源 |
|------|---------|------|
| FedLA | ✅ 真实有效 | IEEE IV 2024，`arXiv:2405.01108`，GitHub: `TixXx1337/...` |
| FedProx+LA | ✅ 真实有效 | 同上论文，mAP +6%，30% 收敛加速 |
| FL-JSDDC | ✅ 真实有效 | Frontiers Neurorobotics 2026，mAP +3%，收敛 2.2x，**无开源代码** |
| SCAFFOLD | ⚠️ 对 YOLO 效果有限 | 原版对 BN 层不友好；需用 BN-SCAFFOLD（`arXiv:2410.03281`） |
| FedProx | ⚠️ 对 YOLO 效果边际 | 前提是解决 BN 问题；`mu` 参考值来自 YOLOv5，对 YOLOv8 需实测 |
| BN-SCAFFOLD | ✅ 解决 BN 问题 | `arXiv:2410.03281`，对 BN 层加 control variate |

### 关键发现（调研报告未提及）

1. **YOLO 含大量 BN 层**：这是 FedProx/SCAFFOLD 在 YOLO 上效果有限的深层原因
2. **FedBN 是关键**：ICLR 2021，`github.com/med-air/FedBN`，聚合时排除 BN 参数，极简单但收益大
3. **BN 层破坏 Non-IID FL**：`arXiv:2301.02982` 理论分析 + FedTAN 解决方案
4. **最优组合**：FedBN（治 BN）+ FedProx（治 Drift）+ FedLA（治标签淹没）三合一

### 参考开源仓库

- `TixXx1337/Federated-Learning-with-Heterogeneous-Data-Handling`：FedLA/FedProx+LA 官方实现
- `med-air/FedBN`：FedBN PyTorch 实现（极简）
- `KarhouTam/SCAFFOLD-PyTorch`：SCAFFOLD 基准实现（含 FedAvg/FedProx/SCAFFOLD 对比）
- `CyprienQuemeneur/fedpylot`：YOLOv7 + 多种聚合算法基准
- `ffyyytt/FLYOLO`：Ultralytics + Flower，FedProx/FedNova 等

---

## §冻结决策时间线

| 日期 | 决策 | 状态 |
|------|------|------|
| 2026-03-31 | DPFLA 主实验必须走 SVD+KMeans，use_validation=True 降为对照 | ✅ 冻结 |
| 2026-04-02 | KMeans 标签映射统一为"多数派=好，少数派=坏" | ✅ 冻结 |
| 2026-04-09 | 日志改为北京时间口径，logs_3/ 按数据集分桶 | ✅ 冻结 |
| 2026-04-10 | opencv-python vs headless 依赖冲突修复 | ✅ 冻结 |
| 2026-04-11 | FedAvg 聚合 YOLO 参数改为 float32 | ✅ 冻结 |
| 2026-04-11 | VisDrone 评估禁用 fallback，失败直接抛错 | ✅ 冻结 |
| 2026-05-14 | 环境从 featurize 迁移到 autodl，路径已统一 | ✅ 冻结 |
| 2026-05-14 | DPFLA YOLO 特征选取待优化（fc2.weight → 检测头参数） | ✅ 冻结（已部分解决） |
| 2026-05-15 | MNIST DPFLA vs FedAvg 实验完成，mAP=95% vs 4.57% | ✅ 冻结 |
| 2026-05-15 | VisDrone mAP 基线过低，根因确诊为 Non-IID Client Drift（聚合失效），必须在聚合环节介入 | 🔄 **进行中** |
| 2026-06-01 | 文献调研+联网核查，重新确诊为三层叠加（Client Drift + BN 层失效 + 标签淹没），新增 FedBN/FedLA/BN-SCAFFOLD 方案 | 🔄 进行中 |
| 2026-06-01 | 00_QuickStart.md 全面更新为三层诊断框架 + 新搜索提示词 | ✅ 完成 |
| 2026-06-06 | 用户确认：目标为让 mAP 收敛到更高位置（≥40%），不是加速收敛；提高学习率无效已验证；确认训练走手写 SGD loop，`augment=True` 无效；推荐换模型（YOLOv8m）+ 换聚合算法（FedBN + FedLA）+ DPFLA 叠加 | 🔄 进行中 |
| 2026-06-06 | 00_QuickStart.md 更新：确认目标为收敛到更高 mAP、确认训练路径、修正方案方向（换模型+换聚合算法） | ✅ 完成 |
| 2026-06-07 | FedBN 实施计划制定：明确 4 个涉及文件（`fed_avg.py` 新增函数、`__init__.py` 导出、`fl_core.py` 新增 rule 分支、新增实验脚本），4 个具体改动步骤，验证步骤。计划已写入 00_QuickStart.md | ✅ 完成 |

---

## §FedBN 实施计划（2026-06-07）

### 原理

FedBN（ICLR 2021，`github.com/med-air/FedBN`）：聚合时**排除 BN 参数**（`running_mean` / `running_var` / `gamma` / `beta`），让各客户端保留自己的 BN 统计量。

Non-IID 下，每个客户端 BN 统计量漂移到各自数据分布。FedAvg 平均后 BN 参数不再代表任何客户端真实分布，推理时 BN 层与实际特征不匹配。

### 涉及文件

| 文件 | 改动 |
|------|------|
| `federated_learning/fl_algorithm/fed_avg.py` | 新增 `average_weights_fedbn()` |
| `federated_learning/fl_algorithm/__init__.py` | 导出新函数 |
| `federated_learning/fl_core.py` | 新增 `rule='fedbn'` 分支 |
| `run-test/visdrone/run_no_attack_baseline.py` | 复制为 FedBN 实验脚本 |

### 关键细节

- **BN 参数判断**：用 key 包含 `bn` 或 `.norm`（不区分大小写）判断。YOLO 模型中 BN 层通常命名为 `model.XX.bn`、`model.XX.m.0.bn` 等。
- **客户端无需改动**：BN 统计量随 `state_dict()` 自动保存/加载，聚合时排除即可。
- **与 DPFLA 完全兼容**：FedBN 修改的是聚合后的 `global_weights`，DPFLA 在其上叠加 SVD+KMeans 无冲突。

### 验证方式

```
1. run_no_attack_baseline.py（rule=fedavg）→ 记录 Round 10 / 20 mAP@0.5
2. 新 FedBN 脚本（rule=fedbn）→ 对比 Round 10 / 20 mAP@0.5
3. 可选：打印聚合前后 BN 参数的值，确认 running_mean/running_var 确实未被平均
```
