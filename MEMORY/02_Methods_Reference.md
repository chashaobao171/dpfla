# 02_Methods_Reference.md — 方法论参考

<!-- 最后更新：2026-06-10 -->
<!-- 算法原理 / 代码骨架 / 设计决策 -->
<!-- new/ 版本为最新，合并后写入此文件 -->

---

## §DPFLA 核心原理

### 防御目标

在联邦学习中，恶意客户端可以通过投毒攻击（标签翻转、高斯噪声等）破坏全局模型。DPFLA 的目标是**在聚合阶段自动识别并降低恶意客户端的权重**，使全局模型免受投毒影响。

### 攻击类型

|| 攻击名称 | 原理 | 实现位置 |
|---------|------|---------|
| 标签翻转 (label_flipping) | 恶意客户端将 source_class 的标签翻转为 target_class | `attack_alg/label_flipping_attack.py` |
| 高斯攻击 (gaussian_attack) | 向本地模型更新注入高斯噪声 | `attack_alg/gaussian_attack.py` |
| 后门攻击 (backdoor) | 在图像注入 trigger pattern，期望被误分类到目标类 | `client.py:participant_update()` |

### 防御算法（聚合规则）

|| 算法 | 原理 | 状态 |
|------|------|------|------|
| **DPFLA**（主路径） | SVD 降维 + K-Means 聚类，检测异常更新 | ✅ 实现 |
| **FedLA** | 按每类标签分布加权，而非按样本数加权 | ✅ 实现，待完整验证 |
| **FedBN** | 聚合时排除 BN 参数（对 VisDrone 标签偏斜无效） | ✅ 实现，FedAvg 全程领先 |
| FedAvg | 简单平均（无防御） | ✅ 实现 |
| FoolsGold | 历史余弦相似度防御 | ✅ 实现 |
| Trimmed Mean | 剔除极端值后平均 | ✅ 实现 |
| Median | 逐坐标取中位数 | ✅ 实现 |
| Multi-Krum | 基于欧氏距离的异常检测 | ✅ 实现 |
| SCAFFOLD | 控制变量显式抵消 client drift | ❌ 未实现 |
| FedProx | 本地训练加近端项惩罚 | ❌ 未实现 |
| BN-SCAFFOLD | BN 层 control variate | ❌ 未实现 |

---

## §DPFLA 算法详解

### 主路径：SVD + K-Means

```python
def DPFLA.score(global_model, local_models, clients_types, selected_clients, ...):
    """
    1. 特征提取：对每个客户端，计算其在检测头参数上的更新向量
    2. 奇异值分解：Z = W @ G^{-1} @ U_mask，其中 G 是正交矩阵
    3. 二维投影：取前两个奇异值对应的特征向量
    4. K-Means 聚类：k=2，多数派簇 = 诚实(权重1)，少数派簇 = 恶意(权重0)
    5. 辅助评分：基于 MAD z-score 的异常度，与 KMeans 标签组合生成软权重
    6. 跨轮惩罚：对连续被判为恶意的客户端逐轮加重惩罚
    """
```

### 关键组件

#### 正交矩阵投影

```python
W = generate_orthogonal_matrix(n=n*n)  # n=选中客户端数
G = concat([W[:, idx*n:(idx+1)*n][0,:].reshape(-1,1) for idx in selected_clients])
U = np.linalg.inv(G) @ U_mask[:, :2]   # 投影到客户端子空间
```

#### K-Means 异常检测

```python
# 轮廓系数判断聚类质量
coefficient = silhouette_score(data, labels)
if coefficient < 0.61:
    return np.ones(n)  # 聚类质量差，全判定为诚实

# 多数派簇 = 诚实，少数派簇 = 恶意
majority_label = unique_labels[np.argmax(counts)]
scores = [1 if lbl == majority_label else 0 for lbl in labels]
```

#### 软权重策略

```
轮廓系数 ≥ 0.72（高置信）：bad 权重可压到 0.02
轮廓系数 [0.58, 0.72)（灰区）：辅助异常分参与柔性惩罚
轮廓系数 < 0.58：仅用辅助特征做温和防守
```

#### 跨轮惩罚/恢复机制

- 连续被标为 bad：指数衰减 `soft = max(0.02, 0.20 * 0.72^(streak-1))`
- 从 bad 恢复：先给缓释权重，再逐轮回到 1.0
- 冷却期：高置信或高异常时触发，防止"一轮好就立刻满权重"

#### 辅助异常评分（MAD z-score）

```python
def _robust_anomaly_score(feature_matrix):
    """基于稳健 z-score(MAD) 的异常度，输出 [0,1]"""
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0)
    z = np.abs((x - med) / (1.4826 * mad))
    z = np.minimum(z, 2.3)
    score = 1.0 / (1.0 + np.exp(-(raw - center) / scale))
    return score
```

---

## §FedLA 标签感知聚合

### 原理

FedLA 按每类标签分布加权，而非按样本数加权。样本少的类（tricycle、awning-tricycle）在 FedAvg 中梯度被淹没，FedLA 确保每类获得公平代表权。

### 公式

```
w_la[i] = Σ_c ( client_i[cls=c] / global_total[cls=c] )
```

### 代码实现

```python
def average_weights_fedla(w, client_hists, num_classes, marks=None, float16_floats=False):
    # Step 1: 计算每类全局总量
    class_totals = [sum(h.get(cls, 0) for h in client_hists) for cls in range(num_classes)]
    # Step 2: 每个客户端权重 = Σ(本地类数/全局类数)
    # Step 3: 归一化使均值为 1.0（与 FedAvg 同一量纲）
    # Step 4: 加权平均
```

### 调用位置

`fl_core.py` → `rule == 'fedla'` 分支 → 收集 `local_hists` → 调用 `average_weights_fedla(local_hists, num_classes)`

---

## §FedBN（对 VisDrone 无效，保留原理）

### 原理

聚合时排除 BN 参数（`running_mean/running_var/gamma/beta`），让各客户端保留自己的 BN 统计量。适用于**特征偏移**（各客户端视觉风格不同但标签分布相似）。

### 对 VisDrone 无效的原因

VisDrone 的 Non-IID 是**标签分布偏斜**（不同区域类别比例不同），而非**特征偏移**（不同客户端图像风格不同）。FedBN 保持 BN 统计量对标签偏斜问题无帮助。

---

## §YOLO + 联邦学习集成

### 整体架构

```
main.py (交互配置)
    ↓
server.py (run_exp → Arguments → FL.run_experiment)
    ↓
fl_core.py (FL 编排器)
    ├── 选客户端
    ├── 并行本地训练（client.py）
    ├── 聚合（fl_algorithm/*）
    └── 评估（test 方法）
```

### YOLOWrapper 设计

`models/yolo_wrapper.py` 封装 Ultralytics YOLOv8：
- 初始化：加载预训练 yolov8*.pt，将 COCO 80 类检测头替换为 nc=10
- criterion (v8DetectionLoss) 移到 GPU，requires_grad=True
- 虚拟 fc2 层：兼容 DPFLA（DPFLA YOLO 走检测头 cv3.weight）

### 检测头参数选择（DPFLA YOLO 路径）

```python
preferred_keys = ['model.22.cv3.0.2.weight', 'model.22.cv3.1.2.weight', 'model.22.cv3.2.2.weight']
```

### ⚠️ fl_core.py val bug（当前卡点）

`fl_core.py` 的子进程 YOLO val 存在 ultralytics 内部机制冲突（`f` 属性、names 匹配问题），导致每次 val 返回 0%。

修复方向：
1. **回退到子进程方式之前的 val 方案**
2. **在 val 子进程里用 visdrone_temp.yaml**，让 ultralytics 自动处理 nc=10
3. **用 `torch.save()` 保存 state_dict**，子进程用 `load_state_dict()` 替代 `YOLO(path)` 加载

---

## §联邦训练流程

### FL 主循环（fl_core.py）

```python
def run_experiment():
    for epoch in global_rounds:
        selected_clients = choose_clients()

        for client in selected_clients:
            update, grad, local_model, loss = client.participant_update(...)
            local_weights.append(update)
            local_grads.append(grad)
            local_losses.append(loss)

        if rule == 'DPFLA':
            scores = dpfla.score(simulation_model, local_models, ...)
            global_weights = average_weights(local_weights, scores)
        elif rule == 'fedavg':
            global_weights = average_weights(local_weights, [1]*n)

        simulation_model.load_state_dict(global_weights)
        accuracy, test_loss = test(simulation_model, ...)
```

### 客户端本地训练（client.py）

```python
def participant_update(global_epoch, model, attack_type, ...):
    # 1. 攻击注入
    if attack_type == 'label_flipping' and self.client_type == 'attacker':
        poisoned_data = label_flipping(self.local_data, mapping)

    # 2. 本地 SGD
    for epoch in local_epochs:
        for batch in train_loader:
            loss, _ = model(data, targets=target)
            loss.backward()
            optimizer.step()
            model.zero_grad()

    return model.state_dict(), client_grad, model, avg_loss
```

### FedAvg 聚合

```python
def average_weights(w, marks, float16_floats=False):
    """加权平均，非浮点参数直接取第一个客户端的值（保护 BN）"""
    for key, v0 in w_avg.items():
        if not v0.is_floating_point():
            w_avg[key] = v0  # 跳过整数/布尔参数
        else:
            acc = v0 * marks[0]
            for i in range(1, len(w)):
                acc = acc + w[i][key] * marks[i]
            w_avg[key] = acc / sum(marks)
```

---

## §VisDrone 数据集

### 10 个类别

```
pedestrian(0), people(1), bicycle(2), car(3), van(4), truck(5),
tricycle(6), awning-tricycle(7), bus(8), motor(9)
```

### YOLO 格式

- 标签格式：`class_id x_center y_center width height`（归一化到 [0,1]）
- 数据集类：`YoloVisDroneDataset`（`datasets/yolo_visdrone_dataset.py`）
- 数据分布：`sample_dirichlet`（Dirichlet 采样），支持 IID / NON_IID / EXTREME_NON_IID

### 数据准备命令

```bash
python convert_visdrone_to_yolo.py --mode visdrone10 \
    --root /root/autodl-tmp/data/visdrone --deploy
```

---

## §指标体系

### 分类任务

- **准确率 (accuracy)**：`correct / n × 100%`
- **被攻击类准确率**：被翻转类的分类准确率
- **标签翻转攻击成功率 (ASR)**：`r[target_class] / sum(r) × 100%`

### VisDrone 目标检测

- **mAP@0.5**：IoU=0.5 时的 mean Average Precision（主指标）
- **mAP@[0.5:0.95]**：COCO 标准，0.5 到 0.95 步长 0.05 的平均
- **Precision / Recall**：精确率和召回率

### 评估实现

⚠️ 当前子进程 val 有 bug（见 §fl_core.py val bug）。正确实现应在**子进程**中执行 YOLO 评估以隔离 inference_mode 副作用：

```python
# fl_core.py: test() - 待修复
m = YOLO(weights_path)
metrics = m.val(data=yaml_path, imgsz=640, conf=0.001, iou=0.5,
                augment=False, plots=False, device=device_py)
# 解析 metrics.box.mp, metrics.box.mr, metrics.box.map50, metrics.box.map
```

---

## §设计决策记录

|| 决策 | 原因 | 状态 |
|------|------|------|
| 评估禁用 fallback | 避免指标口径分叉 | ✅ 冻结 |
| float32 聚合 | float16 累积误差影响 BN 状态 | ✅ 冻结 |
| DPFLA 主路径 = SVD+KMeans | 原始方法学术可验证 | ✅ 冻结 |
| YOLO 标签翻转只改 labels | 避免 bbox 不一致 | ✅ 冻结 |
| 检测头 cv3 权重作为 DPFLA 特征 | 与 COCO 预训练对齐 | ✅ 冻结 |
| FedBN 对 VisDrone 无效 | 标签偏斜 vs 特征偏移 | ✅ 冻结 |
| FedLA 本地标签统计缓存 | `_cached_label_hist` 避免每轮重复扫描 | ✅ 冻结 |
| fl_core.py val 方式 | 子进程方案有 bug，待修复 | 🔄 进行中 |
| 分层学习率 backbone×0.5 / head×2.0 | 适度适配，暂不调整 | ✅ 冻结 |

---

## §Non-IID Client Drift 问题

### 问题根源

FedAvg 在 Non-IID 数据下，客户端本地梯度方向差异大，简单平均导致多个不同方向的力互相抵消，全局模型停在 loss 死区。

### VisDrone 特殊性

VisDrone 的 Non-IID 是**标签分布偏斜**（不同区域类别比例不同），而非**特征偏移**（不同客户端图像风格不同）。这导致：
- FedBN（保留 BN 统计量）对标签偏斜无效
- FedLA（按标签分布加权）是更适合 VisDrone 的方向
- SCAFFOLD 原版对 BN 层不友好，需 BN-SCAFFOLD 或配合 FedBN

### 推荐方案

1. **FedLA**（已实现）：按每类标签分布加权，预期 mAP +5~6%
2. **FedProx + FedLA**（IEEE IV 2024）：近端项 + 标签感知，预期 mAP +6%，收敛加速 30%

---

## §mAP 暴跌根因（快速索引）

完整报告见 `01_Workbench_Memory.md` → §VisDrone mAP 暴跌根因诊断。

```
死亡组合：BS=16 + EPOCH=3 + HEAD_LR×5 + LR=5e-4 + COSINE+WARMUP
好配置：BS=64 + EPOCH=10 + LR=2e-4 + CONSTANT
恢复后实测：R6=21.93%, R8=22.49%
```
