# 02_Methods_Reference.md — 方法论参考
<!-- 最后更新：2026-06-01 -->
<!-- 算法原理 / 代码骨架 / 设计决策 -->
<!-- 需要理解算法或代码设计时查阅 -->

---

## §DPFLA 核心原理

### 防御目标

在联邦学习中，恶意客户端可以通过投毒攻击（标签翻转、高斯噪声等）破坏全局模型。DPFLA 的目标是**在聚合阶段自动识别并降低恶意客户端的权重**，使全局模型免受投毒影响。

### 攻击类型

| 攻击名称 | 原理 | 实现位置 |
|---------|------|---------|
| 标签翻转 (label_flipping) | 恶意客户端将 source_class 的标签翻转为 target_class | `attack_alg/label_flipping_attack.py` |
| 高斯攻击 (gaussian_attack) | 向本地模型更新注入高斯噪声 | `attack_alg/gaussian_attack.py` |
| 后门攻击 (backdoor) | 在图像注入 trigger pattern，期望被误分类到目标类 | `client.py:participant_update()` |

### 防御算法（聚合规则）

| 算法 | 原理 | 文件 |
|------|------|------|
| **DPFLA** | SVD 降维 + K-Means 聚类，检测异常更新 | `fl_algorithm/DPFLA.py` |
| FedAvg | 简单平均（无防御） | `fl_algorithm/fed_avg.py` |
| FoolsGold | 历史余弦相似度防御 | `fl_algorithm/fools_gold.py` |
| Trimmed Mean | 剔除极端值后平均 | `fl_algorithm/t_mean.py` |
| Median | 逐坐标取中位数 | `fl_algorithm/median.py` |
| Multi-Krum | 基于欧氏距离的异常检测 | `fl_algorithm/m_krum.py` |

| **FedProx** | 本地训罚 loss 加近端项，限制偏离全局模型 | Non-IID 收敛 | 待实现 |
| **SCAFFOLD** | 控制变量显式抵消 client drift | Non-IID 收敛 | 待实现 |
| **FedBN** | 聚合时排除 BN 参数，各客户端保留自己的 BN 统计量 | **BN 层问题** | 待实现 |
| **FedLA** | 按每类标签分布加权，而非按样本数加权 | **标签淹没问题** | 待实现 |

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

### 辅助异常评分（MAD z-score）

```python
def _robust_anomaly_score(feature_matrix):
    """基于稳健 z-score(MAD) 的异常度，输出 [0,1]"""
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0)
    z = np.abs((x - med) / (1.4826 * mad))
    z = np.minimum(z, 2.3)  # 小客户端数时防止爆炸
    score = 1.0 / (1.0 + np.exp(-(raw - center) / scale))
    return score
```

特征包含：更新范数的对数、与群体均值的方向余弦相似度、层间更新集中度。

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

```python
class YOLOWrapper(nn.Module):
    # 初始化：加载预训练 yolov8*.pt，将 COCO 80 类检测头替换为 nc=10
    # 关键修复：criterion (v8DetectionLoss) 移到 GPU，requires_grad=True
    # 虚拟 fc2 层：兼容 DPFLA（实际 DPFLA YOLO 走检测头 cv3.weight）
    
    def forward(x, targets=None, return_features=False):
        if self.training and targets is not None:
            # 训练模式：前向 + YOLO 内置损失 + backward
            preds = self.model(x)
            loss_tuple = self.model.criterion(preds, batch)
            return loss_tuple[0], loss_tuple[0]  # (loss, features)
        else:
            # 推理模式
            return self.model(x)
    
    def state_dict():  # 返回 model.state_dict() + fc2 层
    def load_state_dict():  # 分离 fc2，加载 YOLO，再加载 fc2
```

### 检测头参数选择（DPFLA YOLO 路径）

```python
# 优先使用检测头 cv3 末端的类别分支权重
preferred_keys = ['model.22.cv3.0.2.weight', 'model.22.cv3.1.2.weight', 'model.22.cv3.2.2.weight']
# 每个类别一个特征单元：拼接 cv3.{0,1,2}[unit] 的 weight + bias
```

---

## §联邦训练流程

### FL 主循环（fl_core.py）

```python
def run_experiment():
    for epoch in global_rounds:
        # 1. 选客户端
        selected_clients = choose_clients()
        
        # 2. 本地训练
        for client in selected_clients:
            update, grad, local_model, loss = client.participant_update(...)
            local_weights.append(update)
            local_grads.append(grad)
            local_losses.append(loss)
        
        # 3. 聚合
        if rule == 'DPFLA':
            scores = dpfla.score(simulation_model, local_models, ...)
            global_weights = average_weights(local_weights, scores)
        elif rule == 'fedavg':
            global_weights = average_weights(local_weights, [1]*n)
        
        # 4. 更新全局模型
        simulation_model.load_state_dict(global_weights)
        
        # 5. 评估
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
            # 浮点参数加权平均
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

YOLO 原生评估在**子进程**中执行（隔离 inference_mode 副作用）：
```python
# fl_core.py: test()
m = YOLO(weights_path)
metrics = m.val(data=yaml_path, imgsz=640, conf=0.001, iou=0.5,
                augment=False, plots=False, device=device_py)
# 解析 metrics.box.mp, metrics.box.mr, metrics.box.map50, metrics.box.map
```

---

## §设计决策记录

| 决策 | 原因 | 状态 |
|------|------|------|
| 评估禁用 fallback | 避免指标口径分叉 | ✅ 冻结 |
| float32 聚合 | float16 累积误差影响 BN 状态 | ✅ 冻结 |
| DPFLA 主路径 = SVD+KMeans | 原始方法学术可验证 | ✅ 冻结 |
| YOLO 标签翻转只改 labels | 避免 bbox 不一致 | ✅ 冻结 |
| 检测头 cv3 权重作为 DPFLA 特征 | 与 COCO 预训练对齐 | 🔄 待验证 |
| BN 层参数排除在聚合之外 | YOLO Non-IID 场景必须（FedBN 核心） | 🔄 待实现 |
| 聚合时标签分布感知加权 | FedLA 核心，预期 mAP +5~6% | 🔄 待实现 |
---

## §Non-IID Client Drift 问题与解决方案（BN 层方案见下节）

### 问题根源

FedAvg 在 Non-IID 数据下，客户端本地梯度方向差异大，简单平均导致多个不同方向的力互相抵消，全局模型停在 loss 死区。

### 解决方案对比

#### FedProx（最小改动，推荐先做）

**论文**：`FedProx: Federated Optimization in Heterogeneous Networks` — Li et al., MLSys 2020

**核心**：在本地训练的 loss 中加入近端项惩罚：`loss_prox = loss_local + μ/2 * ||w - w_global||²`

**优点**：实现极简单，不需要改 `fl_core.py` 的聚合逻辑

**实现位置**：`client.py` 的 `participant_update()`

#### SCAFFOLD（效果最优，推荐做，但需配合 FedBN 才能在 YOLO 上有效）

**论文**：`SCAFFOLD: Stochastic Controlled Averaging for Federated Learning` — Karimireddy et al., ICML 2020

**核心**：每个客户端维护控制变量 `c_i`，全局服务器维护 `c`，通过方差缩减显式抵消 drift。

**YOLO 注意事项**：原版 SCAFFOLD 对含 BN 层的模型不友好（BN 统计量会导致 control variate 偏差）。YOLO 场景推荐用 BN-SCAFFOLD，或至少配合 FedBN 使用。

**优点**：理论上可证收敛到最优解，可与 DPFLA 叠加

**实现位置**：`client.py`（添加 `c_i`）+ `fl_core.py`（SCAFFOLD 聚合逻辑）

#### 为什么 Multi-Krum/Trimmed Mean 对 Non-IID 收敛问题效果有限

这些算法针对"恶意/异常更新"设计，能过滤明显偏离的客户端，但无法解决**所有客户端都正常但方向不同**的问题。Client drift 下所有客户端都是"正常的坏人"，Krum 反而会选错。

---

## §BN 层问题与解决方案（2026-06-01 新增）

### 问题根源

YOLO 模型含大量 Batch Normalization 层。在 Non-IID 联邦学习中：

1. **本地 BN 统计漂移**：每个客户端的 `running_mean/running_var` 会漂移到各自的数据分布
2. **全局统计失配**：FedAvg 简单平均后，聚合的 BN 参数不再代表任何客户端的真实分布
3. **推理性能下降**：推理时 BN 层使用与实际特征分布严重不匹配的统计量

**后果**：仅靠解决 Client Drift 的算法（FedProx/SCAFFOLD）无法突破，因为 BN 层在底层拖累了所有方向正确的梯度。

### FedBN（最优先）

**论文**：`FedBN: Federated Learning on Non-IID Features via Local Batch Normalization` — Li et al., ICLR 2021

**GitHub**：`github.com/med-air/FedBN`

**核心**：聚合时**排除 BN 参数**（`running_mean/running_var/gamma/beta`），让各客户端保留自己的 BN 统计量。

```python
# average_weights() 中加入：
def _is_bn_param(key):
    return ('running_mean' in key or 'running_var' in key or
            'running_std' in key or
            'bn' in key.lower() or
            ('bias' in key and 'bn' in key.lower()))

# 聚合时：BN 参数取第一个诚实客户端的值（FedBN 核心）
# 其他参数仍然加权平均
```

**优点**：
- 改动极小，只需改 `average_weights()`
- 可叠加于所有其他算法（FedProx、SCAFFOLD、DPFLA）之上
- 预期 mAP +2~5%

### BN-SCAFFOLD

**论文**：`BN-SCAFFOLD: Controlling the Drift of Batch Normalization Statistics` — `arXiv:2410.03281`

**核心**：在 SCAFFOLD 控制变量基础上，对 BN 统计量也加 control variate，显式修正 BN 漂移。

**优点**：理论上比 FedBN 更精确，但实现更复杂

**实现位置**：`fl_core.py`（BN 参数 control variate）+ `client.py`（客户端 BN 控制变量）

---

## §标签感知聚合（2026-06-01 新增）

### 问题根源

FedAvg 按样本数加权，导致：
- 样本多的类（car）主导聚合方向
- 样本少的类（tricycle、awning-tricycle）梯度被淹没

### FedLA（Label-Aware Aggregation）

**论文**：`Label-Aware Aggregation for Improved Federated Learning` — Khalil et al., FMEC 2023
**扩展**：`Federated Learning with Heterogeneous Data Handling for Robust Vehicular Object Detection` — Khalil et al., IEEE IV 2024

**GitHub**：`github.com/TixXx1337/Federated-Learning-with-Heterogeneous-Data-Handling`

**核心**：聚合时按**每类标签分布**（而非样本数）加权。

```python
def compute_label_weights(client_label_hists, num_classes=10):
    """
    client_label_hists: list of dict {class_id: count}
    返回: 归一化后的客户端权重列表
    """
    # Step 1: 计算每类的全局总数量
    class_totals = [sum(h.get(cls, 0) for h in client_label_hists)
                    for cls in range(num_classes)]

    # Step 2: 每个客户端权重 = sum(本地类数/全局类数)
    client_weights = []
    for hist in client_label_hists:
        w = sum(hist.get(cls, 0) / max(class_totals[cls], 1)
                for cls in range(num_classes))
        client_weights.append(w)

    # Step 3: 归一化
    total = sum(client_weights)
    return [w / total for w in client_weights]
```

**项目已有**：客户端 `_build_local_label_hist()` 方法直接返回 `{class_id: count}`，可复用

**预期**：mAP +5~6%，IEEE IV 2024 在 NuScenes 目标检测上验证

### FedProx+LA（最优组合）

**论文**：同上 IEEE IV 2024

**核心**：FedProx 近端项（限制参数漂移）+ FedLA 标签感知（按类分布加权）

**效果**：mAP +6%，收敛速度 +30%

**实现位置**：
- 近端项：`client.py:participant_update()` 的 loss
- 标签感知：`fl_core.py` 聚合逻辑

---

