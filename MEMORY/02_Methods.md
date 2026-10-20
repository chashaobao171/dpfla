# 02_Methods.md — 方法原理

<!-- 最后更新：2026-10-20 -->
<!-- 算法原理 / 代码骨架 / 设计决策。涉及"结论"处标注【实测】/【推断】 -->

## §DPFLA 核心原理

**目标**：联邦学习中，恶意客户端通过投毒（标签翻转 / 高斯噪声 / 后门）破坏全局模型。DPFLA 在**聚合阶段自动识别并降低恶意客户端权重**。

### 主路径：SVD + K-Means

```python
def DPFLA.score(global_model, local_models, clients_types, selected_clients, ...):
    """
    1. 特征提取：对每个客户端，取其检测头参数上的更新向量
    2. 奇异值分解：Z = W @ G^{-1} @ U_mask（G 为正交矩阵投影）
    3. 二维投影：取前两个奇异值对应特征向量
    4. K-Means 聚类(k=2)：多数派簇=诚实(权重1)，少数派簇=恶意(权重0)
    5. 辅助评分：MAD z-score 异常度，与聚类标签组合成软权重
    6. 跨轮惩罚：连续判为恶意的客户端逐轮加重
    """
```

### 关键组件

- **正交矩阵投影**：`W = generate_orthogonal_matrix(n=n*n)`；`U = inv(G) @ U_mask[:, :2]`
- **K-Means 异常检测**：轮廓系数 `silhouette_score < 0.61` → 判聚类质量差，全判诚实
- **软权重**：轮廓系数 ≥0.72 高置信（bad 可压到 0.02）；[0.58,0.72) 灰区柔罚；<0.58 仅温和防守
- **跨轮惩罚/恢复**：`soft = max(0.02, 0.20 * 0.72^(streak-1))`；从 bad 恢复先缓释再回 1.0
- **辅助异常评分（MAD z-score）**：

```python
def _robust_anomaly_score(feature_matrix):
    med = np.median(x, axis=0); mad = np.median(np.abs(x - med), axis=0)
    z = np.abs((x - med) / (1.4826 * mad)); z = np.minimum(z, 2.3)
    return 1.0 / (1.0 + np.exp(-(raw - center) / scale))
```

### 检测头特征选取（YOLO 路径）

```python
preferred_keys = ['model.22.cv3.0.2.weight', 'model.22.cv3.1.2.weight', 'model.22.cv3.2.2.weight']
```

历史：早期用 `fc2.weight`，效果差，改为检测头 `cv3.weight`。【实测】改动后未重跑 DPFLA vs FedAvg 对比（见 00_Current_State）。

## §FedLA 标签感知聚合

**原理**：按每类标签分布加权，而非按样本数加权。样本少的类（tricycle / awning-tricycle）在 FedAvg 中梯度被淹没，FedLA 保证每类公平代表权。

```
w_la[i] = Σ_c ( client_i[cls=c] / global_total[cls=c] )   # 归一化到均值 1.0
```

调用：`fl_core.py` → `rule == 'fedla'` 分支 → `average_weights_fedla(...)`。缓存 `_cached_label_hist` 避免每轮重扫。

【实测】FedLA R8=22.49%，与 FedAvg 接近，未显著超越。

## §FedBN（已实测无效，保留原理）

**原理**：聚合时排除 BN 参数，各客户端保留自己的 BN 统计量，适用于**特征偏移**。

**为何对 VisDrone 无效**：VisDrone 的 Non-IID 是**标签分布偏斜**，不是特征偏移。FedBN 对标签偏斜无帮助。【实测】FedBN 全程 ≤ FedAvg（见 01_Facts §1）。

## §YOLO + 联邦学习集成

### 架构

```
main.py / run-test/*  →  server.py (run_exp→Arguments→FL.run_experiment)
    →  fl_core.py（FL 编排：选客户端 → 并行本地训练 → 聚合 → 评估）
        ├── client.py           本地训练（手写 forward/backward + SGD）
        ├── fl_algorithm/*      聚合（DPFLA / fed_avg / fools_gold / m_krum / median / t_mean / fed_svd）
        └── models/yolo_wrapper.py  封装 Ultralytics YOLOv8
```

### YOLOWrapper 设计

- 初始化：加载 `yolov8*.pt`，将 COCO 80 类检测头替换为 nc=10
- `criterion`（v8DetectionLoss）在 GPU、`requires_grad=True`
- 虚拟 `fc2` 层：兼容 DPFLA 接口

### FedAvg 聚合（`fl_algorithm/fed_avg.py`）

```python
def average_weights(w, marks, float16_floats=False):
    for key, v0 in w_avg.items():
        if not v0.is_floating_point():
            w_avg[key] = v0              # 非浮点参数取第一个客户端（保护 BN）
        else:
            acc = v0 * marks[0]
            for i in range(1, len(w)):
                acc = acc + w[i][key] * marks[i]
            w_avg[key] = acc / sum(marks)
```

**冻结约束**：YOLO 场景统一 `float16_floats=False`（float32），防 BN 状态累积误差。
`num_batches_tracked` 用 `max()` 聚合（Bug3 修复）。

## §联邦训练流程

```python
for epoch in range(global_rounds):
    selected_clients = choose_clients()
    for client in selected_clients:
        update, grad, local_model, loss = client.participant_update(...)
    if rule == 'DPFLA':
        scores = dpfla.score(simulation_model, local_models, ..., use_validation=False)
        global_weights = average_weights(local_weights, scores)
    elif rule == 'fedavg':
        global_weights = average_weights(local_weights, [1]*n, float16_floats=False)
    simulation_model.load_state_dict(global_weights)
    accuracy, test_loss = test(simulation_model, ...)
```

## §VisDrone 数据集

10 类：`pedestrian(0), people(1), bicycle(2), car(3), van(4), truck(5), tricycle(6), awning-tricycle(7), bus(8), motor(9)`

YOLO 标签格式：`class_id x_center y_center width height`（归一化）。
划分：`sample_dirichlet`（IID / NON_IID / EXTREME_NON_IID）。

## §指标体系

- **主指标（唯一）**：YOLO 原生 `val()` 的 `mAP@0.5`（`metrics.box.map50`）
- 辅助：`mAP@0.5:0.95`、Precision、Recall
- 分类任务：accuracy、被攻击类准确率、标签翻转 ASR
- **禁用 fallback 自定义 mAP**（避免指标口径分叉）

## §设计决策记录

| 决策 | 原因 | 状态 |
|------|------|------|
| 评估禁用 fallback | 避免指标口径分叉 | ✅ 冻结 |
| float32 聚合 | float16 累积误差影响 BN | ✅ 冻结 |
| DPFLA 主路径 = SVD+KMeans | 原方法学术可验证（`use_validation=False`）| ✅ 冻结 |
| YOLO 标签翻转只改 labels | 避免 bbox 不一致 | ✅ 冻结 |
| 检测头 cv3 权重作 DPFLA 特征 | 与 COCO 预训练对齐 | ✅ 冻结 |
| FedBN 对 VisDrone 无效 | 标签偏斜 vs 特征偏移 | ✅ 冻结（实测） |
| 分层 LR backbone×0.5 / head×2.0 | 适度适配 | ✅ 保留 |
| 子进程 YOLO val | 隔离 inference_mode 副作用 | ✅ 可用（非 bug，见 04_Retracted） |
| FedAvgM / 梯度裁剪 | 模仿 FedAIoT，但借鉴对象错误 | ❌ 已回退，默认关（见 04_Retracted） |

## §Non-IID Client Drift（本项目不用，仅存档）

> ⚠️ 2026-10-10 用户明确：**本项目只要 IID，不考虑 Non-IID**。VisDrone 脚本均未设 `dd_type`，走默认 `IID`（`alpha=1e6`）。本节仅作背景存档。

VisDrone 的 Non-IID 是**标签分布偏斜**。FedAvg 在 Non-IID 下客户端梯度方向差异大，简单平均可能互相抵消，停在 loss 死区。
- FedBN（保 BN 统计量）对标签偏斜无效
- FedLA（按标签分布加权）更适合 VisDrone
- SCAFFOLD 原版对 BN 不友好，需 BN-SCAFFOLD（未实现）
