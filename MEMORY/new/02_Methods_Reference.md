# 02_Methods_Reference.md — 方法论参考

<!-- 最后更新：2026-06-08 -->
<!-- 算法原理 / 代码骨架 / 设计决策 -->
<!-- 主动实验相关部分完整保留；历史遗留/已弃用方向简化 -->

---

## §DPFLA 核心原理

### 防御目标

恶意客户端通过投毒攻击（标签翻转、高斯噪声等）破坏全局模型。DPFLA 在聚合阶段自动识别并降低恶意客户端权重。

### 攻击类型

| 攻击 | 原理 | 位置 |
|------|------|------|
| 标签翻转 | 源类标签翻转为目标类 | `attack_alg/label_flipping_attack.py` |
| 高斯攻击 | 向本地更新注入高斯噪声 | `attack_alg/gaussian_attack.py` |

### 防御算法

| 算法 | 状态 | 文件 |
|------|------|------|
| **DPFLA**（主路径） | ✅ 实现 | `fl_algorithm/DPFLA.py` |
| **FedLA** | ✅ 实现 | `fl_algorithm/fed_avg.py` |
| **FedBN** | ✅ 实现（对 VisDrone 无效） | `fl_algorithm/fed_avg.py` |
| FedAvg | ✅ 实现 | `fl_algorithm/fed_avg.py` |
| FoolsGold | ✅ 实现 | `fl_algorithm/fools_gold.py` |
| Trimmed Mean | ✅ 实现 | `fl_algorithm/t_mean.py` |
| Median | ✅ 实现 | `fl_algorithm/median.py` |
| Multi-Krum | ✅ 实现 | `fl_algorithm/m_krum.py` |
| SCAFFOLD | 未实现 | — |
| FedProx | 未实现 | — |

---

## §DPFLA 算法详解

### 主路径：SVD + K-Means

```python
def DPFLA.score(global_model, local_models, clients_types, selected_clients, ...):
    # 1. 特征提取：取检测头 cv3.weight 参数
    # 2. 奇异值分解：Z = W @ G^{-1} @ U_mask
    # 3. 二维投影：取前两个奇异值对应的特征向量
    # 4. K-Means 聚类：k=2，多数派=诚实，少数派=恶意
    # 5. 辅助评分：MAD z-score 异常度，与 KMeans 组合生成软权重
    # 6. 跨轮惩罚：连续被判为恶意则逐轮加重
```

### 关键组件

#### K-Means 异常检测

```python
coefficient = silhouette_score(data, labels)
if coefficient < 0.61:
    return np.ones(n)  # 聚类质量差，全判定为诚实

majority_label = unique_labels[np.argmax(counts)]
scores = [1 if lbl == majority_label else 0 for lbl in labels]
```

#### 软权重策略

```
轮廓系数 ≥ 0.72：bad 权重压到 0.02
轮廓系数 [0.58, 0.72)：辅助异常分参与柔性惩罚
轮廓系数 < 0.58：仅用辅助特征做温和防守
```

#### 跨轮惩罚/恢复

- 连续被标为 bad：`soft = max(0.02, 0.20 * 0.72^(streak-1))`
- 从 bad 恢复：缓释权重，逐轮回到 1.0

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

## §VisDrone 数据集

### 10 个类别

```
pedestrian(0), people(1), bicycle(2), car(3), van(4), truck(5),
tricycle(6), awning-tricycle(7), bus(8), motor(9)
```

### 数据准备

```bash
python convert_visdrone_to_yolo.py --mode visdrone10 \
    --root /root/autodl-tmp/data/visdrone --deploy
```

---

## §YOLO + 联邦学习集成

### 整体架构

```
main.py / server.py → fl_core.run_experiment()
    → 选客户端
    → 并行本地训练（client.py）
    → 聚合（fl_algorithm/*）
    → 评估（YOLO val 子进程）
```

### YOLOWrapper

`models/yolo_wrapper.py`：封装 Ultralytics YOLOv8，预训练 backbone + nc=10 检测头替换。

### 检测头参数选择（DPFLA YOLO 路径）

```python
preferred_keys = ['model.22.cv3.0.2.weight', 'model.22.cv3.1.2.weight', 'model.22.cv3.2.2.weight']
```

---

## §指标体系

| 任务 | 指标 |
|------|------|
| 分类 | accuracy |
| VisDrone | mAP@0.5（YOLO `val()`，`metrics.box.map50`） |

评估在子进程中执行，避免 `inference_mode` 污染训练状态。

---

## §设计决策记录

| 决策 | 原因 | 状态 |
|------|------|------|
| 评估禁用 fallback | 避免指标口径分叉 | ✅ 冻结 |
| 聚合精度 float32 | float16 累积误差影响 BN | ✅ 冻结 |
| DPFLA 主路径 = SVD+KMeans | 学术可验证 | ✅ 冻结 |
| FedLA 本地标签统计缓存 | `_cached_label_hist` 避免每轮重复扫描 | ✅ 冻结 |
| FedBN 对 VisDrone 无效 | 标签偏斜非特征偏移 | ✅ 冻结 |
