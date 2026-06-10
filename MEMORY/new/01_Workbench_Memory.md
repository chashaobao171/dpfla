# 01_Workbench_Memory.md — 深度工作台记忆

<!-- 最后更新：2026-06-09 -->
<!-- 每次重要实验完成后追加，记录结论 -->
<!-- 不覆盖历史结论，新增内容追加到顶部 -->

---

## §当前进度（2026-06-09）

### 重大发现：集中式 oracle 结果

| 轮次 | mAP@0.5 | Precision | Recall |
|------|---------|-----------|--------|
| Epoch 1 | 40.6% | 0.508 | 0.435 |
| Epoch 2 | 40.7% | 0.52 | 0.411 |
| Epoch 3 | 41.4% | 0.536 | 0.42 |

- **结论**：集中式 3 epochs ≈ 41%，FedAvg 20 rounds ≈ 22%，差距 2x
- **问题根因**：不在数据集/模型，在联邦训练流程本身
- **排查方向**：B1 conf=0.001（联邦）、B2 drop_last=False、B3 训练/评估对齐

### 已完成实验

| 实验 | 日志 | 结果 |
|------|------|------|
| **集中式 YOLOv8l oracle** | `oracle_yolov8l_20260609_1552.log` | **41.4% (3 epochs)** |
| FedAvg 无攻击基线 | `run_no_attack_baseline_20260608_0755.log` | Round 6: mAP@0.5 = 21.93% |

### 冻结结论

- **FedBN 对 VisDrone Non-IID 无效**：VisDrone 是标签分布偏斜（而非特征偏移），FedBN 的 BN 统计量保留优化不适用。FedAvg 全程领先 FedBN。
- **mAP 停滞在 20-22%**：三层叠加根因（聚合失效 + BN 失效 + 少数类淹没），FedLA 是当前最值得期待的方向。

---

## §实验结果汇总

### MNIST（已冻结）

- 无攻击基线：FedAvg，6 客户端，10 轮 → 94%
- DPFLA 防御：标签翻转，恶意率 30% → 95%（vs FedAvg 4.57%）
- 日志：`logs_3/mnist/run_no_defense_label_flipping_20260515_1020.log`
- 日志：`logs_3/mnist/run_dpfla_label_flipping_20260515_1036.log`

### VisDrone 无攻击基线

```
Round 1:  FedAvg  7.81% | FedBN  6.08%
Round 2:  FedAvg 16.78% | FedBN 15.45%
Round 3:  FedAvg 19.50% | FedBN 17.92%
Round 4:  FedAvg 21.16% | FedBN 20.18%
Round 5:  FedAvg 21.68% | FedBN 20.42%
Round 6:  FedAvg 21.93% | FedBN 20.34%
Round 7:  FedAvg     —   | FedBN 21.55% (峰值)
Round 8:  FedAvg     —   | FedBN 20.94%
```

### VisDrone 标签翻转 + DPFLA（SVD+KMeans）

- 历史结果：mAP 曲线与 FedAvg 高度接近，SVD+KMeans 单簇频繁，恶意命中率 0%
- 原因：YOLO 特征选取（fc2.weight）效果差，后改为检测头 cv3.weight
- 当前状态：待重新运行对比实验

### VisDrone 高斯投毒

- FedAvg：高恶意率下 mAP=0%，无法抵御
- DPFLA（验证集打分）：能评分攻击客户端为 0，恢复 mAP

---

## §环境配置

### 路径映射

| 资源 | 路径 |
|------|------|
| 项目根 | `/root/chashaobao/DPFLA-master` |
| VisDrone | `/root/autodl-tmp/data/visdrone/` |
| YOLO 标签 | `labels_yolo_visdrone10/`（symlink 在 labels/ 下） |
| MNIST | `/root/autodl-tmp/data/MNIST/` |
| CIFAR10 | `/root/chashaobao/data/` |

### 依赖

```bash
pip install -r requirements.txt
pip uninstall opencv-python -y
pip install --force-reinstall --no-deps numpy==1.26.4 opencv-python-headless==4.8.1.78
```

### 依赖路径的文件

`fl_core.py`、`sampling.py`、`visdrone_dataset.py`、`server.py`、`generate_visdrone_yaml.py`、`convert_visdrone_to_yolo.py`、`requirements.txt`

---

## §故障字典

| # | 现象 | 原因 | 修复 |
|---|------|------|------|
| 1 | YOLO 加载失败（libGL.so.1） | opencv-python vs headless | `pip install opencv-python-headless` |
| 2 | YOLO 评估子进程失败 | `device=cpu` 未加引号 | CPU 时用 `"'cpu'"` 字面量 |
| 3 | 评估后 loss 无梯度/NaN | inference_mode 污染 | 评估在子进程执行 |
| 4 | pickle truncate | 缓存损坏 | `rm cache/cache_yolo_*.pkl` |
| 5 | 初始 mAP 全 0 | 换头后冷启动，或标签未部署 | 检查 labels/ symlink |
| 6 | GPU 利用率低 | CPU 在做缓存解析 | 等缓存生成完再观察 |

---

## §TensorBoard

```bash
tensorboard --logdir runs/ --port 6006
```

曲线目录：`runs/VisDrone_YOLO_<rule>_attack-<type>_mr-<rate>/`

---

## §冻结决策时间线

| 日期 | 决策 | 状态 |
|------|------|------|
| 2026-06-08 | FedBN 对 VisDrone 无效（标签偏斜 vs 特征偏移），FedAvg 全程领先 | ✅ 冻结 |
| 2026-06-08 | FedLA 实施完成（`average_weights_fedla()` + `rule='fedla'` 分支），待运行验证 | 🔄 进行中 |
| 2026-06-08 | FedLA 上次运行被手动关闭，已清理残余记录 | ✅ 完成 |
| 2026-06-07 | FedBN 实施完成，训练验证 Round 7 峰值 21.55% | ✅ 冻结 |
| 2026-06-06 | Local epochs = 10（FedLA 论文推荐值，调研确认不改） | ✅ 冻结 |
| 2026-05-15 | DPFLA MNIST 验证成功（mAP 95% vs 4.57%） | ✅ 冻结 |
| 2026-05-14 | 环境迁移到 autodl，路径统一 | ✅ 冻结 |
| 2026-04-11 | YOLO 评估禁用 fallback | ✅ 冻结 |
| 2026-04-11 | YOLO 聚合统一 float32 | ✅ 冻结 |

---

## §历史记录（不活跃）

### 文献调研（2026-06-01，详见 visdrone_fl_research_report.md）

- FedLA：IEEE IV 2024，mAP +5~6%，GitHub: TixXx1337/...
- FL-JSDDC：Frontiers 2026，VisDrone 专用，mAP +3%，收敛 2.2x，无开源
- FedBN：ICLR 2021，对特征偏移有效，对 VisDrone 标签偏斜无效
- BN-SCAFFOLD：arXiv:2410.03281，解决 BN 对 SCAFFOLD 的破坏

### 数据准备（2026-05-14）

```bash
python convert_visdrone_to_yolo.py --mode visdrone10 --root /root/autodl-tmp/data/visdrone --deploy
```

### 标签翻转攻击配置

```python
HIGH_FREQ_POOL = [0, 1, 2]
LOW_FREQ_TARGET_POOL = [9]
MALICIOUS_BEHAVIOR_RATE = 0.26
```

实现：`client.py:participant_update()` → `label_flipping()`，只翻转 labels，不改 boxes。
