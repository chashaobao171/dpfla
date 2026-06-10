# 03_ThirdParty_Skeleton.md — 第三方参考工程最小骨架
<!-- 最后更新：2026-05-15 -->
<!-- 位置：DPFLA-master/YOLOv8_Federated_Learning-main/ -->
<!-- 用途：仅供结构借鉴，不承载主工程方法结论 -->

---

## 核心价值（一句话）

直接调用 Ultralytics `YOLO.train()` / `YOLO.val()` + 对 `model.model.state_dict()` 做参数平均。

---

## 最小可借鉴骨架

### 训练（联邦版）

```python
# 1. 全局模型加载
model = YOLO('yolov8n.pt')

# 2. 复制到各客户端
client_models = [copy.deepcopy(model) for _ in range(num_clients)]

# 3. 本地训练（每个客户端调用一次 train）
for cm in client_models:
    cm.train(
        data=client_yaml,
        epochs=local_epochs,
        batch=16,
        conf=0.001,
        iou=0.5,
        workers=0,
        lr0=1e-7,
        plots=False,
    )

# 4. 聚合 state_dict
def average_models(models):
    averaged = {}
    for key in models[0].model.state_dict():
        tensors = torch.stack([m.model.state_dict()[key].float() for m in models])
        averaged[key] = tensors.mean(dim=0)
    return averaged

model.model.load_state_dict(average_models(client_models))
```

### 评估

```python
metrics = model.val(
    data=global_yaml,
    split='test',
    imgsz=640,
    conf=0.001,
    iou=0.5,
    augment=False,
    plots=False,
    save_json=False,
    workers=0,
    verbose=False,
)
mAP50 = metrics.box.map50
```

---

## 与主工程的差异（不可直接照搬）

| 第三方 | 主工程 |
|--------|--------|
| 每个客户端独立 YAML | Dirichlet 分布划分同一数据集 |
| float16 聚合 | float32 聚合 |
| Ultralytics 原生训练 | YOLOWrapper + 手写 SGD |
| 单类 face 检测 | VisDrone 10 类检测 |
| 无防御 | DPFLA 防御 |

---

## 数据路径（参考）

```
/home/featurize/data/
├── images/trainA / trainB / trainC / trainD   # 第三方用这个
├── images/validation / images/test
├── labels/trainA / trainB / ...
└── labels/validation / labels/test
```

**注意**：主工程 VisDrone 数据在 `/root/autodl-tmp/data/visdrone`，与第三方结构不同。

---

## 迁移时只看这几处

- `YOLOv8_Federated_Learning-main/FedAvg_train.py`：主循环结构
- `YOLOv8_Federated_Learning-main/model_eval.py`：YOLO val 调用方式
- **不迁移**：YAML（Windows 路径）、单类假设、`resume=True`
