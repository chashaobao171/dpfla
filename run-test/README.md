# run-test 脚本说明（按数据集归类）

当前测试脚本按数据集拆分为 3 个子目录：

- `run-test/mnist/`
- `run-test/cifar10/`
- `run-test/visdrone/`

统一原则：

1. DPFLA 主实验固定为 `SVD + K-Means` 路径（`rule='DPFLA'`）。
2. 攻击场景统一使用 `label_flipping`。
3. 每个数据集保留两类脚本：
   - `run_no_attack_baseline.py`：无攻击基线（FedAvg）
   - `run_dpfla_label_flipping.py`：标签翻转攻击 + DPFLA 防御

---

## 运行方式

### MNIST

```bash
python run-test/mnist/run_no_attack_baseline.py
python run-test/mnist/run_dpfla_label_flipping.py
```

### CIFAR10

```bash
python run-test/cifar10/run_no_attack_baseline.py
python run-test/cifar10/run_dpfla_label_flipping.py
```

### VisDrone (YOLO)

```bash
python run-test/visdrone/run_no_attack_baseline.py
python run-test/visdrone/run_dpfla_label_flipping.py
```

---

## 结果解读

- 基线脚本用于给出“无攻击”可达到的性能上限。
- 攻击+防御脚本用于验证：在 `label_flipping` + 恶意率 30% 下，DPFLA(SVD+K-Means) 是否仍保持可用性能。
- 所有日志输出到 `logs_3/`。
