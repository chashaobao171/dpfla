# log/ — 合格可用日志归档

> 只放**经过筛选、可对外引用**的实验日志。原始未筛选日志在 `logs_3/`。

## 收录标准

1. **不以报错告终**：末尾无 `Traceback` / `测试失败` / 非零 `exit`。
2. **指标达标**：基线与防御组需达到预期水平；**"被攻击无防御"允许低分**（攻击成功正是其意义），但其**攻击比例必须是 30%**。
3. **一个数据集至少收录三份**（口径尽量一致）：
   - `01_baseline` —— 无攻击 + FedAvg
   - `02_attack_nodefense` —— 有攻击 + FedAvg（无防御）
   - `03_attack_dpfla` —— 有攻击 + **DPFLA**（防御）
   - 防御方法对比线额外收录 MKrum / TMean 两份
4. 命名：`<序号>_<角色>__<原始文件名>.log`（保留原始名以便溯源）。

## 索引

| 数据集 | 角色 | 指标 | 攻击率 | 原始出处 |
|---|---|---|---|---|
| **VisDrone** | ① 基线 | **mAP@0.5 = 47.64%**（R13 峰值；25 轮平台 ≈47%，R25 45.73%）| 0% | `run_no_attack_baseline_20261021_1421.log` |
| VisDrone | ② 攻击无防御 | **mAP@0.5 = 32.37%**（R10；基线 45.08%，−12.7）| **30%** | `run_no_defense_gaussian_20261021_1231.log`（高斯 σ=0.02）|
| VisDrone | ③ 攻击+DPFLA | ⬜ 待跑 | — | — |
| **MNIST** | ① 基线 | 准确率 **95.81%** | 0% | `demo1_mnist_fedavg_noattack_20260902_1113.log` |
| MNIST | ② 攻击无防御 | 准确率 **4.57%**（被击穿）| **30%** | `run_no_defense_label_flipping_20260515_1020.log` |
| MNIST | ③ 攻击+DPFLA | 准确率 **95.00%**（防住）| **30%** | `run_dpfla_label_flipping_20260515_1036.log` |
| **CIFAR10**（ResNet18CIFAR，2026-10-23 新口径） | ① 基线 | 准确率 **71.47%** | 0% | `01_baseline__run_no_attack_baseline_20261023_1257.log` |
| CIFAR10 | ② 标签翻转 · 无防御 | **65.68%** | 30% | `02_attack_nodefense_label__..._1301.log` |
| CIFAR10 | ③ 标签翻转 · DPFLA | **70.00%** | 30% | `03_attack_dpfla_label__..._1502.log` |
| CIFAR10 | ④ 标签翻转 · Multi-Krum | **67.00%** | 30% | `04_attack_mkrum_label__..._1311.log` |
| CIFAR10 | ⑤ 标签翻转 · Trimmed-Mean | **66.83%** | 30% | `05_attack_tmean_label__..._1316.log` |
| CIFAR10 | ⑥ MinMax · 无防御 | **60.69%**（末5均值 58.78%）| 30% | `06_attack_nodefense_minmax__..._1651.log` |
| CIFAR10 | ⑦ MinMax · DPFLA | **69.25%**（末5均值 **67.75%**）| 30% | `07_attack_dpfla_minmax__..._1651.log` |
| CIFAR10 | ⑧ MinMax · Multi-Krum | **29.19%**（末5均值 29.86%）| 30% | `08_attack_mkrum_minmax__..._1651.log` |
| CIFAR10 | ⑨ MinMax · Trimmed-Mean | **57.01%**（末5均值 54.32%）| 30% | `09_attack_tmean_minmax__..._1651.log` |

> VisDrone ② 已产出（高斯 σ=0.02）；③ 攻击+DPFLA 防御**待跑**，产出并达标后补入本目录。

## 备注

- VisDrone 基线跑在 **v8m @1280 / BS=16 / AMP**（新主机 4090 48G），5 轮达 45.08%，相对旧口径（1024/v8s，15 轮 35.92%）大幅提升。
- MNIST 基线取自 2026-09-02 的一轮完整 no-attack FedAvg（95.81%）；攻击组取自 2026-05-15 的 30% 口径实验（该组是本项目 MNIST 的经典结论）。两组来源批次不同，引用时请注明。
- **CIFAR10 已整体换代为 ResNet18CIFAR 新口径**（2026-10-23）：模型由 `CNNCifar10` 换为 CIFAR 适配版 ResNet（借用 FLPoison 实现），并加入 Multi-Krum / Trimmed-Mean 对比。
  - 旧 CNN 口径的全部产物（60.74% 基线 / 34.87% / 61.99% 及原始日志、TB）**已按用户要求删除**；换模型前的 60.74% 对照值保留在 `MEMORY/01_Facts.md` §5b。
  - 新口径基线 71.47%（10 客户端）、74.25%（7 客户端对照）。
  - 标签翻转线：DPFLA 70.00%，相对 Multi-Krum 67.00 / Trimmed-Mean 66.83 领先 +3.00 / +3.17（R15），末 5 轮均值 +2.20 / +2.11。
  - MinMax 线（服务端注入，Fang NDSS'21）：DPFLA 69.25%，Multi-Krum 29.19%，Trimmed-Mean 57.01%；末 5 轮均值 67.75 vs 29.86 / 54.32 → **领先 +37.90 / +13.43**，三种口径均 ≥2%。
  - 高斯线（σ=0.03–0.06）DPFLA 均 ≥65%，但对 Trimmed-Mean 稳健领先仅 +0.1–0.9，未达 +2%；数值保留在 `MEMORY/01_Facts.md` §5c(c)，日志已按用户要求移出归档。
