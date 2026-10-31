# CIFAR10 标签反转线 50 轮 × 24 step 双指标对比（2026-10-31）

口径：10 客户端 / 50 轮 / 每轮 24 step / lr 0.01 / BS64 / IID / 30% 恶意 / `FL_GRADIENT_SCALE=1`。
- `global` = 全测试集 10 类总准确率（项目主表口径）
- `attacked_class (dog)` = 被攻击源类 dog 的准确率（标签翻转防御是否真的生效）
- `末5/末10` = 最后 5 / 10 轮原始值均值（不做 TensorBoard 平滑）
- TensorBoard 平滑（0.6）只是可视化 EMA，滞后于原始曲线，不作为验收依据。

| 方法 | global R50 | global 末5 | global 末10 | attacked R50 | attacked 末5 | attacked 末10 | attacked best |
|---|---|---|---|---|---|---|---|
| 无攻击基线 | 71.24 | 70.78 | 70.40 | 64.7 | 60.14 | 60.93 | 69.6 |
| 标签 + FedAvg | 68.70 | 68.79 | 68.11 | 24.7 | 25.28 | 23.79 | 31.2 |
| **标签 + DPFLA** | 67.57 | 67.78 | 67.30 | **60.5** | **60.00** | **60.37** | 71.3 |
| 标签 + MKrum | 64.35 | 65.31 | 65.22 | 55.4 | 59.08 | 60.43 | 80.6 |
| 标签 + TMean | **69.76** | **69.66** | 69.05 | 35.8 | 38.02 | 36.71 | 46.5 |

## 读数

- **attacked_class（defense 是否真的防住）**：DPFLA 60.0（末5）≈ 无攻击基线 60.14，**仅次无攻击**；MKrum 59.08；TMean 38.02；FedAvg 25.28。
  - DPFLA vs MKrum：R50 **+5.1**、平滑后 **+3.96**、末5 **+0.92**、末10 **−0.06**。
  - DPFLA vs TMean：R50 +24.7、末5 +21.98（稳健领先）。
- **global（总准确率）**：TMean 69.76 > FedAvg 68.70 > DPFLA 67.57 > MKrum 64.35。
  - TMean 的 global 高是因为它把 dog 判成 cat（cat 61.4 / dog 38.7）——**等于学会了投毒标签**，不是防御成功；其 attacked_class 仅 35.8。
  - DPFLA vs MKrum（global）：R50 +3.22、末5 **+2.47**、末10 **+2.08** → 稳健 ≥2。
- **结论**：按 attacked_class，DPFLA 是唯一接近无攻击基线的防御；对 TMean 领先稳健 ≥2，对 MKrum R50/平滑 ≥2 但末5仅 +0.92。按 global，DPFLA 领先 MKrum ≥2，但低于 TMean（TMean 以 attacked_class 崩溃为代价）。

## MinMax 线 global（50 轮 × 24 step，正式主指标）

| 方法 | R50 | 末5 | 末10 | best |
|---|---|---|---|---|
| 无攻击基线 | 71.24 | 70.78 | 70.40 | 71.44 |
| FedAvg | 56.58 | 56.36 | 55.88 | 56.64 |
| **DPFLA** | **68.12** | **67.84** | **67.39** | **68.15** |
| MKrum | 17.77 | 19.88 | 20.57 | 25.83 |
| TMean | 49.51 | 49.13 | 48.84 | 49.57 |

- DPFLA vs TMean：R50 +18.61、末5 **+18.71**；vs MKrum：+50.35 / **+47.96** → 稳健 ≥2 ✓
- DPFLA vs 无攻击基线：**68.12 < 71.24（−3.12）**，并没有“比无攻击还好”。
- ⚠️ MinMax 线的 `accuracy/attacked_class` 是脚本占位的 `source_class=0`（CIFAR10 的 airplane），不是防御指标；它显示 DPFLA 73.8 > 基线 64.7 只代表 airplane 单类准确率，不能用于判断“防住没有”。MinMax 请看 `accuracy/global`。
