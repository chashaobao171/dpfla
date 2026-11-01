# 正式验收日志索引（2026-11-01）

本目录只收录最终验收矩阵的 **18 条完整 50 轮日志**：CIFAR10 9 条、VisDrone 9 条。`logs_3/` 保留同名原始日志；本目录按角色编号保存副本，便于对外引用。

- 恶意客户端比例均为 30%，不放大梯度。
- 主口径为末 5 轮均值，同时记录 R50。
- CIFAR MinMax 线为 24 step/轮；无攻击基线为 48 step/轮；标签翻转线为 **M5 全类循环映射 + 48 step/轮**。
- VisDrone 为 12 step/轮，指标为 mAP@0.5（%）。
- 旧 5 轮、15 轮、中断、被替换及无关日志/TB 已移至 `/root/dpfla_cleanup_20261101/`，未直接删除。

## CIFAR10（准确率 %）

| 攻击 | 方法 | R50 | 末5 | 收录文件 |
|---|---|---:|---:|---|
| 无攻击 | FedAvg | 73.74 | 73.59 | `cifar10/01_baseline__run_no_attack_baseline_20261101_1200.log` |
| 标签翻转 M5 | FedAvg | 55.66 | 56.16 | `cifar10/02_attack_nodefense_label__run_no_defense_label_flipping_20261101_1103.log` |
| 标签翻转 M5 | DPFLA | **70.25** | **70.25** | `cifar10/03_attack_dpfla_label__run_dpfla_label_flipping_20261101_0952.log` |
| 标签翻转 M5 | Multi-Krum | 67.52 | 67.37 | `cifar10/04_attack_mkrum_label__run_mkrum_label_flipping_20261101_1017.log` |
| 标签翻转 M5 | Trimmed-Mean | 66.85 | 64.17 | `cifar10/05_attack_tmean_label__run_tmean_label_flipping_20261101_1041.log` |
| MinMax | FedAvg | 56.58 | 56.36 | `cifar10/06_attack_nodefense_minmax__run_screen_minmax_fedavg_20261031_1131.log` |
| MinMax | DPFLA | **68.12** | **67.84** | `cifar10/07_attack_dpfla_minmax__run_screen_minmax_DPFLA_20261031_1154.log` |
| MinMax | Multi-Krum | 17.77 | 19.88 | `cifar10/08_attack_mkrum_minmax__run_screen_minmax_mkrum_20261031_1215.log` |
| MinMax | Trimmed-Mean | 49.51 | 49.13 | `cifar10/09_attack_tmean_minmax__run_screen_minmax_tmean_20261031_1237.log` |

标签线末 5 轮：DPFLA 领先 Multi-Krum **+2.88**、Trimmed-Mean **+6.08**。MinMax 线末 5 轮分别领先 **+47.96 / +18.71**。

## VisDrone（mAP@0.5 %）

| 攻击 | 方法 | R50 | 末5 | 收录文件 |
|---|---|---:|---:|---|
| 无攻击 | FedAvg | 39.65 | 39.12 | `visdrone/01_baseline__run_no_attack_baseline_20261031_1258.log` |
| MinMax | FedAvg | 30.23 | 29.89 | `visdrone/02_attack_nodefense_minmax__run_screen_minmax_fedavg_20261031_1729.log` |
| MinMax | DPFLA | **38.36** | **37.85** | `visdrone/03_attack_dpfla_minmax__run_screen_minmax_DPFLA_20261031_1504.log` |
| MinMax | Multi-Krum | 14.09 | 13.86 | `visdrone/04_attack_mkrum_minmax__run_screen_minmax_mkrum_20261031_2212.log` |
| MinMax | Trimmed-Mean | 28.12 | 27.70 | `visdrone/05_attack_tmean_minmax__run_screen_minmax_tmean_20261031_1951.log` |
| IPM | FedAvg | 35.87 | 35.36 | `visdrone/06_attack_nodefense_ipm__run_screen_ipm_fedavg_20261101_0240.log` |
| IPM | DPFLA | **38.32** | **37.99** | `visdrone/07_attack_dpfla_ipm__run_screen_ipm_DPFLA_20261101_0034.log` |
| IPM | Multi-Krum | 26.17 | 25.55 | `visdrone/08_attack_mkrum_ipm__run_screen_ipm_mkrum_20261101_0655.log` |
| IPM | Trimmed-Mean | 32.55 | 32.08 | `visdrone/09_attack_tmean_ipm__run_screen_ipm_tmean_20261101_0446.log` |

MinMax 线末 5 轮：DPFLA 领先 Multi-Krum **+23.99**、Trimmed-Mean **+10.15**。IPM 线分别领先 **+12.44 / +5.91**。两条攻击均满足 DPFLA 相对两种防御领先不少于 1 个百分点的目标。

## TensorBoard

`tb/` 与 `/root/tf-logs/` 均只保留与上述 18 条日志对应的 18 个 run，每个 run 的 `accuracy/global` 均为 50 个步点。原基线 first5 切片和 MNIST 旧 run 已一并移至清理归档。
