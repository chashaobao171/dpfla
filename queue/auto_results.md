# 50 轮全量实验自动汇总（阶段 A）

- 更新时间：2026-10-31 12:58:32
- 说明：主口径为末 5 轮均值；`last` 为第 50 轮；`best` 为 50 轮内最佳。

| # | 实验 | 数据集 | 攻击 | 方法 | 状态 | best | last | 末5 | 末10 | 耗时(min) | 日志 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | cifar10_baseline | cifar10 | no_attack | fedavg | success | 71.44 | 71.24 | 70.78 | 70.4 | 14.1 | run_no_attack_baseline_20261031_1019.log |
| 2 | cifar10_label_fedavg | cifar10 | label_flipping | fedavg | success | 69.19 | 68.7 | 68.79 | 68.11 | 14.1 | run_no_defense_label_flipping_20261031_1033.log |
| 3 | cifar10_label_dpfla | cifar10 | label_flipping | DPFLA | success | 68.28 | 67.57 | 67.78 | 67.3 | 14.7 | run_dpfla_label_flipping_20261031_1047.log |
| 4 | cifar10_label_mkrum | cifar10 | label_flipping | mkrum | success | 66.43 | 64.35 | 65.31 | 65.22 | 14.8 | run_mkrum_label_flipping_20261031_1102.log |
| 5 | cifar10_label_tmean | cifar10 | label_flipping | tmean | success | 70.05 | 69.76 | 69.66 | 69.05 | 14.4 | run_tmean_label_flipping_20261031_1117.log |
| 6 | cifar10_minmax_fedavg | cifar10 | minmax | fedavg | success | 56.64 | 56.58 | 56.36 | 55.88 | 22.5 | run_screen_minmax_fedavg_20261031_1131.log |
| 7 | cifar10_minmax_dpfla | cifar10 | minmax | DPFLA | success | 68.15 | 68.12 | 67.84 | 67.39 | 21.1 | run_screen_minmax_DPFLA_20261031_1154.log |
| 8 | cifar10_minmax_mkrum | cifar10 | minmax | mkrum | success | 25.83 | 17.77 | 19.88 | 20.57 | 21.6 | run_screen_minmax_mkrum_20261031_1215.log |
| 9 | cifar10_minmax_tmean | cifar10 | minmax | tmean | success | 49.57 | 49.51 | 49.13 | 48.84 | 21.4 | run_screen_minmax_tmean_20261031_1237.log |
