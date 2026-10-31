# 50 轮全量实验自动汇总（阶段 A）

- 更新时间：2026-10-31 11:02:15
- 说明：主口径为末 5 轮均值；`last` 为第 50 轮；`best` 为 50 轮内最佳。

| # | 实验 | 数据集 | 攻击 | 方法 | 状态 | best | last | 末5 | 末10 | 耗时(min) | 日志 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | cifar10_baseline | cifar10 | no_attack | fedavg | success | 71.44 | 71.24 | 70.78 | 70.4 | 14.1 | run_no_attack_baseline_20261031_1019.log |
| 2 | cifar10_label_fedavg | cifar10 | label_flipping | fedavg | success | 69.19 | 68.7 | 68.79 | 68.11 | 14.1 | run_no_defense_label_flipping_20261031_1033.log |
| 3 | cifar10_label_dpfla | cifar10 | label_flipping | DPFLA | success | 68.28 | 67.57 | 67.78 | 67.3 | 14.7 | run_dpfla_label_flipping_20261031_1047.log |
