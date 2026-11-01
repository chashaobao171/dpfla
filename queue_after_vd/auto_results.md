# VisDrone 结束后续跑 + CIFAR 多映射重跑

- 更新时间：2026-11-01 10:17:15
- 说明：主口径末 5 轮；标签线同时看 global 与 attacked_class。

| # | 实验 | 数据集 | 攻击 | 方法 | 状态 | best | last | 末5 | 末10 | 耗时(min) | 日志 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | visdrone_ipm_tmean | visdrone | ipm | tmean | success | 32.55 | 32.55 | 32.08 | 31.47 | 129.6 | run_screen_ipm_tmean_20261101_0446.log |
| 2 | visdrone_ipm_mkrum | visdrone | ipm | mkrum | success | 26.17 | 26.17 | 25.55 | 24.73 | 131.2 | run_screen_ipm_mkrum_20261101_0655.log |
| 3 | relabel_DPFLA | cifar10 | label_flipping | DPFLA | success | 70.71 | 70.25 | 70.25 | 70.25 | 24.3 | run_dpfla_label_flipping_20261101_0952.log |
