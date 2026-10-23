"""
Trimmed-Mean 聚合（Yin et al., "Byzantine-Robust Distributed Learning:
Towards Optimal Statistical Rates", ICML'18）。

本项目口径：
- 对每个参数 key，把各客户端的该张量展平后**逐坐标**排序；
- 去掉最大 / 最小各 trim_num 个值，对剩余坐标求均值；
- trim_num = int(trim_ratio * n)，trim_ratio 默认取恶意率（本项目 0.3），必须 < 0.5；
- 非浮点参数（如 `num_batches_tracked`）沿用 FedAvg 语义：计数取最大、其余取第一个客户端。

与 FLPoison（`aggregators/trimmedmean.py`，默认 `beta=0.1`）的差异：
- 本项目默认 `trim_ratio` = 恶意率（30%），否则恶意端无法被裁掉；
- 可用环境变量 `FL_TRIM_RATIO` 覆盖。
"""

import torch
from loguru import logger

from .fed_avg import average_weights


def trimmed_mean(weights, trim_ratio):
    """
    :param weights: list[dict]，每个客户端的 state_dict
    :param trim_ratio: 每端裁剪比例（0 < trim_ratio < 0.5）
    :return: 聚合后的 state_dict
    """
    n = len(weights)
    if n == 0:
        raise ValueError("Trimmed-Mean: 没有客户端权重")

    trim_ratio = float(trim_ratio)
    if trim_ratio <= 0:
        logger.info("Trimmed-Mean: trim_ratio<=0，退化为等权 FedAvg")
        return average_weights(weights, [1 for _ in range(n)], float16_floats=False)
    if trim_ratio >= 0.5:
        raise ValueError(f"Trimmed-Mean: trim_ratio={trim_ratio} 必须 < 0.5")

    trim_num = int(trim_ratio * n)
    keep = n - 2 * trim_num
    if trim_num == 0 or keep <= 0:
        logger.warning(
            f"Trimmed-Mean: n={n}, trim_ratio={trim_ratio} 得到 trim_num={trim_num}, "
            f"保留 {keep} → 退化为等权 FedAvg"
        )
        return average_weights(weights, [1 for _ in range(n)], float16_floats=False)

    result = {}
    for key in weights[0].keys():
        v0 = weights[0][key]
        if not torch.is_tensor(v0):
            result[key] = v0
            continue

        if not v0.is_floating_point():
            # 与 FedAvg 保持一致：num_batches_tracked 取最大，其余非浮点取第一个
            if 'num_batches_tracked' in key:
                result[key] = max(weights[i][key] for i in range(n))
            else:
                result[key] = v0
            continue

        stacked = torch.stack([weights[i][key].float() for i in range(n)], dim=0)
        flat = stacked.reshape(n, -1)
        sorted_flat = flat.sort(dim=0).values
        trimmed = sorted_flat[trim_num:n - trim_num]
        result[key] = trimmed.mean(dim=0).reshape(v0.shape).to(dtype=v0.dtype)

    logger.info(
        f"Trimmed-Mean: n={n}, trim_ratio={trim_ratio:.2f}, 每端裁 {trim_num} 个, 保留 {keep} 个"
    )
    return result
