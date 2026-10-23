"""
Multi-Krum 聚合（Blanchard et al., "Machine Learning with Adversaries:
Byzantine Tolerant Gradient Descent", NeurIPS'17）。

本项目口径（标准定义）：
1. 每个客户端上传完整模型 state_dict（与 FedAvg / DPFLA 同一接口）；
2. 所有浮点张量按 key 排序展平为 1 维向量（各客户端 key 顺序一致）；
3. 客户端 i 的 score = 到其余客户端最近的 (n - f - 2) 个欧氏距离之和；
4. 选 score 最小的 m 个客户端等权平均，m 默认 = n - f - 2。

与 FLPoison（`aggregators/multikrum.py`）的差异（写报告时需注明）：
- FLPoison 的 `krum_compute_scores` 取 n-f-1 个最近距离，默认 `avg_percentage=0.2`；
- 本实现取标准定义的 n-f-2，默认 m=n-f-2；
- 可用环境变量覆盖：`FL_MKRUM_NEIGHBORS`（邻居数）、`FL_MKRUM_M`（选中客户端数）。

注意：距离对"绝对参数"与"相对全局模型的 delta"等价（共同平移项在距离中抵消），
因此本函数不需要额外传入全局权重。
"""

import os

import numpy as np
import torch
from loguru import logger

from .fed_avg import average_weights


def _flatten_state_dict(weights):
    """按 key 排序展平所有浮点张量，保证跨客户端向量顺序一致。"""
    parts = []
    for key in sorted(weights.keys()):
        value = weights[key]
        if torch.is_tensor(value) and value.is_floating_point():
            parts.append(value.detach().cpu().reshape(-1).float())
    if not parts:
        raise ValueError("Multi-Krum: state_dict 中没有可展平的浮点参数")
    return torch.cat(parts)


def _env_int(name):
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"{name}={raw!r} 不是整数，已忽略")
        return None


def multi_krum(weights, num_attackers, m=None):
    """
    :param weights: list[dict]，每个客户端的 state_dict
    :param num_attackers: f，预期恶意客户端数
    :param m: 参与平均的客户端数；None → n-f-2（可用 FL_MKRUM_M 覆盖）
    :return: (聚合后的 state_dict, scores)，scores[i]=1 表示被选中
    """
    n = len(weights)
    f = int(num_attackers)
    if n == 0:
        raise ValueError("Multi-Krum: 没有客户端权重")

    # Krum 的理论前提：n >= 2f + 3；不满足时不硬套，回退等权 FedAvg 并告警
    min_required = 2 * f + 3
    if n < min_required:
        logger.warning(
            f"Multi-Krum 前提不满足：n={n} < 2f+3={min_required}，回退为等权 FedAvg"
        )
        scores = np.ones(n, dtype=float)
        return average_weights(weights, scores, float16_floats=False), scores

    neighbors = _env_int('FL_MKRUM_NEIGHBORS') or (n - f - 2)
    neighbors = max(1, min(neighbors, n - 1))

    if m is None:
        m = _env_int('FL_MKRUM_M') or (n - f - 2)
    m = max(1, min(int(m), n))

    vectors = torch.stack([_flatten_state_dict(w) for w in weights], dim=0)  # [n, d]
    distances = torch.cdist(vectors, vectors, p=2)                          # [n, n]
    distances.fill_diagonal_(float('inf'))                                  # 排除自身
    nearest = torch.topk(distances, neighbors, dim=1, largest=False).values
    krum_scores = nearest.sum(dim=1)

    selected = torch.argsort(krum_scores)[:m].tolist()
    scores = np.zeros(n, dtype=float)
    scores[selected] = 1.0

    logger.info(
        f"Multi-Krum: n={n}, f={f}, 邻居数={neighbors}, 选中 m={m} → {selected}"
    )
    return average_weights(weights, scores, float16_floats=False), scores
