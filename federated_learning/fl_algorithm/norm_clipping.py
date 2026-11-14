"""Norm-Clipping / NDC 聚合（Sun et al., "Can You Really Backdoor Federated Learning?";
Attack of the Tails, NeurIPS'20 的 NDC baseline；FLPoison `aggregators/normclipping.py`）。

本项目适配：
- 输入为客户端 state_dict 列表；按 `w_i - w_global` 计算每个客户端的更新量；
- 对更新量做 L2 范数裁剪，阈值默认 = kappa × median(||delta_i||)（FLAME 的 median-norm 做法）；
- 裁剪后等权平均，再加回全局权重；
- 非浮点 buffer 沿用 FedAvg 语义（num_batches_tracked 取最大，其余取第一个）。
"""

import numpy as np
import torch
from loguru import logger


def _float_keys(base_state):
    return [
        k for k in base_state.keys()
        if torch.is_tensor(base_state[k])
        and base_state[k].is_floating_point()
        and base_state[k].numel() > 0
    ]


def norm_clipping(weights, base_state, kappa=1.0, threshold=None, eps=1e-12):
    """返回裁剪平均后的 state_dict。"""
    if len(weights) == 0:
        raise ValueError("Norm-Clipping: 没有客户端权重")

    keys = _float_keys(base_state)
    deltas = []
    norms = []
    for w in weights:
        parts = []
        d = {}
        for k in keys:
            delta = (
                w[k].detach().cpu().float() - base_state[k].detach().cpu().float()
            ).reshape(-1)
            d[k] = delta
            parts.append(delta)
        vec = torch.cat(parts) if parts else torch.zeros(0)
        deltas.append(d)
        norms.append(float(vec.norm().item()))

    if threshold is None:
        threshold = float(kappa) * float(np.median(norms)) if norms else 0.0
    threshold = max(float(threshold), 0.0)

    scales = [min(1.0, threshold / (n + eps)) for n in norms]
    clipped = sum(1 for s in scales if s < 1.0)

    result = {}
    for k in keys:
        vals = []
        for i, d in enumerate(deltas):
            vals.append(d[k] * scales[i])
        agg = torch.stack(vals, dim=0).mean(dim=0)
        result[k] = (
            base_state[k].detach().cpu().float().reshape(-1) + agg
        ).reshape(base_state[k].shape).to(base_state[k].dtype)

    # 非浮点 buffer / 非 tensor：沿用 FedAvg 语义
    for k, v in weights[0].items():
        if k in result:
            continue
        if torch.is_tensor(v) and not v.is_floating_point():
            if "num_batches_tracked" in k:
                result[k] = max(weights[i][k] for i in range(len(weights)))
            else:
                result[k] = v
        else:
            result[k] = v

    logger.info(
        f"Norm-Clipping: n={len(weights)}, threshold={threshold:.6f} "
        f"(kappa={kappa}), clipped={clipped}/{len(weights)}, "
        f"norm_median={float(np.median(norms)):.6f}"
    )
    return result
