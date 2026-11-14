"""FLAME 聚合（Nguyen et al., USENIX Security'22）的 YOLO/state_dict 适配版。

原版流程：cosine 聚类筛掉离群客户端 → 用更新范数中位数裁剪 → 加高斯噪声 → 聚合。
本实现：
- 用客户端更新量 `w_i - w_global` 作为输入；
- 用 10×10 余弦距离矩阵 + AgglomerativeClustering（precomputed, average linkage）
  做聚类（环境无 hdbscan，用等价的余弦层次聚类替代）；
- 选较大簇为良性；若簇大小相同，选平均余弦距离更小的簇；
- 按全体更新范数中位数裁剪良性更新，平均后加回全局权重；
- 按 FLAME 公式加噪声：std = gamma × median_norm × param.std()（跳过 BN running stats）。
"""

import numpy as np
import torch
from loguru import logger
from sklearn.cluster import AgglomerativeClustering


def _float_keys(base_state):
    return [
        k for k in base_state.keys()
        if torch.is_tensor(base_state[k])
        and base_state[k].is_floating_point()
        and base_state[k].numel() > 0
    ]


def _flatten_delta(w, base_state, keys):
    parts = []
    for k in keys:
        parts.append(
            (w[k].detach().cpu().float() - base_state[k].detach().cpu().float()).reshape(-1)
        )
    return torch.cat(parts) if parts else torch.zeros(0)


def flame(local_weights, base_state, gamma=1.2e-5, seed=None):
    if len(local_weights) == 0:
        raise ValueError("FLAME: 没有客户端权重")

    keys = _float_keys(base_state)
    deltas = [_flatten_delta(w, base_state, keys).numpy() for w in local_weights]
    X = np.stack(deltas, axis=0).astype(np.float32)

    # 1) 余弦距离矩阵 + 层次聚类
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, 1e-12)
    sim = Xn @ Xn.T
    dist = np.clip(1.0 - sim, 0.0, 2.0)
    n = len(local_weights)
    benign_idx = list(range(n))
    if n >= 2:
        try:
            labels = AgglomerativeClustering(
                n_clusters=2, metric="precomputed", linkage="average"
            ).fit_predict(dist)
            c0 = [i for i in range(n) if labels[i] == 0]
            c1 = [i for i in range(n) if labels[i] == 1]
            if len(c0) > len(c1):
                benign_idx = c0
            elif len(c1) > len(c0):
                benign_idx = c1
            else:
                # 平票：选平均余弦距离更小的簇（更靠近大多数）
                mean_d0 = float(np.mean(dist[np.ix_(c0, c0)])) if c0 else 1e9
                mean_d1 = float(np.mean(dist[np.ix_(c1, c1)])) if c1 else 1e9
                benign_idx = c0 if mean_d0 <= mean_d1 else c1
            logger.info(
                f"FLAME: cluster sizes={len(c0)}/{len(c1)}, benign={benign_idx}"
            )
        except Exception as e:  # 聚类失败则退化为全量
            logger.warning(f"FLAME: 聚类失败（{e}），退化为全量客户端")

    # 2) 中位数范数裁剪
    all_norms = norms.reshape(-1)
    median_norm = float(np.median(all_norms))
    scales = np.minimum(1.0, median_norm / np.maximum(all_norms, 1e-12))

    agg_delta = np.mean(
        np.stack([X[i] * scales[i] for i in benign_idx], axis=0), axis=0
    )

    # 3) 聚合 + 噪声
    result = {}
    offset = 0
    noise_scale = float(gamma) * median_norm
    for k in keys:
        v = base_state[k].detach().cpu().float()
        numel = int(v.numel())
        part = agg_delta[offset:offset + numel].reshape(v.shape)
        offset += numel
        agg = v + torch.from_numpy(part.copy()).to(v.dtype)
        if noise_scale > 0 and "running_mean" not in k and "running_var" not in k:
            std = noise_scale * float(agg.std().item())
            if std > 0:
                agg = agg + torch.randn_like(agg) * std
        result[k] = agg.to(base_state[k].dtype)

    # 非浮点 buffer：沿用 FedAvg 语义
    for k, v in local_weights[0].items():
        if k in result:
            continue
        if torch.is_tensor(v) and not v.is_floating_point():
            if "num_batches_tracked" in k:
                result[k] = max(local_weights[i][k] for i in range(len(local_weights)))
            else:
                result[k] = v
        else:
            result[k] = v

    logger.info(
        f"FLAME: n={n}, benign={len(benign_idx)}, median_norm={median_norm:.6f}, "
        f"noise_scale={noise_scale:.6f}"
    )
    return result
