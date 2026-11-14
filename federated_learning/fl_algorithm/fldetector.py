"""FLDetector（简化版）——基于历史更新预测的异常检测聚合。

说明（务必在报告中注明）：
- 原版 FLDetector 用 LBFGS 近似 Hessian 预测下一轮全局梯度，并用 gap statistic
  决定聚类数；本实现为适配 YOLO（10M 参数）与短轮次（~10-15 轮）做了简化：
  1) 用一阶线性外推 `pred = 2*g_{t-1} - g_{t-2}` 代替 LBFGS-Hessian；
  2) 对每个客户端的偏差分数做 KMeans(2)，选平均偏差更低的一簇为良性；
  3) 跳过 gap statistic（直接假设存在攻击簇）。
- 输入仍是 FoolsGold/FLAME 同一套 `w_i - w_global` 更新量，便于同口径对比。
- 前两轮没有预测历史，退化为 FedAvg；通常从第 4 轮开始检测。
"""

import numpy as np
import torch
from loguru import logger
from sklearn.cluster import KMeans


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


class FLDetectorLite:
    def __init__(self, num_clients, window=2):
        self.num_clients = int(num_clients)
        self.window = int(window)
        self.prev_updates = None       # [n, d] float32
        self.prev_global = None        # [d]
        self.prev_prev_global = None   # [d]
        self.score_history = []        # list of [n]

    def aggregate(self, local_weights, base_state):
        if len(local_weights) == 0:
            raise ValueError("FLDetector: 没有客户端权重")

        keys = _float_keys(base_state)
        deltas = [_flatten_delta(w, base_state, keys).numpy() for w in local_weights]
        X = np.stack(deltas, axis=0).astype(np.float32)
        n = X.shape[0]
        benign = list(range(n))

        if (
            self.prev_global is not None
            and self.prev_prev_global is not None
            and self.prev_global.shape[0] == X.shape[1]
        ):
            pred = 2.0 * self.prev_global - self.prev_prev_global
            score = np.linalg.norm(X - pred, axis=1)
            score = score / max(float(score.sum()), 1e-12)
            self.score_history.append(score)
            if len(self.score_history) > self.window:
                self.score_history.pop(0)

            if len(self.score_history) >= self.window:
                mean_score = np.mean(np.stack(self.score_history, axis=0), axis=0)
                try:
                    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(
                        mean_score.reshape(-1, 1)
                    )
                    c0 = [i for i in range(n) if labels[i] == 0]
                    c1 = [i for i in range(n) if labels[i] == 1]
                    if c0 and c1:
                        benign = (
                            c0
                            if float(np.mean(mean_score[c0])) <= float(np.mean(mean_score[c1]))
                            else c1
                        )
                        logger.info(
                            f"FLDetector: scores={np.round(mean_score, 4).tolist()}, "
                            f"benign={benign}"
                        )
                except Exception as e:
                    logger.warning(f"FLDetector: KMeans 失败（{e}），本轮使用全量")

        agg_delta = X[benign].mean(axis=0)

        # 更新历史：用全体更新的均值作为本轮全局更新（与原版 g^t 语义一致）
        self.prev_prev_global = self.prev_global
        self.prev_global = X.mean(axis=0)
        self.prev_updates = X

        result = {}
        offset = 0
        for k in keys:
            v = base_state[k].detach().cpu().float()
            numel = int(v.numel())
            part = agg_delta[offset:offset + numel].reshape(v.shape)
            offset += numel
            result[k] = (v + torch.from_numpy(part.copy())).to(base_state[k].dtype)

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

        logger.info(f"FLDetector: n={n}, benign={len(benign)}/{n}")
        return result

    def state_dict(self):
        return {
            "prev_updates": self.prev_updates,
            "prev_global": self.prev_global,
            "prev_prev_global": self.prev_prev_global,
            "score_history": self.score_history,
        }

    def load_state_dict(self, state):
        if not state:
            return
        self.prev_updates = state.get("prev_updates")
        self.prev_global = state.get("prev_global")
        self.prev_prev_global = state.get("prev_prev_global")
        self.score_history = state.get("score_history", [])
