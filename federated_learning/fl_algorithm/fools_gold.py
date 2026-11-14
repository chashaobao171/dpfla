"""FoolsGold 聚合权重（Fung et al., "Mitigating Sybils in Federated Learning Poisoning"）。

本项目的适配说明：
- 原论文/仓库用 `local_grads[i][-2]` 取某一层梯度；本项目 `client_grad` 是 **dict**，
  且 YOLO26s + AMP 下累积梯度可能含 NaN（2026-11-14 冒烟实测）。
- 因此这里改用 FoolsGold 原文的输入：**客户端模型更新量 `w_i - w_global`**，
  按固定 key 顺序展平全部浮点张量后做余弦相似度。
  这样既避开 NaN 梯度，也能看到服务端注入后的恶意更新。
- memory 按客户端累计历史更新；断点续跑时由 fl_core 从 checkpoint 恢复。
- 若所有客户端权重被压成 0（例如更新全为 0），回退为等权，避免 NaN。
"""

import numpy as np
import sklearn.metrics.pairwise as smp
import torch
from loguru import logger


class FoolsGold:
    def __init__(self, num_peers):
        self.memory = None
        self.wv_history = []
        self.num_peers = num_peers
        self._keys = None
        self._key_numel = None

    def _init_keys_from_state(self, base_state):
        if self._keys is not None:
            return
        keys = []
        for k in sorted(base_state.keys()):
            v = base_state[k]
            if torch.is_tensor(v) and v.is_floating_point() and v.numel() > 0:
                keys.append(k)
        self._keys = keys
        self._key_numel = {k: int(base_state[k].numel()) for k in keys}

    def _flatten_delta(self, weights, base_state):
        parts = []
        for k in self._keys:
            n = self._key_numel[k]
            v = weights.get(k) if isinstance(weights, dict) else None
            b = base_state.get(k) if isinstance(base_state, dict) else None
            if (
                v is not None
                and b is not None
                and torch.is_tensor(v)
                and torch.is_tensor(b)
                and v.is_floating_point()
                and b.is_floating_point()
                and v.numel() == n
                and b.numel() == n
            ):
                delta = v.detach().cpu().float() - b.detach().cpu().float()
                parts.append(delta.reshape(-1))
            else:
                parts.append(torch.zeros(n, dtype=torch.float32))
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return torch.cat(parts).numpy()

    def _score_matrix(self, grads, selected_peers):
        """grads: [m, d] numpy；返回当前轮被选中客户端的 FoolsGold 权重。"""
        grads = np.asarray(grads, dtype=np.float32)
        bad = ~np.isfinite(grads)
        if bad.any():
            logger.warning(
                f"FoolsGold: 输入含 {int(bad.sum())} 个非有限值，已置 0 后继续"
            )
            grads = np.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)

        grad_len = grads.shape[1]
        if self.memory is None or self.memory.shape != (self.num_peers, grad_len):
            self.memory = np.zeros((self.num_peers, grad_len), dtype=np.float32)

        self.memory[selected_peers] += grads
        wv = foolsgold(self.memory)
        self.wv_history.append(wv)
        return wv[selected_peers]

    def score_weights(self, local_weights, base_state, selected_peers):
        """主入口：用模型更新量（state_dict 差值）计算 FoolsGold 权重。"""
        if len(local_weights) == 0:
            return np.zeros(0, dtype=float)

        self._init_keys_from_state(base_state)
        grads = np.stack(
            [self._flatten_delta(w, base_state) for w in local_weights], axis=0
        )
        return self._score_matrix(grads, selected_peers)

    def score_gradients(self, local_grads, selected_peers):
        """兼容旧接口（梯度 dict）。若梯度含 NaN，会先置 0。"""
        if len(local_grads) == 0:
            return np.zeros(0, dtype=float)

        if self._keys is None:
            if isinstance(local_grads[0], dict):
                common = set(local_grads[0].keys())
                for g in local_grads[1:]:
                    if isinstance(g, dict):
                        common &= set(g.keys())
                keys = []
                for k in sorted(common):
                    v = local_grads[0][k]
                    if torch.is_tensor(v) and v.is_floating_point() and v.numel() > 0:
                        keys.append(k)
                self._keys = keys
                self._key_numel = {k: int(local_grads[0][k].numel()) for k in keys}
            else:
                self._keys = list(range(len(local_grads[0])))
                self._key_numel = {
                    i: int(np.prod(local_grads[0][i].shape)) for i in self._keys
                }

        grads = np.stack([self._flatten_grad(g) for g in local_grads], axis=0)
        return self._score_matrix(grads, selected_peers)

    def _flatten_grad(self, client_grad):
        parts = []
        for k in self._keys:
            n = self._key_numel[k]
            if isinstance(client_grad, dict):
                v = client_grad.get(k)
            else:
                v = client_grad[k] if k < len(client_grad) else None

            if (
                v is not None
                and torch.is_tensor(v)
                and v.is_floating_point()
                and v.numel() == n
            ):
                parts.append(v.detach().cpu().reshape(-1).to(torch.float32))
            else:
                parts.append(torch.zeros(n, dtype=torch.float32))

        if not parts:
            return np.zeros(0, dtype=np.float32)
        return torch.cat(parts).numpy()


def foolsgold(grads):
    """按 FoolsGold 原始定义计算每个客户端的权重（长度 = num_peers）。"""
    n_clients = grads.shape[0]
    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)

    # pardoning：相似度相对较低的行按比例压低
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j] if maxcs[j] != 0 else 0.0

    wv = 1 - np.max(cs, axis=1)
    wv = np.clip(wv, 0, 1)

    # 全部为 0（例如更新全 0）时退化为等权，避免 0/0 产生 NaN
    mx = np.max(wv)
    if mx <= 0:
        return np.ones(n_clients, dtype=float)

    wv = wv / mx
    wv[wv == 1] = .99
    wv = np.log(wv / (1 - wv)) + 0.5
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[wv < 0] = 0
    return wv
