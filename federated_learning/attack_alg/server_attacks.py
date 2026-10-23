"""
服务端注入攻击（omniscient attackers，2026-10-23 新增）

用途：在 CIFAR10 上筛选"能让 DPFLA 对多 Krum / Trimmed-Mean 拉开差距"的攻击范式。
参考实现：FLPoison（https://github.com/vio1etus/FLPoison）`attackers/{min,alie,signflipping}.py`，
本项目按其原始设定（攻击者掌握良性端更新）实现，仅把"更新向量"接口适配为
本项目的 state_dict 接口。

支持：
- `minmax`      MinMax（Fang et al., NDSS'21）：在良性端最大成对距离内，沿良性均值反方向做最大扰动，
                 使恶意更新"藏在"良性端之间 → 逐坐标裁剪（Trimmed-Mean）裁不掉；且所有恶意端更新相同
                 → 在距离/inner-product 空间互相靠近（Multi-Krum 的弱点）。
- `alie`        ALIE（Baruch et al., NeurIPS'19）：mean + z_max * std，小扰动专打 Krum。
- `signflipping` 符号翻转（Damaskinos et al., ICML'18）：恶意更新取反 → 直接打坐标级聚合。

调用方式：fl_core 在"所有客户端训练完成、聚合之前"调用 apply_server_attack()，
用构造出的更新替换恶意端上传的 state_dict。
"""

import numpy as np
import torch
from loguru import logger
from scipy.stats import norm

# 参与攻击构造的参数键：跳过 BN 统计量（与项目"高斯/梯度放大跳过 BN"的既有防护一致）
_EXCLUDED_SUBSTRINGS = ('running_mean', 'running_var', 'num_batches_tracked')


def _attackable_keys(state):
    return [
        k for k in state
        if torch.is_tensor(state[k]) and state[k].is_floating_point()
        and not any(s in k for s in _EXCLUDED_SUBSTRINGS)
    ]


def _flatten(state, keys):
    return np.concatenate([
        state[k].detach().cpu().reshape(-1).float().numpy() for k in keys
    ])


def _apply_vec(vec, template_state, keys):
    """把 1D 更新向量写回 state_dict（仅覆盖 keys；其余键沿用 template 的值）。"""
    out = {k: v.detach().clone() for k, v in template_state.items()}
    idx = 0
    for k in keys:
        n = template_state[k].numel()
        chunk = vec[idx:idx + n].reshape(template_state[k].shape)
        out[k] = torch.from_numpy(chunk).to(dtype=template_state[k].dtype)
        idx += n
    return out


def _minmax_update(benign_updates, gamma_init=10.0, stop_threshold=1e-5):
    """MinMax：二分搜索最大可行 λ，使恶意更新落在良性端最大成对距离之内。"""
    benign_mean = benign_updates.mean(axis=0)
    norm = np.linalg.norm(benign_mean)
    deviation = benign_mean / norm if norm > 1e-12 else benign_mean
    metric = lambda x: np.linalg.norm(x, axis=1).max()
    upper_bound = np.max([
        metric(benign_updates - benign_updates[i]) for i in range(len(benign_updates))
    ])
    lamda, step, lamda_succ = float(gamma_init), float(gamma_init) / 2, 0.0
    while abs(lamda_succ - lamda) > stop_threshold:
        mal = benign_mean - lamda * deviation
        if metric(benign_updates - mal) <= upper_bound:
            lamda_succ = lamda
            lamda += step
        else:
            lamda -= step
        step /= 2
    logger.info(f'  MinMax 二分结束：λ*={lamda_succ:.4f}（upper_bound={upper_bound:.4f}）')
    return benign_mean - lamda_succ * deviation


def _alie_update(benign_updates, num_clients, num_attackers, z_max=None):
    """ALIE：mean + z_max * std（z_max 由客户端数/恶意数推得）。"""
    mean = benign_updates.mean(axis=0)
    std = benign_updates.std(axis=0)
    if z_max is None:
        s = np.floor(num_clients / 2 + 1) - num_attackers
        cdf_value = (num_clients - num_attackers - s) / (num_clients - num_attackers)
        z_max = float(norm.ppf(cdf_value))
    logger.info(f'  ALIE：z_max={z_max:.4f}')
    return mean + z_max * std


def apply_server_attack(attack_type, local_weights, clients_types, global_state,
                        num_attackers, num_clients, z_max=None):
    """
    用构造出的恶意更新替换恶意端上传的权重。

    :param local_weights: list[state_dict]，客户端上传顺序
    :param clients_types: list[str]，'Good update' / 'Bad update'（与 local_weights 同序）
    :param global_state: 本轮训练起点（全局模型）的 state_dict
    :param num_attackers: 恶意端数量 f
    :return: (local_weights, 被替换的端索引列表) —— 调用方需据此同步 local_models，
             否则 DPFLA 等防御通过 local_models 读到的是未受攻击的更新。
    """
    good_idx = [i for i, t in enumerate(clients_types) if t == 'Good update']
    bad_idx = [i for i, t in enumerate(clients_types) if t == 'Bad update']
    if not bad_idx:
        logger.warning(f'服务端攻击 {attack_type}：未找到恶意端，跳过')
        return local_weights, []
    if not good_idx:
        logger.warning(f'服务端攻击 {attack_type}：未找到良性端，跳过')
        return local_weights, []

    keys = _attackable_keys(global_state)
    g_vec = _flatten(global_state, keys)

    if attack_type == 'signflipping':
        # 非全知：每个恶意端把自己的更新取反
        for i in bad_idx:
            u = _flatten(local_weights[i], keys) - g_vec
            local_weights[i] = _apply_vec(g_vec - u, local_weights[i], keys)
        logger.warning(f'⚠️  服务端注入攻击 signflipping 已应用（恶意端 {len(bad_idx)} 个）')
        return local_weights, bad_idx

    benign_updates = np.stack([
        _flatten(local_weights[i], keys) - g_vec for i in good_idx
    ])

    if attack_type == 'minmax':
        mal_update = _minmax_update(benign_updates)
    elif attack_type == 'alie':
        mal_update = _alie_update(benign_updates, num_clients, num_attackers, z_max=z_max)
    else:
        raise ValueError(f'未知的 server attack: {attack_type}')

    # 所有恶意端提交相同的构造更新；非攻击参数沿用第一个良性端的值（保证 BN 状态正常）
    for i in bad_idx:
        local_weights[i] = _apply_vec(g_vec + mal_update, local_weights[good_idx[0]], keys)

    logger.warning(
        f'⚠️  服务端注入攻击 {attack_type} 已应用（恶意端 {len(bad_idx)}/{len(local_weights)} 个，'
        f'良性端 {len(good_idx)} 个参与构造）'
    )
    return local_weights, bad_idx
