import numpy as np
import torch

eps = np.finfo(float).eps


def gaussian_attack(update, client_pseudonym, malicious_behavior_rate=0,
                    device='cpu', attack=False, mean=0.0, std=0.5):
    flag = 0
    for key in update.keys():
        # 跳过非浮点类型的参数（如BatchNorm的running_mean, num_batches_tracked等）
        if not update[key].is_floating_point():
            continue
        
        r = np.random.random()
        if r <= malicious_behavior_rate:
            # print('Gausiian noise attack launched by ', client_pseudonym, ' targeting ', key, i+1)
            # 使用与参数相同的dtype和device创建噪声
            noise = torch.randn_like(update[key]) * std + mean
            flag = 1
            # 避免in-place操作，创建新tensor
            update[key] = update[key] + noise
    return update, flag
