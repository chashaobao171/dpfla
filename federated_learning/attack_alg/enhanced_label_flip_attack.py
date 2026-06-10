"""
强化版标签翻转攻击 - All-to-All 全类别循环翻转
策略：
  1. All-to-All 全类别翻转（0→1, 1→2, ..., 9→0）—— 100% 标签污染面
  2. 梯度放大（factor=3）—— 扩大恶意信号在 FedAvg 中的影响力
  3. 始终激活（malicious_behavior_rate=1.0）—— 每轮必执行
"""

import torch
import numpy as np


class EnhancedLabelFlipAttack:
    """
    All-to-All 全类别循环翻转：
    - 每个类映射到下一个类（0→1, 1→2, ..., 9→0）
    - 100% 的训练样本被污染，效果远超部分类翻转
    """

    GRADIENT_FACTOR = 3.0

    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        # All-to-All 循环翻转映射：每个类翻到下一个类
        self.mapping = {i: (i + 1) % num_classes for i in range(num_classes)}

    def apply(self, local_model, global_model):
        with torch.no_grad():
            attacked_local_weights = {}
            for key in local_model.keys():
                delta = local_model[key].float() - global_model[key].float()
                attacked_local_weights[key] = global_model[key].float() + delta * self.GRADIENT_FACTOR
        return attacked_local_weights

    def __repr__(self):
        return (f"EnhancedLabelFlipAttack(all-to-all mapping, "
                f"gradient_factor={self.GRADIENT_FACTOR})")
