import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np

class DifferentialPrivacy:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化差分隐私防御机制
        
        Args:
            config: 差分隐私配置参数
        """
        self.epsilon = config.get('epsilon', 2.0)  # 隐私预算
        self.delta = config.get('delta', 1e-4)     # 失败概率
        self.clip_threshold = config.get('clip_threshold', 1.0)  # 梯度裁剪阈值
        
    def apply(self, model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        对模型参数应用差分隐私保护
        
        Args:
            model_state: 模型状态字典
            
        Returns:
            添加噪声后的模型状态字典
        """
        # 计算噪声尺度
        noise_scale = self._calculate_noise_scale()
        
        # 对每个参数添加噪声
        noisy_state = {}
        for key, param in model_state.items():
            # 梯度裁剪
            clipped_param = self._clip_gradients(param)
            # 添加噪声
            noise = torch.randn_like(clipped_param) * noise_scale
            noisy_state[key] = clipped_param + noise
            
        return noisy_state
    
    def _calculate_noise_scale(self) -> float:
        """
        计算噪声尺度
        
        Returns:
            噪声尺度
        """
        # 使用高斯机制计算噪声尺度
        return self.clip_threshold * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def _clip_gradients(self, param: torch.Tensor) -> torch.Tensor:
        """
        对参数进行梯度裁剪
        
        Args:
            param: 模型参数
            
        Returns:
            裁剪后的参数
        """
        param_norm = torch.norm(param)
        if param_norm > self.clip_threshold:
            return param * (self.clip_threshold / param_norm)
        return param 