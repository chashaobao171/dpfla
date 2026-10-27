import torch
import numpy as np
from typing import Dict, List

class FedAvgAggregator:
    def __init__(self):
        pass
    
    def aggregate(self, client_updates: List[Dict]) -> Dict[str, torch.Tensor]:
        """使用FedAvg算法聚合客户端模型
        
        Args:
            client_updates: 包含客户端更新的列表，每个元素是一个字典，包含:
                - client_id: 客户端ID
                - model_params: 模型参数
                - data_size: 数据大小
                - metrics: 训练指标
        
        Returns:
            聚合后的模型参数字典
        """
        if not client_updates:
            return None
            
        # 获取第一个客户端的模型参数作为模板，并确保是浮点类型
        first_client = client_updates[0]
        aggregated_model = {}
        for param_name, param_value in first_client['model_params'].items():
            aggregated_model[param_name] = torch.zeros_like(param_value, dtype=torch.float32)
            
        # 计算总数据量
        total_data_size = sum(update['data_size'] for update in client_updates)
        
        # 加权平均聚合
        for client_update in client_updates:
            weight = client_update['data_size'] / total_data_size
            for param_name, param_value in client_update['model_params'].items():
                # 确保param_value是浮点类型，然后进行加权和
                aggregated_model[param_name] += param_value.float() * weight
                
        return aggregated_model 