import torch.optim as optim
from typing import Dict, Any, Union, List, Tuple

class OptimizerFactory:
    """优化器工厂类，用于创建不同类型的优化器"""
    
    @staticmethod
    def _convert_betas(betas: Union[List[float], Tuple[float, float], List[str], Tuple[str, str]]) -> Tuple[float, float]:
        """将betas参数转换为浮点数元组格式"""
        if isinstance(betas, (list, tuple)):
            try:
                return (float(betas[0]), float(betas[1]))
            except (ValueError, TypeError) as e:
                print(f"转换betas参数时发生错误: {str(e)}")
                return (0.9, 0.999)  # 默认值
        return (0.9, 0.999)  # 默认值
    
    @staticmethod
    def create_optimizer(optimizer_type: str, model_params, optimizer_config: Dict[str, Any] = None) -> optim.Optimizer:
        """
        创建指定类型的优化器
        
        Args:
            optimizer_type: 优化器类型名称
            model_params: 模型参数
            optimizer_config: 优化器配置参数
            
        Returns:
            创建的优化器实例
        """
        if optimizer_config is None:
            optimizer_config = {}
        # 强制类型转换，防止str类型
        if 'learning_rate' in optimizer_config:
            optimizer_config['learning_rate'] = float(optimizer_config['learning_rate'])
        if 'lr' in optimizer_config:
            optimizer_config['lr'] = float(optimizer_config['lr'])
        if 'momentum' in optimizer_config:
            optimizer_config['momentum'] = float(optimizer_config['momentum'])
        if 'weight_decay' in optimizer_config:
            optimizer_config['weight_decay'] = float(optimizer_config['weight_decay'])
        if 'eps' in optimizer_config:
            optimizer_config['eps'] = float(optimizer_config['eps'])
        if 'betas' in optimizer_config:
            # betas 可能是字符串列表
            betas = optimizer_config['betas']
            if isinstance(betas, (list, tuple)):
                optimizer_config['betas'] = (float(betas[0]), float(betas[1]))
        
        if optimizer_type == "sgd":
            return optim.SGD(
                model_params,
                lr=float(optimizer_config.get('learning_rate', 0.01)),
                momentum=float(optimizer_config.get('momentum', 0.9)),
                weight_decay=float(optimizer_config.get('weight_decay', 0))
            )
        elif optimizer_type == "adam":
            betas = OptimizerFactory._convert_betas(optimizer_config.get('betas', (0.9, 0.999)))
            return optim.Adam(
                model_params,
                lr=float(optimizer_config.get('learning_rate', 0.001)),
                betas=betas,
                eps=float(optimizer_config.get('eps', 1e-8)),
                weight_decay=float(optimizer_config.get('weight_decay', 0))
            )
        elif optimizer_type == "adamw":
            betas = OptimizerFactory._convert_betas(optimizer_config.get('betas', (0.9, 0.999)))
            return optim.AdamW(
                model_params,
                lr=float(optimizer_config.get('learning_rate', 0.001)),
                betas=betas,
                eps=float(optimizer_config.get('eps', 1e-8)),
                weight_decay=float(optimizer_config.get('weight_decay', 0.01))
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}") 