import yaml
import os
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_path: str = 'config/config.yaml'):
        """初始化配置加载器"""
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"加载配置文件失败: {str(e)}")

    def get_server_config(self) -> Dict[str, Any]:
        """获取服务器配置"""
        return self.config.get('server', {})

    def get_client_config(self) -> Dict[str, Any]:
        """获取客户端配置"""
        return self.config.get('client', {})

    def get_defense_config(self):
        """获取防御配置"""
        return self.config.get('defense', {})

    def get_attack_config(self):
        """获取攻击配置"""
        return self.config.get('attack', {})

    def get_logging_config(self):
        """获取日志配置"""
        return self.config.get('logging', {})
        
    def get_visualization_config(self):
        """获取可视化配置"""
        return self.config.get('visualization', {})

    def get_dataset_config(self) -> Dict[str, Any]:
        """获取数据集配置"""
        return self.config.get('dataset', {})
        
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.config.get('training', {})
        
    def get_privacy_config(self) -> Dict[str, Any]:
        """获取隐私配置"""
        return self.config.get('privacy', {})
        
    def get_compression_config(self) -> Dict[str, Any]:
        """获取压缩配置"""
        return self.config.get('compression', {})
        
    def get_model_type(self) -> str:
        """获取模型类型"""
        return self.config.get('model', {}).get('type', 'resnet')
        
    def get_all_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模型配置"""
        return {
            'resnet': {
                'in_channels': 3,  # CIFAR-10 是 3 通道
                'num_classes': 10,  # CIFAR-10 有 10 个类别
            }
        } 