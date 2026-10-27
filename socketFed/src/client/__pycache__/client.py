import socket
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# 添加必要的路径
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'src', 'utils'))

from utils.config_loader import ConfigLoader
from utils.optimizer_factory import OptimizerFactory
from utils.visualization import Visualizer, ModelCheckpoint
from defense.differential_privacy import DifferentialPrivacy
from models.model_factory import ModelFactory
from data.data_loader import DataLoaderFactory

class FederatedClient:
    def __init__(self, client_id: str):
        self.config = ConfigLoader()
        self.client_config = self.config.get_client_config()
        self.client_id = client_id
        
        # 创建模型
        self.model = self._create_model()
        
        # 创建优化器
        self.optimizer = OptimizerFactory.create_optimizer(
            self.client_config['optimizer']['type'],
            self.model.parameters(),
            self.client_config['optimizer']['config']
        )
        
        # 创建防御机制
        self.defense = DifferentialPrivacy(
            self.config.get_defense_config()['differential_privacy']
        )
        
        # 创建可视化工具
        self.visualizer = Visualizer(
            os.path.join(self.config.get_visualization_config()['tensorboard']['log_dir'],
                        f'client_{client_id}')
        )
        
        # 创建模型检查点
        self.checkpoint = ModelCheckpoint(
            os.path.join(self.config.get_visualization_config()['checkpoint']['save_dir'],
                        f'client_{client_id}')
        )
        
        # 设置日志
        logging.basicConfig(
            level=self.config.get_logging_config().get('level', 'INFO'),
            filename=f'client_{client_id}.log'
        )
        
    def _create_model(self) -> nn.Module:
        """创建模型"""
        return ModelFactory.create_model(
            self.client_config['model'],
            self.client_config.get('model_config', {})
        )
    
    def connect_to_server(self):
        """连接到服务器"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_config = self.config.get_server_config()
        self.socket.connect((server_config['host'], server_config['port']))
        logging.info(f"已连接到服务器 {server_config['host']}:{server_config['port']}")
    
    def train(self, local_data: torch.Tensor, local_labels: torch.Tensor, epoch: int):
        """本地训练"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(zip(local_data, local_labels)):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # 记录训练指标
            if batch_idx % self.config.get_visualization_config()['tensorboard']['update_freq'] == 0:
                metrics = {
                    'train_loss': total_loss / (batch_idx + 1),
                    'train_acc': 100. * correct / total
                }
                self.visualizer.log_metrics(metrics, epoch * len(local_data) + batch_idx)
        
        # 保存检查点
        if epoch % self.config.get_visualization_config()['checkpoint']['save_freq'] == 0:
            metrics = {
                'train_loss': total_loss / len(local_data),
                'train_acc': 100. * correct / total
            }
            self.checkpoint.save_model(self.model, self.optimizer, epoch, metrics)
            
        # 应用防御机制
        model_params = self.defense.apply(self.model.state_dict())
        return model_params
    
    def send_model_update(self, model_params: Dict[str, torch.Tensor]):
        """发送模型更新到服务器"""
        message = {
            'type': 'model_update',
            'model': model_params
        }
        self.socket.send(json.dumps(message).encode())
    
    def receive_global_model(self):
        """接收全局模型"""
        data = self.socket.recv(4096)
        if data:
            message = json.loads(data.decode())
            if message['type'] == 'global_model':
                self.model.load_state_dict(message['model'])
                logging.info("已更新全局模型")
    
    def run(self, local_data: torch.Tensor, local_labels: torch.Tensor):
        """运行客户端训练循环"""
        try:
            self.connect_to_server()
            epoch = 0
            
            while True:
                # 训练本地模型
                model_params = self.train(local_data, local_labels, epoch)
                
                # 发送模型更新
                self.send_model_update(model_params)
                
                # 接收全局模型
                self.receive_global_model()
                
                epoch += 1
                
        except Exception as e:
            logging.error(f"客户端运行时发生错误: {str(e)}")
        finally:
            self.socket.close()
            self.visualizer.close()

if __name__ == "__main__":
    # 示例使用
    client = FederatedClient("client_1")
    # 这里需要提供本地数据
    # client.run(local_data, local_labels) 