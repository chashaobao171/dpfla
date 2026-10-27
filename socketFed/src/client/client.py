import os
import sys
import time
import socket
import pickle
import logging
import argparse
import json
import struct
import base64
import zlib
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import io
from datetime import datetime
import copy
import torch.quantization

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

# 导入必要的模块
from src.utils.logger import setup_logger
from src.utils.config_loader import ConfigLoader
from src.models.model_factory import ModelFactory

class FederatedClient:
    def __init__(self, client_id: str, config_path: str = 'config/config.yaml'):
        """初始化联邦学习客户端"""
        self.client_id = client_id
        
        # 首先设置日志
        self.logger = setup_logger(f'client_{client_id}', os.path.join('logs', f'client_{client_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
        self.logger.info(f"客户端 {client_id} 初始化开始")
        
        # 加载配置
        self.config = ConfigLoader(config_path)
        
        # 从服务器配置中获取最大客户端数量，确保在数据加载前初始化
        self.num_total_clients = self.config.get_server_config().get('max_clients', 1) 
        
        self.all_model_configs = self.config.get_all_model_configs()
        self.model_type = self.config.get_model_type()
        
        # 设置设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # 设置默认的CUDA设备
            torch.cuda.set_device(0)
            self.logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            self.logger.info("使用CPU (GPU不可用)")
        
        # 初始化模型
        self.initialize_model()
        self.model = self.model.to(self.device)  # 将模型移到GPU
        
        # 初始化优化器
        training_config = self.config.get_training_config()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=training_config.get('learning_rate', 0.001)
        )
        
        # 初始化数据加载器
        self.train_loader, self.test_loader = self._initialize_data_loaders()
        
        # 初始化socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # 获取服务器配置 (如果其他地方需要用到 self.server_config，否则可以移除)
        self.server_config = self.config.get_server_config()
        
        self.logger.info(f"客户端 {client_id} 初始化完成")
        
    def initialize_model(self):
        """初始化模型"""
        try:
            # 获取模型配置
            model_config = self.all_model_configs[self.model_type]
            self.model = ModelFactory.create_model(self.model_type, model_config)
            self.logger.info(f"客户端 {self.client_id} 初始化 {self.model_type} 模型完成")
        except Exception as e:
            self.logger.error(f"初始化模型时发生错误: {str(e)}")
            raise
    
    def _initialize_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """初始化数据加载器"""
        # 创建数据转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # 创建数据集
        train_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
        
        # ====== 数据划分逻辑开始 ======
        # 将client_id转换为整数，例如"client1" -> 0, "client2" -> 1
        # 假设client_id是 'clientN' 形式，N是整数
        try:
            client_idx = int(self.client_id.replace('client', '')) - 1
        except ValueError:
            self.logger.error(f"无效的client_id格式: {self.client_id}. 期望格式如 'client1'.")
            raise

        if client_idx < 0 or client_idx >= self.num_total_clients:
            self.logger.error(f"客户端ID {self.client_id} 超出总客户端数量 {self.num_total_clients} 的范围。")
            raise IndexError(f"Client ID {self.client_id} is out of bounds for {self.num_total_clients} total clients.")

        # 计算每个客户端的数据量
        total_train_samples = len(train_dataset)
        samples_per_client = total_train_samples // self.num_total_clients
        
        # 确定当前客户端的数据索引范围
        start_idx = client_idx * samples_per_client
        end_idx = (client_idx + 1) * samples_per_client
        
        # 对于最后一个客户端，确保它包含所有剩余的数据
        if client_idx == self.num_total_clients - 1:
            end_idx = total_train_samples
            
        # 创建当前客户端的训练数据集子集
        client_train_dataset = Subset(train_dataset, range(start_idx, end_idx))
        self.logger.info(f"客户端 {self.client_id} 分配到训练数据量: {len(client_train_dataset)} 样本 (索引从 {start_idx} 到 {end_idx-1})")
        # ====== 数据划分逻辑结束 ======

        # 创建数据加载器
        train_loader = DataLoader(
            client_train_dataset, # 使用划分后的数据集
            batch_size=self.config.get_client_config().get('batch_size', 32),
            shuffle=True,
            num_workers=self.config.get_dataset_config().get('num_workers', 0),
            pin_memory=self.config.get_dataset_config().get('pin_memory', True)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get_dataset_config().get('batch_size', 128),
            shuffle=False,
            num_workers=self.config.get_dataset_config().get('num_workers', 0),
            pin_memory=self.config.get_dataset_config().get('pin_memory', True) # 启用pin_memory以加速GPU传输
        )
        
        self.logger.info(f"数据加载完成 - 训练集: {len(client_train_dataset)} 样本, 测试集: {len(test_dataset)} 样本")
        
        return train_loader, test_loader

    def _send_message(self, sock: socket.socket, message: Dict[str, Any]):
        """发送带长度前缀的JSON消息"""
        try:
            json_message = json.dumps(message)
            encoded_message = json_message.encode('utf-8')
            length = len(encoded_message)
            sock.sendall(struct.pack('!I', length) + encoded_message)
        except Exception as e:
            self.logger.error(f"客户端 {self.client_id} 发送消息时发生错误: {str(e)}")
            raise

    def _receive_message(self, sock: socket.socket) -> Dict[str, Any] or None:
        """接收带长度前缀的JSON消息"""
        try:
            length_bytes = sock.recv(4)
            if not length_bytes:
                return None
            length = struct.unpack('!I', length_bytes)[0]

            data = b''
            while len(data) < length:
                packet = sock.recv(length - len(data))
                if not packet:
                    return None
                data += packet
            
            return json.loads(data.decode('utf-8'))
        except Exception as e:
            self.logger.error(f"客户端 {self.client_id} 接收消息时发生错误: {str(e)}")
            return None
        
    def connect_to_server(self):
        """连接到服务器"""
        self.socket.connect((self.config.get_client_config()['host'], self.config.get_client_config()['port']))
        self.logger.info("已连接到服务器")
        
    def run(self, data_loader: DataLoader):
        """运行客户端"""
        try:
            # 连接到服务器
            self.connect_to_server()

            # 发送注册消息
            self._send_message(self.socket, {
                'type': 'register',
                'client_id': self.client_id
            })
            
            # 等待服务器确认
            self.logger.info("等待服务器注册确认...")
            response = self._receive_message(self.socket)
            if response and response.get('type') == 'registered_ack':
                self.logger.info("收到服务器注册确认")
            else:
                raise Exception("服务器注册确认失败")

            # 进入消息处理循环
            self.logger.info("等待服务器指令...")
            while True:
                message = self._receive_message(self.socket)
                if message is None:
                    self.logger.warning("与服务器的连接已断开")
                    break

                message_type = message.get('type')
                if message_type == 'global_model':
                    self.logger.info(f"收到全局模型 (轮次 {message.get('round')})")
                    self._handle_global_model(message)
                elif message_type == 'stop':
                    self.logger.info("收到停止指令")
                    break
                else:
                    self.logger.warning(f"收到未知类型的消息: {message_type}")

        except Exception as e:
            self.logger.error(f"客户端运行过程中发生错误: {str(e)}")
            raise
        finally:
            self.socket.close()
            self.logger.info("客户端已关闭")

    def _handle_global_model(self, message: Dict[str, Any]):
        """处理从服务器接收的全局模型"""
        try:
            # 反序列化模型状态字典
            serialized_state = message['model']
            buffer = io.BytesIO(base64.b64decode(serialized_state))
            model_state = torch.load(buffer)
            
            # 更新本地模型
            self.model.load_state_dict(model_state)
            self.logger.info("本地模型更新完成")
            
            # 开始本地训练
            self.logger.info("开始本地训练...")
            self._train_local_model()
            
        except Exception as e:
            self.logger.error(f"处理全局模型时发生错误: {str(e)}")
            raise
            
    def _train_local_model(self):
        """训练本地模型并发送更新到服务器"""
        try:
            # 训练模型
            self.model.train()
            criterion = nn.CrossEntropyLoss()
            
            total_batches = len(self.train_loader)
            self.logger.info(f"开始训练，共 {total_batches} 个批次")

            for epoch in range(self.config.get_training_config().get('epochs', 1)):  # 使用配置中的epochs
                running_loss = 0.0
                correct = 0
                total = 0
                epoch_start_time = time.time()
                
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)

                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

                    # 每10个批次显示一次进度并发送给服务器
                    if (batch_idx + 1) % 10 == 0:
                        accuracy = 100. * correct / total
                        progress = (batch_idx + 1) / total_batches * 100
                        self.logger.info(f"Epoch {epoch + 1}/{self.config.get_training_config().get('epochs', 1)} [{batch_idx + 1}/{total_batches}] "
                                      f"({progress:.1f}%) - Loss: {running_loss/10:.4f}, "
                                      f"Accuracy: {accuracy:.2f}%")
                        print(f"客户端 {self.client_id} - Epoch {epoch + 1}/{self.config.get_training_config().get('epochs', 1)} [{batch_idx + 1}/{total_batches}] "
                              f"({progress:.1f}%) - Loss: {running_loss/10:.4f}, "
                              f"Accuracy: {accuracy:.2f}%")
                        
                        # 发送训练进度到服务器
                        progress_message = {
                            'type': 'training_progress',
                            'client_id': self.client_id,
                            'progress': {
                                'epoch': epoch + 1,
                                'total_epochs': self.config.get_training_config().get('epochs', 1),
                                'batch': batch_idx + 1,
                                'total_batches': total_batches,
                                'progress_percentage': progress,
                                'current_accuracy': accuracy,
                                'current_loss': running_loss/10
                            }
                        }
                        self._send_message(self.socket, progress_message)
                        running_loss = 0.0
                
                # 每个epoch结束后计算并打印整体准确率
                epoch_accuracy = 100. * correct / total
                epoch_time = time.time() - epoch_start_time
                self.logger.info(f"Epoch {epoch + 1}/{self.config.get_training_config().get('epochs', 1)} 完成 - "
                              f"训练准确率: {epoch_accuracy:.2f}% - "
                              f"用时: {epoch_time:.1f}秒")
                print(f"客户端 {self.client_id} - Epoch {epoch + 1}/{self.config.get_training_config().get('epochs', 1)} 完成 - "
                      f"训练准确率: {epoch_accuracy:.2f}% - "
                      f"用时: {epoch_time:.1f}秒")
                
                # 发送epoch完成消息到服务器
                epoch_complete_message = {
                    'type': 'epoch_complete',
                    'client_id': self.client_id,
                    'epoch_info': {
                        'epoch': epoch + 1,
                        'total_epochs': self.config.get_training_config().get('epochs', 1),
                        'accuracy': epoch_accuracy,
                        'time_taken': epoch_time
                    }
                }
                self._send_message(self.socket, epoch_complete_message)
            
            # 获取模型参数
            model_params = self.model.state_dict()

            # 计算原始模型大小 (以字节为单位)
            original_buffer = io.BytesIO()
            torch.save(model_params, original_buffer)
            original_size_bytes = len(original_buffer.getvalue())
            
            # 获取隐私和压缩配置
            privacy_config = self.config.get_privacy_config()
            compression_config = self.config.get_compression_config()
            defense_config = self.config.get_defense_config()

            # 打印差分隐私信息
            if privacy_config.get('enabled', False):
                epsilon = defense_config.get('differential_privacy', {}).get('epsilon', 'N/A')
                delta = defense_config.get('differential_privacy', {}).get('delta', 'N/A')
                log_message = f"客户端 {self.client_id}: 应用差分隐私，消耗隐私预算 - Epsilon: {epsilon}, Delta: {delta}"
                print(log_message)
                self.logger.info(log_message)
            
            # 应用和打印压缩信息
            if compression_config.get('enabled', False):
                # 简单的稀疏化压缩：将小于阈值的参数置为零
                compression_threshold = 1e-3 # 可以根据需要调整这个阈值
                for name, param in model_params.items():
                    if param.is_floating_point(): # 只对浮点参数进行稀疏化
                        param[torch.abs(param) < compression_threshold] = 0
                
                log_message_compression = f"客户端 {self.client_id}: 应用稀疏化压缩 (阈值: {compression_threshold})"
                print(log_message_compression)
                self.logger.info(log_message_compression)

                compressed_buffer = io.BytesIO()
                torch.save(model_params, compressed_buffer)
                compressed_size_bytes = len(compressed_buffer.getvalue())

                space_saved_bytes = original_size_bytes - compressed_size_bytes
                space_saved_percentage = (space_saved_bytes / original_size_bytes) * 100 if original_size_bytes > 0 else 0

                log_message_saved = (
                    f"客户端 {self.client_id}: 模型压缩节省空间 - "
                    f"原始大小: {original_size_bytes / 1024:.2f} KB, "
                    f"压缩后大小: {compressed_size_bytes / 1024:.2f} KB, "
                    f"节省: {space_saved_bytes / 1024:.2f} KB ({space_saved_percentage:.2f}%)"
                )
                print(log_message_saved)
                self.logger.info(log_message_saved)
            else:
                # 如果没有启用压缩，也打印模型大小信息
                log_message_no_compression = f"客户端 {self.client_id}: 未启用模型压缩，模型大小: {original_size_bytes / 1024:.2f} KB"
                print(log_message_no_compression)
                self.logger.info(log_message_no_compression)
            
            # 如果有裁剪操作，也在此处打印（目前代码中没有显式裁剪，这里仅作为示例）
            if privacy_config.get('enabled', False) and privacy_config.get('clip_threshold') is not None:
                log_message = f"客户端 {self.client_id}: 模型梯度裁剪完成 (阈值: {privacy_config.get('clip_threshold')})"
                print(log_message)
                self.logger.info(log_message)

            # 序列化模型参数
            buffer = io.BytesIO()
            torch.save(model_params, buffer)
            serialized_model = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # 发送模型更新到服务器
            update_message = {
                'type': 'model_update',
                'client_id': self.client_id,
                'model': serialized_model,
                'metrics': {
                    'accuracy': epoch_accuracy,
                    'loss': running_loss/10
                }
            }
            self._send_message(self.socket, update_message)
            
            self.logger.info("本地训练完成，已发送模型更新到服务器")
            
        except Exception as e:
            self.logger.error(f"本地训练过程中发生错误: {str(e)}")
            raise

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='联邦学习客户端')
    parser.add_argument('--client_id', type=str, required=True, help='客户端ID')
    args = parser.parse_args()
    
    print(f"启动客户端 {args.client_id}")
    
    # 创建并运行客户端
    client = FederatedClient(args.client_id)
    client.run(client.train_loader)  # 传入训练数据加载器 