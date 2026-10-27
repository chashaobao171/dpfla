import sys
import os
from datetime import datetime
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# 添加必要的路径
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'src', 'utils'))
sys.path.append(os.path.join(project_root, 'src', 'models'))
sys.path.append(os.path.join(project_root, 'src', 'aggregation'))

# 打印搜索路径
print("Python搜索路径:")
for path in sys.path:
    print(f"- {path}")
import socket
import threading
import json
import logging
from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
import queue # 导入queue模块
import time # 导入time模块
import struct
import io
import zlib
import base64
import select

from utils.config_loader import ConfigLoader
from utils.logger import setup_logger
from aggregation.fedavg import FedAvgAggregator
from models.model_factory import ModelFactory

class FederatedServer:
    def __init__(self):
        """初始化联邦学习服务器"""
        # 加载配置
        self.config = ConfigLoader()
        self.server_config = self.config.get_server_config()
        self.model_type = self.config.get_model_type()
        self.max_rounds = self.config.get_training_config()['rounds']
        self.min_clients = self.config.get_training_config()['min_clients']
        
        # 设置日志
        self.logger = setup_logger('server', os.path.join('logs', f'server_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
        self.logger.info("服务器初始化开始")
        
        # 设置设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # 设置默认的CUDA设备
            torch.cuda.set_device(0)
            self.logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            self.logger.info("使用CPU (GPU不可用)")
        
        # 初始化服务器套接字
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.settimeout(1)  # 设置1秒超时
        
        # 初始化客户端连接管理
        self.clients = {}  # 存储客户端连接信息
        self.client_metrics = {}  # 存储客户端性能指标
        self.client_updates_by_round = {}  # 存储每轮的客户端更新
        self.client_metrics_by_round = {}  # 存储每轮的客户端指标
        
        # 初始化模型配置
        self.all_model_configs = {
            'resnet': {'in_channels': 3, 'num_classes': 10}
        }
        
        # 初始化聚合器
        self.aggregator = self._create_aggregator()
        
        # 初始化全局模型
        self.global_model = self._create_model()  # 初始化全局模型
        self.global_model = self.global_model.to(self.device)  # 将模型移到GPU
        
        # 初始化训练状态
        self.round = 0
        self.training_started = False
        
        # 初始化消息队列
        self.update_queue = queue.Queue()
        
        self.logger.info("服务器初始化完成")
        
        print("\n=== 联邦学习服务器配置 ===")
        print(f"服务器地址: {self.server_config['host']}:{self.server_config['port']}")
        print(f"最大客户端数: {self.server_config['max_clients']}")
        print(f"最小客户端数: {self.min_clients}")
        print("========================\n")
        
    def _send_message(self, sock: socket.socket, message: Dict[str, Any], max_retries: int = 3):
        """发送带长度前缀的JSON消息，带重试机制"""
        retries = 0
        while retries < max_retries:
            try:
                json_message = json.dumps(message)
                encoded_message = json_message.encode('utf-8')
                length = len(encoded_message)
                
                # 发送长度前缀
                sock.sendall(struct.pack('!I', length))
                # 发送消息内容
                sock.sendall(encoded_message)
                
                self.logger.debug(f"成功发送消息，长度: {length} 字节")
                return True
            except Exception as e:
                retries += 1
                self.logger.error(f"发送消息失败 (尝试 {retries}/{max_retries}): {str(e)}")
                if retries == max_retries:
                    raise
                time.sleep(1)  # 等待1秒后重试
        return False

    def _receive_message(self, sock: socket.socket, timeout: int = 300) -> Dict[str, Any] or None:
        """接收带长度前缀的JSON消息，带超时机制"""
        try:
            # 设置接收超时
            sock.settimeout(timeout)
            
            # 接收消息长度
            length_bytes = sock.recv(4)
            if not length_bytes:
                self.logger.warning("客户端连接已关闭 (长度字节为空)")
                return None
                
            length = struct.unpack('!I', length_bytes)[0]
            if length <= 0 or length > 100 * 1024 * 1024:  # 限制最大消息大小为100MB
                self.logger.error(f"无效的消息长度: {length}")
                return None
            
            # 接收消息内容
            data = b''
            bytes_received = 0
            while bytes_received < length:
                try:
                    packet = sock.recv(min(length - bytes_received, 8192))  # 每次最多接收8KB
                    if not packet:
                        self.logger.warning(f"客户端连接已关闭 (在接收 {length} 字节中途)")
                        return None
                    data += packet
                    bytes_received += len(packet)
                except socket.timeout:
                    self.logger.error(f"接收消息超时 (已接收 {bytes_received}/{length} 字节)")
                    return None
            
            # 解析JSON消息
            try:
                message = json.loads(data.decode('utf-8'))
                self.logger.debug(f"成功接收消息，类型: {message.get('type', 'unknown')}")
                return message
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON解析错误: {str(e)}")
                return None
                
        except Exception as e:
            self.logger.error(f"接收消息时发生错误: {str(e)}")
            return None
        finally:
            # 恢复默认超时设置
            sock.settimeout(None)

    def start(self):
        """启动服务器"""
        try:
            self.server_socket.bind((self.server_config['host'], self.server_config['port']))
            self.server_socket.listen(self.server_config['max_clients'])

            print("\n=== 联邦学习服务器已启动 ===")
            print(f"正在监听 {self.server_config['host']}:{self.server_config['port']}")
            print(f"等待至少 {self.min_clients} 个客户端连接...\n")

            # 打印训练参数
            print("\n=== 联邦学习训练参数 ===")
            print(f"模型类型: {self.model_type}")
            print(f"模型配置: {self.all_model_configs[self.model_type]}")
            print(f"训练轮次: {self.max_rounds}")
            print(f"聚合方法: {self.server_config['aggregation_method']}")
            print(f"最小客户端数: {self.min_clients}")
            print(f"最大客户端数: {self.server_config['max_clients']}")
            
            # 添加其他重要的训练参数
            training_config = self.config.get_training_config()
            print(f"每个客户端本地训练Epochs: {training_config.get('epochs', 'N/A')}")
            print(f"学习率: {training_config.get('learning_rate', 'N/A')}")
            print(f"优化器: {training_config.get('optimizer', 'N/A')}")
            print(f"批处理大小: {training_config.get('batch_size', 'N/A')}")

            # 隐私和压缩配置
            privacy_config = self.config.get_privacy_config()
            compression_config = self.config.get_compression_config()
            defense_config = self.config.get_defense_config()

            print(f"隐私保护启用: {privacy_config.get('enabled', False)}")
            if privacy_config.get('enabled', False):
                print(f"  噪声尺度 (noise_scale): {privacy_config.get('noise_scale', 'N/A')}")
                print(f"  裁剪阈值 (clip_threshold): {privacy_config.get('clip_threshold', 'N/A')}")
                print(f"  差分隐私epsilon: {defense_config.get('differential_privacy', {}).get('epsilon', 'N/A')}")
                print(f"  差分隐私delta: {defense_config.get('differential_privacy', {}).get('delta', 'N/A')}")

            print(f"模型压缩启用: {compression_config.get('enabled', False)}")
            if compression_config.get('enabled', False):
                print(f"  量化位数: {compression_config.get('quantization_bits', 'N/A')}")

            print("========================\n")

            # 阶段 1: 客户端注册
            self.logger.info(f"等待客户端注册... (已连接 {len(self.clients)}/{self.min_clients})")
            while len(self.clients) < self.min_clients:
                try:
                    client_socket, address = self.server_socket.accept()
                    self.logger.info(f"客户端连接: {address}")

                    # 处理注册消息并在单独线程中维护连接
                    client_thread = threading.Thread(
                        target=self.handle_client_registration_and_messages,
                        args=(client_socket, address, self.update_queue)
                    )
                    client_thread.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    self.logger.error(f"接受客户端连接时发生错误: {str(e)}")
                    break

            # 阶段 2: 联邦学习训练轮次
            self.training_started = True
            self.logger.info(f"已连接 {len(self.clients)} 个客户端，满足最小客户端数量要求 ({self.min_clients})。开始联邦学习训练轮次。")
            print("\n=== 开始联邦学习训练 ===")

            # 开始训练循环
            for self.round in range(1, self.max_rounds + 1):
                print(f"\n=== 第 {self.round}/{self.max_rounds} 轮训练 ===")

                # 1. 广播全局模型给所有已连接客户端
                self.broadcast_model()
                self.logger.info("全局模型广播完成，等待客户端更新...")
                print("等待客户端训练更新...")

                # 2. 等待并收集客户端发送的模型更新和指标
                required_clients_for_aggregation = len(self.clients)
                current_round_updates = []  # 改为列表存储
                start_time = time.time()
                timeout = 600  # 修改为10分钟超时

                while len(current_round_updates) < required_clients_for_aggregation:
                    try:
                        if time.time() - start_time > timeout:
                            self.logger.warning(f"等待客户端更新超时 ({timeout}秒)")
                            print(f"警告：等待客户端更新超时 ({timeout}秒)")
                            break

                        # 从队列中获取更新
                        update = self.update_queue.get(timeout=1)
                        current_round_updates.append(update)  # 直接添加到列表中
                        self.logger.info(f"收到客户端 {update['client_id']} 的更新 ({len(current_round_updates)}/{required_clients_for_aggregation})")
                        print(f"收到客户端 {update['client_id']} 的更新 ({len(current_round_updates)}/{required_clients_for_aggregation})")

                    except queue.Empty:
                        continue
                    except Exception as e:
                        self.logger.error(f"处理客户端更新时发生错误: {str(e)}")
                        continue

                    

                # 3. 聚合模型更新
                if len(current_round_updates) >= self.min_clients:
                    self.logger.info("开始聚合模型更新...")
                    print("开始聚合模型更新...")
                    try:
                        # 聚合模型
                        new_global_model_params = self.aggregator.aggregate(current_round_updates)
                        if new_global_model_params is not None:
                            self.global_model.load_state_dict(new_global_model_params)
                            self.global_model = self.global_model.to(self.device)
                            self.logger.info("模型聚合完成")
                            print("模型聚合完成")
                        else:
                            self.logger.error("聚合器返回了空的模型参数")
                    except Exception as e:
                        self.logger.error(f"模型聚合时发生错误: {str(e)}")
                        raise
                else:
                    self.logger.warning(f"未收到足够的客户端模型更新 ({len(current_round_updates)}/{required_clients_for_aggregation})，跳过本轮聚合。")
                    print(f"警告：未收到足够的客户端更新 ({len(current_round_updates)}/{required_clients_for_aggregation})，跳过本轮聚合。")

                self.logger.info(f"=== 第 {self.round}/{self.max_rounds} 轮训练完成 ===")
                print(f"=== 第 {self.round}/{self.max_rounds} 轮训练完成 ===")

            # 所有训练轮次完成后，向客户端发送停止指令
            self.logger.info("所有训练轮次完成，向所有客户端发送停止指令...")
            print("所有训练轮次完成，向所有客户端发送停止指令...")
            for client_id, client_info in list(self.clients.items()): # 迭代副本，避免在迭代时修改字典
                try:
                    if 'socket' in client_info and client_info['socket'].fileno() != -1:
                        self._send_message(client_info['socket'], {'type': 'stop'})
                        self.logger.info(f"已向客户端 {client_id} 发送停止指令")
                except Exception as e:
                    self.logger.error(f"向客户端 {client_id} 发送停止指令时发生错误: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"服务器启动错误或训练过程中发生错误: {str(e)}")
            raise
        finally:
            self.logger.info("服务器关闭，关闭所有客户端连接")
            print("服务器关闭，关闭所有客户端连接")
            # 关闭所有客户端连接
            clients_to_remove = []
            for client_id, client_info in list(self.clients.items()):  # 使用list()避免在迭代时修改字典
                if 'socket' in client_info:
                    try:
                        # 检查套接字是否仍然有效
                        if client_info['socket'].fileno() != -1:
                            client_info['socket'].close()
                            self.logger.info(f"客户端 {client_id} 连接已关闭")
                        else:
                            self.logger.info(f"客户端 {client_id} 的套接字已关闭或无效，无需再次关闭")
                    except OSError as e:
                        self.logger.warning(f"关闭客户端 {client_id} 连接时发生错误 (可能已关闭): {e}")
                    except Exception as e:
                        self.logger.error(f"清理客户端 {client_id} 连接时发生未知错误: {e}")
                    finally:
                        clients_to_remove.append(client_id)
                else:
                    clients_to_remove.append(client_id) # 客户端信息可能没有套接字，也要移除

            for client_id in clients_to_remove:
                if client_id in self.clients:
                    del self.clients[client_id]
            
            # 关闭服务器socket
            if self.server_socket: # 检查server_socket对象是否存在
                try:
                    if self.server_socket.fileno() != -1: # 检查套接字是否仍然有效
                        self.server_socket.close() # 只关闭，不需要shutdown
                        self.logger.info("服务器套接字已关闭")
                    else:
                        self.logger.info("服务器套接字已关闭或无效，无需再次关闭")
                except OSError as e:
                    self.logger.warning(f"关闭服务器套接字时发生错误 (可能已关闭): {e}")
                except Exception as e:
                    self.logger.error(f"清理服务器套接字时发生未知错误: {e}")

            self.logger.info("服务器资源清理完成")

    def _deserialize_model(self, model_str: str) -> Dict[str, torch.Tensor]:
        """反序列化模型字符串"""
        try:
            model_bytes = base64.b64decode(model_str)
            buffer = io.BytesIO(model_bytes)
            model_state = torch.load(buffer)
            return model_state
        except Exception as e:
            self.logger.error(f"反序列化模型时发生错误: {e}")
            raise


    def handle_client_registration_and_messages(self, client_socket: socket.socket, address: tuple, update_queue: queue.Queue):
        """处理客户端注册和消息"""
        try:
            # 接收注册消息
            message = self._receive_message(client_socket)
            if not message or message.get('type') != 'register':
                self.logger.error("收到无效的注册消息")
                client_socket.close()
                return
                
            client_id = message.get('client_id')
            if not client_id:
                self.logger.error("注册消息中缺少client_id")
                client_socket.close()
                return
                
            # 发送注册确认
            self._send_message(client_socket, {
                'type': 'registered_ack',
                'client_id': client_id
            })
            
            # 存储客户端信息
            self.clients[client_id] = {
                'socket': client_socket,
                'address': address,
                'last_active': time.time()
            }
            
            self.logger.info(f"客户端 {client_id} 注册成功")
            
            # 进入消息处理循环
            while True:
                message = self._receive_message(client_socket)
                if message is None:
                    break
                    
                message_type = message.get('type')
                if message_type == 'model_update':
                    self.logger.info(f"收到客户端 {client_id} 的模型更新")
                    model_str = message.get('model')
                    metrics = message.get('metrics', {})
                    
                    try:
                        # 反序列化模型参数
                        model_params = self._deserialize_model(model_str)
                        
                        # 准备更新数据
                        update_data = {
                            'client_id': client_id,
                            'model_params': model_params,
                            'data_size': self.clients[client_id].get('data_size', 1),
                            'metrics': metrics
                        }
                        
                        # 将更新放入队列
                        self.update_queue.put(update_data)
                        
                        accuracy = metrics.get('accuracy', 'N/A')
                        loss = metrics.get('loss', 'N/A')
                        print(f"收到客户端 {client_id} 模型更新 - 准确率: {accuracy:.2f}%, 损失: {loss:.4f}")
                        
                    except Exception as e:
                        self.logger.error(f"处理客户端 {client_id} 的模型更新时发生错误: {str(e)}")
                        continue
                elif message_type == 'training_progress':
                    # 处理训练进度消息
                    progress = message.get('progress', {})
                    self.logger.info(
                        f"客户端 {client_id} 训练进度 - "
                        f"Epoch {progress.get('epoch')}/{progress.get('total_epochs')} - "
                        f"Batch {progress.get('batch')}/{progress.get('total_batches')} "
                        f"({progress.get('progress_percentage'):.1f}%) - "
                        f"Accuracy: {progress.get('current_accuracy'):.2f}% - "
                        f"Loss: {progress.get('current_loss'):.4f}"
                    )
                elif message_type == 'epoch_complete':
                    # 处理epoch完成消息
                    epoch_info = message.get('epoch_info', {})
                    self.logger.info(
                        f"客户端 {client_id} 完成 Epoch {epoch_info.get('epoch')}/{epoch_info.get('total_epochs')} - "
                        f"准确率: {epoch_info.get('accuracy'):.2f}% - "
                        f"用时: {epoch_info.get('time_taken'):.1f}秒"
                    )
                else:
                    self.logger.warning(f"收到未知类型的消息: {message_type}")
                    
        except Exception as e:
            self.logger.error(f"处理客户端 {client_id} 消息时发生错误: {str(e)}")
        finally:
            # 清理客户端连接
            if client_id in self.clients:
                del self.clients[client_id]
            client_socket.close()
            self.logger.info(f"客户端 {client_id} 连接已关闭")
            
    def aggregate_model(self, client_model_state: Dict[str, torch.Tensor]):
        """聚合客户端模型更新"""
        try:
            # 反序列化模型状态
            buffer = io.BytesIO(base64.b64decode(client_model_state))
            model_state = torch.load(buffer)
            
            # 实现模型聚合逻辑
            if self.server_config['aggregation_method'] == 'fedavg':
                # 简单的FedAvg实现
                for param, client_param in zip(self.global_model.parameters(), model_state.values()):
                    param.data = (param.data + client_param) / 2
                    
            self.logger.info("模型聚合完成")
            
        except Exception as e:
            self.logger.error(f"模型聚合时发生错误: {str(e)}")
            raise

    def broadcast_model(self):
        """向所有客户端广播全局模型"""
        try:
            # 获取模型状态字典
            model_state = self.global_model.state_dict()
            
            # 序列化模型状态字典
            buffer = io.BytesIO()
            torch.save(model_state, buffer)
            serialized_state = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # 准备广播消息
            broadcast_message = {
                'type': 'global_model',
                'round': self.round,
                'model': serialized_state
            }
            
            # 向每个客户端发送模型
            for client_id, client_info in self.clients.items():
                try:
                    self.logger.info(f"向客户端 {client_id} 广播模型")
                    self._send_message(client_info['socket'], broadcast_message)
                except Exception as e:
                    self.logger.error(f"向客户端 {client_id} 广播模型时发生错误: {str(e)}")
                    # 如果发送失败，从客户端列表中移除
                    del self.clients[client_id]
                    client_info['socket'].close()
            
            self.logger.info("全局模型广播完成，等待客户端更新...")
            print("等待客户端训练更新...")
            
        except Exception as e:
            self.logger.error(f"广播模型时发生错误: {str(e)}")
            raise

    def aggregate_and_broadcast(self):
        """聚合模型并广播全局模型"""
        print("\n=== 开始新一轮聚合 ===")
        print(f"参与聚合的客户端数: {len(self.clients)}")
        
        # 聚合模型
        self.aggregate_model(self.global_model.state_dict())
        
        # 广播全局模型
        self.broadcast_model()
        
        # 清空客户端模型缓存
        self.clients.clear()
        print("=== 本轮聚合完成 ===\n")

    def _create_aggregator(self):
        """创建聚合器"""
        aggregation_method = self.server_config.get('aggregation_method', 'fedavg')
        if aggregation_method == 'fedavg':
            return FedAvgAggregator()
        else:
            raise ValueError(f"不支持的聚合方法: {aggregation_method}")

    def _create_model(self):
        """创建全局模型"""
        try:
            model_config = self.all_model_configs[self.model_type]
            return ModelFactory.create_model(self.model_type, model_config)
        except Exception as e:
            self.logger.error(f"创建模型时发生错误: {str(e)}")
            raise

    def cleanup(self):
        """清理服务器资源"""
        self.logger.info("服务器关闭，关闭所有客户端连接")
        print("服务器关闭，关闭所有客户端连接")
        try:
            # 关闭所有客户端连接
            for client_id, client_info in list(self.clients.items()):  # 使用list()避免在迭代时修改字典
                try:
                    if 'socket' in client_info and client_info['socket'].fileno() != -1:  # 检查套接字是否仍然有效
                        client_info['socket'].shutdown(socket.SHUT_RDWR)  # 尝试优雅关闭
                        client_info['socket'].close()
                        self.logger.info(f"客户端 {client_id} 连接已关闭")
                except OSError as e:
                    self.logger.warning(f"关闭客户端 {client_id} 连接时发生错误 (可能已关闭): {e}")
                except Exception as e:
                    self.logger.error(f"清理客户端 {client_id} 连接时发生未知错误: {e}")
                finally:
                    if client_id in self.clients:  # 确保在处理完后从列表中移除
                        del self.clients[client_id]

            # 关闭服务器socket
            if self.server_socket.fileno() != -1:  # 检查服务器套接字是否仍然有效
                self.server_socket.shutdown(socket.SHUT_RDWR)
                self.server_socket.close()
            self.logger.info("服务器资源清理完成")

        except Exception as e:
            self.logger.error(f"清理服务器资源时发生错误: {str(e)}")
            raise

if __name__ == "__main__":
    server = FederatedServer()
    server.start() 