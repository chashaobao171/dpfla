import os
import sys

# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 添加项目根目录到Python路径
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import threading

from src.client.client import FederatedClient
from src.server.server import FederatedServer
from src.attack.label_flipping import LabelFlippingAttack
from src.utils.config_loader import ConfigLoader

def main():
    # 加载配置
    config_loader = ConfigLoader()
    config = config_loader.config # 获取完整的配置字典

    # 使用DataLoaderFactory根据配置加载数据集
    # 加载整个训练集用于分割
    dataset_config = config.get('dataset', {})
    # 注意: DataLoaderFactory.create_dataloader 返回的是 train_loader, test_loader
    # 我们需要原始的train_dataset来分割
    # 临时加载完整数据集以便分割，这里需要根据dataset.type选择加载方式
    dataset_type = dataset_config.get('type', 'mnist')
    data_dir = dataset_config.get('config', {}).get('data_dir', './data')
    transform = None # Placeholder, DataLoaderFactory handles transforms

    if dataset_type == 'mnist':
         transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
         full_train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    elif dataset_type == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        full_train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


    # 将数据集分割给不同的客户端
    client_count = config.get('server', {}).get('max_clients', 3) # 从server config获取客户端数量，或者使用默认值
    # 确保分割大小正确
    data_per_client = len(full_train_dataset) // client_count
    lengths = [data_per_client] * client_count
    # 处理余数，将剩余样本分配给最后一个客户端
    lengths[-1] += len(full_train_dataset) % client_count

    if sum(lengths) != len(full_train_dataset):
         # 如果计算有误，回退到简单的平均分配，并记录警告
         print(f"Warning: Dataset split calculation mismatch. Falling back to simple split.")
         avg_len = len(full_train_dataset) // client_count
         lengths = [avg_len] * client_count
         lengths[-1] += len(full_train_dataset) % client_count


    client_datasets = random_split(full_train_dataset, lengths)

    # 创建服务器
    server = FederatedServer()
    
    # 创建客户端
    clients = []
    for i in range(len(client_datasets)):
        client = FederatedClient(f"client_{i}")
        
        # 为每个客户端创建数据加载器，使用分割后的数据集和配置中的batch size
        data_loader = DataLoader(
            client_datasets[i],
            batch_size=config.get('client', {}).get('batch_size', 32),
            shuffle=True, # 通常训练时shuffle
            num_workers=dataset_config.get('config', {}).get('num_workers', 2) # 使用配置中的num_workers
        )

        # 如果启用攻击，创建攻击者
        attack_config = config_loader.get_attack_config()
        if attack_config['enabled']:
            attack = LabelFlippingAttack(attack_config)
            if attack.is_attacker():
                # 警告: 攻击逻辑需要根据新的数据加载方式进行调整
                print(f"Warning: Attack logic for client {i} might need adjustments to work with the new data loading.")
                # 示例：如何获取数据进行攻击（可能需要根据实际Attack类调整）
                # 这里只是示意，实际操作取决于Attack类的实现
                # 攻击通常在数据集层面进行，而不是DataLoader层面
                # 您可能需要在分割数据集之前或之后，对client_datasets[i]进行攻击处理
                pass # 暂时跳过攻击逻辑，需要用户根据实际情况实现
        
        clients.append(client)
    
    # 启动服务器
    server_thread = threading.Thread(target=server.start)
    server_thread.start()
    
    # 启动客户端
    client_threads = []
    for i, client in enumerate(clients):
        data_loader = DataLoader(client_datasets[i], batch_size=32)
        thread = threading.Thread(
            target=client.run,
            args=(data_loader,)
        )
        client_threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in client_threads:
        thread.join()
    server_thread.join()

if __name__ == "__main__":
    main() 