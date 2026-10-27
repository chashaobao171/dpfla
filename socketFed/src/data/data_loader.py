import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import numpy as np
from typing import Tuple, List, Dict, Any
import os
import logging
import requests
from tqdm import tqdm
import shutil

logger = logging.getLogger(__name__)

class DataLoaderFactory:
    """数据集加载工厂类"""
    
    @staticmethod
    def create_dataloader(dataset_type: str, dataset_config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
        """
        创建指定类型的数据加载器
        
        Args:
            dataset_type: 数据集类型名称
            dataset_config: 数据集配置参数
            
        Returns:
            训练集和测试集的数据加载器
        """
        if dataset_type == "mnist":
            return DataLoaderFactory._create_mnist_dataloader(dataset_config)
        elif dataset_type == "cifar10":
            return DataLoaderFactory._create_cifar10_dataloader(dataset_config)
        else:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    @staticmethod
    def _create_mnist_dataloader(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
        """创建MNIST数据加载器"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 加载训练集
        train_dataset = datasets.MNIST(
            config.get('data_dir', './data'),
            train=True,
            download=True,
            transform=transform
        )
        
        # 加载测试集
        test_dataset = datasets.MNIST(
            config.get('data_dir', './data'),
            train=False,
            transform=transform
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=True,
            num_workers=config.get('num_workers', 4)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=False,
            num_workers=config.get('num_workers', 4)
        )
        
        return train_loader, test_loader
    
    @staticmethod
    def _download_cifar10(url: str, save_path: str) -> bool:
        """下载CIFAR10数据集"""
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(save_path, 'wb') as f, tqdm(
                desc="下载CIFAR10数据集",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
            return True
        except Exception as e:
            logger.error(f"下载失败: {str(e)}")
            return False

    @staticmethod
    def _create_cifar10_dataloader(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
        """创建CIFAR10数据加载器"""
        data_dir = config.get('data_dir', './data')
        os.makedirs(data_dir, exist_ok=True)

        # 检查已有的数据集目录是否存在以及其中一个批次文件是否存在
        # 您的解压目录是 cifar10，文件在其中
        cifar_extracted_dir = os.path.join(data_dir, 'cifar10')
        batch_file_path = os.path.join(cifar_extracted_dir, 'data_batch_1')

        # 如果已有的解压目录或关键文件不存在，则进行下载和解压
        if not os.path.exists(cifar_extracted_dir) or not os.path.exists(batch_file_path):
            logger.info("未检测到完整的CIFAR10数据集，开始下载...")

            cifar_path = os.path.join(data_dir, 'cifar-10-python.tar.gz')

            # 检查压缩文件是否存在，如果不存在或大小异常（小于100MB认为异常），则下载
            # 注意：这里的下载仍然会使用默认的解压逻辑，生成 cifar-10-batches-py 目录
            if not os.path.exists(cifar_path) or (os.path.exists(cifar_path) and os.path.getsize(cifar_path) < 100 * 1024 * 1024):
                 logger.info("开始从阿里云镜像下载CIFAR10数据集...")
                 url = "https://mirrors.aliyun.com/pytorch-datasets/cifar-10-python.tar.gz"
                 if not DataLoaderFactory._download_cifar10(url, cifar_path):
                     raise Exception("无法从阿里云镜像下载CIFAR10数据集，请检查网络连接")

            # 解压数据集
            logger.info("开始解压CIFAR10数据集...")
            try:
                import tarfile
                with tarfile.open(cifar_path, 'r:gz') as tar:
                    # 提取到 data_dir 目录下，会生成 cifar-10-batches-py 目录
                    tar.extractall(path=data_dir)
                logger.info("CIFAR10数据集解压完成")
                # 如果您希望使用现有的 cifar10 文件夹，并且它包含正确的文件，
                # 可以考虑在这里添加逻辑来验证 cifar10 文件夹，并可能删除新解压的 cifar-10-batches-py
                # 或者直接修改下面的数据集加载路径。

            except Exception as e:
                 logger.error(f"解压CIFAR10数据集失败: {str(e)}")
                 raise Exception(f"解压CIFAR10数据集失败: {str(e)}")

        # 简化数据预处理步骤
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # 加载训练集
        # 使用您现有的 cifar10 目录来加载数据集
        train_dataset = datasets.CIFAR10(
            data_dir,
            train=True,
            download=False,  # 已经手动下载/解压，不需要自动下载
            transform=transform_train
        )

        # 加载测试集
        # 使用您现有的 cifar10 目录来加载数据集
        test_dataset = datasets.CIFAR10(
            data_dir,
            train=False,
            download=False,  # 已经手动下载/解压，不需要自动下载
            transform=transform_test
        )

        # 创建数据加载器，增加num_workers和pin_memory
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=True,
            num_workers=config.get('num_workers', 2),  # 使用配置中的worker数量
            pin_memory=config.get('pin_memory', True),  # 使用配置中的pin_memory设置
            persistent_workers=config.get('persistent_workers', True) # 使用配置中的persistent_workers设置
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=False,
            num_workers=config.get('num_workers', 2),
            pin_memory=config.get('pin_memory', True),
            persistent_workers=config.get('persistent_workers', True)
        )

        return train_loader, test_loader
    
    @staticmethod
    def split_dataset(dataset: Dataset, n_clients: int) -> List[Dataset]:
        """
        将数据集分割给多个客户端
        
        Args:
            dataset: 原始数据集
            n_clients: 客户端数量
            
        Returns:
            分割后的数据集列表
        """
        # 计算每个客户端的数据量
        data_per_client = len(dataset) // n_clients
        lengths = [data_per_client] * n_clients
        
        # 分割数据集
        return random_split(dataset, lengths)

    def load_data(self):
        """加载数据"""
        try:
            # 获取数据集配置
            dataset_config = self.config.get_dataset_config()
            dataset_type = dataset_config['type']
            config = dataset_config['config']
            
            # 设置数据转换
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            # 加载数据集
            if dataset_type == 'cifar10':
                # 加载训练集
                train_dataset = datasets.CIFAR10(
                    root=config['data_dir'],
                    train=True,
                    download=True,
                    transform=transform
                )
                
                # 加载测试集
                test_dataset = datasets.CIFAR10(
                    root=config['data_dir'],
                    train=False,
                    download=True,
                    transform=transform
                )
                
                # 计算每个客户端的数据量
                total_train_size = len(train_dataset)
                num_clients = config.get('num_clients', 3)  # 默认3个客户端
                client_data_size = total_train_size // num_clients
                
                # 为当前客户端分配数据
                client_id = int(self.client_id.split('_')[1])  # 从 client_X 中提取数字
                start_idx = client_id * client_data_size
                
                # 确保最后一个客户端获得所有剩余数据
                if client_id == num_clients - 1:
                    end_idx = total_train_size
                else:
                    end_idx = start_idx + client_data_size
                
                # 创建客户端的数据子集
                indices = list(range(start_idx, end_idx))
                if not indices:  # 如果索引列表为空
                    self.logger.error(f"客户端 {self.client_id} 没有分配到数据")
                    raise ValueError(f"客户端 {self.client_id} 没有分配到数据")
                    
                client_train_dataset = Subset(train_dataset, indices)
                
                # 创建数据加载器
                train_loader = DataLoader(
                    client_train_dataset,
                    batch_size=config['batch_size'],
                    shuffle=True,
                    num_workers=config['num_workers']
                )
                
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=config['batch_size'],
                    shuffle=False,
                    num_workers=config['num_workers']
                )
                
                self.logger.info(f"客户端 {self.client_id} 加载了 {len(client_train_dataset)} 个训练样本")
                self.logger.info(f"客户端 {self.client_id} 加载了 {len(test_dataset)} 个测试样本")
                
                return train_loader, test_loader
                
            else:
                raise ValueError(f"不支持的数据集类型: {dataset_type}")
                
        except Exception as e:
            self.logger.error(f"加载数据时发生错误: {str(e)}")
            raise 