import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import os

class Visualizer:
    """可视化工具类，用于记录和展示训练过程"""
    
    def __init__(self, log_dir: str = "runs"):
        """
        初始化可视化工具
        
        Args:
            log_dir: TensorBoard日志目录
        """
        self.writer = SummaryWriter(log_dir)
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'train_recall': [],
            'test_loss': [],
            'test_acc': [],
            'test_recall': []
        }
        
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        记录训练指标
        
        Args:
            metrics: 指标字典
            step: 当前步数
        """
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
            if name in self.metrics:
                self.metrics[name].append(value)
    
    def log_model_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor):
        """
        记录模型结构
        
        Args:
            model: 模型实例
            input_tensor: 输入张量
        """
        self.writer.add_graph(model, input_tensor)
    
    def plot_metrics(self, save_path: str = None):
        """
        绘制训练指标图表
        
        Args:
            save_path: 图表保存路径
        """
        plt.figure(figsize=(15, 5))
        
        # 绘制损失曲线
        plt.subplot(1, 3, 1)
        plt.plot(self.metrics['train_loss'], label='Train Loss')
        plt.plot(self.metrics['test_loss'], label='Test Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Testing Loss')
        
        # 绘制准确率曲线
        plt.subplot(1, 3, 2)
        plt.plot(self.metrics['train_acc'], label='Train Accuracy')
        plt.plot(self.metrics['test_acc'], label='Test Accuracy')
        plt.xlabel('Steps')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Training and Testing Accuracy')
        
        # 绘制召回率曲线
        plt.subplot(1, 3, 3)
        plt.plot(self.metrics['train_recall'], label='Train Recall')
        plt.plot(self.metrics['test_recall'], label='Test Recall')
        plt.xlabel('Steps')
        plt.ylabel('Recall (%)')
        plt.legend()
        plt.title('Training and Testing Recall')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def close(self):
        """关闭TensorBoard写入器"""
        self.writer.close()

class ModelCheckpoint:
    """模型检查点类，用于保存和加载模型"""
    
    def __init__(self, save_dir: str = "checkpoints"):
        """
        初始化模型检查点
        
        Args:
            save_dir: 模型保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_model(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                  epoch: int, metrics: Dict[str, float], filename: str = None):
        """
        保存模型检查点
        
        Args:
            model: 模型实例
            optimizer: 优化器实例
            epoch: 当前轮次
            metrics: 训练指标
            filename: 保存文件名
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        save_path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, save_path)
    
    def load_model(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                  filename: str) -> Dict:
        """
        加载模型检查点
        
        Args:
            model: 模型实例
            optimizer: 优化器实例
            filename: 检查点文件名
            
        Returns:
            加载的检查点信息
        """
        load_path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(load_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint 