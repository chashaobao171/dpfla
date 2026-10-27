import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.in_channels = 64
        self.num_classes = config['num_classes']
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(config['in_channels'], 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 残差块 - ResNet-32: [3,4,6,3]
        self.layer1 = self._make_layer(64, 3, stride=1)    # 3个残差块
        self.layer2 = self._make_layer(128, 4, stride=2)   # 4个残差块
        self.layer3 = self._make_layer(256, 6, stride=2)   # 6个残差块
        self.layer4 = self._make_layer(512, 3, stride=2)   # 3个残差块
        
        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, self.num_classes)
        
        # 初始化权重
        self._initialize_weights()

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

class ModelFactory:
    """模型工厂类，用于创建不同类型的神经网络模型"""
    
    @staticmethod
    def create_model(model_type: str, model_config: Dict[str, Any]) -> nn.Module:
        """
        创建指定类型的模型
        
        Args:
            model_type: 模型类型名称
            model_config: 模型配置参数
            
        Returns:
            创建的模型实例
        """
        if model_type == "resnet":
            # 创建ResNet18模型
            model = models.resnet18(weights=None)  # 使用weights=None替代pretrained=False
            # 修改第一层以适应MNIST的1通道输入
            model.conv1 = nn.Conv2d(model_config['in_channels'], 64, kernel_size=7, stride=2, padding=3, bias=False)
            # 修改最后一层以适应输出类别数
            model.fc = nn.Linear(model.fc.in_features, model_config['num_classes'])
            return model
        elif model_type == "simple_cnn":
            return SimpleCNN(model_config)
        elif model_type == "simple_mlp":
            return SimpleMLP(model_config)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

class SimpleCNN(nn.Module):
    """简单的CNN模型，用于CIFAR10数据集"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(config['in_channels'], 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 计算全连接层的输入特征数
        self._to_linear = None
        self._get_conv_output_size(config['in_channels'])
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._to_linear, 512),
            nn.ReLU(),
            nn.Dropout(config.get('dropout_rate', 0.1)),
            nn.Linear(512, config['num_classes'])
        )
    
    def _get_conv_output_size(self, in_channels):
        """计算卷积层输出大小"""
        x = torch.randn(1, in_channels, 32, 32)
        x = self.features(x)
        self._to_linear = x.shape[1] * x.shape[2] * x.shape[3]
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class SimpleMLP(nn.Module):
    """简单的MLP模型，用于MNIST数据集"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config['input_size'], config['hidden_sizes'][0]),
            nn.ReLU(),
            nn.Dropout(config.get('dropout_rate', 0.1)),
            nn.Linear(config['hidden_sizes'][0], config['hidden_sizes'][1]),
            nn.ReLU(),
            nn.Dropout(config.get('dropout_rate', 0.1)),
            nn.Linear(config['hidden_sizes'][1], config['num_classes'])
        )
    
    def forward(self, x):
        return self.classifier(x)