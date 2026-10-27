import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积层：输入3通道，输出32通道，3x3卷积核
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # 第二个卷积层：输入32通道，输出64通道，3x3卷积核
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 第三个卷积层：输入64通道，输出128通道，3x3卷积核
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)  # CIFAR-10有10个类别
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 第一个卷积块
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # 第二个卷积块
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # 第三个卷积块
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        # 展平
        x = x.view(-1, 128 * 4 * 4)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x 