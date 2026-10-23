"""
CIFAR-10 专用 ResNet（3x3 stride1 stem、无 maxpool），供本项目联邦训练/评估使用。

来源与许可：
- 结构参考 kuangliu/pytorch-cifar（MIT）中的 ResNet 实现；
- FLPoison（https://github.com/vio1etus/FLPoison）`fl/models/resnet.py` 采用同一结构，
  本项目按"优先复用成熟实现"原则沿用，仅做两处适配：
  1) forward 支持 return_features=True → 返回 (logits, features)：
     `federated_learning/client.py` 的非 VisDrone 分支要求该签名；
  2) 提供 `resnet18_cifar(num_classes)` 工厂，供 `setup_model` 注册使用。

为什么不直接用 torchvision.models.resnet18：
- torchvision 版是 ImageNet 口径（7x7 stride2 stem + maxpool），32x32 输入经 layer4 后
  只剩 1x1 空间分辨率，CIFAR 上系统性吃亏（见 MEMORY/08_Roadmap.md §2）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    """CIFAR 版 ResNet：stem 为 3x3/stride1，末端 avg_pool2d(4)。"""

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetCIFAR, self).__init__()
        self.in_planes = 64

        # 与标准 ImageNet ResNet 的差异：3x3 / stride 1 / padding 1，且不做 maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # 注意：末层名为 linear（非 fc / fc2）；DPFLA 特征键已支持自动探测
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        features = out.view(out.size(0), -1)
        logits = self.linear(features)
        if return_features:
            return logits, features
        return logits


def resnet18_cifar(num_classes=10):
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


if __name__ == '__main__':
    # 快速自检：形状与 return_features 签名
    net = resnet18_cifar(num_classes=10)
    logits, feats = net(torch.randn(2, 3, 32, 32), return_features=True)
    print('logits', tuple(logits.shape), 'features', tuple(feats.shape))
