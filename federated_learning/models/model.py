import torch
from torch import nn
import os
import torch.nn.functional as F
import torchvision as tv
from loguru import logger

from federated_learning.models.bilstm import BiLSTM
from federated_learning.models.cifar_resnet import resnet18_cifar
from federated_learning.models.cnn_cifar_10 import Cifar10CNN
from federated_learning.models.cnn_mnist import CNNMNIST
from federated_learning.models.yolo_wrapper import YOLOWrapper


def setup_model(model_architecture, num_classes=None, tokenizer=None, embedding_dim=None, **kwargs):
    available_models = {
        "CNNMNIST": CNNMNIST,
        "BiLSTM": BiLSTM,
        "CNNCifar10": Cifar10CNN,
        "ResNet18CIFAR": resnet18_cifar,  # CIFAR 专用 stem（3x3/s1，无 maxpool）
        "ResNet18": tv.models.resnet18,   # torchvision ImageNet 版，32x32 上不推荐
        "VGG16": tv.models.vgg16,
        "DN121": tv.models.densenet121,
        "SHUFFLENET": tv.models.shufflenet_v2_x1_0,
        "YOLO": YOLOWrapper
    }
    logger.info('--> Creating {} model.....'.format(model_architecture))
    # variables in pre-trained ImageNet models are model-specific.
    if "ResNet18CIFAR" in model_architecture:
        model = available_models[model_architecture](num_classes=num_classes or 10)
    elif "ResNet18" in model_architecture:
        model = available_models[model_architecture]()
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, num_classes)
    elif "VGG16" in model_architecture:
        model = available_models[model_architecture]()
        n_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(n_features, num_classes)
    elif "SHUFFLENET" in model_architecture:
        model = available_models[model_architecture]()
        model.fc = nn.Linear(1024, num_classes)
    elif 'BiLSTM' in model_architecture:
        model = available_models[model_architecture](num_words=len(tokenizer.word_index), embedding_dim=embedding_dim)
    elif 'YOLO' in model_architecture:
        # YOLO模型配置
        # 可选环境变量（不改训练脚本即可切换）：
        #   YOLO_MODEL_SIZE=n|s|m|l|x  — 使用 Ultralytics 官方 COCO 预训练 yolov8*.pt（默认 s，对齐 23.7% 基线口径）
        #   YOLO_MODEL_PATH=/abs/path.pt — 使用本地权重（如 VisDrone 上微调过的 best.pt），设置时优先于 SIZE
        _size = kwargs.get('model_size') or os.environ.get('YOLO_MODEL_SIZE', 's')
        model_size = _size if _size in ('n', 's', 'm', 'l', 'x') else 's'
        if _size != model_size:
            logger.warning(f"无效的 model_size/YOLO_MODEL_SIZE={_size!r}，回退为 's'")
        pretrained = kwargs.get('pretrained', True)  # 默认使用预训练模型
        model_path = kwargs.get('model_path')
        if model_path is None:
            env_p = os.environ.get('YOLO_MODEL_PATH', '').strip()
            model_path = env_p or None
        model = YOLOWrapper(
            model_size=model_size,
            num_classes=num_classes or 10,
            pretrained=pretrained,
            model_path=model_path
        )
    else:
        model = available_models[model_architecture]()

    if model is None:
        logger.error("Incorrect model architecture specified or architecture not available.")
        raise ValueError(model_architecture)
    logger.info('--> Model has been created!')
    return model
