"""
VisDrone数据集加载器
支持外部路径，动态转换标注格式为YOLO格式
支持缓存机制，加速重复加载
"""

import os
import pickle
import hashlib
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from loguru import logger


# VisDrone类别映射 (原始类别ID -> 新类别ID)
# VisDrone原始: 0=ignored, 1=pedestrian, 2=people, 3=bicycle, 4=car, 
#               5=van, 6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor, 11=others
# 我们只使用1-10类，映射到0-9
VISDRONE_CLASS_MAP = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9,  # motor
}

VISDRONE_CLASSES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]


def _letterbox_xywh_norm(img_rgb, boxes_xywh_norm, out_size=640, pad_value=114):
    """
    与 Ultralytics val 类似的 letterbox：等比缩放 + 对齐 stride 的 padding，并同步变换归一化 xywh 框。
    img_rgb: HWC uint8/float RGB
    boxes_xywh_norm: (N,4) tensor，相对原图尺寸的归一化中心与宽高
    """
    h0, w0 = img_rgb.shape[:2]
    nh = nw = int(out_size)
    r = min(nw / w0, nh / h0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    # 与 Ultralytics LetterBox(auto=False) 一致：铺满目标方形，不用 mod stride（predict 默认 rect=False）
    dw, dh = float(nw - new_w), float(nh - new_h)
    dw, dh = dw / 2.0, dh / 2.0
    img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_out = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value)
    )
    if boxes_xywh_norm is None or boxes_xywh_norm.numel() == 0:
        return img_out, boxes_xywh_norm
    t = boxes_xywh_norm.float()
    xc = t[:, 0] * w0
    yc = t[:, 1] * h0
    bw = t[:, 2] * w0
    bh = t[:, 3] * h0
    x1 = (xc - bw / 2) * r + left
    y1 = (yc - bh / 2) * r + top
    x2 = (xc + bw / 2) * r + left
    y2 = (yc + bh / 2) * r + top
    xc2 = ((x1 + x2) / 2) / nw
    yc2 = ((y1 + y2) / 2) / nh
    w2 = (x2 - x1) / nw
    h2 = (y2 - y1) / nh
    xc2 = xc2.clamp(0.0, 1.0)
    yc2 = yc2.clamp(0.0, 1.0)
    w2 = w2.clamp(0.0, 1.0)
    h2 = h2.clamp(0.0, 1.0)
    new_boxes = torch.stack([xc2, yc2, w2, h2], dim=1)
    return img_out, new_boxes


class VisDroneDataset(Dataset):
    """
    VisDrone目标检测数据集
    
    自动将VisDrone标注格式转换为训练所需格式
    支持缓存机制，第一次加载后会缓存标注数据
    """
    
    def __init__(self, root_path=None, split='train', img_size=640, transform=None, use_cache=True):
        """
        Args:
            root_path: VisDrone数据集根目录（None时自动检测）
            split: 'train' 或 'val'
            img_size: 图像缩放大小
            transform: 图像变换
            use_cache: 是否使用缓存（默认True）
        """
        # 自动检测数据集路径
        if root_path is None:
            if os.path.exists('/root/autodl-tmp/data/visdrone'):
                root_path = '/root/autodl-tmp/data/visdrone'
            elif os.path.exists('/home/featurize/data/visdrone'):
                root_path = '/home/featurize/data/visdrone'
            else:
                root_path = 'D:/Pycharmworkplace/visdrone'
        
        self.root_path = root_path
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.use_cache = use_cache
        
        # 确定数据集路径（处理嵌套目录结构）
        if split == 'train':
            base_path = os.path.join(root_path, 'VisDrone2019-DET-train')
            # 检查是否有嵌套的同名目录
            nested_path = os.path.join(base_path, 'VisDrone2019-DET-train')
            if os.path.exists(nested_path):
                self.data_path = nested_path
            else:
                self.data_path = base_path
        else:
            base_path = os.path.join(root_path, 'VisDrone2019-DET-val')
            # 检查是否有嵌套的同名目录
            nested_path = os.path.join(base_path, 'VisDrone2019-DET-val')
            if os.path.exists(nested_path):
                self.data_path = nested_path
            else:
                self.data_path = base_path
        
        self.images_path = os.path.join(self.data_path, 'images')
        self.annotations_path = os.path.join(self.data_path, 'annotations')
        
        # 获取所有图像文件
        self.image_files = []
        if os.path.exists(self.images_path):
            self.image_files = [f for f in os.listdir(self.images_path) 
                               if f.endswith(('.jpg', '.png', '.jpeg'))]
            self.image_files.sort()
        else:
            logger.warning(f'Images path does not exist: {self.images_path}')
        
        # 尝试加载缓存的标注数据
        self.annotations_cache = None
        if self.use_cache:
            self.annotations_cache = self._load_or_create_cache()
        
        logger.info(f'--> Loaded VisDrone {split} dataset: {len(self.image_files)} images')
    
    def _get_cache_path(self):
        """生成缓存文件路径"""
        # 使用数据集路径和配置生成唯一的缓存文件名
        cache_key = f"{self.data_path}_{self.split}_{len(self.image_files)}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_filename = f"cache_VisDrone_{self.split}_{len(self.image_files)}_{cache_hash}.pkl"
        
        # 缓存文件放在项目根目录的cache文件夹
        cache_dir = os.path.join(os.getcwd(), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, cache_filename)
        return cache_path
    
    def _load_or_create_cache(self):
        """加载或创建缓存"""
        cache_path = self._get_cache_path()
        
        # 尝试加载现有缓存
        if os.path.exists(cache_path):
            try:
                logger.info(f'Loading cached annotations from {cache_path}')
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                logger.info(f'✓ Loaded {len(cache_data)} cached annotations')
                return cache_data
            except Exception as e:
                logger.warning(f'Failed to load cache: {e}, will recreate')
        
        # 创建新缓存
        logger.info(f'Creating annotation cache for {len(self.image_files)} images...')
        cache_data = {}
        
        for idx, img_name in enumerate(self.image_files):
            if idx % 1000 == 0:
                logger.info(f'Processing annotations: {idx}/{len(self.image_files)}')
            
            # 加载标注
            ann_name = os.path.splitext(img_name)[0] + '.txt'
            ann_path = os.path.join(self.annotations_path, ann_name)
            
            # 获取图像尺寸（用于归一化）
            img_path = os.path.join(self.images_path, img_name)
            try:
                with Image.open(img_path) as img:
                    orig_w, orig_h = img.size
            except Exception as e:
                logger.warning(f'Failed to open image {img_name}: {e}')
                continue
            
            boxes = []
            labels = []
            
            if os.path.exists(ann_path):
                with open(ann_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 6:
                            bbox_left = float(parts[0])
                            bbox_top = float(parts[1])
                            bbox_width = float(parts[2])
                            bbox_height = float(parts[3])
                            category = int(parts[5])
                            
                            # 只保留有效类别(1-10)
                            if category in VISDRONE_CLASS_MAP:
                                # 转换为YOLO格式 (归一化的中心点坐标和宽高)
                                x_center = (bbox_left + bbox_width / 2) / orig_w
                                y_center = (bbox_top + bbox_height / 2) / orig_h
                                w = bbox_width / orig_w
                                h = bbox_height / orig_h
                                
                                # 确保坐标在有效范围内
                                x_center = max(0, min(1, x_center))
                                y_center = max(0, min(1, y_center))
                                w = max(0, min(1, w))
                                h = max(0, min(1, h))
                                
                                if w > 0 and h > 0:
                                    boxes.append([x_center, y_center, w, h])
                                    labels.append(VISDRONE_CLASS_MAP[category])
            
            # 保存到缓存
            cache_data[img_name] = {
                'boxes': boxes,
                'labels': labels,
                'orig_size': (orig_w, orig_h)
            }
        
        # 保存缓存文件
        try:
            logger.info(f'Saving annotation cache to {cache_path}')
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f'✓ Cache saved successfully')
        except Exception as e:
            logger.warning(f'Failed to save cache: {e}')
        
        return cache_data
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_path, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # 从缓存加载标注（如果可用）
        if self.annotations_cache and img_name in self.annotations_cache:
            cached = self.annotations_cache[img_name]
            boxes = cached['boxes']
            labels = cached['labels']
        else:
            # 如果没有缓存，实时加载（回退方案）
            orig_w, orig_h = image.size
            ann_name = os.path.splitext(img_name)[0] + '.txt'
            ann_path = os.path.join(self.annotations_path, ann_name)
            
            boxes = []
            labels = []
            
            if os.path.exists(ann_path):
                with open(ann_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 6:
                            bbox_left = float(parts[0])
                            bbox_top = float(parts[1])
                            bbox_width = float(parts[2])
                            bbox_height = float(parts[3])
                            category = int(parts[5])
                            
                            if category in VISDRONE_CLASS_MAP:
                                x_center = (bbox_left + bbox_width / 2) / orig_w
                                y_center = (bbox_top + bbox_height / 2) / orig_h
                                w = bbox_width / orig_w
                                h = bbox_height / orig_h
                                
                                x_center = max(0, min(1, x_center))
                                y_center = max(0, min(1, y_center))
                                w = max(0, min(1, w))
                                h = max(0, min(1, h))
                                
                                if w > 0 and h > 0:
                                    boxes.append([x_center, y_center, w, h])
                                    labels.append(VISDRONE_CLASS_MAP[category])
        
        # Letterbox 到正方形（保持宽高比，对齐 stride），与 Ultralytics 验证管线一致
        img_rgb = np.array(image.convert('RGB'))
        if len(boxes) > 0:
            tb = torch.tensor(boxes, dtype=torch.float32)
        else:
            tb = torch.zeros((0, 4), dtype=torch.float32)
        img_lb, tb = _letterbox_xywh_norm(img_rgb, tb, out_size=self.img_size)
        
        if self.transform:
            image = self.transform(Image.fromarray(img_lb))
        else:
            image = torch.from_numpy(img_lb).permute(2, 0, 1).float() / 255.0
        
        if len(boxes) > 0:
            boxes = tb
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        
        return image, {'boxes': boxes, 'labels': labels}
    
    def get_labels(self):
        """返回数据集中所有图像的标签列表(用于联邦学习数据分配)"""
        all_labels = []
        
        # 如果有缓存，直接从缓存读取（快速）
        if self.annotations_cache:
            for img_name in self.image_files:
                if img_name in self.annotations_cache:
                    labels = self.annotations_cache[img_name]['labels']
                    if len(labels) > 0:
                        all_labels.append(labels[0])
                    else:
                        all_labels.append(-1)
                else:
                    all_labels.append(-1)
        else:
            # 没有缓存，逐个加载（慢）
            for idx in range(len(self)):
                _, target = self[idx]
                if len(target['labels']) > 0:
                    all_labels.append(target['labels'][0].item())
                else:
                    all_labels.append(-1)
        
        return all_labels


def get_visdrone_dataset(root_path=None, split='train', img_size=640, use_cache=True):
    """
    获取VisDrone数据集的便捷函数
    
    Args:
        root_path: 数据集根目录（None时自动检测）
        split: 'train' 或 'val'
        img_size: 图像大小
        use_cache: 是否使用缓存（默认True，强烈推荐）
    
    Returns:
        VisDroneDataset实例
    """
    return VisDroneDataset(root_path=root_path, split=split, img_size=img_size, use_cache=use_cache)
