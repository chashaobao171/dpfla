"""
YOLO 格式（images/labels）数据集加载器（用于主工程联邦学习 + YOLO 原生评估）。

假设标签文件是 Ultralytics YOLO 标准格式：
  class_id x_center y_center width height   （全部为归一化到 [0,1] 的浮点数）

返回给联邦学习框架的目标格式为：
  {'boxes': Tensor[N, 4], 'labels': Tensor[N]}
其中 boxes = [x_center, y_center, w, h]（归一化坐标）
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset

from federated_learning.datasets.visdrone_dataset import _letterbox_xywh_norm


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _list_images_recursively(root: Path) -> List[str]:
    """返回相对 root 的图片 key（用于后续映射 labels 文件）。"""
    keys: List[str] = []
    if not root.exists():
        return keys
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            keys.append(p.relative_to(root).as_posix())
    keys.sort()
    return keys


class YoloVisDroneDataset(Dataset):
    def __init__(
        self,
        root_path: str,
        split: str = "train",
        img_size: int = 640,
        use_cache: bool = True,
    ):
        """
        Args:
            root_path: YOLO 数据根目录（包含 images/ 和 labels/）
            split: 'train' | 'val' | 'test'（主工程目前主要用 train/val）
            img_size: 训练/评估统一使用的输入分辨率
            use_cache: 是否缓存解析后的标注
        """
        self.root_path = str(root_path)
        self.images_base = os.path.join(self.root_path, "images")
        self.labels_base = os.path.join(self.root_path, "labels")
        self.split = split
        self.img_size = img_size
        self.use_cache = use_cache

        if not os.path.exists(self.images_base):
            raise FileNotFoundError(f"images base not found: {self.images_base}")
        if not os.path.exists(self.labels_base):
            raise FileNotFoundError(f"labels base not found: {self.labels_base}")

        # 选择对应 split 的图片目录
        if split == "train":
            # 优先使用合并后的 images/train；否则退化到 trainA~D 合并
            if os.path.exists(os.path.join(self.images_base, "train")):
                split_dir = os.path.join(self.images_base, "train")
                keys = _list_images_recursively(Path(split_dir))
                # 统一为相对 images_base 的 key：train/xxx.jpg
                self.image_keys = [f"train/{k}" for k in keys]
            else:
                keys: List[str] = []
                for sub in ["trainA", "trainB", "trainC", "trainD"]:
                    d = os.path.join(self.images_base, sub)
                    if os.path.exists(d):
                        sub_keys = _list_images_recursively(Path(d))
                        keys.extend([f"{sub}/{k}" for k in sub_keys])
                self.image_keys = sorted(set(keys))
        elif split in ("val", "validation"):
            if os.path.exists(os.path.join(self.images_base, "validation")):
                split_dir = os.path.join(self.images_base, "validation")
                keys = _list_images_recursively(Path(split_dir))
                self.image_keys = [f"validation/{k}" for k in keys]
            else:
                split_dir = os.path.join(self.images_base, "val")
                keys = _list_images_recursively(Path(split_dir))
                self.image_keys = [f"val/{k}" for k in keys]
        elif split == "test":
            split_dir = os.path.join(self.images_base, "test")
            keys = _list_images_recursively(Path(split_dir))
            self.image_keys = [f"test/{k}" for k in keys]
        else:
            raise ValueError(f"Unsupported split for yolo dataset: {split}")

        self.annotations_cache: Dict[str, Dict[str, List]] | None = None

        if self.use_cache:
            self.annotations_cache = self._load_or_create_cache()

        logger.info(f"--> Loaded YOLO dataset {split}: {len(self.image_keys)} images")

    def _cache_path(self) -> str:
        cache_dir = os.path.join(os.getcwd(), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = f"yolo_{self.root_path}_{self.split}_{len(self.image_keys)}"
        cache_hash = str(abs(hash(cache_key)))[:10]
        return os.path.join(cache_dir, f"cache_yolo_{self.split}_{len(self.image_keys)}_{cache_hash}.pkl")

    def _label_path_from_key(self, image_key: str) -> str:
        # image_key 是相对 labels_base 的同级路径（去掉后缀替换为 .txt）
        # 例如：trainA/00000001.jpg -> trainA/00000001.txt
        img_no_ext = os.path.splitext(image_key)[0]
        return os.path.join(self.labels_base, img_no_ext + ".txt")

    def _image_path_from_key(self, image_key: str) -> str:
        return os.path.join(self.images_base, image_key)

    def _parse_label_file(self, label_path: str) -> Tuple[List[List[float]], List[int]]:
        boxes: List[List[float]] = []
        labels: List[int] = []
        if not os.path.exists(label_path):
            return boxes, labels

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

                # clamp：避免极小数值越界导致训练/评估问题
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

                if w <= 0 or h <= 0:
                    continue
                boxes.append([x, y, w, h])
                labels.append(cls)
        return boxes, labels

    def _load_or_create_cache(self) -> Dict[str, Dict[str, List]]:
        cache_path = self._cache_path()
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                logger.info(f"✓ Loaded YOLO annotation cache: {cache_path}")
                return cache_data
            except Exception as e:
                logger.warning(f"Failed to load YOLO cache ({e}), will recreate.")

        logger.info(f"Creating YOLO annotation cache: {cache_path}")
        cache_data: Dict[str, Dict[str, List]] = {}
        for idx, key in enumerate(self.image_keys):
            if idx % 2000 == 0 and idx > 0:
                logger.info(f"  cache progress: {idx}/{len(self.image_keys)}")
            label_path = self._label_path_from_key(key)
            boxes, labels = self._parse_label_file(label_path)
            cache_data[key] = {
                "boxes": boxes,
                "labels": labels,
            }

        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)
        logger.info(f"✓ Saved YOLO annotation cache: {cache_path}")
        return cache_data

    def __len__(self) -> int:
        return len(self.image_keys)

    def __getitem__(self, idx: int):
        image_key = self.image_keys[idx]
        img_path = self._image_path_from_key(image_key)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.annotations_cache is not None:
            cached = self.annotations_cache.get(image_key, {"boxes": [], "labels": []})
            boxes = cached["boxes"]
            labels = cached["labels"]
        else:
            label_path = self._label_path_from_key(image_key)
            boxes, labels = self._parse_label_file(label_path)

        img_rgb = np.array(image)
        if len(boxes) > 0:
            tb = torch.tensor(boxes, dtype=torch.float32)
        else:
            tb = torch.zeros((0, 4), dtype=torch.float32)
        img_lb, tb = _letterbox_xywh_norm(img_rgb, tb, out_size=self.img_size)
        image_t = torch.from_numpy(img_lb).permute(2, 0, 1).float() / 255.0

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)
        else:
            boxes_t = tb
            labels_t = torch.tensor(labels, dtype=torch.long)

        return image_t, {"boxes": boxes_t, "labels": labels_t}


def get_yolo_visdrone_dataset(root_path: str, split: str = "train", img_size: int = 640, use_cache: bool = True):
    return YoloVisDroneDataset(root_path=root_path, split=split, img_size=img_size, use_cache=use_cache)

