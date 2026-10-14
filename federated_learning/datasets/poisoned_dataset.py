import torch
from torch.utils import data


class PoisonedDataset(data.Dataset):
    def __init__(self, dataset, source_class=None, target_class=None, mapping=None, flip_ratio=1.0):
        self.dataset = dataset
        self.source_class = source_class
        self.target_class = target_class
        # mapping: dict[int,int]，支持多类别翻转（例如 {1:7,2:7,3:7}）
        self.mapping = mapping
        # flip_ratio: 被翻实例的比例（1.0 = 全翻，保持历史行为；<1 只翻一部分，用于调弱攻击）
        self.flip_ratio = float(flip_ratio)

    def __getitem__(self, index):
        x, y = self.dataset[index][0], self.dataset[index][1]
        
        # 处理不同的标签格式
        if isinstance(y, dict):
            # 目标检测格式 {'boxes': ..., 'labels': ...}
            # 对于目标检测的标签翻转攻击，翻转所有匹配的标签
            if 'labels' in y and (self.mapping is not None or (self.source_class is not None and self.target_class is not None)):
                # 创建新的副本，避免修改原始数据
                new_boxes = y['boxes'].clone() if torch.is_tensor(y['boxes']) else y['boxes']
                new_labels = y['labels'].clone() if torch.is_tensor(y['labels']) else y['labels']
                
                # 将标签按 mapping 翻转（优先）；否则回退到 source_class→target_class
                if torch.is_tensor(new_labels):
                    # flip_ratio<1 时只翻一部分实例；按样本 index 定种子 → 跨 epoch 稳定、可复现
                    flip_sel = None
                    if self.flip_ratio < 1.0:
                        g = torch.Generator().manual_seed(12345 + int(index))
                        flip_sel = torch.rand(new_labels.numel(), generator=g) < self.flip_ratio

                    if self.mapping is not None:
                        # 逐类替换（保持向量化）
                        for src, tgt in self.mapping.items():
                            mask = (new_labels == int(src))
                            if flip_sel is not None:
                                mask = mask & flip_sel
                            new_labels[mask] = int(tgt)
                    else:
                        mask = (new_labels == self.source_class)
                        if flip_sel is not None:
                            mask = mask & flip_sel
                        new_labels[mask] = self.target_class
                
                # 返回修改后的target（只翻转 labels，不修改 boxes）
                return x, {'boxes': new_boxes, 'labels': new_labels}
            else:
                return x, y
        else:
            # 分类格式 - 标签翻转
            if self.mapping is not None:
                # y 可能是 python int 或 0-dim tensor
                y_val = int(y.item()) if torch.is_tensor(y) else int(y)
                if y_val in self.mapping:
                    y = self.mapping[y_val]
            else:
                if y == self.source_class:
                    y = self.target_class
            return x, y

    def __len__(self):
        return len(self.dataset)
