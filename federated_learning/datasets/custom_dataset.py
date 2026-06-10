from torch.utils import data
import torch


class CustomDataset(data.Dataset):
    def __init__(self, dataset, indices, source_class=None, target_class=None):
        self.dataset = dataset
        self.indices = indices
        self.source_class = source_class
        self.target_class = target_class
        self.contains_source_class = False

    def __getitem__(self, index):
        x, y = self.dataset[int(self.indices[index])][0], self.dataset[int(self.indices[index])][1]
        
        # 处理不同的标签格式
        if isinstance(y, dict):
            # 目标检测格式 {'boxes': ..., 'labels': ...}
            # 对于目标检测，不进行标签翻转（在这个层面）
            # 标签翻转会在PoisonedDataset中处理
            return x, y
        else:
            # 分类格式 - 进行标签翻转
            if y == self.source_class:
                y = self.target_class
            return x, y

    def __len__(self):
        return len(self.indices)