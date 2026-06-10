from federated_learning.datasets import PoisonedDataset


def label_flipping(data, source_class=None, target_class=None, mapping=None):
    """
    标签翻转攻击（支持两种模式）：
    - 单对单：source_class -> target_class
    - 多对多：mapping={src: tgt, ...}（优先使用）
    """
    poisoned_data = PoisonedDataset(data, source_class, target_class, mapping=mapping)
    return poisoned_data

