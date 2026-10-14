from federated_learning.datasets import PoisonedDataset


def label_flipping(data, source_class=None, target_class=None, mapping=None, flip_ratio=1.0):
    """
    标签翻转攻击（支持两种模式）：
    - 单对单：source_class -> target_class
    - 多对多：mapping={src: tgt, ...}（优先使用）
    flip_ratio: 被翻实例的比例（1.0 = 全翻，保持历史行为；<1 只翻一部分，用于调弱攻击）
    """
    poisoned_data = PoisonedDataset(data, source_class, target_class, mapping=mapping, flip_ratio=flip_ratio)
    return poisoned_data

