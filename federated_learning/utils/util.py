import torch
import collections
import numpy as np


def contains_class(dataset, source_class):
    """
    检查数据集是否包含指定类别
    优化版本：优先使用缓存的标签信息，避免遍历整个数据集
    """
    # 尝试使用get_labels方法（如果数据集支持）
    if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'get_labels'):
        # CustomDataset包装的情况
        try:
            all_labels = dataset.dataset.get_labels()
            if hasattr(dataset, 'indices'):
                # 只检查当前客户端的数据
                for idx in dataset.indices:
                    if idx < len(all_labels):
                        label = all_labels[idx]
                        # 对于目标检测，label可能是-1（无标注）或类别ID
                        if label == source_class:
                            return True
                return False
        except Exception:
            pass  # 回退到原始方法
    
    # 回退方案：遍历数据集（慢）
    # 但只检查前100个样本，避免太慢
    max_check = min(100, len(dataset))
    for i in range(max_check):
        try:
            x, y = dataset[i]
            # 处理不同的标签格式
            if isinstance(y, dict):
                # 目标检测格式 {'boxes': ..., 'labels': ...}
                if 'labels' in y and len(y['labels']) > 0:
                    if source_class in y['labels']:
                        return True
            elif isinstance(y, (int, torch.Tensor)):
                # 分类格式
                if isinstance(y, torch.Tensor):
                    y = y.item()
                if y == source_class:
                    return True
        except Exception:
            continue
    
    return False


def reshape_parameter_layer(parameters):
    parameters['conv1.weight'] = parameters['conv1.weight'].view([25, 10])
    parameters['conv1.bias'] = parameters['conv1.bias'].view([1, 10])
    parameters['conv2.weight'] = parameters['conv2.weight'].view([500, 10])
    parameters['conv2.bias'] = parameters['conv2.bias'].view([2, 10])
    parameters['fc1.weight'] = parameters['fc1.weight'].view([1600, 10])
    parameters['fc1.bias'] = parameters['fc1.bias'].view([5, 10])
    parameters['fc2.weight'] = parameters['fc2.weight'].view([50, 10])
    parameters['fc2.bias'] = parameters['fc2.bias'].view([1, 10])
    return parameters


def recover_parameters_shape(parameters):
    parameters['conv1.weight'] = parameters['conv1.weight'].view([10, 1, 5, 5])
    parameters['conv1.bias'] = parameters['conv1.bias'].view([10])
    parameters['conv2.weight'] = parameters['conv2.weight'].view([20, 10, 5, 5])
    parameters['conv2.bias'] = parameters['conv2.bias'].view([20])
    parameters['fc1.weight'] = parameters['fc1.weight'].view([50, 320])
    parameters['fc1.bias'] = parameters['fc1.bias'].view([50])
    parameters['fc2.weight'] = parameters['fc2.weight'].view([10, 50])
    parameters['fc2.bias'] = parameters['fc2.bias'].view([10])
    return parameters


def array_to_parameters(arr):
    parameters = collections.OrderedDict()
    parameters['conv1.weight'] = torch.tensor(arr[0:25, :])
    parameters['conv1.bias'] = torch.tensor(arr[25, :])
    parameters['conv2.weight'] = torch.tensor(arr[26:526, :])
    parameters['conv2.bias'] = torch.tensor(arr[526:528, :])
    parameters['fc1.weight'] = torch.tensor(arr[528:2128, :])
    parameters['fc1.bias'] = torch.tensor(arr[2128:2133, :])
    parameters['fc2.weight'] = torch.tensor(arr[2133:2183, :])
    parameters['fc2.bias'] = torch.tensor(arr[2183, :])
    return parameters


def reshape_parameter_and_to_array(parameter):
    param_array = []
    par = reshape_parameter_layer(parameter)
    for key in par.keys():
        data = par[key].cpu().numpy()
        param_array.append(data)

    res = param_array[0]
    for item in param_array[1:]:
        res = np.concatenate([res, item], axis=0)
    return res
