import random
from random import shuffle
from loguru import logger
import torch
import os  # 添加os导入
import pickle  # 添加pickle导入

random.seed(7)
import numpy as np
from torchvision import datasets, transforms
import codecs
# import tensorflow as tf
import pandas as pd
from federated_learning.datasets.visdrone_dataset import get_visdrone_dataset
from federated_learning.datasets.yolo_visdrone_dataset import get_yolo_visdrone_dataset


def distribute_dataset(dataset_name, num_peers, num_classes, dd_type='IID', classes_per_peer=1, samples_per_class=582,
                       alpha=1, visdrone_root_path=None):
    logger.info("--> Loading of {} dataset".format(dataset_name))
    
    # 自动检测数据集路径（主项目口径：VisDrone 优先 autodl-tmp 目录）
    if visdrone_root_path is None:
        if os.path.exists('/root/autodl-tmp/data/visdrone'):
            visdrone_root_path = '/root/autodl-tmp/data/visdrone'
        elif os.path.exists('/home/featurize/data/visdrone'):
            visdrone_root_path = '/home/featurize/data/visdrone'
        else:
            visdrone_root_path = 'D:/Pycharmworkplace/visdrone'
    
    tokenizer = None
    if dataset_name == 'MNIST':
        trainset, testset = get_mnist()
    elif dataset_name == 'CIFAR10':
        trainset, testset = get_cifar10()
    elif dataset_name == 'IMDB':
        trainset, testset, tokenizer = get_imdb(num_peers=num_peers)
    elif dataset_name == 'VisDrone':
        trainset, testset = get_visdrone(visdrone_root_path)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    if dd_type == 'IID':
        peers_data_dict = sample_dirichlet(trainset, num_peers, alpha=1000000)
    elif dd_type == 'NON_IID':
        peers_data_dict = sample_dirichlet(trainset, num_peers, alpha=alpha)
    elif dd_type == 'EXTREME_NON_IID':
        peers_data_dict = sample_extreme(trainset, num_peers, num_classes, classes_per_peer, samples_per_class)

    logger.info("--> Dataset has been loaded!")
    return trainset, testset, peers_data_dict, tokenizer


# Get the original MNIST data set
def get_mnist():
    # torchvision.MNIST(root=...) 的实际目录是 root/MNIST/raw|processed
    # 因此若已有 /home/featurize/data/MNIST/raw，应把 root 设为 /home/featurize/data
    if os.path.exists('/root/autodl-tmp/data/MNIST/raw'):
        mnist_root = '/root/autodl-tmp/data'
    elif os.path.exists('/root/autodl-tmp/data/MNIST/MNIST/raw'):
        # 兼容历史层级（已多包一层 MNIST）
        mnist_root = '/root/autodl-tmp/data/MNIST'
    elif os.path.exists('/home/featurize/data/MNIST/raw'):
        mnist_root = '/home/featurize/data'
    elif os.path.exists('/home/featurize/data/MNIST/MNIST/raw'):
        # 兼容历史层级（已多包一层 MNIST）
        mnist_root = '/home/featurize/data/MNIST'
    elif os.path.exists('D:/Pycharmworkplace/DPFLA-master/data/MNIST/raw'):
        mnist_root = 'D:/Pycharmworkplace/DPFLA-master/data'
    elif os.path.exists('D:/Pycharmworkplace/DPFLA-master/data/MNIST/MNIST/raw'):
        mnist_root = 'D:/Pycharmworkplace/DPFLA-master/data/MNIST'
    else:
        mnist_root = './data/MNIST'

    logger.info(f'--> MNIST data root: {mnist_root}')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 若本地已有 MNIST 的 processed 文件（training.pt/test.pt），则禁止下载；
    # 否则允许 torchvision 根据 raw/*.gz 生成 processed，必要时自动补下载。
    processed_dir = os.path.join(mnist_root, 'MNIST', 'processed')
    mnist_processed_files = ['training.pt', 'test.pt']
    has_local_mnist = all(os.path.exists(os.path.join(processed_dir, f)) for f in mnist_processed_files)
    download_mnist = not has_local_mnist
    logger.info(f'--> MNIST local cache found: {has_local_mnist}, download={download_mnist}')

    trainset = datasets.MNIST(mnist_root, train=True, download=download_mnist,
                              transform=transform)
    testset = datasets.MNIST(mnist_root, train=False, download=download_mnist,
                             transform=transform)
    return trainset, testset


# Get the original CIFAR10 data set
def get_cifar10():
    # torchvision.CIFAR10 要求 root 目录下存在 cifar-10-batches-py 子目录
    candidate_roots = [
        '/root/autodl-tmp/data/cifar-10-batches-py',    # 直接指定 batches-py 目录（优先）
        '/root/chashaobao/data',                        # chashaobao 数据根（解压后）
        '/root/autodl-tmp/data',                        # autodl 原始路径
        '/root/autodl-tmp/data/cifar',                  # autodl-tmp cifar 子目录
        '/home/featurize/data',                          # 兼容 featurize 历史路径
        '/home/featurize/data/CIFAR10',                 # 兼容 featurize 历史路径
        'D:/Pycharmworkplace/DPFLA-master/data/cifar',
        'data/cifar/'
    ]

    required_files = [
        'batches.meta',
        'data_batch_1',
        'data_batch_2',
        'data_batch_3',
        'data_batch_4',
        'data_batch_5',
        'test_batch'
    ]

    data_dir = candidate_roots[-1]
    has_local_cifar = False
    for root in candidate_roots:
        cifar_batches_dir = os.path.join(root, 'cifar-10-batches-py')
        if not os.path.isdir(cifar_batches_dir):
            continue
        if all(os.path.exists(os.path.join(cifar_batches_dir, f)) for f in required_files):
            data_dir = root
            has_local_cifar = True
            break

    download_cifar = not has_local_cifar
    logger.info(f'--> CIFAR10 data root: {data_dir}, local cache found: {has_local_cifar}, download={download_cifar}')

    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = datasets.CIFAR10(data_dir, train=True, download=download_cifar,
                                transform=apply_transform)

    testset = datasets.CIFAR10(data_dir, train=False, download=download_cifar,
                               transform=apply_transform)
    return trainset, testset


# Get the IMDB data set
def get_imdb(num_peers=10):
    MAX_LEN = 128
    # Read data
    df = pd.read_csv('data/imdb.csv')
    # Convert sentiment columns to numerical values
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == 'positive' else 0)
    # Tokenization
    # use tf.keras for tokenization,  
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.review.values.tolist())

    train_df = df.iloc[:40000].reset_index(drop=True)
    valid_df = df.iloc[40000:].reset_index(drop=True)

    # STEP 3: pad sequence
    xtrain = tokenizer.texts_to_sequences(train_df.review.values)
    xtest = tokenizer.texts_to_sequences(valid_df.review.values)

    # zero padding
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=MAX_LEN)
    xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, maxlen=MAX_LEN)

    # STEP 4: initialize dataset class for training
    trainset = IMDBDataset(reviews=xtrain, targets=train_df.sentiment.values)

    # initialize dataset class for validation
    testset = IMDBDataset(reviews=xtest, targets=valid_df.sentiment.values)

    return trainset, testset, tokenizer


# Get the VisDrone dataset
def get_visdrone(root_path=None):
    """加载VisDrone目标检测数据集"""
    # 自动检测数据集路径
    if root_path is None:
        if os.path.exists('/root/autodl-tmp/data/images') and os.path.exists('/root/autodl-tmp/data/labels'):
            root_path = '/root/autodl-tmp/data'
        elif os.path.exists('/root/autodl-tmp/data/visdrone'):
            root_path = '/root/autodl-tmp/data/visdrone'
        elif os.path.exists('/home/featurize/data/images') and os.path.exists('/home/featurize/data/labels'):
            root_path = '/home/featurize/data'
        elif os.path.exists('/home/featurize/data/visdrone'):
            root_path = '/home/featurize/data/visdrone'
        else:
            root_path = 'D:/Pycharmworkplace/visdrone'

    # 如果根目录是 YOLO 格式（images/labels），就直接使用 YOLO loader
    if os.path.exists(os.path.join(root_path, 'images')) and os.path.exists(os.path.join(root_path, 'labels')):
        trainset = get_yolo_visdrone_dataset(root_path=root_path, split='train', img_size=640)
        testset = get_yolo_visdrone_dataset(root_path=root_path, split='val', img_size=640)
    else:
        # 否则回退到原始 VisDrone 目录结构（VisDrone2019-DET-*/）
        trainset = get_visdrone_dataset(root_path=root_path, split='train', img_size=640)
        testset = get_visdrone_dataset(root_path=root_path, split='val', img_size=640)
    return trainset, testset


def sample_dirichlet(dataset, num_users, alpha=1):
    classes = {}
    total_samples = len(dataset)
    
    # 优化：对于VisDrone数据集，使用缓存的标签
    import sys
    import time
    import pickle
    import hashlib
    
    # 生成数据集的唯一标识
    dataset_id = f"{dataset.__class__.__name__}_{len(dataset)}"
    cache_filename = f"cache_{dataset_id}_labels.pkl"
    
    # 缓存文件放在cache目录
    cache_dir = os.path.join(os.getcwd(), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, cache_filename)
    
    # 尝试从缓存加载
    if os.path.exists(cache_file):
        logger.info(f"从缓存加载标签: {cache_filename}")
        with open(cache_file, 'rb') as f:
            classes = pickle.load(f)
    else:
        # 添加进度显示（对于大数据集）
        if total_samples > 1000:
            # 使用简单的百分比输出，兼容性更好
            last_progress = -1
            start_time = time.time()
            update_interval = max(50, total_samples // 200)  # 大约更新200次
            
            iterator = enumerate(dataset)
            for idx, x in iterator:
                _, label = x
                # 处理目标检测数据集格式 (image, {'boxes': boxes, 'labels': labels})
                if isinstance(label, dict) and 'labels' in label:
                    # 对于目标检测，使用图像中出现的主要类别（第一个标签）
                    if len(label['labels']) > 0:
                        label = label['labels'][0].item() if isinstance(label['labels'][0], torch.Tensor) else label['labels'][0]
                    else:
                        # 如果没有标注，跳过
                        continue
                elif type(label) == torch.Tensor:
                    label = label.item()
                
                if label in classes:
                    classes[label].append(idx)
                else:
                    classes[label] = [idx]
                
                # 定期输出进度（避免刷屏）
                if idx % update_interval == 0 or idx == total_samples - 1:
                    progress = int((idx + 1) / total_samples * 100)
                    if progress != last_progress:
                        # 使用\r在同一行刷新
                        sys.stdout.write(f'\r处理数据集: {progress:3d}% | {idx+1}/{total_samples}')
                        sys.stdout.flush()
                        last_progress = progress
            
            # 完成后换行
            sys.stdout.write('\n')
            sys.stdout.flush()
        else:
            # 小数据集，不使用进度条
            iterator = enumerate(dataset)
            for idx, x in iterator:
                _, label = x
                # 处理目标检测数据集格式 (image, {'boxes': boxes, 'labels': labels})
                if isinstance(label, dict) and 'labels' in label:
                    # 对于目标检测，使用图像中出现的主要类别（第一个标签）
                    if len(label['labels']) > 0:
                        label = label['labels'][0].item() if isinstance(label['labels'][0], torch.Tensor) else label['labels'][0]
                    else:
                        # 如果没有标注，跳过
                        continue
                elif type(label) == torch.Tensor:
                    label = label.item()
                
                if label in classes:
                    classes[label].append(idx)
                else:
                    classes[label] = [idx]
        
        # 保存到缓存
        with open(cache_file, 'wb') as f:
            pickle.dump(classes, f)
        logger.info(f"标签已缓存到: {cache_filename}")
    
    # 计算类别数
    num_classes = len(classes.keys())

    peers_data_dict = {i: {'data': np.array([]), 'labels': []} for i in range(num_users)}

    for n in range(num_classes):
        random.shuffle(classes[n])
        class_size = len(classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(np.array(num_users * [alpha]))
        for user in range(num_users):
            num_imgs = int(round(sampled_probabilities[user]))
            sampled_list = classes[n][:min(len(classes[n]), num_imgs)]
            peers_data_dict[user]['data'] = np.concatenate((peers_data_dict[user]['data'], np.array(sampled_list)),
                                                           axis=0)
            if num_imgs > 0:
                peers_data_dict[user]['labels'].append((n, num_imgs))

            classes[n] = classes[n][min(len(classes[n]), num_imgs):]

    return peers_data_dict


def _extract_labels_from_dataset(dataset):
    """从数据集中提取标签，支持分类和目标检测数据集"""
    labels = []
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        # 处理目标检测数据集格式
        if isinstance(label, dict) and 'labels' in label:
            if len(label['labels']) > 0:
                label_val = label['labels'][0].item() if isinstance(label['labels'][0], torch.Tensor) else label['labels'][0]
            else:
                label_val = -1  # 无标注
        elif isinstance(label, torch.Tensor):
            label_val = label.item()
        else:
            label_val = label
        labels.append(label_val)
    return np.array(labels)


def sample_extreme(dataset, num_users, num_classes, classes_per_peer, samples_per_class):
    n = len(dataset)
    num_classes = 10
    peers_data_dict = {i: {'data': np.array([]), 'labels': []} for i in range(num_users)}
    idxs = np.arange(n)
    
    # 支持目标检测数据集（没有targets属性）
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = _extract_labels_from_dataset(dataset)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    label_indices = {l: [] for l in range(num_classes)}
    for l in label_indices:
        label_idxs = np.where(labels == l)
        label_indices[l] = list(idxs[label_idxs])

    labels = [i for i in range(num_classes)]

    for i in range(num_users):
        user_labels = np.random.choice(labels, classes_per_peer, replace=False)
        for l in user_labels:
            peers_data_dict[i]['labels'].append(l)
            lab_idxs = label_indices[l][:samples_per_class]
            label_indices[l] = list(set(label_indices[l]) - set(lab_idxs))
            if len(label_indices[l]) < samples_per_class:
                labels = list(set(labels) - set([l]))
            peers_data_dict[i]['data'] = np.concatenate(
                (peers_data_dict[i]['data'], lab_idxs), axis=0)

    return peers_data_dict
