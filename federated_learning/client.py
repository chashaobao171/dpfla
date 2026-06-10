import copy
import math
import os
import time

import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
import torch
import random

from federated_learning.attack_alg import label_flipping, gaussian_attack
from federated_learning.optimizer import PerturbedGradientDescent


def _visdrone_dataloader_workers() -> int:
    """VisDrone DataLoader 进程数：默认 4（吃满 CPU 预取），可用环境变量覆盖，上限 8。"""
    raw = os.environ.get("VISDRONE_DATALOADER_WORKERS", "4").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 4
    return max(0, min(n, 8))


def detection_collate_fn(batch):
    """
    自定义collate函数，用于处理目标检测数据集中变长的boxes和labels。
    
    Args:
        batch: list of (image, target) tuples
               - image: tensor of shape (C, H, W)
               - target: dict with 'boxes' and 'labels', or tensor for classification
    
    Returns:
        images: stacked tensor of shape (B, C, H, W)
        targets: list of dicts (detection) or stacked tensor (classification)
    """
    images = []
    targets = []
    
    for item in batch:
        image, target = item
        images.append(image)
        targets.append(target)
    
    # Stack images into a batch
    images = torch.stack(images, dim=0)
    
    # Check if this is detection data (dict with boxes) or classification data (tensor)
    if isinstance(targets[0], dict):
        # Detection: keep targets as a list of dicts (variable length boxes)
        return images, targets
    else:
        # Classification: stack targets into a tensor
        targets = torch.stack(targets, dim=0) if torch.is_tensor(targets[0]) else torch.tensor(targets)
        return images, targets


class Client():
    # Class variable shared among all the instances
    _performed_attacks = 0

    @property
    def performed_attacks(self):
        return type(self)._performed_attacks

    @performed_attacks.setter
    def performed_attacks(self, val):
        type(self)._performed_attacks = val

    def __init__(self, client_id, client_pseudonym, local_data, labels, criterion,
                 device, local_epochs, local_bs, local_lr,
                 local_momentum, client_type='honest'):

        self.client_id = client_id
        self.client_pseudonym = client_pseudonym
        self.local_data = local_data
        self.labels = labels
        self.criterion = criterion
        self.device = device
        self.local_epochs = local_epochs
        self.local_bs = local_bs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.client_type = client_type
        self._cached_label_hist = None

    def _build_local_label_hist(self):
        """
        统计客户端本地数据中各类别出现频次（检测任务按标注框计数）。
        缓存后复用，避免每轮重复全量扫描。
        """
        if self._cached_label_hist is not None:
            return self._cached_label_hist

        hist = {}
        for i in range(len(self.local_data)):
            _, y = self.local_data[i]
            if isinstance(y, dict) and 'labels' in y:
                labels = y['labels']
                if torch.is_tensor(labels):
                    labels = labels.detach().cpu().tolist()
                for cls_id in labels:
                    k = int(cls_id)
                    hist[k] = hist.get(k, 0) + 1
            else:
                cls_id = int(y.item()) if torch.is_tensor(y) else int(y)
                hist[cls_id] = hist.get(cls_id, 0) + 1

        self._cached_label_hist = hist
        return hist

    def _build_dynamic_label_flip_mapping(self, global_epoch, strategy_cfg):
        """
        每轮动态生成 label flipping 映射：
        - 高频优先（基于客户端本地标签频次）
        - 随机补充覆盖更多类别
        - 低频目标池按轮次轮换目标类
        """
        num_classes = int(strategy_cfg.get("num_classes", 10))
        high_freq_pool = [int(x) for x in strategy_cfg.get("high_freq_pool", [0, 1, 2, 3, 4])]
        low_freq_target_pool = [int(x) for x in strategy_cfg.get("low_freq_target_pool", [8, 9])]
        pick_from_high = int(strategy_cfg.get("pick_from_high", 4))
        pick_from_others = int(strategy_cfg.get("pick_from_others", 3))
        rotate_target = bool(strategy_cfg.get("rotate_target_each_round", True))
        flip_all_visible = bool(strategy_cfg.get("flip_all_visible_classes", False))
        expand_target_pool = bool(strategy_cfg.get("expand_target_pool_with_non_source", True))
        base_seed = int(strategy_cfg.get("seed", 20260402))

        all_classes = list(range(num_classes))
        hist = self._build_local_label_hist()
        available = [c for c in all_classes if hist.get(c, 0) > 0]
        if not available:
            available = all_classes

        rng = random.Random(base_seed + 1009 * int(global_epoch) + 9173 * int(self.client_id))

        if flip_all_visible:
            # 全可见类翻转：攻击者本地出现过的所有类别都作为源类
            picked_sources = sorted(set(available))
        else:
            # 高频优先：先取“在高频池中且本地出现过”的类别，并按本地频次排序后抽样
            high_candidates = [c for c in high_freq_pool if c in available]
            high_candidates = sorted(high_candidates, key=lambda c: hist.get(c, 0), reverse=True)
            k1 = min(pick_from_high, len(high_candidates))
            picked_high = rng.sample(high_candidates, k1) if k1 > 0 else []

            # 随机补充：从其余可用类别中补抽
            other_candidates = [c for c in available if c not in picked_high]
            k2 = min(pick_from_others, len(other_candidates))
            picked_other = rng.sample(other_candidates, k2) if k2 > 0 else []

            picked_sources = sorted(set(picked_high + picked_other))
            if not picked_sources:
                fallback = sorted(available, key=lambda c: hist.get(c, 0), reverse=True)
                picked_sources = fallback[:min(3, len(fallback))]

        valid_target_pool = [c for c in low_freq_target_pool if c in all_classes]
        if expand_target_pool:
            non_source_pool = [c for c in all_classes if c not in picked_sources]
            # 目标池 = 低频池优先 + 非源类池扩展（去重，保持顺序）
            merged = valid_target_pool + non_source_pool
            valid_target_pool = list(dict.fromkeys(merged))
        if not valid_target_pool:
            valid_target_pool = all_classes

        if rotate_target:
            target = valid_target_pool[int(global_epoch) % len(valid_target_pool)]
        else:
            target = valid_target_pool[0]

        picked_sources = [c for c in picked_sources if c != target]
        if not picked_sources:
            fallback = [c for c in available if c != target]
            if not fallback:
                fallback = [c for c in all_classes if c != target]
            picked_sources = rng.sample(fallback, min(3, len(fallback)))

        mapping = {src: target for src in picked_sources}
        return mapping

    # ======================================= Start of training function ===========================================================#
    def participant_update(self, global_epoch, model, attack_type='no_attack', malicious_behavior_rate=0,
                           source_class=None, target_class=None, dataset_name=None, untarget=False,
                           label_flip_mapping=None):

        # 保存全局模型快照，用于梯度放大计算
        global_model_snapshot = copy.deepcopy(model)

        # 初始化backdoor_pattern（根据数据集类型）
        backdoor_pattern = None
        x_offset, y_offset = 0, 0
        
        if dataset_name == 'MNIST':
            backdoor_pattern = torch.tensor([[2.8238, 2.8238, 2.8238],
                                             [2.8238, 2.8238, 2.8238],
                                             [2.8238, 2.8238, 2.8238]])
            x_offset, y_offset = backdoor_pattern.shape[0], backdoor_pattern.shape[1]
        elif dataset_name == 'CIFAR10':
            backdoor_pattern = torch.tensor([[[2.5141, 2.5141, 2.5141],
                                              [2.5141, 2.5141, 2.5141],
                                              [2.5141, 2.5141, 2.5141]],

                                             [[2.5968, 2.5968, 2.5968],
                                              [2.5968, 2.5968, 2.5968],
                                              [2.5968, 2.5968, 2.5968]],

                                             [[2.7537, 2.7537, 2.7537],
                                              [2.7537, 2.7537, 2.7537],
                                              [2.7537, 2.7537, 2.7537]]])
            x_offset, y_offset = backdoor_pattern.shape[1], backdoor_pattern.shape[2]
        elif dataset_name == 'VisDrone':
            # VisDrone数据集的后门模式（5x5像素块，RGB格式）
            backdoor_pattern = torch.tensor([[[2.5141, 2.5141, 2.5141, 2.5141, 2.5141],
                                              [2.5141, 2.5141, 2.5141, 2.5141, 2.5141],
                                              [2.5141, 2.5141, 2.5141, 2.5141, 2.5141],
                                              [2.5141, 2.5141, 2.5141, 2.5141, 2.5141],
                                              [2.5141, 2.5141, 2.5141, 2.5141, 2.5141]],

                                             [[2.5968, 2.5968, 2.5968, 2.5968, 2.5968],
                                              [2.5968, 2.5968, 2.5968, 2.5968, 2.5968],
                                              [2.5968, 2.5968, 2.5968, 2.5968, 2.5968],
                                              [2.5968, 2.5968, 2.5968, 2.5968, 2.5968],
                                              [2.5968, 2.5968, 2.5968, 2.5968, 2.5968]],

                                             [[2.7537, 2.7537, 2.7537, 2.7537, 2.7537],
                                              [2.7537, 2.7537, 2.7537, 2.7537, 2.7537],
                                              [2.7537, 2.7537, 2.7537, 2.7537, 2.7537],
                                              [2.7537, 2.7537, 2.7537, 2.7537, 2.7537],
                                              [2.7537, 2.7537, 2.7537, 2.7537, 2.7537]]])
            x_offset, y_offset = backdoor_pattern.shape[1], backdoor_pattern.shape[2]


        if untarget:
            timestamp = int(time.time())
            target_class = timestamp % 10
            if target_class == source_class and target_class != 9:
                target_class += 1
            if target_class == source_class and target_class == 9:
                target_class -= 1

        epochs = self.local_epochs
        
        # 根据数据集类型选择collate_fn
        if dataset_name == 'VisDrone':
            # 训练时让 CPU 并行预取数据，提升 GPU 利用率（不改变数据语义）
            pin_memory = ('cuda' in str(self.device).lower())
            num_workers = _visdrone_dataloader_workers()
            train_loader = DataLoader(
                self.local_data,
                self.local_bs,
                shuffle=True,
                drop_last=False,
                collate_fn=detection_collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=(num_workers > 0),
                prefetch_factor=2 if num_workers > 0 else None,
            )
        else:
            train_loader = DataLoader(
                self.local_data,
                self.local_bs,
                shuffle=True,
                drop_last=True,
                num_workers=2,
                pin_memory=('cuda' in str(self.device).lower()),
                persistent_workers=True,
                prefetch_factor=2,
            )
        
        attacked = 0
        
        # 调试日志：显示客户端类型和攻击配置
        logger.info(f'🔍 Client {self.client_pseudonym}: type={self.client_type}, attack_type={attack_type}, malicious_behavior_rate={malicious_behavior_rate}')
        
        # Get the poisoned training data of the client in case of label-flipping or backdoor attacks
        if (attack_type == 'label_flipping') and (self.client_type == 'attacker'):
            logger.info(f'  → Client {self.client_pseudonym} is attacker, checking malicious_behavior_rate...')
            r = np.random.random()
            logger.info(f'  → Random value: {r:.4f}, threshold: {malicious_behavior_rate}')
            if r <= malicious_behavior_rate:
                if dataset_name != 'IMDB':
                    if label_flip_mapping is not None:
                        # 仅做 labels 翻转（不修改 bbox）
                        if isinstance(label_flip_mapping, dict) and label_flip_mapping.get("mode") == "dynamic_round_highfreq":
                            mapping = self._build_dynamic_label_flip_mapping(global_epoch, label_flip_mapping)
                            logger.info(
                                f"  → Dynamic label flipping mapping (round={global_epoch}, size={len(mapping)}): {mapping}"
                            )
                        elif isinstance(label_flip_mapping, dict) and "mapping" in label_flip_mapping:
                            # 兼容 server.py 传入的结构：{"mapping": {...}}
                            mapping = label_flip_mapping.get("mapping")
                        else:
                            mapping = label_flip_mapping

                        logger.info(f'  → Applying label flipping mapping (size={len(mapping)}): {mapping}')
                        poisoned_data = label_flipping(self.local_data, mapping=mapping)
                    else:
                        logger.info(f'  → Applying label flipping: {source_class} → {target_class}')
                        poisoned_data = label_flipping(self.local_data, source_class, target_class)
                    
                    # 验证标签是否被修改（仅打印摘要，避免大段 tensor 刷屏）
                    sample_count = min(5, len(poisoned_data))
                    if dataset_name == 'VisDrone':
                        visdrone_label_hist = {}
                        visdrone_box_count = 0
                        for i in range(sample_count):
                            _, label = poisoned_data[i]
                            if isinstance(label, dict) and 'labels' in label:
                                lbl = label['labels']
                                visdrone_box_count += int(len(lbl))
                                for c in lbl.tolist():
                                    ci = int(c)
                                    visdrone_label_hist[ci] = visdrone_label_hist.get(ci, 0) + 1
                        logger.info(
                            f'  → Verify label flip summary (first {sample_count} samples): '
                            f'boxes={visdrone_box_count}, class_hist={dict(sorted(visdrone_label_hist.items()))}'
                        )
                    else:
                        cls_hist = {}
                        for i in range(sample_count):
                            _, label = poisoned_data[i]
                            if torch.is_tensor(label):
                                li = int(label.item())
                            else:
                                li = int(label)
                            cls_hist[li] = cls_hist.get(li, 0) + 1
                        logger.info(
                            f'  → Verify label flip summary (first {sample_count} samples): '
                            f'class_hist={dict(sorted(cls_hist.items()))}'
                        )
                    
                    # 对于VisDrone，需要使用detection_collate_fn
                    if dataset_name == 'VisDrone':
                        pin_memory = ('cuda' in str(self.device).lower())
                        num_workers = _visdrone_dataloader_workers()
                        train_loader = DataLoader(
                            poisoned_data,
                            self.local_bs,
                            shuffle=True,
                            drop_last=False,
                            collate_fn=detection_collate_fn,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            persistent_workers=(num_workers > 0),
                            prefetch_factor=2 if num_workers > 0 else None,
                        )
                    else:
                        train_loader = DataLoader(poisoned_data, self.local_bs, shuffle=True, drop_last=False)
                self.performed_attacks += 1
                attacked = 1
                if label_flip_mapping is not None:
                    logger.info(f'✅ Label flipping attack launched by {self.client_pseudonym} with mapping(size={len(mapping)})')
                else:
                    logger.info(f'✅ Label flipping attack launched by {self.client_pseudonym} to flip class {source_class} → {target_class}')
            else:
                logger.info(f'  → Random value {r:.4f} > threshold {malicious_behavior_rate}, attack skipped this round')
        elif attack_type == 'label_flipping' and self.client_type != 'attacker':
            logger.debug(f'  → Client {self.client_pseudonym} is not attacker, skipping label flip')
        elif attack_type != 'label_flipping':
            logger.debug(f'  → Attack type is {attack_type}, not label_flipping')
        lr = self.local_lr
        # VisDrone 走手写反传，不是 ultralytics 的 model.train()，因此 lr0/warmup/cos_lr 等训练器参数不生效。
        # 可选：按全局轮次余弦衰减（由运行脚本设置 FL_VISDRONE_LR_SCHEDULE=cosine 与 FL_GLOBAL_ROUNDS 等）。
        if dataset_name == 'VisDrone':
            sched = os.environ.get('FL_VISDRONE_LR_SCHEDULE', 'constant').strip().lower()
            if sched == 'cosine':
                total = os.environ.get('FL_GLOBAL_ROUNDS')
                lr_min_s = os.environ.get('FL_LR_MIN')
                if total is None or lr_min_s is None:
                    logger.warning(
                        'FL_VISDRONE_LR_SCHEDULE=cosine 但未设置 FL_GLOBAL_ROUNDS 或 FL_LR_MIN，使用恒定 local_lr'
                    )
                else:
                    try:
                        total_r = max(1, int(total))
                        lr_min = float(lr_min_s)
                        eta_max = float(self.local_lr)
                        if lr_min > eta_max:
                            lr_min, eta_max = eta_max, lr_min
                        T = max(1, total_r - 1)
                        t = float(global_epoch)
                        lr = lr_min + 0.5 * (eta_max - lr_min) * (1.0 + math.cos(math.pi * t / T))
                        logger.info(
                            f'  → VisDrone LR (cosine w.r.t. global round): {lr:.2e} '
                            f'(round {int(global_epoch) + 1}/{total_r}, η_max={eta_max:.2e}, η_min={lr_min:.2e})'
                        )
                    except (TypeError, ValueError) as e:
                        logger.warning(f'VisDrone cosine LR 解析失败 ({e})，回退恒定 local_lr')

        server_model = copy.deepcopy(model)

        if dataset_name == 'IMDB':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif dataset_name == 'VisDrone':
            # YOLO 手写反传：使用标准 SGD+momentum+weight_decay（不再被 FedProx PGD 覆盖）
            optimizer = optim.SGD(
                model.parameters(), lr=lr, momentum=self.local_momentum, weight_decay=5e-4
            )
        else:
            optimizer = PerturbedGradientDescent(model.parameters(), lr=lr, mu=0.001)

        model.train()
        epoch_loss = []
        client_grad = {}  # 使用字典而不是列表
        t = 0
        
        # 计算总batch数用于显示进度
        total_batches = len(train_loader)
        logger.info(f'🔄 {self.client_pseudonym} 开始训练: {total_batches} batches, {epochs} epoch(s)')

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                # 每50个batch或最后一个batch显示进度
                if batch_idx % 50 == 0 or batch_idx == total_batches - 1:
                    progress = (batch_idx + 1) / total_batches * 100
                    logger.info(f'   {self.client_pseudonym} - Epoch {epoch+1}/{epochs} - {progress:.0f}% ({batch_idx+1}/{total_batches})')
                
                # 处理不同数据格式
                if dataset_name == 'VisDrone':
                    # 目标检测: target是dict列表，需要单独处理
                    data = data.to(self.device, non_blocking=True)
                    # 将每个target dict中的tensor移到device
                    target = [{k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v 
                              for k, v in t.items()} for t in target]
                else:
                    # 分类任务: target是tensor
                    data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

                if dataset_name == 'IMDB':
                    target = target.view(-1, 1) * (1 - attacked)

                if (attack_type == 'backdoor') and (self.client_type == 'attacker') and (
                        np.random.random() <= malicious_behavior_rate):
                    if backdoor_pattern is None:
                        logger.warning(f'Backdoor pattern not defined for dataset {dataset_name}, skipping backdoor attack')
                    elif dataset_name == 'VisDrone':
                        # 目标检测的后门攻击需要特殊处理
                        logger.warning('Backdoor attack for VisDrone detection not fully implemented yet')
                    else:
                        pdata = data.clone()
                        ptarget = target.clone()
                        keep_idxs = (target == source_class)
                        pdata = pdata[keep_idxs]
                        ptarget = ptarget[keep_idxs]
                        if len(pdata) > 0:
                            pdata[:, :, -x_offset:, -y_offset:] = backdoor_pattern
                            ptarget[:] = target_class
                            data = torch.cat([data, pdata], dim=0)
                            target = torch.cat([target, ptarget], dim=0)

                # 对于VisDrone，传递targets给模型以使用YOLO内置训练
                if dataset_name == 'VisDrone':
                    # YOLO模型在训练模式下会直接返回损失
                    loss, features = model(data, return_features=True, targets=target)
                    # 不需要再调用criterion，loss已经是YOLO原生损失
                else:
                    output, features = model(data, return_features=True)
                    # 计算损失
                    loss = self.criterion(output, target)

                loss.backward()
                epoch_loss.append(loss.item())
                # get gradients - 使用字典而不是列表，避免索引问题
                cur_time = time.time()
                if epoch == 0 and batch_idx == 0:
                    # 第一次：初始化梯度字典
                    client_grad = {}
                    for name, params in model.named_parameters():
                        if params.requires_grad and params.grad is not None:
                            client_grad[name] = params.grad.clone()
                else:
                    # 后续：累加梯度
                    for name, params in model.named_parameters():
                        if params.requires_grad and params.grad is not None:
                            if name in client_grad:
                                client_grad[name] += params.grad.clone()
                            else:
                                client_grad[name] = params.grad.clone()
                t += time.time() - cur_time
                if isinstance(optimizer, PerturbedGradientDescent):
                    optimizer.step(server_model.parameters(), self.device)
                else:
                    optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()

            # epoch结束时显示平均损失
            avg_loss = np.mean(epoch_loss) if epoch_loss else 0
            logger.info(f'   ✓ {self.client_pseudonym} - Epoch {epoch+1}/{epochs} 完成 - 平均损失: {avg_loss:.4f}')

        # 训练完成
        logger.info(f'✅ {self.client_pseudonym} 训练完成！总损失: {np.mean(epoch_loss):.4f}')

        # 强化版标签翻转攻击：梯度大幅放大后上传
        # 原理：在返回 state_dict 前，将恶意更新的幅度乘以大系数
        # 效果：等效提升恶意端在 FedAvg 中的影响力，打破 7:3 的压制
        if (attack_type == 'label_flipping'
                and self.client_type == 'attacker'
                and attacked == 1):
            GRADIENT_SCALE = 15.0  # 放大倍数，需足够大使恶意信号超过诚实端（30%*15 > 70%）
            global_weights = global_model_snapshot.state_dict()
            with torch.no_grad():
                for key in model.state_dict():
                    if model.state_dict()[key].is_floating_point():
                        delta = model.state_dict()[key].float() - global_weights[key].float()
                        model.state_dict()[key].copy_(
                            global_weights[key].float() + delta * GRADIENT_SCALE
                        )
            logger.warning(
                f'⚠️  {self.client_pseudonym} 强化攻击：梯度放大 ×{GRADIENT_SCALE} 已执行'
            )

        if (attack_type == 'gaussian' and self.client_type == 'attacker'):
            update, flag = gaussian_attack(model.state_dict(), self.client_pseudonym,
                                           malicious_behavior_rate=malicious_behavior_rate, device=self.device)
            if flag == 1:
                self.performed_attacks += 1
                attacked = 1
            model.load_state_dict(update)

        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        # print("Number of Attacks:{}".format(self.performed_attacks))
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        model = model.cpu()
        return model.state_dict(), client_grad, model, np.mean(epoch_loss), attacked, t
