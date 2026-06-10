"""
目标检测评估指标模块
包含mAP（mean Average Precision）计算
"""

import torch
import numpy as np
from collections import defaultdict


def box_iou(boxes1, boxes2):
    """
    计算两组boxes的IoU (Intersection over Union)
    
    Args:
        boxes1: tensor [N, 4] (x_center, y_center, width, height) 归一化坐标
        boxes2: tensor [M, 4] (x_center, y_center, width, height) 归一化坐标
    
    Returns:
        iou: tensor [N, M]
    """
    # 转换为 (x1, y1, x2, y2) 格式
    def xywh2xyxy(boxes):
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    boxes1_xyxy = xywh2xyxy(boxes1)
    boxes2_xyxy = xywh2xyxy(boxes2)
    
    # 计算交集
    x1 = torch.max(boxes1_xyxy[:, None, 0], boxes2_xyxy[None, :, 0])
    y1 = torch.max(boxes1_xyxy[:, None, 1], boxes2_xyxy[None, :, 1])
    x2 = torch.min(boxes1_xyxy[:, None, 2], boxes2_xyxy[None, :, 2])
    y2 = torch.min(boxes1_xyxy[:, None, 3], boxes2_xyxy[None, :, 3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # 计算并集
    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])
    union = area1[:, None] + area2[None, :] - intersection
    
    iou = intersection / (union + 1e-6)
    return iou


def compute_ap(recall, precision):
    """
    计算AP (Average Precision)
    使用11点插值法
    
    Args:
        recall: list of recall values
        precision: list of precision values
    
    Returns:
        ap: Average Precision
    """
    # 添加哨兵值
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    
    # 计算precision包络
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    
    # 计算AP (11点插值)
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0
    
    return ap


def calculate_map_simple(predictions, targets, num_classes=10, iou_threshold=0.5):
    """
    简化的mAP计算（用于联邦学习快速评估）
    
    Args:
        predictions: list of dicts, 每个dict包含:
            - 'boxes': tensor [N, 4] 预测的boxes
            - 'scores': tensor [N] 置信度分数
            - 'labels': tensor [N] 预测的类别
        targets: list of dicts, 每个dict包含:
            - 'boxes': tensor [M, 4] 真实的boxes
            - 'labels': tensor [M] 真实的类别
        num_classes: 类别数量
        iou_threshold: IoU阈值
    
    Returns:
        mAP: mean Average Precision
        ap_per_class: 每个类别的AP
    """
    # 收集所有预测和真实标注
    all_predictions = defaultdict(list)  # {class_id: [(score, is_correct), ...]}
    all_targets = defaultdict(int)  # {class_id: count}
    
    for pred, target in zip(predictions, targets):
        # 处理真实标注
        if len(target['boxes']) > 0:
            for label in target['labels']:
                all_targets[label.item()] += 1
        
        # 处理预测
        if len(pred['boxes']) > 0 and len(target['boxes']) > 0:
            # 计算IoU矩阵
            ious = box_iou(pred['boxes'], target['boxes'])
            
            for i, (box, score, label) in enumerate(zip(pred['boxes'], pred['scores'], pred['labels'])):
                label_id = label.item()
                
                # 找到与该预测box匹配的真实box
                if label_id in target['labels']:
                    # 找到相同类别的真实boxes
                    same_class_mask = target['labels'] == label_id
                    if same_class_mask.any():
                        # 获取该预测与相同类别真实boxes的IoU
                        class_ious = ious[i][same_class_mask]
                        max_iou = class_ious.max().item()
                        
                        # 判断是否为正确预测
                        is_correct = max_iou >= iou_threshold
                        all_predictions[label_id].append((score.item(), is_correct))
                    else:
                        all_predictions[label_id].append((score.item(), False))
                else:
                    all_predictions[label_id].append((score.item(), False))
        elif len(pred['boxes']) > 0:
            # 有预测但没有真实标注，都是false positive
            for score, label in zip(pred['scores'], pred['labels']):
                all_predictions[label.item()].append((score.item(), False))
    
    # 计算每个类别的AP
    ap_per_class = {}
    for class_id in range(num_classes):
        if class_id not in all_targets or all_targets[class_id] == 0:
            # 该类别没有真实样本
            ap_per_class[class_id] = 0.0
            continue
        
        if class_id not in all_predictions or len(all_predictions[class_id]) == 0:
            # 该类别没有预测
            ap_per_class[class_id] = 0.0
            continue
        
        # 按置信度排序
        predictions_sorted = sorted(all_predictions[class_id], key=lambda x: x[0], reverse=True)
        
        # 计算precision和recall
        tp = 0
        fp = 0
        precisions = []
        recalls = []
        
        for score, is_correct in predictions_sorted:
            if is_correct:
                tp += 1
            else:
                fp += 1
            
            precision = tp / (tp + fp)
            recall = tp / all_targets[class_id]
            
            precisions.append(precision)
            recalls.append(recall)
        
        # 计算AP
        if len(precisions) > 0:
            ap = compute_ap(np.array(recalls), np.array(precisions))
            ap_per_class[class_id] = ap
        else:
            ap_per_class[class_id] = 0.0
    
    # 计算mAP
    valid_aps = [ap for ap in ap_per_class.values() if ap > 0]
    mAP = np.mean(valid_aps) if len(valid_aps) > 0 else 0.0
    
    return mAP, ap_per_class


def yolo_output_to_predictions(output, conf_threshold=0.25, num_classes=10, img_size=640):
    """
    将YOLO的原始输出转换为预测格式
    注意：YOLO输出80个COCO类别，我们映射到VisDrone的10个类别
    
    Args:
        output: YOLO模型输出，格式为 [batch, 4+80, num_anchors]
        conf_threshold: 置信度阈值
        num_classes: 目标类别数（VisDrone=10）
        img_size: 图像尺寸
    
    Returns:
        predictions: list of dicts
    """
    from loguru import logger
    
    # COCO到VisDrone的类别映射（简化版）
    # COCO: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck, ...
    # VisDrone: 0=pedestrian, 1=people, 2=bicycle, 3=car, 4=van, 5=truck, 6=tricycle, 7=awning-tricycle, 8=bus, 9=motor
    COCO_TO_VISDRONE = {
        0: 0,   # person -> pedestrian
        1: 2,   # bicycle -> bicycle  
        2: 3,   # car -> car
        3: 9,   # motorcycle -> motor
        5: 8,   # bus -> bus
        7: 5,   # truck -> truck
    }
    
    predictions = []
    
    if isinstance(output, (list, tuple)):
        if len(output) > 0:
            output = output[0]
        else:
            return [{'boxes': torch.zeros((0, 4)), 'scores': torch.zeros(0), 'labels': torch.zeros(0, dtype=torch.long)}]
    
    if not isinstance(output, torch.Tensor):
        return [{'boxes': torch.zeros((0, 4)), 'scores': torch.zeros(0), 'labels': torch.zeros(0, dtype=torch.long)}]
    
    # YOLO输出: [batch, 84, 8400] (4+80)
    # 转置为: [batch, 8400, 84]
    if output.shape[1] == 84:  # 4 + 80 COCO classes
        output = output.permute(0, 2, 1)
    
    batch_size = output.shape[0]
    device = output.device
    
    for batch_idx in range(batch_size):
        batch_output = output[batch_idx]  # [8400, 84]
        
        # 提取boxes和类别概率
        boxes = batch_output[:, :4]  # [8400, 4]
        class_probs = batch_output[:, 4:]  # [8400, 80]
        
        # 归一化坐标
        boxes = boxes / img_size
        boxes = torch.clamp(boxes, 0, 1)
        
        # 获取最大类别概率
        class_scores, coco_class_ids = torch.max(class_probs, dim=1)
        
        # 过滤：只保留能映射到VisDrone类别的预测
        valid_mask = torch.zeros(len(coco_class_ids), dtype=torch.bool, device=device)
        visdrone_labels = torch.zeros(len(coco_class_ids), dtype=torch.long, device=device)
        
        for coco_id, visdrone_id in COCO_TO_VISDRONE.items():
            mask = coco_class_ids == coco_id
            valid_mask |= mask
            visdrone_labels[mask] = visdrone_id
        
        # 同时满足：1) 能映射到VisDrone类别  2) 置信度足够高
        final_mask = valid_mask & (class_scores > conf_threshold)
        
        if final_mask.sum() == 0:
            predictions.append({
                'boxes': torch.zeros((0, 4), device=device),
                'scores': torch.zeros(0, device=device),
                'labels': torch.zeros(0, dtype=torch.long, device=device)
            })
            continue
        
        filtered_boxes = boxes[final_mask]
        filtered_scores = class_scores[final_mask]
        filtered_labels = visdrone_labels[final_mask]
        
        # 限制预测数量
        max_predictions = 100
        if len(filtered_scores) > max_predictions:
            top_k_indices = torch.topk(filtered_scores, max_predictions).indices
            filtered_boxes = filtered_boxes[top_k_indices]
            filtered_scores = filtered_scores[top_k_indices]
            filtered_labels = filtered_labels[top_k_indices]
        
        predictions.append({
            'boxes': filtered_boxes,
            'scores': filtered_scores,
            'labels': filtered_labels
        })
    
    return predictions
