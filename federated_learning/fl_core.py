import time
import copy
import numpy as np
import random
import torch
import os
from loguru import logger
from contextlib import contextmanager
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix
import gc
from tqdm import tqdm

from federated_learning.client import Client, detection_collate_fn
from federated_learning.datasets import CustomDataset
from federated_learning.fl_algorithm import FoolsGold, average_weights, simple_median, trimmed_mean, Krum, \
    FedSVD, DPFLA
from federated_learning.models import setup_model
from federated_learning.utils import distribute_dataset, contains_class
from federated_learning.utils.metrics import calculate_map_simple, yolo_output_to_predictions
import subprocess
import tempfile
import re


class FL:
    def __init__(self, dataset_name, model_name, dd_type, num_clients, frac_clients,
                 seed, test_batch_size, criterion, global_rounds, local_epochs, local_bs, local_lr,
                 local_momentum, labels_dict, device, attackers_ratio=0,
                 class_per_client=2, samples_per_class=250, rate_unbalance=1, alpha=1, source_class=None,
                 visdrone_root_path=None):

        # 自动检测数据集路径
        if visdrone_root_path is None:
            # 优先使用 autodl-tmp 数据目录
            if os.path.exists('/root/autodl-tmp/data/images') and os.path.exists('/root/autodl-tmp/data/labels'):
                visdrone_root_path = '/root/autodl-tmp/data'
            elif os.path.exists('/root/autodl-tmp/data/visdrone'):
                visdrone_root_path = '/root/autodl-tmp/data/visdrone'
            elif os.path.exists('/home/featurize/data/images') and os.path.exists('/home/featurize/data/labels'):
                visdrone_root_path = '/home/featurize/data'
            elif os.path.exists('/home/featurize/data/visdrone'):
                visdrone_root_path = '/home/featurize/data/visdrone'
            else:
                visdrone_root_path = 'D:/Pycharmworkplace/visdrone'
        
        FL._history = np.zeros(num_clients)
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_clients = num_clients
        self.clients_pseudonyms = ['Client ' + str(i + 1) for i in range(self.num_clients)]
        self.frac_clients = frac_clients
        self.seed = seed
        self.test_batch_size = test_batch_size
        self.criterion = criterion
        self.global_rounds = global_rounds
        self.local_epochs = local_epochs
        self.local_bs = local_bs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.labels_dict = labels_dict
        self.num_classes = len(self.labels_dict)
        self.device = device
        self.attackers_ratio = attackers_ratio
        self.class_per_client = class_per_client
        self.samples_per_class = samples_per_class
        self.rate_unbalance = rate_unbalance
        self.source_class = source_class
        self.dd_type = dd_type
        self.alpha = alpha
        self.embedding_dim = 100
        self.clients = []
        self.trainset, self.testset = None, None

        self.score_history = np.zeros([self.num_clients], dtype=float)

        # Fix the random state of the environment
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)

        # Loading of data
        self.trainset, self.testset, user_groups_train, tokenizer = distribute_dataset(self.dataset_name,
                                                                                       self.num_clients,
                                                                                       self.num_classes,
                                                                                       self.dd_type,
                                                                                       self.class_per_client,
                                                                                       self.samples_per_class,
                                                                                       self.alpha,
                                                                                       visdrone_root_path=visdrone_root_path)

        # 根据数据集类型选择collate_fn
        if self.dataset_name == 'VisDrone':
            self.test_loader = DataLoader(self.testset, batch_size=self.test_batch_size,
                                          shuffle=False, num_workers=0, collate_fn=detection_collate_fn)
        else:
            self.test_loader = DataLoader(self.testset, batch_size=self.test_batch_size,
                                          shuffle=False, num_workers=0)

        # Creating model
        self.global_model = setup_model(model_architecture=self.model_name, num_classes=self.num_classes,
                                        tokenizer=tokenizer, embedding_dim=self.embedding_dim)
        self.global_model = self.global_model.to(self.device)

        # Dividing the training set among clients
        self.local_data = []
        self.have_source_class = []
        self.labels = []
        logger.info('--> Distributing training data among clients')
        for p in user_groups_train:
            self.labels.append(user_groups_train[p]['labels'])
            indices = user_groups_train[p]['data']
            client_data = CustomDataset(self.trainset, indices=indices)
            self.local_data.append(client_data)
            if self.source_class in user_groups_train[p]['labels']:
                self.have_source_class.append(p)
        logger.info('--> Training data have been distributed among clients')

        # Creating clients instances
        logger.info('--> Creating peets instances')
        m_ = 0
        self.num_attackers = 0  # 初始化为0，避免无攻击场景报错
        if self.attackers_ratio > 0:
            # pick m random participants from the workers list
            # k_src = len(self.have_source_class)
            # print('# of clients who have source class examples:', k_src)
            m_ = int(self.attackers_ratio * self.num_clients)
            self.num_attackers = copy.deepcopy(m_)

        clients = list(np.arange(self.num_clients))
        random.shuffle(clients)
        
        logger.info(f'--> Creating {len(clients)} client instances...')
        for idx, i in enumerate(clients):
            logger.info(f'  Creating client {idx+1}/{len(clients)} (ID: {i})')
            
            # 简化恶意客户端分配逻辑：直接按比例分配，不检查是否包含source_class
            # 原因：contains_class只检查前100个样本，可能导致误判
            if m_ > 0:
                self.clients.append(Client(i, self.clients_pseudonyms[i],
                                           self.local_data[i], self.labels[i],
                                           self.criterion, self.device, self.local_epochs, self.local_bs, self.local_lr,
                                           self.local_momentum, client_type='attacker'))
                m_ -= 1
                logger.info(f'  ✓ Client {idx+1} created as ATTACKER')
            else:
                self.clients.append(Client(i, self.clients_pseudonyms[i],
                                           self.local_data[i], self.labels[i],
                                           self.criterion, self.device, self.local_epochs, self.local_bs, self.local_lr,
                                           self.local_momentum))
                logger.info(f'  ✓ Client {idx+1} created as HONEST')
        
        logger.info(f'--> All {len(clients)} clients created successfully!')
        logger.info(f'  Total ATTACKERS: {self.num_attackers}')
        logger.info(f'  Total HONEST: {len(clients) - self.num_attackers}')
        del self.local_data

    # ======================================= Start of testning function ===========================================================#
    def test(self, model, device, test_loader, dataset_name=None):
        model.eval()
        test_loss = []
        correct = 0
        n = 0
        
        # 清理显存
        torch.cuda.empty_cache()
        
        # 对于VisDrone数据集，优先使用ultralytics官方val流程，但放到子进程中执行：
        # - ultralytics 的 BaseValidator/__call__ 使用 inference_mode（装饰器），在同进程内可能污染训练状态
        # - 子进程评估可以完全隔离副作用，同时得到与 `yolo val ...` 一致的mAP
        if dataset_name == 'VisDrone' and hasattr(model, 'use_yolo') and model.use_yolo:
            try:
                logger.info('→ 使用YOLO原生评估方法（子进程 yolo val，隔离inference_mode）')
                
                # 动态生成visdrone.yaml配置文件
                from generate_visdrone_yaml import generate_visdrone_yaml
                yaml_path = generate_visdrone_yaml('visdrone_temp.yaml')

                # 导出当前权重为临时.pt（ultralytics格式），用 yolo CLI 评估
                if not hasattr(model, 'yolo'):
                    raise RuntimeError('YOLOWrapper缺少yolo属性，无法导出权重进行评估')

                original_device = next(model.parameters()).device
                with tempfile.TemporaryDirectory() as td:
                    weights_path = os.path.join(td, "tmp_eval.pt")
                    model.yolo.save(weights_path)

                    # 说明：直接解析 `yolo val` CLI 文本在 subprocess(capture_output=True) 下可能拿不到输出（stdout/stderr 为空）。
                    # 为了让指标口径稳定，改用 Ultralytics Python API 并输出 JSON，父进程从 JSON 中读取 mAP@0.5。
                    import json

                    # 必须生成合法 Python 字面量：CPU 为 device='cpu'，不能写成 device=cpu
                    device_py = "0" if original_device.type == "cuda" else "'cpu'"
                    py_code = (
                        "import json; "
                        "from ultralytics import YOLO; "
                        f"m=YOLO({weights_path!r}); "
                        # 显式关闭 Ultralytics 的可视化/保存开关，避免在每次评估时产生日志图片（runs/detect/*）。
                        f"metrics=m.val(data={yaml_path!r}, imgsz=640, conf=0.001, iou=0.5, augment=False, plots=False, save_json=False, save_txt=False, device={device_py}, verbose=False); "
                        "res={"
                        "  'precision': float(metrics.box.mp),"
                        "  'recall': float(metrics.box.mr),"
                        "  'mAP50': float(metrics.box.map50),"
                        "  'mAP50_95': float(metrics.box.map)"
                        "}; "
                        "print(json.dumps(res))"
                    )

                    cmd = ["python3", "-c", py_code]
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
                    if proc.returncode != 0:
                        raise RuntimeError(f"yolo python val failed (code={proc.returncode}). Output:\n{out[-3000:]}")

                    out_clean = (proc.stdout or "").strip()
                    last_line = out_clean.splitlines()[-1] if out_clean else ""
                    try:
                        res = json.loads(last_line)
                    except Exception as je:
                        raise RuntimeError(
                            "无法解析子进程输出 JSON（last_line 可能不是 JSON）。"
                            f"\nlast_line:\n{last_line}\nFull tail:\n{out[-3000:]}"
                        ) from je

                    precision = float(res["precision"])
                    recall = float(res["recall"])
                    mAP50 = float(res["mAP50"])
                    mAP50_95 = float(res["mAP50_95"])

                logger.info(f'📊 YOLO原生mAP@0.5: {mAP50*100:.2f}%')
                logger.info(f'   mAP@0.5:0.95: {mAP50_95*100:.2f}%')
                logger.info(f'   Precision: {precision*100:.2f}%')
                logger.info(f'   Recall: {recall*100:.2f}%')

                # 评估后确保模型训练状态
                model.to(original_device)
                model.train()
                for param in model.parameters():
                    param.requires_grad = True
                if hasattr(model, 'model') and hasattr(model.model, 'criterion') and model.model.criterion is not None:
                    if hasattr(model, '_move_criterion_to_device'):
                        model._move_criterion_to_device(model.model.criterion, original_device)

                # 计算平均测试损失（快速估计）
                test_loss = []
                for batch_idx, (data, target) in enumerate(test_loader):
                    data = data.to(self.device)
                    target = [{k: v.to(self.device) if torch.is_tensor(v) else v for k, v in t.items()} for t in target]
                    with torch.no_grad():
                        loss, _ = model(data, return_features=True, targets=target)
                        test_loss.append(loss.item())
                    del data, loss
                    torch.cuda.empty_cache()
                test_loss = np.mean(test_loss) if test_loss else 0.0

                return mAP50 * 100.0, test_loss
            except Exception as e:
                # 主实验指标固定为 YOLO 原生 val：禁用 fallback，避免指标口径混用
                logger.error(f'❌ YOLO原生评估出错（已禁用fallback，不进入自定义mAP）：{e}')
                import traceback
                logger.debug(f'Traceback: {traceback.format_exc()}')
                # 确保模型回到原设备并恢复训练状态
                if 'original_device' in locals():
                    model.to(original_device)
                    model.train()
                    for param in model.parameters():
                        param.requires_grad = True
                raise
        
        # 自定义评估（fallback方案，用于非YOLO模型或YOLO评估失败时）
        all_predictions = []
        all_targets = []
        for batch_idx, (data, target) in enumerate(test_loader):
            # 处理不同数据格式
            if dataset_name == 'VisDrone':
                # 目标检测: target是dict列表
                data = data.to(self.device)
                target = [{k: v.to(self.device) if torch.is_tensor(v) else v 
                          for k, v in t.items()} for t in target]
                
                with torch.no_grad():  # 测试时不需要梯度，节省显存
                    output = model(data)
                    
                    # 计算损失
                    loss = self.criterion(output, target)
                    
                    # 使用ultralytics的non_max_suppression进行后处理
                    try:
                        from ultralytics.utils.ops import non_max_suppression
                    except (ImportError, AttributeError):
                        try:
                            from ultralytics.yolo.utils.ops import non_max_suppression
                        except (ImportError, AttributeError):
                            # 如果找不到NMS，使用自定义解析
                            from federated_learning.utils.metrics import yolo_output_to_predictions
                            predictions = yolo_output_to_predictions(output, num_classes=self.num_classes, img_size=640)
                            all_predictions.extend(predictions)
                            all_targets.extend(target)
                            test_loss.append(loss.item())
                            n += len(target)
                            del data, output, loss
                            torch.cuda.empty_cache()
                            continue
                    
                    # NMS参数
                    conf_thres = 0.001  # 降低阈值以获取更多预测
                    iou_thres = 0.6
                    max_det = 300
                    
                    # 对YOLO输出应用NMS（会自动应用sigmoid和解码）
                    predictions_nms = non_max_suppression(
                        output[0] if isinstance(output, (list, tuple)) else output,
                        conf_thres=conf_thres,
                        iou_thres=iou_thres,
                        max_det=max_det
                    )
                    
                    # 转换为我们需要的格式
                    predictions = []
                    for pred in predictions_nms:
                        if pred is not None and len(pred) > 0:
                            # pred格式: [num_boxes, 6] - [x1, y1, x2, y2, conf, cls]
                            boxes_xyxy = pred[:, :4]  # 像素坐标
                            
                            # 转换为归一化的中心点格式 [x_center, y_center, w, h]
                            x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
                            x_center = (x1 + x2) / 2.0 / 640.0  # 归一化
                            y_center = (y1 + y2) / 2.0 / 640.0
                            w = (x2 - x1) / 640.0
                            h = (y2 - y1) / 640.0
                            
                            boxes_xywh = torch.stack([x_center, y_center, w, h], dim=1)
                            scores = pred[:, 4]
                            labels = pred[:, 5].long()
                            
                            # COCO到VisDrone类别映射
                            COCO_TO_VISDRONE = {
                                0: 0,   # person -> pedestrian
                                1: 2,   # bicycle -> bicycle  
                                2: 3,   # car -> car
                                3: 9,   # motorcycle -> motor
                                5: 8,   # bus -> bus
                                7: 5,   # truck -> truck
                            }
                            
                            # 过滤并映射类别
                            valid_mask = torch.zeros(len(labels), dtype=torch.bool, device=labels.device)
                            visdrone_labels = torch.zeros_like(labels)
                            
                            for coco_id, visdrone_id in COCO_TO_VISDRONE.items():
                                mask = labels == coco_id
                                valid_mask |= mask
                                visdrone_labels[mask] = visdrone_id
                            
                            if valid_mask.sum() > 0:
                                predictions.append({
                                    'boxes': boxes_xywh[valid_mask],
                                    'scores': scores[valid_mask],
                                    'labels': visdrone_labels[valid_mask]
                                })
                            else:
                                predictions.append({
                                    'boxes': torch.zeros((0, 4)),
                                    'scores': torch.zeros(0),
                                    'labels': torch.zeros(0, dtype=torch.long)
                                })
                        else:
                            predictions.append({
                                'boxes': torch.zeros((0, 4)),
                                'scores': torch.zeros(0),
                                'labels': torch.zeros(0, dtype=torch.long)
                            })
                    
                    all_predictions.extend(predictions)
                    all_targets.extend(target)
                
                test_loss.append(loss.item())
                n += len(target)
                
                # 每个batch后清理显存
                del data, output, loss
                torch.cuda.empty_cache()
                
            elif dataset_name == 'IMDB':
                data, target = data.to(self.device), target.to(self.device)
                with torch.no_grad():
                    output = model(data)
                    test_loss.append(self.criterion(output, target.view(-1, 1)).item())
                    pred = output > 0.5
                    correct += pred.eq(target.view_as(pred)).sum().item()
                n += target.shape[0]
            else:
                # 分类任务
                data, target = data.to(self.device), target.to(self.device)
                with torch.no_grad():
                    output = model(data)
                    test_loss.append(self.criterion(output, target).item())
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                n += target.shape[0]
        
        test_loss = np.mean(test_loss)
        
        if dataset_name == 'VisDrone':
            # 计算mAP（自定义方法）
            try:
                from federated_learning.utils.metrics import calculate_map_simple
                
                # 检查是否有有效的预测
                total_predictions = sum(len(p['boxes']) for p in all_predictions)
                total_targets = sum(len(t['boxes']) for t in all_targets)
                
                logger.debug(f'🔍 自定义mAP计算统计:')
                logger.debug(f'   总预测框数: {total_predictions}')
                logger.debug(f'   总真实框数: {total_targets}')
                
                # 调试：检查坐标范围
                if total_predictions > 0 and len(all_predictions) > 0:
                    sample_pred = all_predictions[0]
                    if len(sample_pred['boxes']) > 0:
                        pred_boxes = sample_pred['boxes']
                        logger.debug(f'   预测框样本 (前3个):')
                        for i in range(min(3, len(pred_boxes))):
                            box = pred_boxes[i]
                            logger.debug(f'     框{i}: x={box[0]:.3f}, y={box[1]:.3f}, w={box[2]:.3f}, h={box[3]:.3f}, score={sample_pred["scores"][i]:.3f}, label={sample_pred["labels"][i]}')
                
                if total_targets > 0 and len(all_targets) > 0:
                    sample_target = all_targets[0]
                    if len(sample_target['boxes']) > 0:
                        target_boxes = sample_target['boxes']
                        logger.debug(f'   真实框样本 (前3个):')
                        for i in range(min(3, len(target_boxes))):
                            box = target_boxes[i]
                            logger.debug(f'     框{i}: x={box[0]:.3f}, y={box[1]:.3f}, w={box[2]:.3f}, h={box[3]:.3f}, label={sample_target["labels"][i]}')
                
                if total_predictions == 0:
                    logger.warning('⚠️  没有任何预测框！')
                    mAP = 0.0
                elif total_targets == 0:
                    logger.warning('⚠️  没有任何真实标注框！')
                    mAP = 0.0
                else:
                    mAP, ap_per_class = calculate_map_simple(all_predictions, all_targets, 
                                                             num_classes=self.num_classes, 
                                                             iou_threshold=0.5)
                    logger.info(f'📊 自定义mAP@0.5: {mAP*100:.2f}%')
                    
                    # 显示每个类别的AP（只显示AP>0的类别）
                    class_names = list(self.labels_dict.keys())
                    high_ap_classes = [(class_names[cid], ap) for cid, ap in ap_per_class.items() if ap > 0.1]
                    if high_ap_classes:
                        logger.debug(f'   Top classes: {", ".join([f"{name}={ap*100:.1f}%" for name, ap in high_ap_classes[:5]])}')
            except Exception as e:
                logger.error(f'❌ mAP计算失败: {e}')
                import traceback
                logger.error(traceback.format_exc())
                mAP = 0.0
            
            # 目标检测：返回mAP作为准确率
            logger.debug('Average test loss: {:.4f}, Tested on {} images, mAP: {:.4f}'.format(test_loss, n, mAP))
            return mAP * 100.0, test_loss  # 返回mAP作为百分比
        else:
            logger.debug('Average test loss: {:.4f}, Test accuracy: {}/{} ({:.2f}%)'.format(test_loss, correct, n,
                                                                                            100 * correct / n))
            return 100.0 * (float(correct) / n), test_loss
            logger.debug('Average test loss: {:.4f}, Test accuracy: {}/{} ({:.2f}%)'.format(test_loss, correct, n,
                                                                                            100 * correct / n))
            return 100.0 * (float(correct) / n), test_loss

    # ======================================= End of testning function =============================================================#
    # Test label prediction function
    def test_label_predictions(self, model, device, test_loader, dataset_name=None):
        model.eval()
        actuals = []
        predictions = []
        
        if dataset_name == 'VisDrone':
            # 目标检测任务：返回空列表（不计算混淆矩阵）
            logger.debug('Skipping label predictions for detection task')
            return [], []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                if dataset_name == 'IMDB':
                    prediction = output > 0.5
                else:
                    prediction = output.argmax(dim=1, keepdim=True)

                actuals.extend(target.view_as(prediction))
                predictions.extend(prediction)
        return [i.item() for i in actuals], [i.item() for i in predictions]

    # choose random set of clients
    def choose_clients(self):
        # pick m random clients from the available list of clients
        m = max(int(self.frac_clients * self.num_clients), 1)
        selected_clients = np.random.choice(range(self.num_clients), m, replace=False)

        # print('\nSelected Clients\n')
        # for i, p in enumerate(selected_clients):
        #     print(i+1, ': ', self.clients[p].client_pseudonym, ' is ', self.clients[p].client_type)
        return selected_clients

    def test_backdoor(self, model, device, test_loader, backdoor_pattern, source_class, target_class):
        model.eval()
        correct = 0
        n = 0
        x_offset, y_offset = backdoor_pattern.shape[0], backdoor_pattern.shape[1]
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            keep_idxs = (target == source_class)
            bk_data = copy.deepcopy(data[keep_idxs])
            bk_target = copy.deepcopy(target[keep_idxs])
            bk_data[:, :, -x_offset:, -y_offset:] = backdoor_pattern
            bk_target[:] = target_class
            output = model(bk_data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(bk_target.view_as(pred)).sum().item()
            n += bk_target.shape[0]
        return np.round(100.0 * (float(correct) / n), 2)

    def run_experiment(self, attack_type='no_attack', malicious_behavior_rate=0,
                       source_class=None, target_class=None, rule='fedavg', resume=False, model_name=None,
                       untarget=False, label_flip_mapping=None):
        simulation_model = copy.deepcopy(self.global_model)

        logger.info("===>Simulation started...")
        
        # 测试初始模型（预训练权重的能力）
        logger.info("====>Testing initial pretrained model...")
        initial_accuracy, initial_loss = self.test(
            simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name
        )
        if self.dataset_name == 'VisDrone':
            logger.info(f"✓ 初始模型 - 📊 mAP@0.5: {initial_accuracy:.2f}% | 损失: {initial_loss:.4f}")
            if initial_accuracy == 0.0:
                coco_ref = getattr(simulation_model, "coco_pretrained_baseline", None)
                if coco_ref:
                    logger.info(
                        "ℹ️  初始 mAP@0.5≈0% 属正常冷启动现象：当前为 VisDrone 10 类新检测头。"
                        f" 上方已给出换头前 COCO 参考（{coco_ref.get('dataset', 'coco128')}）mAP。"
                    )
                else:
                    logger.info(
                        "ℹ️  初始 mAP@0.5≈0% 在换头后的首评估通常是正常的。"
                        " 若后续 2-3 轮仍接近 0%，再排查 labels/yaml 配置。"
                    )
                    logger.info(
                        "   快速自检：val images 同级需有 labels/*.txt；"
                        "可运行: python convert_visdrone_to_yolo.py --root /path/to/visdrone --mode visdrone10 "
                        "并在各 split 下 ln -sfn labels_yolo_visdrone10 labels"
                    )
                    logger.info(
                        "   需要显示换头前 COCO 参考 mAP：设置 FL_RUN_COCO_PRETRAINED_VAL=1（会触发 coco128 下载/缓存）。"
                    )
        else:
            logger.info(f"✓ 初始模型 - 准确率: {initial_accuracy:.2f}% | 损失: {initial_loss:.4f}")
            if initial_accuracy == 0.0:
                logger.warning("⚠️  初始模型准确率为0%，这不正常！")
                logger.warning("   可能原因：1) 模型初始化/加载异常  2) 标签映射异常  3) 数据读取异常")
        
        fg = FoolsGold(self.num_clients)
        fed_svd = FedSVD()
        dpfla = DPFLA()
        # copy weights
        global_weights = simulation_model.state_dict()
        last10_updates = []
        test_losses = []
        global_accuracies = []
        source_class_accuracies = []
        cpu_runtimes = []
        noise_scalar = 1.0
        # best_accuracy = 0.0
        mapping = {'honest': 'Good update', 'attacker': 'Bad update'}

        # TensorBoard 可视化：写入两个目录
        # 1. runs/ - 本地工程目录（已有）
        # 2. /root/tf-logs/ - AutoDL 平台监控目录
        log_dir_local = os.path.join(
            "runs",
            f"{self.dataset_name}_{self.model_name}_{rule}_attack-{attack_type}_mr-{self.attackers_ratio}"
        )
        log_dir_autodl = os.path.join(
            "/root/tf-logs",
            f"{self.dataset_name}_{self.model_name}_{rule}_attack-{attack_type}_mr-{self.attackers_ratio}"
        )
        os.makedirs(log_dir_autodl, exist_ok=True)
        writer_local = SummaryWriter(log_dir=log_dir_local)
        writer_autodl = SummaryWriter(log_dir=log_dir_autodl)

        def tb_write(scalar_dict, step):
            for tag, value in scalar_dict.items():
                writer_local.add_scalar(tag, value, step)
                writer_autodl.add_scalar(tag, value, step)

        # start training
        start_round = 0
        if resume:
            logger.info('Loading last saved checkpoint..')
            checkpoint = torch.load(
                './checkpoints/' + self.dataset_name + '_' + self.model_name + '_' + self.dd_type + '_' + rule + '_' + str(
                    self.attackers_ratio) + '_' + str(self.local_epochs) + '.t7')
            simulation_model.load_state_dict(checkpoint['state_dict'])
            start_round = checkpoint['epoch'] + 1
            last10_updates = checkpoint['last10_updates']
            test_losses = checkpoint['test_losses']
            global_accuracies = checkpoint['global_accuracies']
            source_class_accuracies = checkpoint['source_class_accuracies']

            logger.info('>>checkpoint loaded!')
        logger.info("====>Global model training started...")
        for epoch in tqdm(range(start_round, self.global_rounds)):
            gc.collect()
            torch.cuda.empty_cache()

            # if epoch % 20 == 0:
            #     clear_output()
            logger.debug(f'| Global training round : {epoch + 1}/{self.global_rounds} |')
            selected_clients = self.choose_clients()
            logger.info(f'选中 {len(selected_clients)} 个客户端进行训练')
            
            local_weights, local_grads, local_models, local_losses, performed_attacks = [], [], [], [], []
            clients_types = []
            i = 1
            attacks = 0
            Client._performed_attacks = 0
            
            for client in selected_clients:
                clients_types.append(mapping[self.clients[client].client_type])
                logger.info(f'>>> 客户端 {i}/{len(selected_clients)}: {self.clients_pseudonyms[client]} 开始训练...')
                
                client_update, client_grad, client_local_model, client_loss, attacked, t = self.clients[
                    client].participant_update(
                    epoch,
                    copy.deepcopy(simulation_model),
                    untarget=untarget,
                    attack_type=attack_type, malicious_behavior_rate=malicious_behavior_rate,
                    source_class=source_class, target_class=target_class, dataset_name=self.dataset_name,
                    label_flip_mapping=label_flip_mapping)
                
                logger.info(f'<<< 客户端 {i}/{len(selected_clients)}: {self.clients_pseudonyms[client]} 训练完成 (损失: {client_loss:.4f})')
                
                local_weights.append(client_update)
                local_grads.append(client_grad)
                local_losses.append(client_loss)
                local_models.append(client_local_model)
                attacks += attacked
                # print('{} ends training in global round:{} |\n'.format((self.clients_pseudonyms[client]), (epoch + 1)))
                i += 1
            
            # 所有客户端训练完成
            logger.info(f'✓ 所有客户端训练完成！平均损失: {np.mean(local_losses):.4f}')
            logger.info(f'开始聚合参数...')
            
            # loss_avg = sum(local_losses) / len(local_losses)
            # print('Average of clients\' local losses: {:.6f}'.format(loss_avg))
            # aggregated global weights
            scores = np.zeros(len(local_weights))
            # Expected malicious clients
            f = int(self.num_clients * self.attackers_ratio)
            if rule == 'median':
                cur_time = time.time()
                global_weights = simple_median(local_weights)
                cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'tmean':
                cur_time = time.time()
                # trim_ratio = self.attackers_ratio * self.num_clients / len(selected_clients)
                trim_ratio = self.attackers_ratio * self.num_clients / len(selected_clients)
                global_weights = trimmed_mean(local_weights, trim_ratio=trim_ratio)
                cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'mkrum':
                cur_time = time.time()
                good_updates = Krum(local_models, f=f, multi=True)
                scores[good_updates] = 1
                global_weights = average_weights(local_weights, scores)
                cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'foolsgold':
                cur_time = time.time()
                scores = fg.score_gradients(local_grads, selected_clients)

                logger.debug("Defense result:")
                for i, pt in enumerate(clients_types):
                    logger.info(str(pt) + ' scored ' + str(scores[i]))

                global_weights = average_weights(local_weights, scores)
                cpu_runtimes.append(time.time() - cur_time + t)

            elif rule == 'fedavg':
                cur_time = time.time()
                # YOLO 检测头与 BN 状态对数值敏感：float16 聚合会累积误差、拖 mAP；统一 float32。
                global_weights = average_weights(
                    local_weights,
                    [1 for i in range(len(local_weights))],
                    float16_floats=False,
                )
                cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'fed_svd':
                print("--------------------------")
                cur_time = time.time()
                new_global_weights = fed_svd.aggregation(copy.deepcopy(global_weights),
                                                         copy.deepcopy(local_weights),
                                                         [])
                global_weights = new_global_weights
                cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'DPFLA':
                cur_time = time.time()
                if model_name == "CNNMNIST":
                    m = 50
                    n = 10
                elif model_name == "CNNCifar10":
                    m = 128
                    n = 10
                elif model_name == "YOLO":
                    m = 128
                    n = 10
                else:
                    raise Exception('Undefined model name!!!')

                # DPFLA 主路径（冻结）：使用原始 SVD + K-Means 异常检测
                # - 通过 DPFLA.score(..., use_validation=False) 触发 SVD+k-means 逻辑
                # - use_validation=True 的 loss 打分版只在需要时手动开启，不再作为主实验结论
                scores = dpfla.score(copy.deepcopy(simulation_model),
                                     copy.deepcopy(local_models),
                                     clients_types=clients_types,
                                     selected_clients=selected_clients, p=m, w=n,
                                     model_name=model_name,
                                     val_loader=None,
                                     device=None,
                                     use_validation=False)
                global_weights = average_weights(local_weights, scores)
                t = time.time() - cur_time
                logger.debug('Aggregation took', np.round(t, 4))
                cpu_runtimes.append(t)

            else:
                global_weights = average_weights(local_weights, [1 for i in range(len(local_weights))])
                ##############################################################################################

            g_model = copy.deepcopy(simulation_model)
            simulation_model.load_state_dict(global_weights)
            if epoch >= self.global_rounds - 10:
                last10_updates.append(global_weights)

            logger.info(f'开始测试全局模型（Round {epoch + 1}/{self.global_rounds}）...')
            current_accuracy, test_loss = self.test(simulation_model, self.device, self.test_loader,
                                                    dataset_name=self.dataset_name)
            if self.dataset_name == 'VisDrone':
                logger.info(f'✓ Round {epoch + 1}/{self.global_rounds} - 📊 mAP@0.5: {current_accuracy:.2f}% | 损失: {test_loss:.4f}')
            else:
                logger.info(f'✓ Round {epoch + 1}/{self.global_rounds} - 准确率: {current_accuracy:.2f}%, 损失: {test_loss:.4f}')

            # 写入 TensorBoard：
            # - 统一用 "accuracy/global" 作为主曲线（分类任务=accuracy，VisDrone=YOLO mAP@0.5）
            # - 对 VisDrone 额外写一条 "mAP50/global" 方便区分
            tb_write({
                "accuracy/global": current_accuracy,
                "loss/test": test_loss,
            }, epoch + 1)
            if self.dataset_name == "VisDrone":
                tb_write({"mAP50/global": current_accuracy}, epoch + 1)

            if np.isnan(test_loss):
                simulation_model = copy.deepcopy(g_model)
                noise_scalar = noise_scalar * 0.5

            global_accuracies.append(np.round(current_accuracy, 2))
            test_losses.append(np.round(test_loss, 4))
            performed_attacks.append(attacks)

            backdoor_asr = 0.0
            backdoor_pattern = None
            if attack_type == 'backdoor':
                if self.dataset_name == 'MNIST':
                    backdoor_pattern = torch.tensor([[2.8238, 2.8238, 2.8238],
                                                     [2.8238, 2.8238, 2.8238],
                                                     [2.8238, 2.8238, 2.8238]])
                elif self.dataset_name == 'CIFAR10':
                    backdoor_pattern = torch.tensor([[[2.5141, 2.5141, 2.5141],
                                                      [2.5141, 2.5141, 2.5141],
                                                      [2.5141, 2.5141, 2.5141]],

                                                     [[2.5968, 2.5968, 2.5968],
                                                      [2.5968, 2.5968, 2.5968],
                                                      [2.5968, 2.5968, 2.5968]],

                                                     [[2.7537, 2.7537, 2.7537],
                                                      [2.7537, 2.7537, 2.7537],
                                                      [2.7537, 2.7537, 2.7537]]])
                elif self.dataset_name == 'VisDrone':
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

                if backdoor_pattern is not None:
                    backdoor_asr = self.test_backdoor(simulation_model, self.device, self.test_loader,
                                                      backdoor_pattern, source_class, target_class)
            logger.info('Backdoor ASR {}'.format(backdoor_asr))

            state = {
                'epoch': epoch,
                'state_dict': simulation_model.state_dict(),
                'global_model': g_model,
                'local_models': copy.deepcopy(local_models),
                'last10_updates': last10_updates,
                'test_losses': test_losses,
                'global_accuracies': global_accuracies,
                'source_class_accuracies': source_class_accuracies
            }
            # savepath = './checkpoints/' + self.dataset_name + '_' + self.model_name + '_' + self.dd_type + '_' + rule + '_' + str(
            #     self.attackers_ratio) + '_' + str(self.local_epochs) + '.t7'
            # torch.save(state, savepath)
            del local_models
            del local_weights
            del local_grads
            gc.collect()
            torch.cuda.empty_cache()
            # print("***********************************************************************************")
            # print and show confusion matrix after each global round
            actuals, predictions = self.test_label_predictions(simulation_model, self.device, self.test_loader,
                                                               dataset_name=self.dataset_name)
            classes = list(self.labels_dict.keys())

            # 只对分类任务计算混淆矩阵
            if self.dataset_name != 'VisDrone' and len(actuals) > 0:
                logger.debug('{0:10s} - {1}'.format('Class', 'Accuracy'))
                attacked_class_acc = None
                for i, r in enumerate(confusion_matrix(actuals, predictions)):
                    acc_i = r[i] / np.sum(r) * 100
                    logger.info('{0:10s} - {1:.1f}'.format(classes[i], acc_i))
                    if i == source_class:
                        attacked_class_acc = np.round(acc_i, 2)
                        source_class_accuracies.append(attacked_class_acc)
                # 将被攻击类的准确率写入 TensorBoard（如果有定义 source_class）
                if attacked_class_acc is not None:
                    tb_write({"accuracy/attacked_class": attacked_class_acc}, epoch + 1)
            else:
                # 目标检测任务：添加占位值
                source_class_accuracies.append(0.0)

            if epoch == self.global_rounds - 1:
                logger.info('Last 10 updates results')
                global_weights = average_weights(last10_updates,
                                                 np.ones([len(last10_updates)]))
                simulation_model.load_state_dict(global_weights)
                current_accuracy, test_loss = self.test(simulation_model, self.device, self.test_loader,
                                                        dataset_name=self.dataset_name)
                global_accuracies.append(np.round(current_accuracy, 2))
                test_losses.append(np.round(test_loss, 4))
                performed_attacks.append(attacks)
                logger.info("***********************************************************************************")
                # print and show confusion matrix after each global round
                actuals, predictions = self.test_label_predictions(simulation_model, self.device, self.test_loader,
                                                                   dataset_name=self.dataset_name)
                classes = list(self.labels_dict.keys())
                asr = 0.0
                
                # 只对分类任务计算混淆矩阵
                if self.dataset_name != 'VisDrone' and len(actuals) > 0:
                    logger.info('{0:10s} - {1}'.format('Class', 'Accuracy'))
                    for i, r in enumerate(confusion_matrix(actuals, predictions)):
                        logger.info('{0:10s} - {1:.1f}'.format(classes[i], r[i] / np.sum(r) * 100))
                        if i == source_class:
                            source_class_accuracies.append(np.round(r[i] / np.sum(r) * 100, 2))
                            asr = np.round(r[target_class] / np.sum(r) * 100, 2)
                else:
                    # 目标检测任务：添加占位值
                    source_class_accuracies.append(0.0)

                backdoor_asr = 0.0
                if attack_type == 'backdoor':
                    if self.dataset_name == 'MNIST':
                        backdoor_pattern = torch.tensor([[2.8238, 2.8238, 2.8238],
                                                         [2.8238, 2.8238, 2.8238],
                                                         [2.8238, 2.8238, 2.8238]])
                    elif self.dataset_name == 'CIFAR10':
                        backdoor_pattern = torch.tensor([[[2.5141, 2.5141, 2.5141],
                                                          [2.5141, 2.5141, 2.5141],
                                                          [2.5141, 2.5141, 2.5141]],

                                                         [[2.5968, 2.5968, 2.5968],
                                                          [2.5968, 2.5968, 2.5968],
                                                          [2.5968, 2.5968, 2.5968]],

                                                         [[2.7537, 2.7537, 2.7537],
                                                          [2.7537, 2.7537, 2.7537],
                                                          [2.7537, 2.7537, 2.7537]]])

                    backdoor_asr = self.test_backdoor(simulation_model, self.device, self.test_loader,
                                                      backdoor_pattern, source_class, target_class)

        state = {
            'state_dict': simulation_model.state_dict(),
            'test_losses': test_losses,
            'global_accuracies': global_accuracies,
            'source_class_accuracies': source_class_accuracies,
            'asr': asr,
            'backdoor_asr': backdoor_asr,
            'avg_cpu_runtime': np.mean(cpu_runtimes)
        }
        # savepath = './results/' + self.dataset_name + '_' + self.model_name + '_' + self.dd_type + '_' + rule + '_' + str(
        #     self.attackers_ratio) + '_' + str(self.local_epochs) + '.t7'
        # torch.save(state, savepath)

        writer_local.close()
        writer_autodl.close()

        logger.debug('Global accuracies: {}'.format(global_accuracies))
        logger.debug('Class {} accuracies: {}'.format(source_class, source_class_accuracies))
        logger.debug("Test loss: {}".format(test_losses))
        logger.debug("Label-flipping Attack success rate: {}".format(asr))
        logger.debug('Backdoor attack succes rate: {}'.format(backdoor_asr))
        logger.debug("Average CPU aggregation runtime: {}".format(np.mean(cpu_runtimes)))

    def update_score_history(self, scores, selected_peers, epoch):
        print('-> Update score history')
        self.score_history[selected_peers] += scores
        q1 = np.quantile(self.score_history, 0.25)
        trust = self.score_history - q1
        trust = trust / trust.max()
        trust[(trust < 0)] = 0
        return trust[selected_peers]
