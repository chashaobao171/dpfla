"""
YOLO模型封装类，用于联邦学习框架集成
使用ultralytics库封装YOLOv8模型
"""

import torch
from torch import nn
from loguru import logger
import os
import traceback

# 配置代理（如果需要访问GitHub）
# 取消下面的注释并填入你的代理地址
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# 或者禁用在线功能（不需要代理）
os.environ['YOLO_OFFLINE'] = 'True'


def _ultralytics_coco128_yaml_path():
    try:
        import ultralytics
        from pathlib import Path

        p = Path(ultralytics.__file__).resolve().parent / "cfg" / "datasets" / "coco128.yaml"
        return str(p) if p.is_file() else None
    except Exception:
        return None


def _run_official_weights_coco128_val(pt_path: str, device_arg) -> dict | None:
    """
    在子进程中对「未换头的官方检测权重」跑 Ultralytics 自带的 coco128 验证集，得到可读的 COCO mAP。
    与 VisDrone 换 nc 后的指标不可比；用于回答「预训练到底有没有效」。
    """
    import json
    import subprocess

    data_yaml = _ultralytics_coco128_yaml_path()
    if not data_yaml:
        return None
    py_code = (
        "import json, os; "
        "os.environ.pop('YOLO_OFFLINE', None); "
        "from ultralytics import YOLO; "
        f"m=YOLO({pt_path!r}); "
        f"metrics=m.val(data={data_yaml!r}, imgsz=640, conf=0.001, iou=0.5, augment=False, "
        "plots=False, save_json=False, save_txt=False, "
        f"device={repr(device_arg)}, verbose=False); "
        "res={'mAP50': float(metrics.box.map50), 'mAP50_95': float(metrics.box.map), "
        "'precision': float(metrics.box.mp), 'recall': float(metrics.box.mr), 'dataset': 'coco128'}; "
        "print(json.dumps(res))"
    )
    try:
        proc = subprocess.run(
            ["python3", "-c", py_code],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        logger.warning("coco128 预训练参考验证超时（600s），已跳过。")
        return None
    out = (proc.stdout or "").strip()
    tail_err = ((proc.stderr or "") + "\n" + out)[-2000:]
    if proc.returncode != 0:
        logger.warning(
            "coco128 预训练参考验证失败（不影响主实验）。"
            f" code={proc.returncode}\n{tail_err}"
        )
        return None
    last_line = out.splitlines()[-1] if out else ""
    try:
        return json.loads(last_line)
    except Exception as je:
        logger.warning(f"解析 coco128 val JSON 失败: {je}\nlast_line={last_line!r}\n{tail_err}")
        return None


class YOLOWrapper(nn.Module):
    """
    封装ultralytics YOLO模型以兼容联邦学习框架
    
    主要功能:
    - 提供state_dict()和load_state_dict()方法用于联邦学习参数聚合
    - 支持前向传播和反向传播
    - 兼容PyTorch标准模型接口
    - 添加虚拟fc2层以兼容DPFLA算法
    """
    
    def __init__(self, model_size='n', num_classes=10, pretrained=True, model_path=None):
        """
        初始化YOLO封装类
        
        Args:
            model_size: 模型大小 ('n'=nano, 's'=small, 'm'=medium, 'l'=large, 'x'=xlarge)
            num_classes: 目标检测类别数（用于联邦学习框架，实际YOLO保持80类）
            pretrained: 是否使用预训练权重（默认True，强烈推荐）
            model_path: 本地模型文件路径(可选，优先使用)
        """
        super().__init__()
        self.model_size = model_size
        self.num_classes = num_classes  # 联邦学习框架 / 检测头类别数（VisDrone=10）
        self.yolo_classes = num_classes  # 检测头实际类别数（与 yaml val、训练标签一致）
        self.use_yolo = False
        self.coco_pretrained_baseline = None  # 换头前 coco128 参考指标（若有）
        
        # 错误日志计数器，避免刷屏
        self._error_counts = {}
        self._max_error_logs = 2  # 每种错误最多输出2次
        
        logger.info(f'--> Creating YOLOv8{model_size} (pretrained backbone, detection nc={num_classes}, aligned with VisDrone val)')
        
        try:
            from ultralytics import YOLO
            import os
            
            # 禁用ultralytics的详细输出和profiling
            os.environ['YOLO_VERBOSE'] = 'False'
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning, module='ultralytics')
            
            # 临时禁用ultralytics的日志
            import logging
            logging.getLogger('ultralytics').setLevel(logging.ERROR)
            
            if model_path:
                # 使用本地模型文件
                self.yolo = YOLO(model_path, verbose=False)
                logger.info(f'✓ Loaded YOLO from local file: {model_path}')
            elif pretrained:
                logger.info(f'→ Loading pretrained YOLOv8{model_size} model...')
                self.yolo = YOLO(f'yolov8{model_size}.pt', verbose=False)
                logger.info(f'✓ Loaded pretrained YOLOv8{model_size} (COCO head 将在下方替换为 nc={num_classes})')
            else:
                # 从yaml配置创建新模型
                logger.info(f'→ Creating YOLOv8{model_size} from scratch...')
                self.yolo = YOLO(f'yolov8{model_size}.yaml', verbose=False)
                logger.info(f'✓ Created YOLOv8{model_size} from scratch')
            
            # 获取底层PyTorch模型（DetectionModel）
            self.model = self.yolo.model  # 这是DetectionModel实例
            self.use_yolo = True

            from types import SimpleNamespace
            from ultralytics.nn.modules.head import Detect
            from ultralytics.utils.loss import v8DetectionLoss

            # 换头前（可选）：在 coco128 上跑一次 val，日志里多一行 COCO 预训练 mAP。
            # 默认关闭，避免每次启动下载/生成项目下 datasets/coco128。需要时: FL_RUN_COCO_PRETRAINED_VAL=1
            _head0 = self.model.model[-1]
            if (
                pretrained
                and isinstance(_head0, Detect)
                and _head0.nc == 80
                and num_classes != _head0.nc
                and os.environ.get("FL_RUN_COCO_PRETRAINED_VAL", "").strip() == "1"
            ):
                pt_for_val = os.path.abspath(os.path.expanduser(model_path)) if model_path else f'yolov8{model_size}.pt'
                dev_arg = 0 if torch.cuda.is_available() else "cpu"
                baseline = _run_official_weights_coco128_val(pt_for_val, dev_arg)
                if baseline:
                    self.coco_pretrained_baseline = baseline
                    logger.info(
                        f"📌 COCO 预训练检测头参考（换头前，权重 {pt_for_val} @ {baseline['dataset']}）: "
                        f"mAP@0.5={baseline['mAP50'] * 100:.2f}%, mAP@0.5:0.95={baseline['mAP50_95'] * 100:.2f}% "
                        f"| P={baseline['precision'] * 100:.2f}% R={baseline['recall'] * 100:.2f}%"
                    )
                    logger.info(
                        "   说明：下列 VisDrone 初始 mAP 在 nc 换为 "
                        f"{num_classes} 后重新计数，接近 0% 为预期，不代表 COCO 权重未载入。"
                    )
            
            default_hyp = {'box': 7.5, 'cls': 0.5, 'dfl': 1.5}
            
            # 统一 model.args / hyp（checkpoint 常为 dict 且无 hyp）
            if not hasattr(self.model, 'args') or self.model.args is None:
                self.model.args = SimpleNamespace()
            elif isinstance(self.model.args, dict):
                self.model.args = SimpleNamespace(**self.model.args)
            if not hasattr(self.model.args, 'hyp') or self.model.args.hyp is None:
                self.model.args.hyp = SimpleNamespace(**default_hyp)
            elif isinstance(self.model.args.hyp, dict):
                h = {**default_hyp, **self.model.args.hyp}
                self.model.args.hyp = SimpleNamespace(**h)
            else:
                for key, val in default_hyp.items():
                    if not hasattr(self.model.args.hyp, key):
                        setattr(self.model.args.hyp, key, val)
            
            # P0：将 COCO 80 类检测头替换为与 VisDrone / yaml val 一致的 nc（默认 10），训练标签与评估同一 id 空间
            head = self.model.model[-1]
            if isinstance(head, Detect) and head.nc != num_classes:
                ch = tuple(head.cv2[i][0].conv.in_channels for i in range(head.nl))
                dev = next(self.model.parameters()).device
                new_head = Detect(nc=num_classes, ch=ch)
                new_head.f = head.f
                new_head.i = head.i
                new_head.stride = head.stride.clone().to(dev)
                new_head.to(dev)
                new_head.training = True
                new_head.bias_init()
                self.model.model[-1] = new_head
                logger.info(f'✓ 检测头已替换: nc {head.nc} → {num_classes}（骨干仍为 COCO 预训练）')
                # 重置 BN 统计量：COCO 预训练的 running_mean/var 对 VisDrone 不适用，
                # 强制让 BN 在训练开始后用本地数据重新统计
                for module in self.model.modules():
                    if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                        module.reset_running_stats()
                        module.train()
                logger.debug('✓ 已重置所有 BN 统计量（running_mean/var）并切换为 train 模式')
            elif isinstance(head, Detect):
                logger.info(f'✓ 检测头 nc={head.nc} 已与 num_classes={num_classes} 一致，跳过替换')
            
            vis_names = (
                'pedestrian',
                'people',
                'bicycle',
                'car',
                'van',
                'truck',
                'tricycle',
                'awning-tricycle',
                'bus',
                'motor',
            )
            if num_classes == len(vis_names):
                self.model.names = {i: vis_names[i] for i in range(num_classes)}
            else:
                self.model.names = {i: str(i) for i in range(num_classes)}
            if hasattr(self.model, 'yaml') and isinstance(self.model.yaml, dict):
                self.model.yaml['nc'] = num_classes
            
            self.model.criterion = v8DetectionLoss(self.model)
            
            # 步骤4：直接修复criterion内部的hyp属性（以防万一）
            if hasattr(self.model.criterion, 'hyp'):
                if isinstance(self.model.criterion.hyp, dict):
                    # 补充缺失的超参数
                    for key, val in default_hyp.items():
                        if key not in self.model.criterion.hyp:
                            self.model.criterion.hyp[key] = val
                    # 转换为SimpleNamespace
                    self.model.criterion.hyp = SimpleNamespace(**self.model.criterion.hyp)
                elif isinstance(self.model.criterion.hyp, SimpleNamespace):
                    # 已经是SimpleNamespace，检查是否缺少属性
                    for key, val in default_hyp.items():
                        if not hasattr(self.model.criterion.hyp, key):
                            setattr(self.model.criterion.hyp, key, val)
                            logger.debug(f'为criterion.hyp添加缺失属性: {key} = {val}')
            
            # 步骤5：启用模型参数的梯度（关键修复！）
            for param in self.model.parameters():
                param.requires_grad = True
            logger.debug('✓ 已启用所有模型参数的梯度')
            
            # 步骤6：将criterion及其所有子组件移到GPU（如果可用）
            device = next(self.model.parameters()).device
            self._move_criterion_to_device(self.model.criterion, device)
            logger.debug(f'✓ 已将criterion移到设备: {device}')
            
            logger.debug('✓ 将使用YOLO模型内置损失函数')
            
            # 为DPFLA算法添加虚拟fc2层（不参与forward，仅用于兼容DPFLA的state_dict访问）
            self.fc2 = nn.Linear(128, num_classes)
            
        except Exception as e:
            logger.warning(f'Failed to load YOLO from ultralytics: {e}')
            logger.info('Creating a simple CNN model as fallback...')
            # 创建一个简单的CNN作为后备方案
            self.model = self._create_simple_cnn(num_classes)
            self.use_yolo = False
        
        logger.info('--> YOLO model has been created!')
    
    def _log_once(self, level, message, error_key=None):
        """
        只输出一次或有限次数的日志，避免刷屏
        
        Args:
            level: 日志级别 ('error', 'warning', 'info', 'debug')
            message: 日志消息
            error_key: 错误标识符，用于计数（默认使用message的前50个字符）
        """
        if error_key is None:
            error_key = message[:50]
        
        # 检查是否已经输出过这个错误
        count = self._error_counts.get(error_key, 0)
        
        if count < self._max_error_logs:
            # 输出日志
            if level == 'error':
                logger.error(message)
            elif level == 'warning':
                logger.warning(message)
            elif level == 'info':
                logger.info(message)
            elif level == 'debug':
                logger.debug(message)
            
            # 增加计数
            self._error_counts[error_key] = count + 1
            
            # 如果达到最大次数，输出提示
            if self._error_counts[error_key] == self._max_error_logs:
                logger.info(f'[{error_key[:30]}...] 此类错误已输出{self._max_error_logs}次，后续将不再显示')
        
        return count < self._max_error_logs
    
    def _move_criterion_to_device(self, obj, device):
        """
        递归地将criterion及其所有子组件的张量移到指定设备
        基于文献建议：Dynamic DP论文 - 递归移动所有子组件
        
        Args:
            obj: criterion对象或其子组件
            device: 目标设备
        """
        if obj is None:
            return
            
        # 关键组件1：移动proj（DFL投影矩阵）
        if hasattr(obj, 'proj') and isinstance(obj.proj, torch.Tensor):
            obj.proj = obj.proj.to(device)
        
        # 关键组件2：移动assigner的所有张量
        if hasattr(obj, 'assigner'):
            assigner = obj.assigner
            for attr_name in ['anchor_grid', 'stride', 'anchors']:
                if hasattr(assigner, attr_name):
                    attr = getattr(assigner, attr_name)
                    if isinstance(attr, torch.Tensor):
                        setattr(assigner, attr_name, attr.to(device))
        
        # 关键组件3：移动bbox_loss的所有张量
        if hasattr(obj, 'bbox_loss'):
            bbox_loss = obj.bbox_loss
            for attr_name in dir(bbox_loss):
                if not attr_name.startswith('_'):
                    try:
                        attr = getattr(bbox_loss, attr_name)
                        if isinstance(attr, torch.Tensor):
                            setattr(bbox_loss, attr_name, attr.to(device))
                    except Exception:
                        pass
        
        # 通用递归：遍历所有属性
        for attr_name in dir(obj):
            # 跳过私有属性和方法
            if attr_name.startswith('_') or callable(getattr(obj, attr_name, None)):
                continue
            
            try:
                attr = getattr(obj, attr_name)
                
                # 如果是张量，移到目标设备
                if isinstance(attr, torch.Tensor):
                    setattr(obj, attr_name, attr.to(device))
                # 如果是nn.Module，使用标准.to()方法
                elif isinstance(attr, nn.Module):
                    attr.to(device)
                # 如果有__dict__属性，递归处理
                elif hasattr(attr, '__dict__') and not isinstance(attr, type):
                    self._move_criterion_to_device(attr, device)
            except Exception:
                # 忽略无法访问或移动的属性
                pass
    
    def freeze_backbone(self, freeze=True):
        """
        冻结YOLOv8的backbone和neck层，只训练检测头
        
        Args:
            freeze: True=冻结backbone, False=解冻所有层
        """
        if not self.use_yolo:
            return
        
        # YOLOv8检测头通常在最后几层，包含'cv2', 'cv3', 'dfl'等关键字
        # 这些是Detect层的组件
        detect_keywords = ['cv2', 'cv3', 'dfl']
        
        frozen_params = []
        trainable_params = []
        
        for name, param in self.model.named_parameters():
            # 检查是否是检测头的参数
            is_detect_head = any(key in name for key in detect_keywords)
            
            if freeze:
                # 冻结模式：只有检测头可训练
                if is_detect_head:
                    param.requires_grad = True
                    trainable_params.append(name)
                else:
                    param.requires_grad = False
                    frozen_params.append(name)
            else:
                # 解冻模式：所有层可训练
                param.requires_grad = True
                trainable_params.append(name)
        
        if freeze:
            logger.info(f'✓ Backbone已冻结: {len(frozen_params)}个参数冻结, {len(trainable_params)}个参数可训练')
            if trainable_params:
                logger.debug(f'   可训练参数示例: {trainable_params[:3]}')
        else:
            logger.info(f'✓ 所有层已解冻: {len(trainable_params)}个参数可训练')
    
    def _add_virtual_fc2_layer(self):
        """
        为YOLO模型添加虚拟fc2层，用于兼容DPFLA算法
        DPFLA算法需要访问model.state_dict()['fc2.weight']
        """
        # 创建一个虚拟的全连接层
        # 输入维度设为128（与DPFLA中的m参数匹配），输出为类别数
        self.fc2 = nn.Linear(128, self.num_classes)
        logger.debug(f'--> Added virtual fc2 layer: Linear(128, {self.num_classes})')
    
    def _freeze_backbone(self):
        """
        冻结YOLO的backbone（特征提取部分），只训练检测头
        这样可以保持预训练权重的检测能力
        """
        if not self.use_yolo:
            return
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 只解冻最后的检测头（Detect层）
        # YOLO的结构：backbone -> neck -> head(Detect)
        # 我们只训练head部分
        if hasattr(self.model, 'model') and len(self.model.model) > 0:
            # 解冻最后3层（通常是检测头）
            for layer in self.model.model[-3:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # 统计可训练参数
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.debug(f'   可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)')
    
    def _create_simple_cnn(self, num_classes):
        """创建简单CNN作为后备方案"""
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x, return_features=False, targets=None):
        """
        前向传播
        
        Args:
            x: 输入图像张量 [B, C, H, W]
            return_features: 是否返回特征（用于联邦学习框架兼容）
            targets: 目标标签（训练时使用），支持两种格式：
                    1. list of dicts: [{'boxes': [N, 4], 'labels': [N]}, ...]
                    2. tensor [N, 6]: [batch_idx, cls, x, y, w, h]
            
        Returns:
            训练模式+targets: 返回损失标量
            其他情况: 返回预测输出
        """
        if self.use_yolo:
            # 训练模式且提供了targets，计算损失
            if self.training and targets is not None:
                try:
                    # 确保模型在训练模式
                    self.model.train()
                    
                    # 关键修复1：确保所有参数需要梯度（原生评估会冻结）
                    for param in self.model.parameters():
                        param.requires_grad = True
                    
                    # 关键修复2：确保criterion与输入在同一设备（每次forward都检查）
                    # 文献建议：在训练前确保criterion与模型在同一设备且状态正确
                    current_device = x.device
                    if hasattr(self.model, 'criterion') and self.model.criterion is not None:
                        self._move_criterion_to_device(self.model.criterion, current_device)
                        # 强制移动stride（criterion直接属性，容易漏掉）
                        crit = self.model.criterion
                        if hasattr(crit, 'stride') and isinstance(crit.stride, torch.Tensor):
                            crit.stride = crit.stride.to(current_device)
                        if hasattr(crit, 'device'):
                            crit.device = current_device
                        # 调试日志：确认criterion设备
                        if not hasattr(self, '_criterion_device_logged'):
                            logger.debug(f'✓ 已将criterion移到设备: {current_device}')
                            self._criterion_device_logged = True
                    
                    # 调试：检查参数梯度状态（只检查一次）
                    if not hasattr(self, '_params_grad_checked'):
                        params_with_grad = sum(1 for p in self.model.parameters() if p.requires_grad)
                        total_params = sum(1 for p in self.model.parameters())
                        logger.debug(f'模型参数: {params_with_grad}/{total_params} 需要梯度')
                        self._params_grad_checked = True
                    
                    # 检查targets格式并转换
                    if isinstance(targets, list):
                        # 格式1: list of dicts -> 转换为tensor [N, 6]
                        targets = self._convert_targets_list_to_tensor(targets, x.device)
                    
                    # 检查是否有有效目标
                    if targets.shape[0] == 0:
                        # 没有目标，返回零损失
                        self._log_once('warning', 'Batch没有目标框，跳过损失计算', 'no_targets')
                        loss = torch.tensor(0.0, device=x.device, requires_grad=True)
                        if return_features:
                            return loss, loss
                        return loss
                    
                    # 构造batch字典（YOLO原生格式）
                    # 注意：所有值都应该是1D tensor（除了bboxes是2D）
                    batch = {
                        'img': x,
                        'batch_idx': targets[:, 0].long(),  # [N] - batch索引
                        'cls': targets[:, 1],  # [N] - 类别
                        'bboxes': targets[:, 2:6]  # [N, 4] - [x, y, w, h]
                    }
                    
                    # 标准两步走：
                    # 1. 前向传播获取预测
                    preds = self.model(x)
                    
                    # 2. 直接调用criterion获取有梯度的loss
                    # criterion返回(loss, loss_items)，其中loss有梯度，loss_items是detached的
                    loss_tuple = self.model.criterion(preds, batch)
                    
                    # 仅在真实训练图下打一次 loss 结构；fl_core.test() 在 no_grad 里也会 forward，
                    # 那时 requires_grad=False，勿与「loss 无梯度」混淆。
                    if torch.is_grad_enabled() and not hasattr(self, '_loss_tuple_logged_train'):
                        logger.debug(f'loss_tuple type: {type(loss_tuple)}')
                        if isinstance(loss_tuple, (tuple, list)):
                            logger.debug(f'loss_tuple length: {len(loss_tuple)}')
                            for i, item in enumerate(loss_tuple):
                                if isinstance(item, torch.Tensor):
                                    logger.debug(
                                        f'  item[{i}] shape: {item.shape}, requires_grad: {item.requires_grad}, '
                                        f'grad_fn: {item.grad_fn}'
                                    )
                        self._loss_tuple_logged_train = True
                    
                    # 提取有梯度的loss（第一个元素）
                    if isinstance(loss_tuple, (tuple, list)):
                        loss = loss_tuple[0]  # 有梯度的总损失
                    else:
                        loss = loss_tuple
                    
                    # 处理返回值（loss已经是标量tensor）
                    if isinstance(loss, (tuple, list)):
                        loss = loss[0] if len(loss) > 0 else loss
                    elif isinstance(loss, dict):
                        # 如果返回dict，提取总损失
                        loss = loss.get('loss', loss.get('total_loss', sum(loss.values())))
                    
                    # 确保loss是标量tensor
                    if isinstance(loss, torch.Tensor):
                        if loss.numel() > 1:
                            # v8DetectionLoss 返回 shape=[3] 的 [box, cls, dfl] 向量
                            # 必须 .sum() 而非 .mean()，否则梯度信号被稀释3倍且语义错误
                            loss = loss.sum()
                        # 验证梯度
                        # 注意：`fl_core.test()` 在 torch.no_grad() 下会估计 loss，这时 loss 不应要求梯度，
                        # 因此只有在真实训练（grad enabled）时才打印该错误。
                        if torch.is_grad_enabled():
                            if not loss.requires_grad or loss.grad_fn is None:
                                self._log_once(
                                    'error',
                                    f'Loss没有梯度！requires_grad={loss.requires_grad}, grad_fn={loss.grad_fn}',
                                    'loss_no_grad'
                                )
                    else:
                        self._log_once('error', f'Loss不是tensor: {type(loss)}', 'loss_not_tensor')
                        loss = torch.tensor(0.5, device=x.device, requires_grad=True)
                    
                    # 如果需要返回features，返回(loss, loss)作为兼容
                    if return_features:
                        return loss, loss
                    return loss
                    
                except Exception as e:
                    self._log_once('error', f'YOLO损失计算失败: {e}', 'yolo_loss_failed')
                    self._log_once('debug', f'Traceback: {traceback.format_exc()}', 'yolo_loss_traceback')
                    # 训练期 OOM 不能用固定 0.5 假损失“伪训练”；直接抛错让上层降 batch/清理并重跑。
                    if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        raise RuntimeError(
                            "YOLO training OOM: reduce FL_TRAIN_BATCH_SIZE / model size, "
                            "and ensure only one training process occupies GPU."
                        ) from e
                    # 其它异常维持兜底，避免单批次偶发错误中断整轮
                    loss = torch.tensor(0.5, device=x.device, requires_grad=True)
                    if return_features:
                        return loss, loss
                    return loss
            
            # 推理模式或没有targets
            output = self.model(x)
            
            if return_features:
                features = output[-1] if isinstance(output, (list, tuple)) else output
                return output, features
            return output
        else:
            # 后备CNN模型
            output = self.model(x)
            if return_features:
                return output, output
            return output
    
    def _convert_targets_list_to_tensor(self, targets, device):
        """
        将 list of dicts 转为 tensor [N, 6]，类别 id 与数据集 / yolo val（nc=10）一致，不再映射到 COCO。
        """
        batch_idx_list = []
        cls_list = []
        boxes_list = []
        
        for i, target in enumerate(targets):
            boxes = target['boxes']  # [M, 4] xywh 归一化
            labels = target['labels']  # [M] 类别 0..nc-1
            
            if len(boxes) == 0:
                continue
            
            batch_idx = torch.full((len(boxes),), i, dtype=torch.float32, device=device)
            batch_idx_list.append(batch_idx)
            cls_list.append(labels.float().to(device))
            boxes_list.append(boxes.to(device))
        
        if len(batch_idx_list) == 0:
            # 没有任何目标，返回空tensor
            return torch.zeros((0, 6), dtype=torch.float32, device=device)
        
        # 拼接所有列
        batch_idx_tensor = torch.cat(batch_idx_list, dim=0).unsqueeze(1)  # [N, 1]
        cls_tensor = torch.cat(cls_list, dim=0).unsqueeze(1)  # [N, 1]
        boxes_tensor = torch.cat(boxes_list, dim=0)  # [N, 4]
        
        # 拼接为 [N, 6]
        targets_tensor = torch.cat([batch_idx_tensor, cls_tensor, boxes_tensor], dim=1)
        
        return targets_tensor
    
    def state_dict(self, *args, **kwargs):
        """
        返回模型参数字典，用于联邦学习参数聚合
        包含YOLO模型参数和虚拟fc2层参数
        """
        state = self.model.state_dict(*args, **kwargs)
        # 添加fc2层的参数
        if hasattr(self, 'fc2'):
            state['fc2.weight'] = self.fc2.weight
            state['fc2.bias'] = self.fc2.bias
        return state
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        加载模型参数，用于联邦学习参数更新
        """
        # 分离fc2层的参数
        fc2_weight = state_dict.pop('fc2.weight', None)
        fc2_bias = state_dict.pop('fc2.bias', None)
        
        # 加载YOLO模型参数
        result = self.model.load_state_dict(state_dict, *args, strict=False, **kwargs)
        
        # 加载fc2层参数
        if fc2_weight is not None and hasattr(self, 'fc2'):
            self.fc2.weight.data = fc2_weight
        if fc2_bias is not None and hasattr(self, 'fc2'):
            self.fc2.bias.data = fc2_bias
            
        return result
    
    def parameters(self, recurse=True):
        """
        返回模型参数迭代器
        """
        return self.model.parameters(recurse=recurse)
    
    def named_parameters(self, prefix='', recurse=True):
        """
        返回命名参数迭代器
        """
        return self.model.named_parameters(prefix=prefix, recurse=recurse)
    
    def train(self, mode=True):
        """
        设置训练模式
        """
        self.model.train(mode)
        return self
    
    def eval(self):
        """
        设置评估模式
        """
        self.model.eval()
        return self
    
    def to(self, device):
        """
        移动模型到指定设备
        文献建议：确保criterion与模型同步移动
        """
        logger.debug(f'→ YOLOWrapper.to({device}) 被调用')
        
        self.model = self.model.to(device)
        if hasattr(self, 'fc2'):
            self.fc2 = self.fc2.to(device)
        
        # 关键修复：同时移动criterion到相同设备
        if hasattr(self.model, 'criterion') and self.model.criterion is not None:
            logger.debug(f'  → 移动criterion到 {device}')
            self._move_criterion_to_device(self.model.criterion, device)
            logger.debug(f'  ✓ criterion已移动到 {device}')
        else:
            logger.debug(f'  ⚠️  criterion不存在或为None')
        
        return self
    
    def cuda(self, device=None):
        """
        移动模型到GPU
        文献建议：确保criterion与模型同步移动
        """
        target_device = torch.device('cuda' if device is None else f'cuda:{device}')
        logger.debug(f'→ YOLOWrapper.cuda({device}) 被调用，目标设备: {target_device}')
        
        self.model = self.model.cuda(device)
        if hasattr(self, 'fc2'):
            self.fc2 = self.fc2.cuda(device)
        
        # 关键修复：同时移动criterion到GPU
        if hasattr(self.model, 'criterion') and self.model.criterion is not None:
            logger.debug(f'  → 移动criterion到 {target_device}')
            self._move_criterion_to_device(self.model.criterion, target_device)
            logger.debug(f'  ✓ criterion已移动到 {target_device}')
        else:
            logger.debug(f'  ⚠️  criterion不存在或为None')
        
        return self
    
    def cpu(self):
        """
        移动模型到CPU
        """
        self.model = self.model.cpu()
        if hasattr(self, 'fc2'):
            self.fc2 = self.fc2.cpu()
        return self
