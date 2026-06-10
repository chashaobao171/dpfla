import copy
import time
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import sklearn.metrics.pairwise as smp
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F


class DPFLA:
    def __init__(self):
        # 跨轮历史：用于“稳定惩罚/恢复”机制，提升小样本(如7客户端)下的鲁棒性
        self.client_bad_streak = {}
        self.client_prev_soft = {}
        self.client_cooldown = {}

    @staticmethod
    def _robust_anomaly_score(feature_matrix):
        """
        基于稳健 z-score(MAD) 的异常度评分，输出 [0,1]。
        feature_matrix: shape [n_clients, n_features]
        小客户端数(n<=8)时 MAD 极易放大噪声，故对 z 做逐维裁剪并略抬高 logistic 中心，减少诚实客户端被标成 ~1.0。
        """
        if feature_matrix is None or len(feature_matrix) == 0:
            return np.array([], dtype=float)
        x = np.asarray(feature_matrix, dtype=float)
        n = int(x.shape[0])
        med = np.median(x, axis=0, keepdims=True)
        mad = np.median(np.abs(x - med), axis=0, keepdims=True)
        mad = np.where(mad < 1e-8, 1e-8, mad)
        z = np.abs((x - med) / (1.4826 * mad))
        # 防止单维爆炸把多数客户端打成“全异常”
        z_cap = 2.3 if n <= 8 else 3.5
        z = np.minimum(z, z_cap)
        raw = np.mean(z, axis=1)
        raw = np.minimum(raw, 2.8)
        # n 小时略提高阈值，使辅助分更保守
        center = 1.55 if n <= 8 else 1.2
        scale = 0.85 if n <= 8 else 0.7
        score = 1.0 / (1.0 + np.exp(-(raw - center) / scale))
        return np.clip(score, 0.0, 1.0)
    
    def score_validation_based(self, global_model, local_models, clients_types, val_loader, device, model_name='CNNMNIST'):
        """
        增强版DPFLA：基于验证集评估客户端更新质量
        
        :param global_model: 全局模型
        :param local_models: 本地模型列表
        :param clients_types: 客户端类型列表（用于日志）
        :param val_loader: 验证集DataLoader
        :param device: 设备
        :param model_name: 模型名称
        :return: scores列表，值越高表示更新质量越好
        """
        logger.info("=" * 70)
        logger.info("🔍 增强版DPFLA：基于验证集的客户端评分")
        logger.info("=" * 70)
        
        # 1. 计算全局模型在验证集上的基准损失
        # 注意：YOLO/检测任务的loss需要走 model(data, targets=...) 且通常要求处于 train() 模式
        global_model.to(device)
        if model_name == 'YOLO':
            global_model.train()
        else:
            global_model.eval()
        global_loss = 0.0
        total_samples = 0
        
        # 限制验证样本数量（只取一个batch的前max_samples个样本，快速且一致地评估）
        # 说明：
        #   - 当前YOLO+VisDrone实验中 max_samples 固定为128，用于在“有攻击 + DPFLA防御”下快速区分好/坏客户端。
        #   - 该设置在 Gaussian 攻击 + 4客户端恶意率25%、global_round=10 的实验中已验证收敛稳定（mAP50≈70%→95%）。
        #   - 如后续需要更精细的评分信号，可放宽 max_samples 或切换到基于检测指标差分的打分方式。
        max_samples = 128
        
        # 🔁 关键改动：全局模型与各客户端使用**同一批**验证样本
        with torch.no_grad():
            try:
                val_data, val_target = next(iter(val_loader))
            except StopIteration:
                logger.warning("验证集为空，DPFLA 验证评分回退为均匀权重")
                return np.ones(len(local_models), dtype=float)

            val_data = val_data[:max_samples].to(device)
            total_samples = len(val_data)

            # 分类任务：target是Tensor，可直接CE
            # 检测任务(VisDrone)：target通常是 list[dict]，需要走YOLO原生loss
            if model_name == 'YOLO' or isinstance(val_target, (list, tuple)) or isinstance(val_target, dict):
                # 截取与data一致的targets数量
                if isinstance(val_target, (list, tuple)):
                    val_target = val_target[:max_samples]
                    # 将target内部tensor移动到device（与Client一致）
                    val_target = [
                        {k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()}
                        for t in val_target
                    ]
                elif isinstance(val_target, dict):
                    val_target = {k: v.to(device) if torch.is_tensor(v) else v for k, v in val_target.items()}
                
                loss = global_model(val_data, targets=val_target)
                # YOLOWrapper在return_features=True时返回(loss, features)，这里兜底处理
                if isinstance(loss, (tuple, list)):
                    loss = loss[0]
                global_loss = float(loss.item()) if torch.is_tensor(loss) else float(loss)
            else:
                val_target = val_target[:max_samples].to(device)
                output = global_model(val_data)
                global_loss = F.cross_entropy(output, val_target).item()
        
        logger.info(f"📊 全局模型基准损失: {global_loss:.4f} (验证样本数: {total_samples})")
        
        # 2. 评估每个客户端更新的效果（沿用同一批 val_data/val_target）
        improvements = []
        
        for i, local_model in enumerate(local_models):
            local_model.to(device)
            if model_name == 'YOLO':
                local_model.train()
            else:
                local_model.eval()
            with torch.no_grad():
                if model_name == 'YOLO' or isinstance(val_target, (list, tuple)) or isinstance(val_target, dict):
                    loss = local_model(val_data, targets=val_target)
                    if isinstance(loss, (tuple, list)):
                        loss = loss[0]
                    local_loss = float(loss.item()) if torch.is_tensor(loss) else float(loss)
                else:
                    output = local_model(val_data)
                    local_loss = F.cross_entropy(output, val_target).item()
            
            # 计算改进量（负值表示变差）
            improvement = global_loss - local_loss
            improvements.append(improvement)
            logger.info(f"  客户端 {i+1} ({clients_types[i]}): 损失={local_loss:.4f}, 改进={improvement:.4f}")

        improvements = np.array(improvements, dtype=float)

        # 3. 将改进量映射为聚合权重
        #
        # 情况A：存在至少一个客户端使验证loss下降（improvement > 0）
        #   - 只对 improvement>0 的客户端分配权重，按改进量线性归一化
        #   - improvement<=0 的客户端权重=0
        # 情况B：所有客户端都使验证loss变差或几乎不变（max_improvement<=0）
        #   - 说明“当前这一轮整体是坏的更新”，此时不应把攻击者均匀混回去
        #   - 策略：只保留“最不差”的那个客户端（improvement 最大者），其权重=1，其它为0

        max_improvement = float(improvements.max()) if improvements.size > 0 else 0.0

        if max_improvement > 0:
            positive = np.maximum(improvements, 0.0)
            sum_positive = positive.sum()
            if sum_positive <= 0:
                # 理论上不会到这里，但为稳妥再兜底一次
                scores = np.ones_like(positive)
            else:
                scores = positive / sum_positive
            logger.info("✅ 至少存在使验证损失下降的客户端：按正向改进量线性归一化分配权重")
        else:
            # 所有改进量 <= 0：选择“最不差”的客户端
            if improvements.size == 0:
                scores = np.ones(0, dtype=float)
            else:
                best_idx = int(np.argmax(improvements))  # 最大的（最不负）改进量
                scores = np.zeros_like(improvements)
                scores[best_idx] = 1.0
            logger.warning("⚠️ 所有客户端更新在验证集上都未带来改进，仅保留最不差的一个客户端参与聚合")
        
        if scores.size > 0:
            logger.info("=" * 70)
            logger.info(f"📈 得分统计: 最小={scores.min():.4f}, 最大={scores.max():.4f}, 平均={scores.mean():.4f}")
            logger.info("=" * 70)
        
        return scores

    def score(self, global_model, local_models, clients_types, selected_clients, p, w, model_name='CNNMNIST', 
              val_loader=None, device=None, use_validation=True):
        """
        DPFLA评分函数（支持两种模式）
        
        :param use_validation: True=使用验证集评分（增强版），False=使用聚类评分（原版）
        """
        
        # 如果提供了验证集且启用验证模式，使用增强版
        if use_validation and val_loader is not None and device is not None:
            return self.score_validation_based(global_model, local_models, clients_types, 
                                               val_loader, device, model_name)
        
        # 否则使用原始的 SVD+KMeans 聚类方法（当前 DPFLA 主实验路径）
        logger.info("✅ 使用 SVD+KMeans 聚类评分（主实验路径）")

        n = len(selected_clients)
        W = generate_orthogonal_matrix(n=n * n, reuse=True)
        Ws = [W[:, e * n: e * n + n][0, :].reshape(-1, 1) for e in range(n)]

        param_diff = []
        param_diff_mask = []

        m_len = len(local_models)

        detect_res_list = []
        start_model_layer_param_list = []

        # 根据模型类型选择要分析的层
        if model_name == 'YOLO':
            state_keys = set(global_model.state_dict().keys())
            # 优先使用检测头分类分支的可比较参数（YOLOv8n 常见键）
            preferred_keys = [
                'model.22.cv3.0.2.weight',
                'model.22.cv3.1.2.weight',
                'model.22.cv3.2.2.weight',
            ]
            layer_keys = [k for k in preferred_keys if k in state_keys]

            # 兼容不同Ultralytics版本：自动抓取检测头cv3末端卷积
            if not layer_keys:
                layer_keys = [
                    k for k in global_model.state_dict().keys()
                    if k.startswith('model.22.cv3.') and k.endswith('.2.weight')
                ]
                layer_keys = sorted(layer_keys)

            # 兜底：若未找到检测头参数，退回历史fc2（仅保证可运行）
            if not layer_keys:
                if 'fc2.weight' in state_keys:
                    layer_keys = ['fc2.weight']
                    logger.warning("YOLO检测头参数未命中，暂时回退到 fc2.weight（兜底）")
                else:
                    raise RuntimeError("YOLO特征层选择失败：未找到检测头参数，也不存在 fc2.weight")

            # 以“按类别切片”的方式构建特征单元，语义上对齐 fc2 的按类比较
            # 每个类别的向量由多尺度检测头该类别的 weight(+bias) 拼接而成
            yolo_bias_keys = [k.replace('.weight', '.bias') for k in layer_keys]
            has_bias = all(k in state_keys for k in yolo_bias_keys)
            class_counts = [int(global_model.state_dict()[k].shape[0]) for k in layer_keys]
            num_classes = min(class_counts) if class_counts else 0

            if num_classes > 0:
                feature_units = list(range(num_classes))
                logger.info(
                    f"YOLO特征层选择: {layer_keys}, has_bias={has_bias}, 按类别切片单元数={num_classes}"
                )
                use_yolo_class_slice = True
            else:
                # 兜底：无法识别类别维时，退回整层向量拼接
                feature_units = ['__all_head_concat__']
                logger.warning("YOLO检测头类别维识别失败，回退为整层拼接向量")
                use_yolo_class_slice = False
            is_yolo_head_mode = True
        else:
            # CNN模型保持原有按类别切片的fc2分析方式
            layer_key = 'fc2.weight'
            num_classes = 10
            feature_units = list(range(num_classes))
            is_yolo_head_mode = False

        # 扩展防御特征面：在保持 SVD+KMeans 主判别不变前提下，引入辅助统计特征
        # 特征包含：更新范数、与群体均值方向余弦相似度、层间更新集中度
        client_feature_rows = []
        if is_yolo_head_mode:
            global_full_parts = []
            for lk in layer_keys:
                global_full_parts.append(global_model.state_dict()[lk].cpu().detach().reshape(-1).numpy())
                bk = lk.replace('.weight', '.bias')
                if bk in global_model.state_dict():
                    global_full_parts.append(global_model.state_dict()[bk].cpu().detach().reshape(-1).numpy())
            global_full_vec = np.concatenate(global_full_parts, axis=0)

            local_full_deltas = []
            local_layer_norms = []
            for i in range(m_len):
                local_parts = []
                per_layer_norm = []
                for lk in layer_keys:
                    lv = local_models[i].state_dict()[lk].cpu().detach().reshape(-1).numpy()
                    gv = global_model.state_dict()[lk].cpu().detach().reshape(-1).numpy()
                    local_parts.append(lv)
                    per_layer_norm.append(float(np.linalg.norm(lv - gv)))
                    bk = lk.replace('.weight', '.bias')
                    if bk in local_models[i].state_dict() and bk in global_model.state_dict():
                        lb = local_models[i].state_dict()[bk].cpu().detach().reshape(-1).numpy()
                        gb = global_model.state_dict()[bk].cpu().detach().reshape(-1).numpy()
                        local_parts.append(lb)
                        per_layer_norm.append(float(np.linalg.norm(lb - gb)))
                local_full_vec = np.concatenate(local_parts, axis=0)
                delta = local_full_vec - global_full_vec
                local_full_deltas.append(delta)
                local_layer_norms.append(per_layer_norm)

            mean_delta = np.mean(np.stack(local_full_deltas, axis=0), axis=0) if local_full_deltas else None
            mean_delta_norm = float(np.linalg.norm(mean_delta)) if mean_delta is not None else 0.0

            for i in range(m_len):
                delta = local_full_deltas[i]
                delta_norm = float(np.linalg.norm(delta))
                if delta_norm < 1e-12 or mean_delta_norm < 1e-12:
                    cos_sim = 1.0
                else:
                    cos_sim = float(np.dot(delta, mean_delta) / (delta_norm * mean_delta_norm))
                layer_norms = np.array(local_layer_norms[i], dtype=float)
                layer_sum = float(np.sum(layer_norms)) + 1e-12
                concentration = float(np.max(layer_norms / layer_sum)) if layer_norms.size > 0 else 0.0
                client_feature_rows.append([np.log1p(delta_norm), cos_sim, concentration])
        else:
            # 非YOLO任务保持轻量辅助特征
            layer_key = 'fc2.weight'
            g = global_model.state_dict()[layer_key].cpu().detach().reshape(-1).numpy()
            deltas = []
            for i in range(m_len):
                l = local_models[i].state_dict()[layer_key].cpu().detach().reshape(-1).numpy()
                deltas.append(l - g)
            mean_delta = np.mean(np.stack(deltas, axis=0), axis=0) if deltas else None
            mean_delta_norm = float(np.linalg.norm(mean_delta)) if mean_delta is not None else 0.0
            for delta in deltas:
                delta_norm = float(np.linalg.norm(delta))
                if delta_norm < 1e-12 or mean_delta_norm < 1e-12:
                    cos_sim = 1.0
                else:
                    cos_sim = float(np.dot(delta, mean_delta) / (delta_norm * mean_delta_norm))
                client_feature_rows.append([np.log1p(delta_norm), cos_sim, 1.0])

        anomaly_scores = self._robust_anomaly_score(client_feature_rows)

        for unit in feature_units:
            if is_yolo_head_mode:
                if use_yolo_class_slice and isinstance(unit, int):
                    unit_parts = []
                    for lk in layer_keys:
                        unit_parts.append(
                            global_model.state_dict()[lk][unit].cpu().detach().reshape(-1).numpy()
                        )
                        bk = lk.replace('.weight', '.bias')
                        if bk in global_model.state_dict():
                            unit_parts.append(
                                global_model.state_dict()[bk][unit].cpu().detach().reshape(-1).numpy()
                            )
                    start_vec = np.concatenate(unit_parts, axis=0)
                    layer_key = f'yolo_head_class_{unit}'
                else:
                    unit_parts = []
                    for lk in layer_keys:
                        unit_parts.append(global_model.state_dict()[lk].cpu().detach().reshape(-1).numpy())
                        bk = lk.replace('.weight', '.bias')
                        if bk in global_model.state_dict():
                            unit_parts.append(global_model.state_dict()[bk].cpu().detach().reshape(-1).numpy())
                    start_vec = np.concatenate(unit_parts, axis=0)
                    layer_key = 'yolo_head_concat'
            else:
                layer_key = 'fc2.weight'
                start_vec = global_model.state_dict()[layer_key][unit].cpu().detach().reshape(-1).numpy()

            local_param_diff_mask = []
            # 计算每个本地模型与全局模型在选定特征上的参数差
            for i in range(m_len):
                if is_yolo_head_mode:
                    if use_yolo_class_slice and isinstance(unit, int):
                        unit_parts = []
                        for lk in layer_keys:
                            unit_parts.append(
                                local_models[i].state_dict()[lk][unit].cpu().detach().reshape(-1).numpy()
                            )
                            bk = lk.replace('.weight', '.bias')
                            if bk in local_models[i].state_dict():
                                unit_parts.append(
                                    local_models[i].state_dict()[bk][unit].cpu().detach().reshape(-1).numpy()
                                )
                        end_vec = np.concatenate(unit_parts, axis=0)
                    else:
                        unit_parts = []
                        for lk in layer_keys:
                            unit_parts.append(local_models[i].state_dict()[lk].cpu().detach().reshape(-1).numpy())
                            bk = lk.replace('.weight', '.bias')
                            if bk in local_models[i].state_dict():
                                unit_parts.append(local_models[i].state_dict()[bk].cpu().detach().reshape(-1).numpy())
                        end_vec = np.concatenate(unit_parts, axis=0)
                else:
                    end_vec = local_models[i].state_dict()[layer_key][unit].cpu().detach().reshape(-1).numpy()

                gradient = calculate_parameter_gradients(start_vec, end_vec).flatten()
                # 使用与当前特征维度一致的投影矩阵，避免维度错配
                p_current = int(gradient.shape[0])
                P_current = generate_orthogonal_matrix(n=p_current, reuse=True)
                X_mask = Ws[i] @ gradient.reshape(1, -1) @ P_current
                local_param_diff_mask.append(X_mask)

            Z_mask = sum(local_param_diff_mask)
            U_mask, sigma, VT_mask = svd(Z_mask)

            G = Ws[0]
            for idx, val in enumerate(selected_clients):
                if idx == 0:
                    continue
                G = np.concatenate((G, Ws[idx]), axis=1)

            U = np.linalg.inv(G) @ U_mask
            U = U[:, :2]
            res = U * sigma[:2]
            detect_res_list.append(res)

        coefficient_list, score_list = batch_detect_outliers_kmeans(detect_res_list)

        max_sc = max(coefficient_list)
        max_sc_idx = coefficient_list.index(max_sc)
        # 双阈值策略：
        # - >=0.72: 高置信检测，强执行
        # - [0.58,0.72): 灰区检测，结合辅助异常特征进行柔性惩罚
        # - <0.58: 仅使用辅助特征做温和防守，避免纯噪声误杀
        kmeans_labels = score_list[max_sc_idx] if max_sc >= 0.58 else np.ones(n, dtype=int)
        minority_count = int(np.sum(kmeans_labels == 0))

        logger.debug("-------------------------------------")
        logger.debug("Max Silhouette Coefficient: " + str(max_sc))
        logger.debug("Detect Class: " + str(max_sc_idx))
        logger.debug("Defense result:")
        #
        # 经过 batch_detect_outliers_kmeans 规范化后：
        #   - scores[i] == 1 表示该客户端位于“多数派簇”（认为是诚实客户端）
        #   - scores[i] == 0 表示该客户端位于“少数派簇”（认为是恶意/异常客户端）
        #
        # 在聚合时，average_weights 使用 scores 作为权重：
        #   w_avg = Σ marks[i] * w[i]，其中 marks[i] = scores[i]
        # 因此：
        #   - scores[i] = 1 → 客户端完整参与聚合
        #   - scores[i] = 0 → 客户端被完全剔除
        # 软权重聚合 + 跨轮惩罚/恢复：
        # - 避免 0/1 硬切在误杀时造成大幅性能损失
        # - 对连续判为 bad 的客户端逐轮加重惩罚
        # - 对“刚从bad恢复”的客户端做缓释恢复，降低抖动
        high_conf = max_sc >= 0.72
        gray_zone = (max_sc >= 0.58) and (max_sc < 0.72)
        confidence = float(np.clip((max_sc - 0.50) / 0.45, 0.0, 1.0))
        base_bad_weight = 0.20 - 0.12 * confidence  # 更激进：高置信时 bad 初始更低
        soft_scores = np.ones(n, dtype=float)

        for i, pt in enumerate(clients_types):
            cid = int(selected_clients[i])
            km_label = int(kmeans_labels[i])
            a_score = float(anomaly_scores[i]) if i < len(anomaly_scores) else 0.0
            # 多数簇内：辅助分只做“轻度纠偏”，避免 n=7 时把诚实客户端打成 aux≈1
            a_for_risk = float(a_score)
            if km_label == 1:
                cap = 0.38 if gray_zone else (0.48 if not high_conf else 0.55)
                a_for_risk = min(a_for_risk, cap)

            # 组合风险分数（0低风险 ~ 1高风险）
            if km_label == 0:
                # 仅 1 个少数簇样本且辅助异常不高：多为 KMeans 抖动误杀诚实客户端，减轻惩罚
                if minority_count == 1 and a_for_risk < 0.48:
                    risk = 0.42 + 0.28 * a_for_risk
                else:
                    risk = 0.68 + 0.28 * a_for_risk
            else:
                if high_conf:
                    risk = 0.06 + 0.26 * a_for_risk
                elif gray_zone:
                    risk = 0.10 + 0.28 * a_for_risk
                else:
                    risk = 0.08 + 0.20 * a_for_risk
            risk = float(np.clip(risk, 0.0, 0.98))

            if km_label == 0:
                streak = self.client_bad_streak.get(cid, 0) + 1
                self.client_bad_streak[cid] = streak
                # 高置信或高异常时启用冷却期，防止“一轮好就立刻满权重”
                if high_conf or a_score >= 0.65:
                    self.client_cooldown[cid] = 2
                # 连续 bad 指数衰减，最小权重更低以加强抑制
                soft = max(0.02, float(base_bad_weight * (0.72 ** (streak - 1))))
                # 融合风险分数后进一步压低
                soft = min(soft, max(0.02, 1.0 - risk))
                # 不确定的“单点少数簇”保底权重，避免诚实被压到 ~0.12 拖垮全局
                if minority_count == 1 and a_score < 0.48:
                    soft = max(soft, 0.28)
                soft_scores[i] = soft
            else:
                prev_streak = self.client_bad_streak.get(cid, 0)
                self.client_bad_streak[cid] = 0
                cooldown_left = int(self.client_cooldown.get(cid, 0))
                base_soft = max(0.05, float(1.0 - risk))
                # 从连续bad恢复时先给一个缓释权重，再逐轮回到1.0
                if prev_streak >= 2 or cooldown_left > 0:
                    prev_soft = self.client_prev_soft.get(cid, 0.35)
                    recover_soft = max(0.45, min(0.80, prev_soft + 0.15))
                    if cooldown_left > 0:
                        # 冷却期内上限收紧
                        recover_soft = min(recover_soft, 0.60)
                        self.client_cooldown[cid] = cooldown_left - 1
                    soft_scores[i] = min(base_soft, recover_soft)
                else:
                    # 灰区：仍限制略低于 1，但多数簇保底更高，减少误伤
                    if gray_zone:
                        soft_scores[i] = max(0.78, min(0.94, base_soft))
                    else:
                        soft_scores[i] = min(1.0, base_soft)

            self.client_prev_soft[cid] = float(soft_scores[i])
            logger.info(
                f"{pt} | KMeans标签={km_label} | aux异常={a_score:.3f} | risk={risk:.3f} | streak={self.client_bad_streak.get(cid, 0)} | cooldown={self.client_cooldown.get(cid, 0)} | 聚合权重={soft_scores[i]:.3f}"
            )

        return soft_scores


def generate_orthogonal_matrix(n, reuse=False, block_size=None):
    orthogonal_matrix_cache_dir = 'orthogonal_matrices'
    if os.path.isdir(orthogonal_matrix_cache_dir) is False:
        os.makedirs(orthogonal_matrix_cache_dir, exist_ok=True)
    file_list = os.listdir(orthogonal_matrix_cache_dir)
    existing = [e.split('.')[0] for e in file_list]

    file_name = str(n)
    if block_size is not None:
        file_name += '_blc%s' % block_size

    if reuse and file_name in existing:
        with open(os.path.join(orthogonal_matrix_cache_dir, file_name + '.pkl'), 'rb') as f:
            return pickle.load(f)
    else:
        if block_size is not None:
            qs = [block_size] * int(n / block_size)
            if n % block_size != 0:
                qs[-1] += (n - np.sum(qs))
            q = np.zeros([n, n])
            for i in range(len(qs)):
                sub_n = qs[i]
                tmp = generate_orthogonal_matrix(sub_n, reuse=False, block_size=sub_n)
                index = int(np.sum(qs[:i]))
                q[index:index + sub_n, index:index + sub_n] += tmp
        else:
            q, _ = np.linalg.qr(np.random.randn(n, n), mode='full')
        if reuse:
            with open(os.path.join(orthogonal_matrix_cache_dir, file_name + '.pkl'), 'wb') as f:
                pickle.dump(q, f, protocol=4)
        return q


def calculate_parameter_gradients(params_1, params_2):
    return np.array([x for x in np.subtract(params_1, params_2)])


def detect_outliers_kmeans(data, n_clusters=2):
    """
    对单一特征矩阵做 KMeans 聚类，并将“多数派簇→1（好客户端）”，“少数派簇→0（坏客户端）”。

    说明：
    - KMeans 原始标签是任意的（哪一簇是 0/1 没有语义），我们根据簇大小重新赋义：
      - 样本数最多的簇视为“多数派簇”，记为 1（诚实客户端）
      - 其余簇视为“少数派簇”，记为 0（恶意/异常客户端）
    """
    # 初始化 K-means 模型并训练
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data)
    labels = kmeans.predict(data)

    # 所有样本都在同一簇时，认为“没有明显异常”
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        logger.debug("Only one cluster found in detect_outliers_kmeans, all clients are considered good")
        return np.ones(len(data), dtype=int)

    # 计算轮廓系数
    coefficient = silhouette_score(data, labels)
    logger.debug("Silhouette Coefficient：{}", coefficient)
    # calinski_harabasz = calinski_harabasz_score(data, labels)
    # logger.debug("Calinski Harabasz：{}", calinski_harabasz)

    # 轮廓系数过低：簇结构不明显，直接认为“全部为好客户端”，避免误杀
    if coefficient < 0.61:
        return np.ones(len(data), dtype=int)

    # 多数派簇 → 1，少数派簇 → 0
    majority_label = unique_labels[np.argmax(counts)]
    scores = np.array([1 if lbl == majority_label else 0 for lbl in labels], dtype=int)

    return scores


def batch_detect_outliers_kmeans(list, n_clusters=2):
    # 初始化K-means模型
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    coefficient_list = []
    score_list = []

    for data in list:
        try:
            # 训练模型
            kmeans.fit(data)
            # 预测聚类标签
            labels = kmeans.predict(data)
            
            # 检查是否所有样本都在同一个类别
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                # 如果只有一个类别，说明没有明显异常：所有客户端视为“好”
                logger.debug("Only one cluster found, all clients are considered good")
                coefficient = 0.0
                scores = np.ones(len(data), dtype=int)  # 所有客户端得分=1（好的）
            else:
                # 计算轮廓系数
                coefficient = silhouette_score(data, labels)

                # 统一约定：多数派簇=诚实(1)，少数派簇=恶意(0)
                #
                # 这里不再使用“标签求和 < N/2 时翻转”的复杂逻辑，
                # 而是直接按照簇大小重新编码标签，保证语义稳定：
                #   - clients in majority cluster → 1（好客户端）
                #   - clients in minority cluster(s) → 0（坏客户端）
                _, counts = np.unique(labels, return_counts=True)
                majority_label = unique_labels[np.argmax(counts)]
                scores = np.array([1 if lbl == majority_label else 0 for lbl in labels], dtype=int)
            
            coefficient_list.append(coefficient)
            score_list.append(scores)
            
        except ValueError as e:
            # 处理KMeans异常（比如样本数太少）
            logger.warning(f"KMeans clustering failed: {e}, treating all clients as good")
            coefficient_list.append(0.0)
            score_list.append(np.ones(len(data), dtype=int))
    
    # 日志去噪：按批次汇总输出轮廓系数，避免每个切片刷屏
    if len(coefficient_list) > 0:
        coeff_np = np.array(coefficient_list, dtype=float)
        logger.debug(
            "Silhouette统计: n={} | min={:.4f} | p25={:.4f} | median={:.4f} | p75={:.4f} | max={:.4f} | mean={:.4f}",
            len(coeff_np),
            float(np.min(coeff_np)),
            float(np.percentile(coeff_np, 25)),
            float(np.median(coeff_np)),
            float(np.percentile(coeff_np, 75)),
            float(np.max(coeff_np)),
            float(np.mean(coeff_np)),
        )

    return coefficient_list, score_list


def svd(x):
    m, n = x.shape
    if m >= n:
        return np.linalg.svd(x)
    else:
        u, s, v = np.linalg.svd(x.T)
        return v.T, s, u.T


def draw(data, clients_types, scores):
    SAVE_NAME = str(time.time()) + '.jpg'

    fig = plt.figure(figsize=(20, 6))
    fig1 = plt.subplot(121)
    for i, pt in enumerate(clients_types):
        if pt == 'Good update':
            plt.scatter(data[i, 0], data[i, 1], facecolors='none', edgecolors='black', marker='o', s=800,
                        label="Good update")
        else:
            plt.scatter(data[i, 0], data[i, 1], facecolors='black', edgecolors='black', marker='o', s=800,
                        label="Bad update")

    fig2 = plt.subplot(122)
    for i, pt in enumerate(clients_types):
        if scores[i] == 1:
            plt.scatter(data[i, 0], data[i, 1], color="orange", s=800, label="Good update")
        else:
            plt.scatter(data[i, 0], data[i, 1], color="blue", marker="x", linewidth=3, s=800, label="Bad update")

    plt.grid(False)
    # plt.show()
    plt.savefig(SAVE_NAME, bbox_inches='tight', pad_inches=0.1, dpi=400)
