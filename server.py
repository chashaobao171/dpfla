from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.fl_core import FL
from federated_learning.my_dict import get_cifar10_labels_dict, get_mnist_labels_dict, get_visdrone_labels_dict
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import time

BEIJING_TZ = ZoneInfo("Asia/Shanghai")


def _set_process_timezone_to_beijing():
    """Force process-local timezone for log timestamps and filenames."""
    if os.environ.get("TZ") != "Asia/Shanghai":
        os.environ["TZ"] = "Asia/Shanghai"
        if hasattr(time, "tzset"):
            time.tzset()


def generate_log_file(args, experiment_tag=None, start_time=None):
    """
    生成简洁的日志文件名，仅包含：
    - 数据集子目录（按 dataset_name 自动分桶）
    - 运行脚本名（experiment_tag）
    - 时间戳（到分钟，格式 YYYYMMDD_HHMM）
    示例:
    logs_3/visdrone/run_dpfla_label_flipping_20260331_1255.log
    """
    if start_time is None:
        start_time = datetime.now(BEIJING_TZ)

    timestamp = start_time.strftime('%Y%m%d_%H%M')
    tag = experiment_tag or "unknown_script"
    dataset_name = (args.get_dataset_name() or "unknown_dataset").strip().lower()
    log_dir = os.path.join("./logs_3", dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    savepath = os.path.join(log_dir, f"{tag}_{timestamp}.log")
    return savepath


def run_exp(num_workers, frac_workers, attack_type, rule, replace_method, dataset, malicious_rate,
            malicious_behavior_rate, global_round, local_epoch, untarget, experiment_tag=None):
    _set_process_timezone_to_beijing()
    args = Arguments(logger)
    args.set_num_workers(num_workers)
    args.set_frac_workers(frac_workers)
    args.set_rule(rule)
    # 支持三种 label_flipping 配置：
    # - 传统固定翻转：{'source_class': 1, 'target_class': 7}
    # - 固定 mapping：{'mapping': {...}, 'source_class': 1, 'target_class': 7}
    # - 动态策略配置：{'mode': 'dynamic_round_highfreq', ...}
    label_flip_mapping = replace_method if isinstance(replace_method, dict) else None
    if isinstance(replace_method, dict) and "mapping" in replace_method:
        fixed_mapping = replace_method.get("mapping") or {}
        if fixed_mapping:
            first_src = next(iter(fixed_mapping.keys()))
            first_tgt = fixed_mapping[first_src]
            args.set_source_class(int(first_src))
            args.set_target_class(int(first_tgt))
        else:
            args.set_source_class(replace_method.get('source_class', 0))
            args.set_target_class(replace_method.get('target_class', 1))
    else:
        args.set_source_class(replace_method['source_class'])
        args.set_target_class(replace_method['target_class'])
    args.set_attack_type(attack_type)
    args.set_malicious_rate(malicious_rate)
    args.set_malicious_behavior_rate(malicious_behavior_rate)
    args.set_global_rounds(global_round)
    args.set_local_epochs(local_epoch)

    args.set_dataset_name(dataset['dataset_name'])
    args.set_model_name(dataset['model_name'])
    if dataset['dataset_name'] == 'MNIST':
        args.set_labels_dict(get_mnist_labels_dict())
    elif dataset['dataset_name'] == 'CIFAR10':
        args.set_labels_dict(get_cifar10_labels_dict())
    elif dataset['dataset_name'] == 'VisDrone':
        args.set_labels_dict(get_visdrone_labels_dict())
    else:
        raise Exception('Undefined dataset!!!')
    
    # 根据数据集类型更新损失函数
    args.update_loss_function_for_dataset()

    log_files = generate_log_file(args, experiment_tag=experiment_tag)
    # Initialize logger
    handler = logger.add(log_files, enqueue=True)

    args.log()

    # VisDrone数据集路径配置（主项目强制走 autodl-tmp 目录）
    # 约束：
    # - 主项目统一使用 /root/autodl-tmp/data/visdrone 下的 train/val
    # - 第三方项目如需使用 /root/autodl-tmp/data/images|labels，不在这里处理
    import os
    if os.path.exists('/root/autodl-tmp/data/visdrone'):
        visdrone_root_path = '/root/autodl-tmp/data/visdrone'  # autodl-tmp VisDrone 目录
        print("✓ 主项目使用 autodl-tmp VisDrone 目录: /root/autodl-tmp/data/visdrone")
    elif os.path.exists('/home/featurize/data/visdrone'):
        visdrone_root_path = '/home/featurize/data/visdrone'  # featurize 旧路径
        print("✓ 主项目使用 featurize VisDrone 目录: /home/featurize/data/visdrone")
    else:
        visdrone_root_path = 'D:/Pycharmworkplace/visdrone'  # 本地路径
        print("✓ 主项目使用本地原始 VisDrone 目录")
    
    # 处理malicious_rate可能是列表的情况
    malicious_rate_value = args.get_malicious_rate()
    if isinstance(malicious_rate_value, list):
        attackers_ratio = malicious_rate_value[0] if malicious_rate_value else 0
    else:
        attackers_ratio = malicious_rate_value
    
    flEnv = FL(dataset_name=args.get_dataset_name(), model_name=args.get_model_name(), dd_type=args.get_dd_type(),
               num_clients=args.get_num_workers(), frac_clients=args.get_frac_workers(), seed=args.get_seed(),
               test_batch_size=args.get_test_batch_size(), criterion=args.get_loss_function(),
               global_rounds=args.get_global_rounds(),
               local_epochs=args.get_local_epochs(), local_bs=args.get_batch_size(), local_lr=args.get_lr(),
               local_momentum=args.get_momentum(), labels_dict=args.get_labels_dict(), device=args.get_device(),
               attackers_ratio=attackers_ratio,
               class_per_client=args.get_class_per_workers(), samples_per_class=args.get_samples_per_class(),
               rate_unbalance=args.get_rate_unbalance(), alpha=args.get_alpha(), source_class=args.get_source_class(),
               visdrone_root_path=visdrone_root_path)

    flEnv.run_experiment(attack_type=args.get_attack_type(), malicious_behavior_rate=args.get_malicious_behavior_rate(),
                         source_class=args.get_source_class(), target_class=args.get_target_class(),
                         rule=args.get_rule(), resume=False, model_name=args.get_model_name(), untarget=untarget,
                         label_flip_mapping=label_flip_mapping)

    logger.remove(handler)
