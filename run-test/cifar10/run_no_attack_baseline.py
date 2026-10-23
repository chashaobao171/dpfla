"""
CIFAR10 + ResNet18CIFAR + 无攻击 + FedAvg 基线（15 轮）

口径见 cifar_fed_hparams.py（10 客户端 / 15 轮 / 1 local epoch / lr 0.01 / BS 64 / IID）。
"""

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CIF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _CIF)

import time
import torch

from cifar_fed_hparams import (
    DATASET_NAME,
    MODEL_NAME,
    NUM_WORKERS,
    FRAC_WORKERS,
    GLOBAL_ROUNDS,
    LOCAL_EPOCHS,
    FLIP_MAPPING,
    apply_arguments_patch,
    apply_common_env,
    print_config,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

apply_arguments_patch(device)
apply_common_env()

from server import run_exp

print_config("无攻击", "FedAvg（基线）", extra_lines=[f"- 设备: {device}"], malicious_rate=0.0)

start_time = time.time()

try:
    run_exp(
        num_workers=NUM_WORKERS,
        frac_workers=FRAC_WORKERS,
        attack_type='no_attack',
        rule='fedavg',
        replace_method={'source_class': 5, 'target_class': 3},
        dataset={'dataset_name': DATASET_NAME, 'model_name': MODEL_NAME},
        malicious_rate=0.0,
        malicious_behavior_rate=0.0,
        global_round=GLOBAL_ROUNDS,
        local_epoch=LOCAL_EPOCHS,
        untarget=False,
        experiment_tag=os.path.splitext(os.path.basename(__file__))[0],
    )
    elapsed = time.time() - start_time
    print(f"\n✓ 测试完成，总耗时: {elapsed / 60:.1f} 分钟")
except KeyboardInterrupt:
    print("\n测试被中断")
except Exception as e:
    print(f"\n测试失败: {e}")
    import traceback

    traceback.print_exc()
