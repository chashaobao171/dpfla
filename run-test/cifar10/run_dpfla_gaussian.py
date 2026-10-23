"""
CIFAR10 + ResNet18CIFAR + 高斯攻击 + DPFLA 防御（SVD+KMeans 主路径）

攻击口径与 run_no_defense_gaussian.py 完全一致（仅聚合规则不同，单变量对比）：
10 客户端 / 15 轮 / 1 local epoch / lr 0.01 / BS 64 / IID；30% 恶意 / 行为率 100% / 不放大。
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
    MALICIOUS_RATE,
    MALICIOUS_BEHAVIOR_RATE,
    GAUSSIAN_STD,
    apply_arguments_patch,
    apply_common_env,
    print_config,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

apply_arguments_patch(device)
apply_common_env()

from server import run_exp

PLACEHOLDER_CLASSES = {'source_class': 0, 'target_class': 1}

print_config(
    "高斯攻击",
    "DPFLA 防御（SVD+KMeans，k=2）",
    extra_lines=[
        f"- 设备: {device}",
        f"- 噪声标准差 FL_GAUSSIAN_STD={GAUSSIAN_STD}（调强度用该环境变量）",
        f"- 恶意行为率: {MALICIOUS_BEHAVIOR_RATE:.2f}",
    ],
)

start_time = time.time()

try:
    run_exp(
        num_workers=NUM_WORKERS,
        frac_workers=FRAC_WORKERS,
        attack_type='gaussian',
        rule='DPFLA',
        replace_method=PLACEHOLDER_CLASSES,
        dataset={'dataset_name': DATASET_NAME, 'model_name': MODEL_NAME},
        malicious_rate=MALICIOUS_RATE,
        malicious_behavior_rate=MALICIOUS_BEHAVIOR_RATE,
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
