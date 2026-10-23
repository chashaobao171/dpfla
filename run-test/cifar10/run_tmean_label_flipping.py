"""
CIFAR10 + ResNet18CIFAR + 标签翻转攻击 + Trimmed-Mean 防御

口径与 run_no_defense_label_flipping.py 完全一致（仅聚合规则不同，单变量对比）：
10 客户端 / 15 轮 / 1 local epoch / lr 0.01 / BS 64 / IID；
30% 恶意、行为率 100%、固定映射 {5(dog)->3(cat)}、**FL_GRADIENT_SCALE=1（不放大）**。

Trimmed-Mean 参数：默认 trim_ratio = 恶意率（0.3，每端裁 3/10）；可用 FL_TRIM_RATIO 覆盖。
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
    FLIP_MAPPING,
    FLIP_RATIO,
    apply_arguments_patch,
    apply_common_env,
    print_config,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

apply_arguments_patch(device)
apply_common_env()

from server import run_exp

print_config(
    "标签翻转攻击",
    "Trimmed-Mean 防御",
    extra_lines=[
        f"- 设备: {device}",
        f"- 翻转映射: {FLIP_MAPPING}（flip_ratio={FLIP_RATIO}）",
        f"- 恶意行为率: {MALICIOUS_BEHAVIOR_RATE:.2f}",
        f"- 裁剪比例: {os.environ.get('FL_TRIM_RATIO', f'{MALICIOUS_RATE:.2f}')}（默认=恶意率）",
    ],
)

start_time = time.time()

try:
    run_exp(
        num_workers=NUM_WORKERS,
        frac_workers=FRAC_WORKERS,
        attack_type='label_flipping',
        rule='tmean',
        replace_method={'mapping': FLIP_MAPPING, 'flip_ratio': FLIP_RATIO},
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
