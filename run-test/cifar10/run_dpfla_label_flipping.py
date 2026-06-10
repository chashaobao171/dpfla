"""
CIFAR10 + CNNCifar10 + 标签翻转攻击 + DPFLA(SVD+KMeans) 防御
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import time

NUM_WORKERS = 7
MALICIOUS_RATE = 0.30
GLOBAL_ROUNDS = 8
LOCAL_EPOCHS = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from federated_learning import arguments
original_args_init = arguments.Arguments.__init__


def patched_args_init(self, logger):
    original_args_init(self, logger)
    if device == 'cpu':
        self.device = 'cpu'


arguments.Arguments.__init__ = patched_args_init

from server import run_exp

print("=" * 70)
print("CIFAR10：标签翻转攻击 + DPFLA(SVD+KMeans)")
print("=" * 70)
print(f"- 客户端: {NUM_WORKERS}")
print(f"- 恶意率: {MALICIOUS_RATE*100:.0f}%")
print(f"- 轮次: {GLOBAL_ROUNDS} x {LOCAL_EPOCHS}")
print(f"- 设备: {device}")
print("=" * 70)

start_time = time.time()

try:
    run_exp(
        num_workers=NUM_WORKERS,
        frac_workers=1.0,
        attack_type='label_flipping',
        rule='DPFLA',
        replace_method={'source_class': 1, 'target_class': 7},
        dataset={'dataset_name': 'CIFAR10', 'model_name': 'CNNCifar10'},
        malicious_rate=MALICIOUS_RATE,
        malicious_behavior_rate=1.0,
        global_round=GLOBAL_ROUNDS,
        local_epoch=LOCAL_EPOCHS,
        untarget=True,
        experiment_tag=os.path.splitext(os.path.basename(__file__))[0]
    )
    elapsed = time.time() - start_time
    print(f"\n✓ 测试完成，总耗时: {elapsed / 60:.1f} 分钟")
except KeyboardInterrupt:
    print("\n测试被中断")
except Exception as e:
    print(f"\n测试失败: {e}")
    import traceback
    traceback.print_exc()

