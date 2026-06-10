"""
MNIST + CNNMNIST + 无攻击 + FedAvg 基线
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import time

NUM_WORKERS = 10
GLOBAL_ROUNDS = 20
LOCAL_EPOCHS = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from federated_learning import arguments
original_args_init = arguments.Arguments.__init__


def patched_args_init(self, logger):
    original_args_init(self, logger)
    self.lr = 0.01
    if device == 'cpu':
        self.device = 'cpu'


arguments.Arguments.__init__ = patched_args_init

from server import run_exp

print("=" * 70)
print("MNIST 基线：无攻击 + FedAvg")
print("=" * 70)

start_time = time.time()

try:
    run_exp(
        num_workers=NUM_WORKERS,
        frac_workers=1.0,
        attack_type='no_attack',
        rule='fedavg',
        replace_method={'source_class': 1, 'target_class': 7},
        dataset={'dataset_name': 'MNIST', 'model_name': 'CNNMNIST'},
        malicious_rate=0.0,
        malicious_behavior_rate=0.0,
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
