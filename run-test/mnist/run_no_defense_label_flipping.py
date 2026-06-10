"""
MNIST + CNNMNIST + 强化标签翻转攻击 + FedAvg（无防御）
强化策略：
  1. All-to-All 全类别循环翻转（0→1, 1→2, ..., 9→0）—— 100% 标签污染
  2. 梯度放大（×3）—— 放大攻击信号
  3. 恶意率 50% —— 恶意与诚实势均力敌
  4. 始终激活（malicious_behavior_rate=1.0）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import time

NUM_WORKERS = 10
MALICIOUS_RATE = 0.30
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
print("MNIST：强化标签翻转攻击 + FedAvg（无防御）")
print("=" * 70)

start_time = time.time()

try:
    run_exp(
        num_workers=NUM_WORKERS,
        frac_workers=1.0,
        attack_type='label_flipping',
        rule='fedavg',
        replace_method={'mapping': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 0}},
        dataset={'dataset_name': 'MNIST', 'model_name': 'CNNMNIST'},
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
