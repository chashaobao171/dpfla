"""快速对照实验入口（CIFAR-10）：给老师演示用

实验1: 无攻击 + 无防御 (FedAvg, no_attack)
实验2: 有攻击 + 有防御 (label_flipping 30% 恶意客户端 + DPFLA 防御)
实验3: 有攻击 + 无防御 (label_flipping 30% + FedAvg) —— 展示攻击破坏性

用法:
  python run_demo_cifar.py 1     # 只跑实验1
  python run_demo_cifar.py 2     # 只跑实验2
  python run_demo_cifar.py 3     # 只跑实验3
  python run_demo_cifar.py all   # 顺序跑三个实验
"""
import sys

from server import run_exp
from federated_learning.my_dict import run_cifar10, replace_dog_with_cat, fed_avg, DPFLA

# ===== 学习率补丁（仅本次演示生效，不改项目源码）=====
# 项目默认 lr=0.0001 是为保护 VisDrone 预训练 YOLO 权重设置的，
# 对从零训练的 CIFAR-10 CNN 太小，此处运行时覆盖为 0.01。
from federated_learning.arguments import Arguments

_orig_init = Arguments.__init__


def _patched_init(self, logger):
    _orig_init(self, logger)
    self.lr = 0.01


Arguments.__init__ = _patched_init

# ===== 快速演示参数（与 MNIST demo 口径一致）=====
NUM_WORKERS = 10            # 客户端数
FRAC_WORKERS = 1.0          # 每轮全部参与
GLOBAL_ROUND = 15           # 全局轮数
LOCAL_EPOCH = 1             # 本地训练 1 轮
MALICIOUS_RATE = 0.3        # 恶意客户端比例
MALICIOUS_BEHAVIOR_RATE = 1.0
UNTARGET = False

DATASET = run_cifar10()
REPLACE_METHOD = replace_dog_with_cat()  # 标签翻转 dog(5) -> cat(3)


def exp1_baseline():
    """实验1: 无攻击无防御"""
    run_exp(NUM_WORKERS, FRAC_WORKERS, 'no_attack', fed_avg(), REPLACE_METHOD, DATASET,
            [0], MALICIOUS_BEHAVIOR_RATE, GLOBAL_ROUND, LOCAL_EPOCH, UNTARGET,
            experiment_tag='cifar1_fedavg_noattack')


def exp2_attack_defense():
    """实验2: 有攻击(标签翻转, 30%恶意) + 有防御(DPFLA)"""
    run_exp(NUM_WORKERS, FRAC_WORKERS, 'label_flipping', DPFLA(), REPLACE_METHOD, DATASET,
            [MALICIOUS_RATE], MALICIOUS_BEHAVIOR_RATE, GLOBAL_ROUND, LOCAL_EPOCH, UNTARGET,
            experiment_tag='cifar2_attack_dpfla')


def exp3_attack_no_defense():
    """实验3: 有攻击(标签翻转, 30%恶意) + 无防御(FedAvg)"""
    run_exp(NUM_WORKERS, FRAC_WORKERS, 'label_flipping', fed_avg(), REPLACE_METHOD, DATASET,
            [MALICIOUS_RATE], MALICIOUS_BEHAVIOR_RATE, GLOBAL_ROUND, LOCAL_EPOCH, UNTARGET,
            experiment_tag='cifar3_attack_nodefense')


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('1', 'all'):
        exp1_baseline()
    if which in ('2', 'all'):
        exp2_attack_defense()
    if which in ('3', 'all'):
        exp3_attack_no_defense()
    print('DONE:', which)
