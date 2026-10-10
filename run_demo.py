"""快速对照实验入口（MNIST）：给老师演示用

实验1: 无攻击 + 无防御 (FedAvg, no_attack)
实验2: 有攻击 + 有防御 (label_flipping 20% 恶意客户端 + DPFLA 防御)

用法:
  python run_demo.py 1     # 只跑实验1
  python run_demo.py 2     # 只跑实验2
  python run_demo.py all   # 顺序跑两个实验
"""
import sys

from server import run_exp
from federated_learning.my_dict import run_mnist, replace_1_with_7, fed_avg, DPFLA

# ===== 学习率补丁（仅本次演示生效，不改项目源码）=====
# 项目默认 lr=0.0001 是为保护 VisDrone 预训练 YOLO 权重设置的，
# 对从零训练的 MNIST CNN 太小（15 轮只能到 ~13%）。
# 此处运行时覆盖为 0.01，与历史上达到 95% 的 MNIST 实验一致。
from federated_learning.arguments import Arguments

_orig_init = Arguments.__init__


def _patched_init(self, logger):
    _orig_init(self, logger)
    self.lr = 0.01


Arguments.__init__ = _patched_init

# ===== 快速演示参数（刻意调小，追求快速出结果）=====
# 口径与 5 月历史实验一致（10 客户端 / 30% 恶意），保证攻击强度足以形成对照
NUM_WORKERS = 10            # 客户端数
FRAC_WORKERS = 1.0          # 每轮全部参与
GLOBAL_ROUND = 15           # 全局轮数
LOCAL_EPOCH = 1             # 本地训练 1 轮
MALICIOUS_RATE = 0.3        # 恶意客户端比例
MALICIOUS_BEHAVIOR_RATE = 1.0
UNTARGET = False

DATASET = run_mnist()
REPLACE_METHOD = replace_1_with_7()  # 标签翻转 1 -> 7


def exp1_baseline():
    """实验1: 无攻击无防御"""
    run_exp(NUM_WORKERS, FRAC_WORKERS, 'no_attack', fed_avg(), REPLACE_METHOD, DATASET,
            [0], MALICIOUS_BEHAVIOR_RATE, GLOBAL_ROUND, LOCAL_EPOCH, UNTARGET,
            experiment_tag='demo1_mnist_fedavg_noattack')


def exp2_attack_defense():
    """实验2: 有攻击(标签翻转, 20%恶意) + 有防御(DPFLA)"""
    run_exp(NUM_WORKERS, FRAC_WORKERS, 'label_flipping', DPFLA(), REPLACE_METHOD, DATASET,
            [MALICIOUS_RATE], MALICIOUS_BEHAVIOR_RATE, GLOBAL_ROUND, LOCAL_EPOCH, UNTARGET,
            experiment_tag='demo2_mnist_attack_dpfla')


def exp3_attack_no_defense():
    """实验3: 有攻击(标签翻转, 20%恶意) + 无防御(FedAvg) —— 展示攻击破坏性"""
    run_exp(NUM_WORKERS, FRAC_WORKERS, 'label_flipping', fed_avg(), REPLACE_METHOD, DATASET,
            [MALICIOUS_RATE], MALICIOUS_BEHAVIOR_RATE, GLOBAL_ROUND, LOCAL_EPOCH, UNTARGET,
            experiment_tag='demo3_mnist_attack_nodefense')


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('1', 'all'):
        exp1_baseline()
    if which in ('2', 'all'):
        exp2_attack_defense()
    if which in ('3', 'all'):
        exp3_attack_no_defense()
    print('DONE:', which)
