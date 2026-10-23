"""
CIFAR10 联邦实验共用训练超参（与 run-test/visdrone/visdrone_fed_hparams.py 同构）。

默认口径（2026-10-23 定稿，见 MEMORY/08_Roadmap.md §2）：
- 模型 **ResNet18CIFAR**（CIFAR 专用 stem：3x3/s1、无 maxpool；不用 torchvision ImageNet 版）
- 10 客户端 / **15 轮（全局上限）** / 1 local epoch / BS 64 / IID
- lr 0.01（from-scratch；与唯一保留的 CIFAR 基线归档一致）
- 攻击口径：恶意率 30% / 行为率 100% / 标签翻转 {5(dog) -> 3(cat)} / **不放大梯度**

可覆盖的环境变量：
FL_NUM_WORKERS / FL_GLOBAL_ROUNDS / FL_LOCAL_EPOCHS / FL_TRAIN_BATCH_SIZE /
FL_TEST_BATCH_SIZE / FL_LOCAL_LR / FL_MOMENTUM / FL_MALICIOUS_RATE /
FL_MALICIOUS_BEHAVIOR_RATE / FL_FLIP_RATIO / FL_GAUSSIAN_STD / FL_GRADIENT_SCALE /
CIFAR_MODEL
"""

from __future__ import annotations

import os

DATASET_NAME = "CIFAR10"
MODEL_NAME = (os.environ.get("CIFAR_MODEL") or "ResNet18CIFAR").strip() or "ResNet18CIFAR"

NUM_WORKERS = int(os.environ.get("FL_NUM_WORKERS", "10"))
FRAC_WORKERS = float(os.environ.get("FL_FRAC_WORKERS", "1.0"))
GLOBAL_ROUNDS = int(os.environ.get("FL_GLOBAL_ROUNDS", "15"))   # 全局上限 15（2026-10-23）
LOCAL_EPOCHS = int(os.environ.get("FL_LOCAL_EPOCHS", "1"))
TRAIN_BATCH_SIZE = int(os.environ.get("FL_TRAIN_BATCH_SIZE", "64"))
TEST_BATCH_SIZE = int(os.environ.get("FL_TEST_BATCH_SIZE", "1000"))
LOCAL_LR = float(os.environ.get("FL_LOCAL_LR", "0.01"))
MOMENTUM = float(os.environ.get("FL_MOMENTUM", "0.9"))

MALICIOUS_RATE = float(os.environ.get("FL_MALICIOUS_RATE", "0.3"))
MALICIOUS_BEHAVIOR_RATE = float(os.environ.get("FL_MALICIOUS_BEHAVIOR_RATE", "1.0"))

NUM_CLASSES = 10
# 固定映射（原历史归档口径）：dog(5) -> cat(3)；不使用 untarget，避免目标类每轮漂移
FLIP_MAPPING = {5: 3}
FLIP_RATIO = float(os.environ.get("FL_FLIP_RATIO", "1.0"))

# 高斯攻击噪声 std（CIFAR 未测过，默认先给 0.05；实际档位按 5 轮阶梯筛）
GAUSSIAN_STD = float(os.environ.get("FL_GAUSSIAN_STD", "0.05"))


def apply_common_env() -> None:
    """统一攻击口径：CIFAR 新实验一律不放大梯度（代码默认 15 不改，仅用 env 覆盖）。"""
    os.environ.setdefault("FL_GRADIENT_SCALE", "1")


def apply_arguments_patch(device: str) -> None:
    """
    覆盖 Arguments 默认值：lr / batch / test batch / momentum。
    （Arguments 默认 lr=1e-4 是为保护 VisDrone 预训练 YOLO 权重设置的，对 from-scratch CIFAR 过小）
    """
    from federated_learning import arguments

    if getattr(arguments.Arguments, "_cifar_patch_applied", False):
        return

    original_init = arguments.Arguments.__init__

    def patched_init(self, logger):
        original_init(self, logger)
        self.batch_size = TRAIN_BATCH_SIZE
        self.test_batch_size = TEST_BATCH_SIZE
        self.lr = LOCAL_LR
        self.momentum = MOMENTUM
        if device == "cpu":
            self.device = "cpu"

    arguments.Arguments.__init__ = patched_init
    arguments.Arguments._cifar_patch_applied = True


def print_config(attack_desc: str, rule_desc: str, extra_lines=(), malicious_rate=None) -> None:
    """统一的配置打印，便于日志参数块逐项比对。"""
    mr = MALICIOUS_RATE if malicious_rate is None else malicious_rate
    print("=" * 70)
    print(f"CIFAR10 | {attack_desc} | {rule_desc}")
    print("=" * 70)
    print(f"- 模型: {MODEL_NAME}")
    print(f"- {NUM_WORKERS} 客户端，恶意率 {mr * 100:.0f}%"
          f"（约 {int(mr * NUM_WORKERS)} 个恶意）")
    print(f"- 轮次: {GLOBAL_ROUNDS} x {LOCAL_EPOCHS} epoch（上限 15）")
    print(f"- train_bs={TRAIN_BATCH_SIZE}, test_bs={TEST_BATCH_SIZE}")
    print(f"- lr={LOCAL_LR}, momentum={MOMENTUM}, 数据分布=IID")
    print(f"- 梯度放大 FL_GRADIENT_SCALE={os.environ.get('FL_GRADIENT_SCALE', '15.0')}"
          f"（1=不放大）")
    for line in extra_lines:
        print(line)
    print("=" * 70)
