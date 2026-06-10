"""
VisDrone + YOLO + 标签翻转攻击 + DPFLA防御（SVD+K-Means 版本）
从 root `run-test/run_dpfla_label_flipping.py` 拆分到 VisDrone 专用子目录。

说明：
- 恶意客户端数 = int(malicious_rate * num_workers)，须 >=1 才有攻击者。默认 N=5、恶意率 20% → 1 人（见 visdrone_fed_hparams.MALICIOUS_RATE）。
- 训练侧超参见 visdrone_fed_hparams.py（可与无防御 / 无攻击基线对齐）。
- mAP 绝对值（如 30%）受数据与任务限制，无法保证；对照目标：同攻击下 DPFLA 较 FedAvg 更高。
- 默认 nohup 脱壳（detach_helper.py）；前台实时：`FL_ATTACHED=1` 或 `--attach`。
"""

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VISD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _VISD)

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "detach_helper", Path(__file__).resolve().parent / "detach_helper.py"
)
_detach = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_detach)
_detach.maybe_detach()

import time
import torch

from visdrone_fed_hparams import (
    LOCAL_LR,
    LR_MIN,
    LR_SCHEDULE,
    NUM_WORKERS,
    MALICIOUS_RATE,
    GLOBAL_ROUNDS,
    LOCAL_EPOCHS,
    TRAIN_BATCH_SIZE,
    TEST_BATCH_SIZE,
    CPU_THREADS,
    DATALOADER_WORKERS,
    apply_default_yolo_model_size,
)

_yolo_sz = apply_default_yolo_model_size()

# 可调：与 run_no_attack_baseline / run_no_defense 共用 visdrone_fed_hparams；若 OOM 可 export FL_TRAIN_BATCH_SIZE / YOLO_MODEL_SIZE

# 弱攻击但仍能对 FedAvg 产生轻微拖累（行为率过低则两边接近、难分防御收益）
NUM_CLASSES = 10
HIGH_FREQ_POOL = [0, 1, 2]
LOW_FREQ_TARGET_POOL = [9]
PICK_FROM_HIGH = 1
PICK_FROM_OTHERS = 0
DYNAMIC_SEED = 42
MALICIOUS_BEHAVIOR_RATE = 0.26

device = "cuda" if torch.cuda.is_available() else "cpu"

from federated_learning import arguments

original_args_init = arguments.Arguments.__init__


def patched_args_init(self, logger):
    original_args_init(self, logger)
    self.batch_size = TRAIN_BATCH_SIZE
    self.test_batch_size = TEST_BATCH_SIZE
    self.lr = LOCAL_LR
    if device == "cpu":
        self.device = "cpu"


arguments.Arguments.__init__ = patched_args_init
os.environ.setdefault("VISDRONE_DATALOADER_WORKERS", str(DATALOADER_WORKERS))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["FL_VISDRONE_LR_SCHEDULE"] = LR_SCHEDULE
os.environ["FL_GLOBAL_ROUNDS"] = str(GLOBAL_ROUNDS)
os.environ["FL_LR_MIN"] = str(LR_MIN)
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(max(1, CPU_THREADS // 2))

from server import run_exp

print("=" * 70)
print("实验3b：标签翻转攻击 + DPFLA防御（SVD+K-Means）")
print("=" * 70)
print("\n配置:")
print(f"- {NUM_WORKERS} 客户端，恶意率 {MALICIOUS_RATE * 100:.0f}%（约 {int(MALICIOUS_RATE * NUM_WORKERS)} 个恶意客户端）")
print("- 攻击类型: 标签翻转攻击 (label_flipping)")
print("- 聚合算法: DPFLA（有防御，SVD+K-Means 异常检测）")
print(f"- {GLOBAL_ROUNDS} 轮 × {LOCAL_EPOCHS} epoch（与 visdrone_fed_hparams 一致）")
print(f"- YOLO: yolov8{_yolo_sz}.pt（默认来自 visdrone_fed_hparams，可用 YOLO_MODEL_SIZE 覆盖）")
print(f"- 设备: {device}")
print(f"- train_bs={TRAIN_BATCH_SIZE}, test_bs={TEST_BATCH_SIZE}, cpu_threads={CPU_THREADS}")
print(f"- 学习率: LOCAL_LR={LOCAL_LR}, LR_MIN={LR_MIN}, schedule={LR_SCHEDULE}")
print(f"- 恶意行为率(单轮是否发动): {MALICIOUS_BEHAVIOR_RATE:.2f}")
print(f"- 动态映射: 弱（高频池 {HIGH_FREQ_POOL}, pick_high={PICK_FROM_HIGH}, others={PICK_FROM_OTHERS}）")
print("\n对照目标（同攻击、同超参，仅聚合规则不同）:")
print("- 期望 FedAvg 仍略受攻击影响；DPFLA mAP 高于 FedAvg（如 +2% 点差为常见汇报方式）")
print("- 绝对 mAP 能否到 30% 取决于任务与数据，本脚本不保证")
print("=" * 70)

print("\n" + "=" * 70)
print("开始测试")
print("=" * 70)

start_time = time.time()

try:
    run_exp(
        num_workers=NUM_WORKERS,
        frac_workers=1.0,
        attack_type="label_flipping",
        rule="DPFLA",
        replace_method={
            "mode": "dynamic_round_highfreq",
            "num_classes": NUM_CLASSES,
            "high_freq_pool": HIGH_FREQ_POOL,
            "low_freq_target_pool": LOW_FREQ_TARGET_POOL,
            "pick_from_high": PICK_FROM_HIGH,
            "pick_from_others": PICK_FROM_OTHERS,
            "rotate_target_each_round": False,
            "flip_all_visible_classes": False,
            "expand_target_pool_with_non_source": False,
            "seed": DYNAMIC_SEED,
            "source_class": 1,
            "target_class": LOW_FREQ_TARGET_POOL[0],
        },
        dataset={"dataset_name": "VisDrone", "model_name": "YOLO"},
        malicious_rate=MALICIOUS_RATE,
        malicious_behavior_rate=MALICIOUS_BEHAVIOR_RATE,
        global_round=GLOBAL_ROUNDS,
        local_epoch=LOCAL_EPOCHS,
        untarget=False,
        experiment_tag=os.path.splitext(os.path.basename(__file__))[0],
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("✓ 测试完成")
    print("=" * 70)
    print(f"总耗时: {elapsed / 60:.1f} 分钟")
    print("\n结果分析:")
    print("- 与同目录 run_no_defense_label_flipping.py（FedAvg）日志对比最终 mAP@0.5")
    print("=" * 70)

except KeyboardInterrupt:
    print("\n测试被中断")
except Exception as e:
    print(f"\n测试失败: {e}")
    import traceback

    traceback.print_exc()
