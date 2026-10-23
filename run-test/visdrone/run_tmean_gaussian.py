"""
VisDrone + YOLO + 高斯攻击 + Trimmed-Mean 防御

攻击口径与 run_no_defense_gaussian.py 完全一致（仅聚合规则不同，单变量对比）：
高斯 σ 由 FL_GAUSSIAN_STD 控制（本项目 VisDrone ② 的可用档为 0.02）；
30% 恶意 / 行为率 100% / 不放大 / 训练侧超参共用 visdrone_fed_hparams。

Trimmed-Mean 参数：默认 trim_ratio = 恶意率（0.3，每端裁 3/10）；可用 FL_TRIM_RATIO 覆盖。
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

GAUSSIAN_STD = float(os.environ.get("FL_GAUSSIAN_STD", "0.5"))
MALICIOUS_BEHAVIOR_RATE = 1.0

PLACEHOLDER_CLASSES = {"source_class": 0, "target_class": 1}

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
os.environ.setdefault("FL_GRADIENT_SCALE", "1")
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(max(1, CPU_THREADS // 2))

from server import run_exp

print("=" * 70)
print("VisDrone：高斯攻击 + Trimmed-Mean 防御")
print("=" * 70)
print("\n配置:")
print(f"- {NUM_WORKERS} 客户端，恶意率 {MALICIOUS_RATE * 100:.0f}%（约 {int(MALICIOUS_RATE * NUM_WORKERS)} 个恶意客户端）")
print("- 攻击类型: 高斯噪声攻击 (gaussian) —— 纯加噪，不放大")
print(f"- 噪声标准差 FL_GAUSSIAN_STD={GAUSSIAN_STD}（VisDrone 可用档 0.02）")
print(f"- 聚合算法: Trimmed-Mean（裁剪比例 {os.environ.get('FL_TRIM_RATIO', f'{MALICIOUS_RATE:.2f}')}）")
print(f"- YOLO: yolov8{_yolo_sz}.pt（可用 YOLO_MODEL_SIZE 覆盖）")
print(f"- {GLOBAL_ROUNDS} 轮 × {LOCAL_EPOCHS} epoch（全局上限 15）")
print(f"- train_bs={TRAIN_BATCH_SIZE}, test_bs={TEST_BATCH_SIZE}, cpu_threads={CPU_THREADS}")
print(f"- 学习率: LOCAL_LR={LOCAL_LR}, LR_MIN={LR_MIN}, schedule={LR_SCHEDULE}")
print(f"- 恶意行为率: {MALICIOUS_BEHAVIOR_RATE:.2f}")
print("=" * 70)

start_time = time.time()

try:
    run_exp(
        num_workers=NUM_WORKERS,
        frac_workers=1.0,
        attack_type="gaussian",
        rule="tmean",
        replace_method=PLACEHOLDER_CLASSES,
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
except KeyboardInterrupt:
    print("\n测试被中断")
except Exception as e:
    print(f"\n测试失败: {e}")
    import traceback

    traceback.print_exc()
