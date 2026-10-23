"""
VisDrone 标签翻转攻击 × 四臂（FedAvg / DPFLA / Multi-Krum / Trimmed-Mean）

与 run_screen_server_attack.py 同一套口径与保护：
- v8m @1280 / train_bs16 / test_bs64 / AMP / 10 客户端 IID / 30% 恶意 / 行为率 1.0 / 不放大
- 攻击为**多标签全类循环反转** `{i: (i+1)%10}`、`flip_ratio=1.0`（用户 2026-10-23：VisDrone 用最强档，
  不要求与 CIFAR 映射一致）

用法：
  FL_RULE=fedavg|DPFLA|mkrum|tmean
  FL_GLOBAL_ROUNDS=5
"""

import os
import sys

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

# VisDrone 正式对比口径（防止漏设 env 回落 v8s@640）
os.environ.setdefault("YOLO_MODEL_SIZE", "m")
os.environ.setdefault("FL_IMG_SIZE", "1280")
os.environ.setdefault("FL_AMP", "1")
os.environ.setdefault("FL_TRAIN_BATCH_SIZE", "16")
os.environ.setdefault("FL_TEST_BATCH_SIZE", "64")

import time

import torch

# 容器 GPU 偶发瞬时不可用（CUDA Error 304）：重试，避免静默回落到 CPU 空转
for _ in range(60):
    if torch.cuda.is_available():
        break
    time.sleep(1)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA 不可用：已重试 60s，终止以免在 CPU 上跑完整实验")

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

RULE = (os.environ.get("FL_RULE") or "DPFLA").strip()
NUM_CLASSES = 10
# 全类循环反转（all-to-all）：i -> (i+1)%10，100% 覆盖，每类既源又目标（最强标签档）
FLIP_MAPPING = {i: (i + 1) % NUM_CLASSES for i in range(NUM_CLASSES)}
FLIP_RATIO = float(os.environ.get("FL_FLIP_RATIO", "1.0"))
MALICIOUS_BEHAVIOR_RATE = 1.0

_yolo_sz = apply_default_yolo_model_size()
_img_sz = os.environ.get("FL_IMG_SIZE")

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
print(f"VisDrone：多标签全翻转（{FLIP_MAPPING}）+ {RULE}")
print("=" * 70)
print(f"- {NUM_WORKERS} 客户端，恶意率 {MALICIOUS_RATE * 100:.0f}%，翻转比例 {FLIP_RATIO:.2f}")
print(f"- YOLO: yolov8{_yolo_sz}.pt @ {_img_sz}, bs={TRAIN_BATCH_SIZE}, test_bs={TEST_BATCH_SIZE}")
print(f"- {GLOBAL_ROUNDS} 轮 × {LOCAL_EPOCHS} epoch")
print("=" * 70)

start_time = time.time()

try:
    run_exp(
        num_workers=NUM_WORKERS,
        frac_workers=1.0,
        attack_type="label_flipping",
        rule=RULE,
        replace_method={"mapping": FLIP_MAPPING, "flip_ratio": FLIP_RATIO},
        dataset={"dataset_name": "VisDrone", "model_name": "YOLO"},
        malicious_rate=MALICIOUS_RATE,
        malicious_behavior_rate=MALICIOUS_BEHAVIOR_RATE,
        global_round=GLOBAL_ROUNDS,
        local_epoch=LOCAL_EPOCHS,
        untarget=False,
        experiment_tag=f"run_screen_label_{RULE}",
    )
    elapsed = time.time() - start_time
    print(f"\n✓ 测试完成，总耗时: {elapsed / 60:.1f} 分钟")
except KeyboardInterrupt:
    print("\n测试被中断")
except Exception as e:
    print(f"\n测试失败: {e}")
    import traceback

    traceback.print_exc()
