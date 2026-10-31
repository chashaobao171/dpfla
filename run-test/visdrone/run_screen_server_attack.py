"""
VisDrone 服务端注入攻击筛选/正式对比（MinMax / ALIE / SignFlipping）

与 CIFAR10 的 run_screen_server_attack.py 同一套攻击实现（attack_alg/server_attacks.py），
口径对齐 VisDrone 正式基线：v8m @1280 / BS16 / AMP / 10 客户端 IID / 30% 恶意 /
行为率 1.0 / 不放大。

用法（环境变量控制）：
  FL_SERVER_ATTACK=minmax|alie|signflipping   # 攻击范式（默认 minmax）
  FL_RULE=fedavg|DPFLA|mkrum|tmean            # 聚合/防御方法（默认 DPFLA）
  FL_GLOBAL_ROUNDS=5                          # 筛选用短轮（正式对比 15）
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
# 评估 batch 与既有 VisDrone 口径一致（256 在 1280 下会 OOM）
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

SERVER_ATTACK = (os.environ.get("FL_SERVER_ATTACK") or "minmax").strip().lower()
RULE = (os.environ.get("FL_RULE") or "DPFLA").strip()
MALICIOUS_BEHAVIOR_RATE = 1.0

_yolo_sz = apply_default_yolo_model_size()
_img_sz = os.environ.get("FL_IMG_SIZE")
_amp = os.environ.get("FL_AMP")

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
# 50 轮全量口径（2026-10-31 用户要求）：降低每轮本地步数，使 50 轮总本地更新量≈历史 5 轮（约 15 epoch）
if GLOBAL_ROUNDS >= 50:
    os.environ.setdefault("FL_LOCAL_STEPS", "12")
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(max(1, CPU_THREADS // 2))

from server import run_exp

print("=" * 70)
print(f"VisDrone：服务端注入攻击 {SERVER_ATTACK}（omniscient）+ {RULE}")
print("=" * 70)
print("\n配置:")
print(f"- {NUM_WORKERS} 客户端，恶意率 {MALICIOUS_RATE * 100:.0f}%（约 {int(MALICIOUS_RATE * NUM_WORKERS)} 个）")
print(f"- 攻击: {SERVER_ATTACK}（服务端聚合前注入，不放大）")
print(f"- 聚合: {RULE}")
print(f"- YOLO: yolov8{_yolo_sz}.pt @ {_img_sz}, bs={TRAIN_BATCH_SIZE}, AMP={_amp}")
print(f"- {GLOBAL_ROUNDS} 轮 × {LOCAL_EPOCHS} epoch（全局上限 15）")
print(f"- 学习率: LOCAL_LR={LOCAL_LR}, LR_MIN={LR_MIN}, schedule={LR_SCHEDULE}")
print("=" * 70)

start_time = time.time()

try:
    run_exp(
        num_workers=NUM_WORKERS,
        frac_workers=1.0,
        attack_type=SERVER_ATTACK,
        rule=RULE,
        replace_method=PLACEHOLDER_CLASSES,
        dataset={"dataset_name": "VisDrone", "model_name": "YOLO"},
        malicious_rate=MALICIOUS_RATE,
        malicious_behavior_rate=MALICIOUS_BEHAVIOR_RATE,
        global_round=GLOBAL_ROUNDS,
        local_epoch=LOCAL_EPOCHS,
        untarget=False,
        experiment_tag=f"run_screen_{SERVER_ATTACK}_{RULE}",
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
