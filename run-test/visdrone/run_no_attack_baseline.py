"""
VisDrone + YOLO + 无攻击 + FedAvg 基线

与 run_dpfla_label_flipping.py 使用相同训练侧超参（客户端数、轮数、batch、LR），
仅 attack_type=no_attack、malicious_rate=0，用于估计「这条联邦管线」的上界走势。

关于学习率 vs 第三方 YOLOv8_Federated_Learning-main：
  第三方 FedAvg_train.py 里 lr=1e-7 且走 ultralytics YOLO().train(lr0=lrf=lr)，含其内部 warmup/调度。
  主项目是手写前向反传 + PerturbedGradientDescent，**不能把 1e-7 直接当同一语义抄过来**；
  当前 LOCAL_LR 以主项目 Arguments 量级与 VisDrone 实验为准，第三方仅作「极小 lr 微调」的定性参考。

默认自动 nohup 脱壳（关终端仍跑，无第二份日志）；要前台实时：FL_ATTACHED=1 或 --attach。
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
    GLOBAL_ROUNDS,
    LOCAL_EPOCHS,
    TRAIN_BATCH_SIZE,
    TEST_BATCH_SIZE,
    CPU_THREADS,
    DATALOADER_WORKERS,
    apply_default_yolo_model_size,
)

_yolo_sz = apply_default_yolo_model_size()

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
print("VisDrone 基线：无攻击 + FedAvg（与 DPFLA / FedAvg+攻击 共用 visdrone_fed_hparams）")
print("=" * 70)
print(f"- YOLO: yolov8{_yolo_sz}.pt（可用 YOLO_MODEL_SIZE 覆盖）")
print(f"- {NUM_WORKERS} 客户端, {GLOBAL_ROUNDS} 轮 × {LOCAL_EPOCHS} local epoch")
print(f"- train_bs={TRAIN_BATCH_SIZE}, test_bs={TEST_BATCH_SIZE}, LOCAL_LR={LOCAL_LR}, schedule={LR_SCHEDULE}")
print("=" * 70)

start_time = time.time()

try:
    run_exp(
        num_workers=NUM_WORKERS,
        frac_workers=1.0,
        attack_type="no_attack",
        rule="fedavg",
        replace_method={"source_class": 1, "target_class": 7},
        dataset={"dataset_name": "VisDrone", "model_name": "YOLO"},
        malicious_rate=0.0,
        malicious_behavior_rate=0.0,
        global_round=GLOBAL_ROUNDS,
        local_epoch=LOCAL_EPOCHS,
        untarget=False,
        experiment_tag=os.path.splitext(os.path.basename(__file__))[0],
    )
    elapsed = time.time() - start_time
    print(f"\n✓ 测试完成，总耗时: {elapsed / 60:.1f} 分钟")
except KeyboardInterrupt:
    print("\n测试被中断")
except Exception as e:
    print(f"\n测试失败: {e}")
    import traceback

    traceback.print_exc()
