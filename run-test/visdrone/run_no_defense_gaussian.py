"""
VisDrone + YOLO + 高斯攻击 + 无防御（FedAvg）

攻击方式：gaussian —— 恶意客户端训练后、上传前，对模型更新叠加高斯噪声
（`federated_learning/attack_alg/gaussian_attack.py`），**纯加噪、不做任何放大**。
强度杠杆 = 噪声标准差 `FL_GAUSSIAN_STD`（默认 0.5，见 client.py 高斯分支）。

与 run_dpfla_* / run_no_defense_label_flipping.py 共用 visdrone_fed_hparams 的训练侧超参，
仅 attack_type 与 rule 不同，便于同口径对比。
默认 nohup 脱壳；前台实时：`FL_ATTACHED=1` 或 `--attach`。
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

# 高斯攻击强度：噪声 std（client.py 读取该环境变量；默认 0.5，不放大）
GAUSSIAN_STD = float(os.environ.get("FL_GAUSSIAN_STD", "0.5"))
MALICIOUS_BEHAVIOR_RATE = 1.0   # 100%：每个恶意客户端每轮必发动（高斯按参数键采样）

# 高斯不使用 source/target class；此占位 dict 仅用于满足 server.py 的 replace_method 校验，
# 保证不触发 KeyError（攻击类型为 gaussian 时客户端不会读取它）。
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
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(max(1, CPU_THREADS // 2))

from server import run_exp

print("=" * 70)
print("VisDrone：高斯攻击 + 无防御（FedAvg）")
print("=" * 70)
print("\n配置:")
print(f"- {NUM_WORKERS} 客户端，恶意率 {MALICIOUS_RATE * 100:.0f}%（约 {int(MALICIOUS_RATE * NUM_WORKERS)} 个恶意客户端）")
print("- 攻击类型: 高斯噪声攻击 (gaussian) —— 纯加噪，不放大")
print(f"- 噪声标准差 FL_GAUSSIAN_STD={GAUSSIAN_STD}（调强度用该环境变量）")
print("- 聚合算法: FedAvg（无防御）")
print(f"- YOLO: yolov8{_yolo_sz}.pt（可用 YOLO_MODEL_SIZE 覆盖）")
print(f"- {GLOBAL_ROUNDS} 轮 × {LOCAL_EPOCHS} epoch")
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
        rule="fedavg",
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
