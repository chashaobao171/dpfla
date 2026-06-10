"""
全量 VisDrone 联邦实验共用训练超参（非 smoke）。

默认按 **单卡 RTX 4090 48GB** 取稳健上限（尽量吃满算力、避免频繁 OOM）：
- 模型默认 yolov8s（可 export YOLO_MODEL_SIZE=m 进一步增大）
- batch / local epoch / 客户端数默认更保守，便于在 4090 上稳定跑满

启动前可 export：YOLO_MODEL_SIZE、FL_GLOBAL_ROUNDS、FL_TRAIN_BATCH_SIZE、FL_LOCAL_EPOCHS、
FL_NUM_WORKERS、FL_MALICIOUS_RATE（攻击脚本）、FL_CPU_THREADS、FL_DATALOADER_WORKERS 等。

Smoke 脚本不引用本模块。
"""

from __future__ import annotations

import os

# 4090 默认 small；需要更强可 export YOLO_MODEL_SIZE=m
DEFAULT_YOLO_MODEL_SIZE = "s"


def apply_default_yolo_model_size() -> str:
    cur = (os.environ.get("YOLO_MODEL_SIZE") or "").strip().lower()
    if cur in ("n", "s", "m", "l", "x"):
        return cur
    os.environ["YOLO_MODEL_SIZE"] = DEFAULT_YOLO_MODEL_SIZE
    return DEFAULT_YOLO_MODEL_SIZE


# 4090 默认：学习率不过激，先保证稳定收敛；需要更猛再通过 FL_LOCAL_LR 调大
LOCAL_LR = float(os.environ.get("FL_LOCAL_LR", "0.01"))      # 2e-4 → 0.01（配合 FedAvgM）
LR_MIN = float(os.environ.get("FL_LR_MIN", "5e-6"))
LR_SCHEDULE = "constant"  # cosine 在 Batch 2 加

NUM_WORKERS = int(os.environ.get("FL_NUM_WORKERS", "10"))
# 攻击脚本默认 10%（N=10 → 1 人）；若改为 N=5，记得配套把 FL_MALICIOUS_RATE 调到 0.2
MALICIOUS_RATE = float(os.environ.get("FL_MALICIOUS_RATE", "0.1"))

GLOBAL_ROUNDS = int(os.environ.get("FL_GLOBAL_ROUNDS", "50"))   # 20 → 50
LOCAL_EPOCHS = int(os.environ.get("FL_LOCAL_EPOCHS", "1"))     # 10 → 1（高频低深）
TRAIN_BATCH_SIZE = int(os.environ.get("FL_TRAIN_BATCH_SIZE", "64"))
TEST_BATCH_SIZE = int(os.environ.get("FL_TEST_BATCH_SIZE", "256"))
CPU_THREADS = int(os.environ.get("FL_CPU_THREADS", "16"))
DATALOADER_WORKERS = int(os.environ.get("FL_DATALOADER_WORKERS", "6"))
