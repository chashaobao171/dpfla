"""
全量 VisDrone 联邦实验共用训练超参（非 smoke）。

默认按 **单卡 32GB GPU** 取稳健上限（尽量吃满算力、避免频繁 OOM）：
- 2026-11-09 起默认 **YOLO26s**（边缘终端首选）；YOLOv8s 仅作备选
- 历史复现：`YOLO_MODEL_FAMILY=8 YOLO_MODEL_SIZE=m`（v8m）或 `...=8 ...=s`（v8s）
- batch / local epoch / 客户端数默认更保守，便于稳定跑满

启动前可 export：YOLO_MODEL_FAMILY、YOLO_MODEL_SIZE、FL_GLOBAL_ROUNDS、FL_TRAIN_BATCH_SIZE、
FL_LOCAL_EPOCHS、FL_NUM_WORKERS、FL_MALICIOUS_RATE（攻击脚本）、FL_CPU_THREADS、FL_DATALOADER_WORKERS 等。

Smoke 脚本不引用本模块。
"""

from __future__ import annotations

import json
import os

# 2026-11-09 起：YOLO26s 为边缘终端首选；YOLOv8s 仅作备选。
# 历史 v8m/v8s 复现需显式 export YOLO_MODEL_FAMILY=8 YOLO_MODEL_SIZE=m/s。
DEFAULT_YOLO_MODEL_FAMILY = "26"


def apply_default_yolo_model_family() -> str:
    cur = (os.environ.get("YOLO_MODEL_FAMILY") or "").strip()
    if cur in ("8", "26"):
        return cur
    os.environ["YOLO_MODEL_FAMILY"] = DEFAULT_YOLO_MODEL_FAMILY
    return DEFAULT_YOLO_MODEL_FAMILY


def yolo_weights_name(size: str | None = None) -> str:
    """返回当前 family/size 对应的权重文件名（yolo26s.pt / yolov8s.pt）。"""
    fam = apply_default_yolo_model_family()
    sz = (size or apply_default_yolo_model_size()).strip().lower()
    return f"yolo26{sz}.pt" if fam == "26" else f"yolov8{sz}.pt"


# 默认 small；需要更强可 export YOLO_MODEL_SIZE=m
DEFAULT_YOLO_MODEL_SIZE = "s"


def apply_default_yolo_model_size() -> str:
    apply_default_yolo_model_family()
    cur = (os.environ.get("YOLO_MODEL_SIZE") or "").strip().lower()
    if cur in ("n", "s", "m", "l", "x"):
        return cur
    os.environ["YOLO_MODEL_SIZE"] = DEFAULT_YOLO_MODEL_SIZE
    return DEFAULT_YOLO_MODEL_SIZE


# VisDrone 10 类：0 pedestrian, 1 people, 2 bicycle, 3 car, 4 van,
# 5 truck, 6 tricycle, 7 awning-tricycle, 8 bus, 9 motor
LABEL_FLIP_PRESETS = {
    # 全类循环：i -> (i+1)%10（原默认，10 类全覆盖）
    "full_cycle": {i: (i + 1) % 10 for i in range(10)},
    "full_cycle_reverse": {i: (i - 1) % 10 for i in range(10)},
    # 5 对互换：0<->1, 2<->3, ...
    "pair_swap": {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 9, 9: 8},
    # 只翻 1 类 / 2 类 / 3 类 / 4 类，用于“反转个数”筛选
    "single_pair": {0: 1},
    "one_class": {0: 1},
    "two_class": {0: 1, 1: 0},
    "three_class": {0: 1, 1: 2, 2: 0},
    "four_class": {0: 1, 1: 2, 2: 3, 3: 0},
    # 语义相近类互换，攻击更隐蔽
    "person_pair": {0: 1, 1: 0},
    "car_van": {3: 4, 4: 3},
    # 只翻车辆类（3..9），行人保持正确
    "vehicle_cycle": {3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 3},
    # 多对一：所有非 car 类 -> car（强破坏）
    "all_to_car": {0: 3, 1: 3, 2: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3},
    # 偏移 2 / 5 类，覆盖不同目标分布
    "offset2": {i: (i + 2) % 10 for i in range(10)},
    "offset5": {i: (i + 5) % 10 for i in range(10)},
}


def resolve_label_flip_mapping(num_classes: int = 10) -> tuple[dict[int, int], str]:
    """解析标签翻转映射。

    优先级：
      1) FL_FLIP_MAPPING：自定义映射，支持 JSON（{"0":1,"1":2}）或 "0:1,1:2"
      2) FL_FLIP_PATTERN：预设名（见 LABEL_FLIP_PRESETS），默认 full_cycle
    返回 (mapping, name)，name 用于日志。
    """
    raw = (os.environ.get("FL_FLIP_MAPPING") or "").strip()
    if raw:
        try:
            if raw.startswith("{"):
                obj = json.loads(raw)
                mapping = {int(k): int(v) for k, v in obj.items()}
            else:
                mapping = {}
                for part in raw.split(","):
                    k, v = part.split(":")
                    mapping[int(k)] = int(v)
        except Exception as exc:
            raise ValueError(f"无法解析 FL_FLIP_MAPPING={raw!r}: {exc}") from exc
        name = f"custom:{raw}"
    else:
        pattern = (os.environ.get("FL_FLIP_PATTERN") or "full_cycle").strip().lower()
        if pattern not in LABEL_FLIP_PRESETS:
            raise ValueError(
                f"未知 FL_FLIP_PATTERN={pattern!r}；可选: {sorted(LABEL_FLIP_PRESETS)}"
            )
        mapping = dict(LABEL_FLIP_PRESETS[pattern])
        name = pattern

    bad = {k: v for k, v in mapping.items()
           if not (0 <= int(k) < num_classes and 0 <= int(v) < num_classes)}
    if bad:
        raise ValueError(f"标签翻转映射越界（num_classes={num_classes}）: {bad}")
    if not mapping:
        raise ValueError("标签翻转映射为空")
    return mapping, name


# 4090 默认：学习率不过激，先保证稳定收敛；需要更猛再通过 FL_LOCAL_LR 调大
LOCAL_LR = float(os.environ.get("FL_LOCAL_LR", "0.0002"))    # 0.01 → 0.0002（历史好配置，2e-4）
LR_MIN = float(os.environ.get("FL_LR_MIN", "5e-6"))
LR_SCHEDULE = "constant"  # cosine 在 Batch 2 加

NUM_WORKERS = int(os.environ.get("FL_NUM_WORKERS", "10"))
# 恶意率固定 30%（项目任务要求，2026-10-21 用户明确；N=10 → 3 人）
MALICIOUS_RATE = float(os.environ.get("FL_MALICIOUS_RATE", "0.3"))

GLOBAL_ROUNDS = int(os.environ.get("FL_GLOBAL_ROUNDS", "15"))   # 2026-10-23 起全局上限 15（原 50）
LOCAL_EPOCHS = int(os.environ.get("FL_LOCAL_EPOCHS", "3"))      # 1 → 3（历史好配置）
TRAIN_BATCH_SIZE = int(os.environ.get("FL_TRAIN_BATCH_SIZE", "64"))
TEST_BATCH_SIZE = int(os.environ.get("FL_TEST_BATCH_SIZE", "256"))
CPU_THREADS = int(os.environ.get("FL_CPU_THREADS", "16"))
DATALOADER_WORKERS = int(os.environ.get("FL_DATALOADER_WORKERS", "6"))
