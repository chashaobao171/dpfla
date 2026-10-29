"""项目内数据集路径探测。

默认先使用各环境的历史绝对路径；都找不到时，自动在项目根目录的
``data/`` 下查找 CIFAR10 / VisDrone，便于把整个 data 文件夹随项目一起拷贝。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CIFAR_BATCH_FILES = (
    "batches.meta",
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
    "test_batch",
)


def _cifar_batches_dir(root: Path) -> Optional[Path]:
    """返回包含 CIFAR batches 文件的目录。"""
    if root.name == "cifar-10-batches-py" and root.is_dir():
        if all((root / name).is_file() for name in CIFAR_BATCH_FILES):
            return root
        return None
    candidate = root / "cifar-10-batches-py"
    if candidate.is_dir() and all((candidate / name).is_file() for name in CIFAR_BATCH_FILES):
        return candidate
    return None


def find_cifar10_root() -> Optional[str]:
    """在项目根目录的 data/ 下查找 torchvision CIFAR10 的 root。"""
    data_dir = PROJECT_ROOT / "data"
    candidates: List[Path] = [
        data_dir,
        data_dir / "cifar",
        data_dir / "cifar10",
        data_dir / "CIFAR10",
        data_dir / "cifar-10-batches-py",
        PROJECT_ROOT / "cifar",
        PROJECT_ROOT / "cifar10",
        PROJECT_ROOT / "cifar-10-batches-py",
    ]
    for root in candidates:
        batches = _cifar_batches_dir(root)
        if batches is None:
            continue
        # torchvision 要求传入 batches 的父目录
        return str(root.parent if root.name == "cifar-10-batches-py" else root)
    return None


def _visdrone_split_dir(root: Path, split: str) -> Optional[Path]:
    """返回 VisDrone split 目录（兼容多包一层同名目录）。"""
    base = root / f"VisDrone2019-DET-{split}"
    nested = base / f"VisDrone2019-DET-{split}"
    if (nested / "images").is_dir():
        return nested
    if (base / "images").is_dir():
        return base
    return None


def is_visdrone_root(root: Path) -> bool:
    """判断 root 是否可作为 VisDrone 根目录。"""
    if not root.is_dir():
        return False
    if (root / "images").is_dir() and (root / "labels").is_dir():
        return True
    return _visdrone_split_dir(root, "train") is not None or _visdrone_split_dir(root, "val") is not None


def find_visdrone_root() -> Optional[str]:
    """在项目根目录的 data/ 下查找 VisDrone 根目录。"""
    data_dir = PROJECT_ROOT / "data"
    candidates: List[Path] = [
        data_dir,
        data_dir / "visdrone",
        data_dir / "VisDrone",
        data_dir / "VisDrone2019-DET",
        PROJECT_ROOT / "visdrone",
        PROJECT_ROOT / "VisDrone",
        PROJECT_ROOT / "datasets" / "visdrone",
    ]
    for root in candidates:
        if is_visdrone_root(root):
            return str(root)
    return None


def visdrone_yaml_paths(root_path: str) -> Optional[tuple[str, str]]:
    """返回 (train 相对路径, val 相对路径)，用于生成 YOLO yaml。"""
    root = Path(root_path)
    train_dir = _visdrone_split_dir(root, "train")
    val_dir = _visdrone_split_dir(root, "val")
    if train_dir is None or val_dir is None:
        return None
    return (
        (train_dir / "images").relative_to(root).as_posix(),
        (val_dir / "images").relative_to(root).as_posix(),
    )
