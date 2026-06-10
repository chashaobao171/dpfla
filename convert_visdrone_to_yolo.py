"""
将 VisDrone 原始标注转换为 YOLO 标签。

VisDrone格式:
<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

YOLO格式:
<class> <x_center> <y_center> <width> <height>  (归一化坐标)
"""

import argparse
import os
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


VISDRONE_ORIG_TO_COCO = {
    1: 0,   # pedestrian -> person
    2: 0,   # people -> person
    3: 1,   # bicycle -> bicycle
    4: 2,   # car -> car
    5: 2,   # van -> car (COCO无van)
    6: 7,   # truck -> truck
    7: 1,   # tricycle -> bicycle
    8: 1,   # awning-tricycle -> bicycle
    9: 5,   # bus -> bus
    10: 3,  # motor -> motorcycle
}


def map_category(category: int, mode: str):
    # 跳过 ignored region (0) 和 others (11)
    if category in (0, 11):
        return None
    if mode == "coco":
        return VISDRONE_ORIG_TO_COCO.get(category)
    if mode == "visdrone10":
        # 保留 VisDrone 10 类 ID 到 [0,9]
        if 1 <= category <= 10:
            return category - 1
    return None


def split_paths(visdrone_root: Path, split: str):
    if split == "train":
        split_dir = visdrone_root / "VisDrone2019-DET-train" / "VisDrone2019-DET-train"
    else:
        split_dir = visdrone_root / "VisDrone2019-DET-val" / "VisDrone2019-DET-val"
    return split_dir, split_dir / "annotations", split_dir / "images"


def convert_visdrone_to_yolo(visdrone_root, mode="coco", output_root=None):
    visdrone_root = Path(visdrone_root)
    print(f"转换模式: {mode} ({'COCO映射' if mode == 'coco' else '保留VisDrone 10类'})")

    for split in ["train", "val"]:
        split_dir, annotations_dir, images_dir = split_paths(visdrone_root, split)
        if not split_dir.exists():
            print(f"⚠️ 跳过 {split}，目录不存在: {split_dir}")
            continue
        if not annotations_dir.exists():
            print(f"⚠️ 跳过 {split}，annotations目录不存在")
            continue

        if output_root:
            # 与当前工程目录结构对齐：labels/train, labels/validation
            split_out = "train" if split == "train" else "validation"
            labels_dir = Path(output_root) / split_out
        else:
            suffix = "labels_yolo_coco" if mode == "coco" else "labels_yolo_visdrone10"
            labels_dir = split_dir / suffix
        labels_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n处理 {split} 集...")
        print(f"  输入: {annotations_dir}")
        print(f"  输出: {labels_dir}")

        ann_files = list(annotations_dir.glob("*.txt"))
        converted, skipped = 0, 0

        for ann_file in tqdm(ann_files, desc=f"转换{split}"):
            img_file = images_dir / ann_file.name.replace(".txt", ".jpg")
            if not img_file.exists():
                skipped += 1
                continue

            try:
                img = Image.open(img_file)
                img_width, img_height = img.size
            except Exception as e:
                print(f"无法读取图像 {img_file}: {e}")
                skipped += 1
                continue

            yolo_lines = []
            with open(ann_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 6:
                        continue
                    try:
                        bbox_left = int(parts[0])
                        bbox_top = int(parts[1])
                        bbox_width = int(parts[2])
                        bbox_height = int(parts[3])
                        category = int(parts[5])

                        yolo_class = map_category(category, mode)
                        if yolo_class is None:
                            continue

                        x_center = (bbox_left + bbox_width / 2) / img_width
                        y_center = (bbox_top + bbox_height / 2) / img_height
                        norm_width = bbox_width / img_width
                        norm_height = bbox_height / img_height

                        x_center = max(0.0, min(1.0, x_center))
                        y_center = max(0.0, min(1.0, y_center))
                        norm_width = max(0.0, min(1.0, norm_width))
                        norm_height = max(0.0, min(1.0, norm_height))
                        if norm_width <= 0 or norm_height <= 0:
                            continue

                        yolo_lines.append(
                            f"{yolo_class} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                        )
                    except (ValueError, IndexError):
                        continue

            output_file = labels_dir / ann_file.name
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))
            converted += 1

        print(f"✓ {split}集转换完成: {converted}个文件, {skipped}个跳过")

    print("\n" + "=" * 70)
    print("转换完成！")
    print("=" * 70)


def symlink_labels_next_to_images(visdrone_root: Path, mode: str) -> None:
    """
    在每个 split 目录下创建 labels -> labels_yolo_*，与 images/ 同级，
    供 Ultralytics `yolo val`（generate_visdrone_yaml 指向的 val images）自动读 GT。
    """
    suffix = "labels_yolo_coco" if mode == "coco" else "labels_yolo_visdrone10"
    for split in ("train", "val"):
        split_dir, _, _ = split_paths(visdrone_root, split)
        if not split_dir.exists():
            continue
        src = split_dir / suffix
        dst = split_dir / "labels"
        if not src.exists():
            print(f"⚠️ 跳过 symlink {split}：不存在 {src}")
            continue
        if dst.is_symlink():
            dst.unlink()
        elif dst.exists():
            print(f"⚠️ 已存在 {dst}（非 symlink 或非空则勿动），跳过以免覆盖")
            continue
        try:
            dst.symlink_to(suffix, target_is_directory=True)
        except OSError as e:
            print(f"⚠️ 无法创建 symlink {dst} -> {suffix}: {e}")
            continue
        print(f"✓ Ultralytics 可读: {dst} -> {suffix}")


def deploy_labels_to_data_root(visdrone_root: Path, mode: str):
    train_split_dir, _, _ = split_paths(visdrone_root, "train")
    val_split_dir, _, _ = split_paths(visdrone_root, "val")
    suffix = "labels_yolo_coco" if mode == "coco" else "labels_yolo_visdrone10"
    src_train = train_split_dir / suffix
    src_val = val_split_dir / suffix
    dst_root = Path("/root/autodl-tmp/data/labels")
    dst_train = dst_root / "train"
    dst_val = dst_root / "validation"

    if not src_train.exists() or not src_val.exists():
        raise FileNotFoundError("转换结果目录不存在，无法部署。请先完成转换。")

    dst_root.mkdir(parents=True, exist_ok=True)
    for d in [dst_train, dst_val]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    for f in src_train.glob("*.txt"):
        shutil.copy2(f, dst_train / f.name)
    for f in src_val.glob("*.txt"):
        shutil.copy2(f, dst_val / f.name)
    print(f"✓ 已部署到 {dst_root}（mode={mode}）")


if __name__ == "__main__":
    default_root = "/root/autodl-tmp/data/visdrone" if os.path.exists("/root/autodl-tmp/data/visdrone") else "/home/featurize/data/visdrone" if os.path.exists("/home/featurize/data/visdrone") else "D:/Pycharmworkplace/visdrone"

    parser = argparse.ArgumentParser(description="Convert VisDrone annotations to YOLO labels")
    parser.add_argument("--root", default=default_root, help="VisDrone 原始数据根目录")
    parser.add_argument("--mode", choices=["coco", "visdrone10"], default="visdrone10",
                        help="类别映射模式：coco=映射到COCO类，visdrone10=保留VisDrone 10类ID")
    parser.add_argument("--output-root", default=None, help="输出目录（默认写到原始数据目录下）")
    parser.add_argument("--deploy", action="store_true",
                        help="转换后：(1) 在各 split 下创建 labels -> labels_yolo_* 供 yolo val；(2) 复制到 /root/autodl-tmp/data/labels/")
    args = parser.parse_args()

    print("=" * 70)
    print("VisDrone to YOLO 格式转换")
    print("=" * 70)
    convert_visdrone_to_yolo(args.root, mode=args.mode, output_root=args.output_root)
    if args.deploy:
        root = Path(args.root)
        symlink_labels_next_to_images(root, mode=args.mode)
        try:
            deploy_labels_to_data_root(root, mode=args.mode)
        except FileNotFoundError as e:
            print(f"⚠️ 跳过复制到 /root/autodl-tmp/data/labels: {e}")
