"""
动态生成visdrone.yaml配置文件
根据当前环境自动检测数据集路径
"""

import os

def generate_visdrone_yaml(output_path='visdrone_temp.yaml'):
    """
    生成VisDrone数据集的YOLO配置文件
    
    Args:
        output_path: 输出文件路径
        
    Returns:
        生成的yaml文件路径
    """
    # 主项目口径：强制优先 autodl-tmp 目录
    if os.path.exists('/root/autodl-tmp/data/visdrone'):
        visdrone_root = '/root/autodl-tmp/data/visdrone'
        train_path = 'VisDrone2019-DET-train/VisDrone2019-DET-train/images'
        val_path = 'VisDrone2019-DET-val/VisDrone2019-DET-val/images'
        test_path = None
    elif os.path.exists('/home/featurize/data/visdrone'):
        visdrone_root = '/home/featurize/data/visdrone'
        train_path = 'VisDrone2019-DET-train/VisDrone2019-DET-train/images'
        val_path = 'VisDrone2019-DET-val/VisDrone2019-DET-val/images'
        test_path = None
    else:
        visdrone_root = 'D:/Pycharmworkplace/visdrone'
        train_path = 'VisDrone2019-DET-train/VisDrone2019-DET-train/images'
        val_path = 'VisDrone2019-DET-val/VisDrone2019-DET-val/images'
        test_path = None
    
    # 使用 VisDrone 原始 10 类定义（保留官方类别语义）
    visdrone_names = [
        'pedestrian',
        'people',
        'bicycle',
        'car',
        'van',
        'truck',
        'tricycle',
        'awning-tricycle',
        'bus',
        'motor',
    ]
    names_block = "\n".join([f"  {i}: {name}" for i, name in enumerate(visdrone_names)])

    test_block = f"test: {test_path}\n" if test_path else ""
    yaml_content = f"""# VisDrone数据集配置文件（YOLO格式）
# 自动生成，用于YOLO原生评估
# 注意：主项目口径固定为原始 VisDrone train/val；类别按 VisDrone 10 类定义

# 数据集路径
path: {visdrone_root}  # 数据集根目录
train: {train_path}  # 训练集图像路径（相对于path）
val: {val_path}      # 验证集图像路径（相对于path）
{test_block}# 类别数量（VisDrone 10类）
nc: 10

# 类别名称（VisDrone 10类）
names:
{names_block}
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    # 避免 `python ... | head` 等场景下 stdout 已关触发 BrokenPipeError，中断 fl_core 评估
    try:
        print(f"✓ 生成YOLO配置文件: {output_path}")
        print(f"  数据集路径: {visdrone_root}")
    except BrokenPipeError:
        pass

    return output_path


if __name__ == '__main__':
    generate_visdrone_yaml()
