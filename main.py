"""
DPFLA联邦学习主程序

使用方式:
1. 交互式配置 (推荐): python main.py
2. 直接配置模式: python main.py --direct
"""
from server import run_exp
from federated_learning.my_dict import replace_1_with_7, replace_dog_with_cat, replace_car_with_truck
from federated_learning.my_dict import run_mnist, run_cifar10, run_visdrone_yolo
from federated_learning.my_dict import fed_avg, simple_median, trimmed_mean, multi_krum, fools_gold, DPFLA


# ========== 交互式配置函数 ==========
def print_separator():
    print("=" * 70)


def print_title(title):
    print_separator()
    print(f"  {title}")
    print_separator()


def get_choice(prompt, options, default=None, show_recommend=True):
    """
    获取用户选择（使用数字选项）
    
    Args:
        prompt: 提示信息
        options: 选项列表，格式: [(key, description, recommended), ...]
        default: 默认选项的key
        show_recommend: 是否显示推荐标记
    """
    print(f"\n{prompt}")
    
    # 创建数字到key的映射
    num_to_key = {}
    default_num = None
    
    for idx, (key, desc, recommended) in enumerate(options, start=1):
        num_to_key[str(idx)] = key
        if key == default:
            default_num = str(idx)
        
        mark = "⭐ [推荐]" if recommended and show_recommend else ""
        default_mark = " [默认]" if key == default else ""
        print(f"  [{idx}] {desc}{mark}{default_mark}")
    
    while True:
        default_hint = f"默认: {default_num}" if default_num else "无默认值"
        choice_input = input(f"\n请选择 ({default_hint}): ").strip()
        
        # 如果为空且有默认值，返回默认值
        if not choice_input and default_num:
            return default
        
        # 检查是否是有效的数字选择
        if choice_input in num_to_key:
            return num_to_key[choice_input]
        
        # 也支持直接输入key（向后兼容，转为小写）
        choice_lower = choice_input.lower()
        for opt_key, _, _ in options:
            if choice_lower == opt_key.lower():
                return opt_key
        
        print("❌ 无效选择，请重新输入！")


def get_int_input(prompt, default, min_val=None, max_val=None):
    """获取整数输入"""
    while True:
        try:
            value_str = input(f"{prompt} (默认: {default}): ").strip()
            if not value_str:
                return default
            value = int(value_str)
            if min_val is not None and value < min_val:
                print(f"❌ 值不能小于 {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"❌ 值不能大于 {max_val}")
                continue
            return value
        except ValueError:
            print("❌ 请输入有效的整数！")


def get_float_input(prompt, default, min_val=None, max_val=None):
    """获取浮点数输入"""
    while True:
        try:
            value_str = input(f"{prompt} (默认: {default}): ").strip()
            if not value_str:
                return default
            value = float(value_str)
            if min_val is not None and value < min_val:
                print(f"❌ 值不能小于 {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"❌ 值不能大于 {max_val}")
                continue
            return value
        except ValueError:
            print("❌ 请输入有效的数字！")


def interactive_config():
    """交互式配置"""
    print_title("DPFLA 联邦学习实验配置")
    print("\n欢迎使用交互式配置工具！")
    print("您可以通过以下步骤配置实验参数。")
    
    config = {}
    
    # 1. 选择数据集
    print_title("步骤 1: 选择数据集")
    dataset_choice = get_choice(
        "请选择数据集:",
        [
            ("mnist", "MNIST (手写数字识别)", False),
            ("cifar10", "CIFAR-10 (图像分类)", False),
            ("visdrone", "VisDrone (目标检测) + YOLO", True),
        ],
        default="visdrone"
    )
    
    if dataset_choice == "mnist":
        config['dataset'] = run_mnist()
        config['replace_method'] = replace_1_with_7()
        config['num_classes'] = 10
    elif dataset_choice == "cifar10":
        config['dataset'] = run_cifar10()
        config['replace_method'] = replace_dog_with_cat()
        config['num_classes'] = 10
    else:  # visdrone
        config['dataset'] = run_visdrone_yolo()
        config['replace_method'] = replace_car_with_truck()
        config['num_classes'] = 10
    
    # 2. 选择防御算法
    print_title("步骤 2: 选择聚合/防御算法")
    algorithm_choice = get_choice(
        "请选择聚合算法:",
        [
            ("fedavg", "FedAvg (联邦平均)", False),
            ("median", "Median (中位数)", False),
            ("tmean", "Trimmed Mean (截断均值)", False),
            ("mkrum", "Multi-Krum", False),
            ("foolsgold", "FoolsGold", False),
            ("dpfla", "DPFLA (隐私保护投毒防御)", True),
        ],
        default="dpfla"
    )
    
    algorithm_map = {
        "fedavg": fed_avg,
        "median": simple_median,
        "tmean": trimmed_mean,
        "mkrum": multi_krum,
        "foolsgold": fools_gold,
        "dpfla": DPFLA,
    }
    config['rule'] = algorithm_map[algorithm_choice]()
    
    # 3. 客户端配置
    print_title("步骤 3: 客户端配置")
    config['num_workers'] = get_int_input(
        "客户端总数",
        default=50,
        min_val=1,
        max_val=1000
    )
    
    config['frac_workers'] = get_float_input(
        "每轮参与客户端比例 (0.0-1.0)",
        default=1.0,
        min_val=0.1,
        max_val=1.0
    )
    
    # 4. 训练配置
    print_title("步骤 4: 训练配置")
    config['global_round'] = get_int_input(
        "全局训练轮数",
        default=100,
        min_val=1,
        max_val=1000
    )
    
    config['local_epoch'] = get_int_input(
        "本地训练轮数",
        default=1,
        min_val=1,
        max_val=50
    )
    
    # 5. 攻击配置
    print_title("步骤 5: 攻击配置")
    has_attack = get_choice(
        "是否启用攻击?",
        [
            ("n", "否 - 无攻击训练", True),
            ("y", "是 - 启用投毒攻击", False),
        ],
        default="n"
    )
    
    if has_attack == "y":
        attack_type = get_choice(
            "选择攻击类型:",
            [
                ("label_flipping", "标签翻转攻击", True),
                ("backdoor", "后门攻击", False),
            ],
            default="label_flipping"
        )
        config['attack_type'] = attack_type
        
        malicious_rate = get_float_input(
            "恶意客户端比例 (0.0-0.5)",
            default=0.2,
            min_val=0.0,
            max_val=0.5
        )
        config['malicious_rate'] = [malicious_rate]
    else:
        config['attack_type'] = "no_attack"
        config['malicious_rate'] = [0]
    
    config['malicious_behavior_rate'] = 1.0
    config['untarget'] = False
    
    # 6. 数据分布配置
    print_title("步骤 6: 数据分布配置")
    dd_type = get_choice(
        "选择数据分布类型:",
        [
            ("iid", "IID (独立同分布)", True),
            ("non_iid", "Non-IID (非独立同分布)", False),
            ("extreme_non_iid", "Extreme Non-IID (极端非独立同分布)", False),
        ],
        default="iid"
    )
    config['dd_type'] = dd_type.upper()
    
    # 显示配置摘要
    print_title("配置摘要")
    print(f"数据集: {config['dataset']['dataset_name']}")
    print(f"模型: {config['dataset']['model_name']}")
    print(f"聚合算法: {algorithm_choice.upper()}")
    print(f"客户端总数: {config['num_workers']}")
    print(f"参与比例: {config['frac_workers']}")
    print(f"全局轮数: {config['global_round']}")
    print(f"本地轮数: {config['local_epoch']}")
    print(f"攻击类型: {config['attack_type']}")
    print(f"恶意比例: {config['malicious_rate'][0] * 100:.1f}%")
    print(f"数据分布: {config['dd_type']}")
    
    # 警告信息
    if config['dataset']['dataset_name'] == 'VisDrone' and algorithm_choice == 'dpfla':
        print_separator()
        print("⚠️  注意: VisDrone + DPFLA 可能需要适配YOLO模型结构")
        print("   如果遇到错误，可以先用 fedavg 测试数据集是否正常")
        print_separator()
    
    # 确认
    print_separator()
    confirm = get_choice(
        "\n确认开始实验?",
        [
            ("y", "是 - 开始运行", True),
            ("n", "否 - 取消", False),
        ],
        default="y"
    )
    
    if confirm == "n":
        print("\n❌ 已取消实验")
        return None
    
    return config


# ========== 主程序入口 ==========
if __name__ == '__main__':
    import sys
    
    # 检查是否使用交互式配置
    use_interactive = True
    
    # 如果命令行参数包含 --direct 或 -d，则使用直接配置模式
    if len(sys.argv) > 1 and (sys.argv[1] == '--direct' or sys.argv[1] == '-d'):
        use_interactive = False
    
    if use_interactive:
        # 使用交互式配置
        print("=" * 70)
        print("  使用交互式配置模式")
        print("=" * 70)
        print("\n提示: 如果想使用直接配置模式，请运行: python main.py --direct")
        print("=" * 70 + "\n")
        
        try:
            config = interactive_config()
            if config:
                print("\n✅ 配置完成，开始运行实验...\n")
                run_exp(
                    num_workers=config['num_workers'],
                    frac_workers=config['frac_workers'],
                    attack_type=config['attack_type'],
                    rule=config['rule'],
                    replace_method=config['replace_method'],
                    dataset=config['dataset'],
                    malicious_rate=config['malicious_rate'],
                    malicious_behavior_rate=config['malicious_behavior_rate'],
                    global_round=config['global_round'],
                    local_epoch=config['local_epoch'],
                    untarget=config['untarget']
                )
                print("\n✅ 实验完成！")
        except KeyboardInterrupt:
            print("\n\n❌ 用户中断实验")
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        # 直接配置模式（需要手动编辑代码）
        print("=" * 70)
        print("  使用直接配置模式")
        print("=" * 70)
        print("\n提示: 如果想使用交互式配置，请直接运行: python main.py")
        print("=" * 70 + "\n")
        
        # ========== 在这里配置参数 ==========
        NUM_WORKERS = 10
        FRAC_WORKERS = 1
        # ATTACK_TYPE = "label_flipping"
        ATTACK_TYPE = "backdoor"
        GLOBAL_ROUND = 3
        LOCAL_EPOCH = 1
        UNTARGET = False

        # MNIST配置
        # REPLACE_METHOD = replace_1_with_7()
        # RULE = DPFLA()
        # DATASET = run_mnist()

        # CIFAR10配置
        # REPLACE_METHOD = replace_dog_with_cat()
        # RULE = DPFLA()
        # DATASET = run_cifar10()

        # VisDrone + YOLO配置
        REPLACE_METHOD = replace_car_with_truck()
        RULE = DPFLA()
        DATASET = run_visdrone_yolo()

        MALICIOUS_RATE = [0]  # 0, 0.1, 0.2, 0.3, 0.4, 0.5
        MALICIOUS_BEHAVIOR_RATE = 1

        for rate in MALICIOUS_RATE:
            run_exp(NUM_WORKERS, FRAC_WORKERS, ATTACK_TYPE, RULE, REPLACE_METHOD, DATASET, rate,
                    MALICIOUS_BEHAVIOR_RATE, GLOBAL_ROUND, LOCAL_EPOCH, UNTARGET)
