"""
CIFAR10 服务端注入攻击筛选脚本（用于挑出让 DPFLA 领先基线防御 ≥2% 的攻击范式）

用法（环境变量控制）：
  FL_SERVER_ATTACK=minmax|alie|signflipping   # 攻击范式（默认 minmax）
  FL_RULE=fedavg|DPFLA|mkrum|tmean            # 聚合/防御方法（默认 DPFLA）
  FL_GLOBAL_ROUNDS=5                          # 筛选用短轮（正式对比请用 15）

口径与 run_dpfla_gaussian.py 一致：ResNet18CIFAR / 10 客户端 / BS64 / lr0.01 / IID /
30% 恶意 / 行为率 1.0 / FL_GRADIENT_SCALE=1。
攻击由服务端在聚合前注入（omniscient，FLPoison 原始设定），见 attack_alg/server_attacks.py。
"""

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CIF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _CIF)

import time
import torch

from cifar_fed_hparams import (
    DATASET_NAME,
    MODEL_NAME,
    NUM_WORKERS,
    FRAC_WORKERS,
    GLOBAL_ROUNDS,
    LOCAL_EPOCHS,
    MALICIOUS_RATE,
    MALICIOUS_BEHAVIOR_RATE,
    apply_arguments_patch,
    apply_common_env,
    print_config,
)

SERVER_ATTACK = (os.environ.get('FL_SERVER_ATTACK') or 'minmax').strip().lower()
RULE = (os.environ.get('FL_RULE') or 'DPFLA').strip()

device = "cuda" if torch.cuda.is_available() else "cpu"

apply_arguments_patch(device)
apply_common_env()

from server import run_exp

PLACEHOLDER_CLASSES = {'source_class': 0, 'target_class': 1}

print_config(
    f"服务端注入攻击 {SERVER_ATTACK}（omniscient）",
    f"{RULE}",
    extra_lines=[
        f"- 设备: {device}",
        f"- 恶意行为率: {MALICIOUS_BEHAVIOR_RATE:.2f}",
        f"- 攻击实现: attack_alg/server_attacks.py（服务端聚合前注入）",
    ],
)

start_time = time.time()

try:
    run_exp(
        num_workers=NUM_WORKERS,
        frac_workers=FRAC_WORKERS,
        attack_type=SERVER_ATTACK,
        rule=RULE,
        replace_method=PLACEHOLDER_CLASSES,
        dataset={'dataset_name': DATASET_NAME, 'model_name': MODEL_NAME},
        malicious_rate=MALICIOUS_RATE,
        malicious_behavior_rate=MALICIOUS_BEHAVIOR_RATE,
        global_round=GLOBAL_ROUNDS,
        local_epoch=LOCAL_EPOCHS,
        untarget=False,
        experiment_tag=f'run_screen_{SERVER_ATTACK}_{RULE}',
    )
    elapsed = time.time() - start_time
    print(f"\n✓ 测试完成，总耗时: {elapsed / 60:.1f} 分钟")
except KeyboardInterrupt:
    print("\n测试被中断")
except Exception as e:
    print(f"\n测试失败: {e}")
    import traceback

    traceback.print_exc()
