#!/usr/bin/env python3
"""交互式分布式启动器。

一条命令进入：
    python launch_distributed.py

按段询问：
1) 数据集 / 攻击 / 防御 / 配置档位
2) 角色（prepare / server / client / orchestrator）
3) 是否生成分区、分区路径
4) 训练参数
5) 网络与日志参数

最后调用 run_distributed_smoke.py，`distributed_fl.py` 保持不变。
"""

from __future__ import annotations

import argparse
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parent
SMOKE = ROOT / "run_distributed_smoke.py"


ROLE_OPTIONS = [
    ("prepare", "准备 / 重新生成数据集分区"),
    ("server", "启动服务器"),
    ("client", "启动客户端"),
    ("orchestrator", "本机一键单条编排"),
    ("exit", "退出"),
]
DATASET_OPTIONS = [
    ("cifar10", "CIFAR10 + ResNet18CIFAR"),
    ("visdrone", "VisDrone + YOLOv8"),
]
SCENARIO_OPTIONS = [
    ("baseline", "无攻击"),
    ("label_flipping", "标签翻转（客户端侧）"),
    ("minmax", "MinMax（服务端注入）"),
    ("ipm", "IPM（服务端注入）"),
    ("alie", "ALIE（服务端注入）"),
    ("signflipping", "SignFlipping（服务端注入）"),
]
DEFENSE_OPTIONS = [
    ("fedavg", "FedAvg（无防御）"),
    ("dpfla", "DPFLA（SVD + KMeans）"),
    ("mkrum", "Multi-Krum"),
    ("tmean", "Trimmed-Mean"),
]
PROFILE_OPTIONS = [
    ("smoke", "冒烟：少客户端 / 少样本 / 少轮"),
    ("formal", "单机正式参数：10 客户端 / 全量 / 30% 恶意"),
    ("custom", "自定义（以冒烟默认值为起点逐项修改）"),
]


@dataclass
class Config:
    role: str = "server"
    dataset: str = "cifar10"
    scenario: str = "baseline"
    defense: str = "dpfla"
    profile: str = "smoke"
    num_clients: int = 5
    rounds: int = 2
    full_data: bool = False
    max_samples: int = 32
    batch_size: int = 16
    test_batch_size: int = 64
    test_samples: Optional[int] = 256
    lr: float = 0.01
    local_epochs: int = 1
    model_size: str = "n"
    image_size: int = 32
    amp: int = 0
    malicious_rate: Optional[float] = None
    seed: int = 42
    device: str = "cuda"
    partition: str = "./dist_run/partitions.json"
    workdir: str = "./dist_run/server"
    port: int = 6000
    bind_host: str = "0.0.0.0"
    server_host: str = "127.0.0.1"
    client_id: int = 0
    timeout: int = 1800
    visdrone_root: str = "/root/autodl-tmp/data/visdrone"


def _section(title: str) -> None:
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def _ask_text(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or default


def _ask_int(prompt: str, default: int, minimum: int = 0, maximum: Optional[int] = None) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("  请输入整数。")
            continue
        if value < minimum or (maximum is not None and value > maximum):
            print(f"  范围应为 {minimum}..{maximum if maximum is not None else '∞'}。")
            continue
        return value


def _ask_float(prompt: str, default: float, minimum: float = 0.0) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("  请输入数字。")
            continue
        if value < minimum:
            print(f"  不能小于 {minimum}。")
            continue
        return value


def _ask_optional_rate(prompt: str, default: Optional[float]) -> Optional[float]:
    shown = "留空=1个客户端" if default is None else f"{default}"
    while True:
        raw = input(f"{prompt} [{shown}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("  请输入 0..1 之间的小数，或留空。")
            continue
        if not 0.0 < value <= 1.0:
            print("  恶意率应在 (0, 1] 之间。")
            continue
        return value


def _ask_bool(prompt: str, default: bool) -> bool:
    shown = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{shown}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes", "1", "是"):
            return True
        if raw in ("n", "no", "0", "否"):
            return False
        print("  请输入 y 或 n。")


def _ask_choice(title: str, options: Sequence[tuple[str, str]], default: str) -> str:
    print(title)
    for index, (key, label) in enumerate(options, 1):
        print(f"  {index}) {label}  ({key})")
    index_map = {str(i): key for i, (key, _) in enumerate(options, 1)}
    keys = {key for key, _ in options}
    while True:
        raw = input(f"选择 [{default}]: ").strip()
        if not raw:
            return default
        if raw in keys:
            return raw
        if raw in index_map:
            return index_map[raw]
        print("  请输入编号或括号内的名称。")


def _profile_defaults(profile: str, dataset: str) -> Config:
    """返回该档位/数据集下的默认值。"""
    cfg = Config(dataset=dataset, profile=profile)
    if dataset == "cifar10":
        cfg.partition = "./dist_run/cifar10/partitions.json"
        cfg.workdir = "./dist_run/cifar10/server"
        cfg.max_samples = 32
        cfg.batch_size = 16
        cfg.test_batch_size = 64
        cfg.test_samples = 256
        cfg.lr = 0.01
        cfg.image_size = 32
        cfg.amp = 0
        cfg.num_clients = 5
        cfg.malicious_rate = None
        cfg.rounds = 2
        cfg.seed = 42
        cfg.timeout = 1800
        if profile == "formal":
            cfg.num_clients = 10
            cfg.rounds = 15
            cfg.full_data = True
            cfg.batch_size = 64
            cfg.test_batch_size = 1000
            cfg.test_samples = 0
            cfg.malicious_rate = 0.3
            cfg.seed = 7
            cfg.timeout = 7200
    else:
        cfg.partition = "./dist_run/visdrone/partitions.json"
        cfg.workdir = "./dist_run/visdrone/server"
        cfg.max_samples = 4
        cfg.batch_size = 2
        cfg.test_batch_size = 8
        cfg.test_samples = None
        cfg.lr = 2e-4
        cfg.local_epochs = 1
        cfg.model_size = "n"
        cfg.image_size = 320
        cfg.amp = 0
        cfg.num_clients = 3
        cfg.malicious_rate = None
        cfg.rounds = 2
        cfg.seed = 42
        cfg.timeout = 1800
        if profile == "formal":
            cfg.num_clients = 10
            cfg.rounds = 15
            cfg.full_data = True
            cfg.batch_size = 16
            cfg.test_batch_size = 64
            cfg.lr = 2e-4
            cfg.local_epochs = 3
            cfg.model_size = "m"
            cfg.image_size = 1280
            cfg.amp = 1
            cfg.malicious_rate = 0.3
            cfg.seed = 7
            cfg.timeout = 43200
    return cfg


def _ask_common(cfg: Config) -> None:
    _section("第 1 段：数据集、攻击与防御")
    cfg.dataset = _ask_choice("训练数据集：", DATASET_OPTIONS, cfg.dataset)
    cfg.scenario = _ask_choice("攻击样式：", SCENARIO_OPTIONS, cfg.scenario)
    cfg.defense = _ask_choice("防御样式：", DEFENSE_OPTIONS, cfg.defense)
    cfg.profile = _ask_choice("参数档位：", PROFILE_OPTIONS, cfg.profile)
    scenario, defense = cfg.scenario, cfg.defense
    base = _profile_defaults(cfg.profile, cfg.dataset)
    for key, value in vars(base).items():
        setattr(cfg, key, value)
    cfg.scenario = scenario
    cfg.defense = defense

    _section("第 2 段：角色与分区")
    cfg.role = _ask_choice("本机角色：", ROLE_OPTIONS, cfg.role)
    if cfg.role == "exit":
        return
    cfg.num_clients = _ask_int("客户端数量（进程数，不是机器数）", cfg.num_clients, 1, 100)
    if cfg.defense == "mkrum" and cfg.num_clients < 5:
        print("  Multi-Krum 需要至少 5 个客户端（n >= 2f+3）。")
        cfg.num_clients = _ask_int("请重新输入客户端数量", 5, 5, 100)
    if cfg.defense == "dpfla" and cfg.num_clients < 3:
        print("  注意：2 客户端时 DPFLA 的 KMeans 会退化，只能验证链路。")
    if cfg.scenario == "minmax" and cfg.num_clients == 2:
        print("  注意：2 客户端时 MinMax 只有 1 个良性端，构造会退化，建议 IPM。")
    if cfg.scenario == "label_flipping" and cfg.dataset == "visdrone":
        print("  注意：分布式 VisDrone 标签翻转目前是 5→3；单机演示 car→truck 是 3→5。")

    cfg.full_data = _ask_bool("是否使用全量训练集（否则按 max-samples 截断）", cfg.full_data)
    if not cfg.full_data:
        cfg.max_samples = _ask_int("每个客户端保留的样本数", cfg.max_samples, 1)
    cfg.seed = _ask_int("随机种子（正式实验用 7）", cfg.seed, 0)
    cfg.partition = _ask_text("分区 JSON 路径", cfg.partition)
    if cfg.dataset == "visdrone":
        cfg.visdrone_root = _ask_text("VisDrone 数据根", cfg.visdrone_root)
    cfg.device = _ask_text("设备（cuda/cpu）", cfg.device)


def _run_prepare(cfg: Config) -> int:
    argv = _build_argv(cfg, "prepare")
    return _dispatch(argv)


def _ask_training(cfg: Config) -> None:
    _section("第 3 段：训练参数")
    cfg.rounds = _ask_int("全局轮数", cfg.rounds, 1)
    cfg.local_epochs = _ask_int("每轮本地 epoch", cfg.local_epochs, 1)
    cfg.batch_size = _ask_int("训练 batch size", cfg.batch_size, 1)
    cfg.test_batch_size = _ask_int("评估 batch size", cfg.test_batch_size, 1)
    if cfg.dataset == "cifar10":
        cfg.test_samples = _ask_int(
            "CIFAR 测试样本数（0=全量 10000）",
            0 if cfg.test_samples is None else cfg.test_samples,
            0,
        )
    else:
        cfg.model_size = _ask_choice(
            "YOLO 模型尺寸：",
            [("n", "nano"), ("s", "small"), ("m", "medium"), ("l", "large"), ("x", "xlarge")],
            cfg.model_size,
        )
        cfg.image_size = _ask_int("YOLO 输入分辨率", cfg.image_size, 32)
        cfg.amp = 1 if _ask_bool("是否开启 AMP（1280 建议开启）", bool(cfg.amp)) else 0
    cfg.lr = _ask_float("学习率", cfg.lr, 1e-8)
    if cfg.scenario == "baseline":
        cfg.malicious_rate = None
    else:
        cfg.malicious_rate = _ask_optional_rate("恶意客户端比例", cfg.malicious_rate)


def _ask_network(cfg: Config) -> None:
    _section("第 4 段：网络与日志")
    cfg.timeout = _ask_int("server/client 超时秒数", cfg.timeout, 1)
    if cfg.role == "server":
        cfg.bind_host = _ask_text("server 绑定地址", cfg.bind_host)
        cfg.port = _ask_int("server 端口", cfg.port, 1, 65535)
        cfg.workdir = _ask_text("server 日志目录", cfg.workdir)
    elif cfg.role == "client":
        cfg.server_host = _ask_text("server 地址", cfg.server_host)
        cfg.port = _ask_int("server 端口", cfg.port, 1, 65535)
        cfg.client_id = _ask_int("客户端 ID（0..N-1）", cfg.client_id, 0, cfg.num_clients - 1)
        default_workdir = f"./dist_run/{cfg.dataset}/client{cfg.client_id}"
        cfg.workdir = _ask_text("client 日志目录", default_workdir)
    elif cfg.role == "orchestrator":
        cfg.workdir = _ask_text("编排输出目录", "./logs_dist_smoke")
    elif cfg.role == "prepare":
        pass


def _build_argv(cfg: Config, role: Optional[str] = None) -> list[str]:
    role = role or cfg.role
    argv = [
        "--role", role,
        "--dataset", cfg.dataset,
        "--scenario", cfg.scenario,
        "--defense", cfg.defense,
        "--num-clients", str(cfg.num_clients),
        "--rounds", str(cfg.rounds),
        "--seed", str(cfg.seed),
        "--batch-size", str(cfg.batch_size),
        "--test-batch-size", str(cfg.test_batch_size),
        "--lr", str(cfg.lr),
        "--local-epochs", str(cfg.local_epochs),
        "--model-size", cfg.model_size,
        "--image-size", str(cfg.image_size),
        "--amp", str(cfg.amp),
        "--device", cfg.device,
    ]
    if cfg.dataset == "cifar10":
        argv += ["--test-samples", str(0 if cfg.test_samples is None else cfg.test_samples)]
    if cfg.full_data:
        argv += ["--full-data"]
    else:
        argv += ["--max-samples", str(cfg.max_samples)]
    if cfg.malicious_rate is not None and cfg.scenario != "baseline":
        argv += ["--malicious-rate", str(cfg.malicious_rate)]
    if cfg.dataset == "visdrone":
        argv += ["--visdrone-root", cfg.visdrone_root]
    argv += ["--partition", cfg.partition]

    if role == "server":
        argv += [
            "--bind-host", cfg.bind_host,
            "--port", str(cfg.port),
            "--workdir", cfg.workdir,
            "--timeout", str(cfg.timeout),
        ]
    elif role == "client":
        argv += [
            "--server-host", cfg.server_host,
            "--port", str(cfg.port),
            "--client-id", str(cfg.client_id),
            "--workdir", cfg.workdir,
            "--timeout", str(cfg.timeout),
        ]
    elif role == "orchestrator":
        argv += ["--workdir", cfg.workdir, "--timeout", str(cfg.timeout)]
    return argv


def _dispatch(argv: list[str]) -> int:
    command = [sys.executable, str(SMOKE), *argv]
    print("\n将执行：")
    print(" ".join(shlex.quote(part) for part in command))
    if _DRY_RUN:
        print("\n[dry-run] 未执行。")
        return 0
    from run_distributed_smoke import main as smoke_main

    return smoke_main(argv)


def _summary(cfg: Config) -> None:
    _section("确认")
    print(f"角色        : {cfg.role}")
    print(f"数据集      : {cfg.dataset}")
    print(f"攻击/防御   : {cfg.scenario} / {cfg.defense}")
    print(f"客户端/轮数 : {cfg.num_clients} / {cfg.rounds}")
    print(f"数据        : {'全量' if cfg.full_data else f'每端 {cfg.max_samples} 条'}")
    print(f"batch/lr    : {cfg.batch_size} / {cfg.lr}")
    print(f"本地 epoch  : {cfg.local_epochs}")
    print(f"设备        : {cfg.device}")
    if cfg.dataset == "visdrone":
        print(f"YOLO        : {cfg.model_size} @ {cfg.image_size}, AMP={cfg.amp}")
    print(f"恶意率      : {'1 个客户端' if cfg.malicious_rate is None else cfg.malicious_rate}")
    print(f"分区        : {cfg.partition}")
    if cfg.role in ("server", "client", "orchestrator"):
        print(f"日志目录    : {cfg.workdir}")
    if cfg.role == "server":
        print(f"监听        : {cfg.bind_host}:{cfg.port}")
    if cfg.role == "client":
        print(f"连接        : {cfg.server_host}:{cfg.port}, client-id={cfg.client_id}")


def _interactive() -> int:
    cfg = Config()
    _ask_common(cfg)
    if cfg.role == "exit":
        print("已退出。")
        return 0

    if cfg.role == "prepare":
        return _run_prepare(cfg)

    if cfg.role in ("server", "client"):
        has_partition = Path(cfg.partition).exists()
        print(f"\n分区文件：{cfg.partition}（{'存在' if has_partition else '不存在'}）")
        do_prepare = _ask_bool("是否先在本机生成/重新生成分区", not has_partition)
        if do_prepare:
            rc = _run_prepare(cfg)
            if rc != 0:
                return rc
            if not _ask_bool("分区已完成，继续启动本机角色", True):
                return 0

    _ask_training(cfg)
    _ask_network(cfg)
    _summary(cfg)
    if not _ask_bool("确认执行", True):
        print("已取消。")
        return 0
    return _dispatch(_build_argv(cfg))


_DRY_RUN = False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="交互式启动 DPFLA 分布式实验")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令，不真正启动")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _DRY_RUN
    args = _build_parser().parse_args(argv)
    _DRY_RUN = args.dry_run
    try:
        return _interactive()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
