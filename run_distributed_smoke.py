"""run_distributed_smoke.py — CIFAR10 / VisDrone 分布式冒烟测试。

复用单终端项目的训练 / 攻击 / 防御 / 评估代码，通过 `distributed_fl.py`
的 TCP 传输层跑 2 轮小规模实验，验证分布式链路可用。

用法：
    # 跑 Phase A 全部 10 条（CIFAR 5 条 + VisDrone 5 条）
    python run_distributed_smoke.py --phase a

    # 只跑一条
    python run_distributed_smoke.py --dataset cifar10 --scenario label_flipping --defense dpfla

    # 多终端手动模式（server 与 client 在不同机器）：
    # 1) 生成分区（在 server 机器或任意机器生成后拷贝到各终端）
    python run_distributed_smoke.py --role prepare --dataset cifar10 --num-clients 5 \
        --scenario minmax --partition /shared/partitions.json --max-samples 32
    # 2) server（监听所有网卡）
    python run_distributed_smoke.py --role server --dataset cifar10 --scenario minmax \
        --defense fedavg --num-clients 5 --rounds 2 --bind-host 0.0.0.0 --port 5000 \
        --partition /shared/partitions.json --workdir /shared/server_run --timeout 1800
    # 3) 每台 client 机器（同一份代码 + 同一份数据 + 拷贝 partitions.json）
    python run_distributed_smoke.py --role client --dataset cifar10 --scenario minmax \
        --defense fedavg --num-clients 5 --rounds 2 --server-host <server-ip> --port 5000 \
        --client-id 0 --partition /shared/partitions.json --workdir ./client0 --timeout 1800

设计说明：
- 只验证链路，不追求指标；参数显著小于正式实验。
- 每个 client 是独立进程，server 也是独立进程，和真实分布式形态一致。
- 多终端模式下 server 用 `--bind-host 0.0.0.0`，client 用 `--server-host <server-ip>`；
  各终端需有相同的 repo 与数据集路径（VisDrone 可用 `--visdrone-root` 覆盖）。
- 分区使用 `make_partitions` 精确 IID，再按 max_samples 截断；不等同于
  单终端 `distribute_dataset` 的 dirichlet(α=1e6) 划分。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from distributed_fl import FLClient, FLServer, ClientContext, load_partition, make_partitions, setup_logging

DATASET_CIFAR = "cifar10"
DATASET_VISDRONE = "visdrone"
VISDRONE_ROOT = Path("/root/autodl-tmp/data/visdrone")
DEFENSES = ("fedavg", "dpfla", "mkrum", "tmean")
SCENARIOS = ("baseline", "label_flipping", "minmax", "ipm", "alie", "signflipping")
SERVER_ATTACKS = ("minmax", "ipm", "alie", "signflipping")


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def _default_num_clients(dataset: str, defense: str) -> int:
    if dataset == DATASET_CIFAR:
        return 5
    # VisDrone Phase A 用 3 端；MKrum 需要 n >= 2f+3（f=1 → n>=5）
    return 5 if defense == "mkrum" else 3


def _default_max_samples(dataset: str) -> int:
    return 32 if dataset == DATASET_CIFAR else 4


def _default_batch(dataset: str) -> int:
    return 16 if dataset == DATASET_CIFAR else 2


def _default_lr(dataset: str) -> float:
    return 0.01 if dataset == DATASET_CIFAR else 2e-4


@dataclass
class SmokeConfig:
    role: str
    dataset: str
    scenario: str
    defense: str
    num_clients: int
    rounds: int = 2
    seed: int = 42
    max_samples: int = 32
    batch_size: int = 16
    lr: float = 0.01
    device: str = "cuda"
    port: int = 0
    bind_host: str = "127.0.0.1"
    server_host: str = "127.0.0.1"
    visdrone_root: str = str(VISDRONE_ROOT)
    workdir: Path = Path("logs_dist_smoke")
    partition_path: Optional[Path] = None
    client_id: Optional[str] = None
    timeout_s: float = 600.0
    phase: str = "a"

    @property
    def is_cifar(self) -> bool:
        return self.dataset == DATASET_CIFAR

    @property
    def dataset_name(self) -> str:
        return "CIFAR10" if self.is_cifar else "VisDrone"

    @property
    def model_name(self) -> str:
        return "ResNet18CIFAR" if self.is_cifar else "YOLO"

    @property
    def malicious_ids(self) -> Tuple[int, ...]:
        if self.scenario == "baseline":
            return ()
        return (0,)

    @property
    def attack_type(self) -> str:
        return "no_attack" if self.scenario == "baseline" else self.scenario


def _config_from_args(args: argparse.Namespace) -> SmokeConfig:
    dataset = args.dataset
    defense = args.defense or "fedavg"
    scenario = args.scenario or "baseline"
    num_clients = args.num_clients or _default_num_clients(dataset, defense)
    if defense == "mkrum":
        if args.num_clients is not None and int(args.num_clients) < 5:
            raise ValueError(
                "Multi-Krum requires at least 5 clients (f=1); "
                "for a 2-client multi-machine test use --defense fedavg or dpfla"
            )
        num_clients = max(num_clients, 5)
    max_samples = args.max_samples or _default_max_samples(dataset)
    batch_size = args.batch_size or _default_batch(dataset)
    lr = args.lr if args.lr is not None else _default_lr(dataset)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return SmokeConfig(
        role=args.role,
        dataset=dataset,
        scenario=scenario,
        defense=defense,
        num_clients=num_clients,
        rounds=args.rounds,
        seed=args.seed,
        max_samples=max_samples,
        batch_size=batch_size,
        lr=lr,
        device=device,
        port=args.port,
        bind_host=args.bind_host,
        server_host=args.server_host,
        visdrone_root=args.visdrone_root,
        workdir=Path(args.workdir).resolve() if args.workdir else (ROOT / "logs_dist_smoke").resolve(),
        partition_path=Path(args.partition).resolve() if args.partition else None,
        client_id=args.client_id,
        timeout_s=args.timeout,
        phase=args.phase,
    )


def _apply_env(cfg: SmokeConfig) -> None:
    """子进程和父进程都设置；确保导入项目模块前生效。"""
    is_visdrone = cfg.dataset == DATASET_VISDRONE
    os.environ["YOLO_MODEL_SIZE"] = "n"
    os.environ["FL_IMG_SIZE"] = "320" if is_visdrone else "32"
    os.environ["FL_AMP"] = "0"
    os.environ["FL_TRAIN_BATCH_SIZE"] = str(cfg.batch_size)
    os.environ["FL_TEST_BATCH_SIZE"] = "8" if is_visdrone else "64"
    os.environ["FL_LOCAL_EPOCHS"] = "1"
    os.environ["FL_GLOBAL_ROUNDS"] = str(cfg.rounds)
    os.environ["FL_LOCAL_LR"] = str(cfg.lr)
    os.environ["FL_MOMENTUM"] = "0.9"
    os.environ["FL_GRADIENT_SCALE"] = "1"
    os.environ["FL_VISDRONE_LR_SCHEDULE"] = "constant"
    os.environ["VISDRONE_DATALOADER_WORKERS"] = "0"
    os.environ["FL_VISDRONE_ROOT"] = cfg.visdrone_root
    os.environ["FL_IPM_SCALE"] = os.environ.get("FL_IPM_SCALE", "0.1")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# 数据 / 模型 / 分区
# ---------------------------------------------------------------------------

def _get_visdrone(cfg: SmokeConfig):
    """与 server.py 对齐：主项目优先使用 /root/autodl-tmp/data/visdrone。"""
    from federated_learning.utils.sampling import get_visdrone

    root = cfg.visdrone_root if Path(cfg.visdrone_root).exists() else None
    return get_visdrone(root_path=root)


def _build_trainset(cfg: SmokeConfig):
    if cfg.is_cifar:
        from federated_learning.utils.sampling import get_cifar10

        trainset, _ = get_cifar10()
        return trainset
    trainset, _ = _get_visdrone(cfg)
    return trainset


def _build_testloader(cfg: SmokeConfig):
    if cfg.is_cifar:
        from federated_learning.utils.sampling import get_cifar10

        _, testset = get_cifar10()
        subset = Subset(testset, list(range(min(256, len(testset)))))
        return DataLoader(subset, batch_size=64, shuffle=False, num_workers=0)

    from federated_learning.client import detection_collate_fn

    _, testset = _get_visdrone(cfg)
    return DataLoader(
        testset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate_fn,
    )


def _build_model(cfg: SmokeConfig):
    from federated_learning.models import setup_model

    return setup_model(cfg.model_name, num_classes=10)


def _create_partition(cfg: SmokeConfig, path: Path) -> None:
    trainset = _build_trainset(cfg)
    data = make_partitions(len(trainset), cfg.num_clients, seed=cfg.seed, mode="iid")
    parts = {
        str(client): list(indices)[: cfg.max_samples]
        for client, indices in data["partitions"].items()
    }

    # CIFAR label flipping：保证恶意端至少含 min(8, max_samples) 个 class 5 样本
    if cfg.is_cifar and cfg.scenario == "label_flipping" and cfg.malicious_ids:
        targets = getattr(trainset, "targets", None)
        if targets is not None:
            class5 = [index for index, target in enumerate(targets) if int(target) == 5]
            malicious_key = str(cfg.malicious_ids[0])
            keep = class5[: min(8, cfg.max_samples)]
            keep_set = set(keep)
            rest = [index for index in parts[malicious_key] if index not in keep_set]
            parts[malicious_key] = keep + rest[: max(0, cfg.max_samples - len(keep))]

    data["partitions"] = parts
    data["smoke"] = {
        "dataset": cfg.dataset,
        "scenario": cfg.scenario,
        "defense": cfg.defense,
        "num_clients": cfg.num_clients,
        "max_samples": cfg.max_samples,
        "seed": cfg.seed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# 客户端适配器
# ---------------------------------------------------------------------------

def _make_train_fn(cfg: SmokeConfig):
    from federated_learning.client import Client
    from federated_learning.datasets import CustomDataset

    if cfg.partition_path is None:
        raise ValueError("client role requires --partition")
    indices = load_partition(str(cfg.partition_path), cfg.client_id)
    trainset = _build_trainset(cfg)
    local_data = CustomDataset(trainset, indices=indices)

    client_type = "attacker" if int(cfg.client_id) in cfg.malicious_ids else "honest"
    client = Client(
        int(cfg.client_id),
        f"Client {cfg.client_id}",
        local_data,
        labels=[],
        criterion=nn.CrossEntropyLoss(),
        device=cfg.device,
        local_epochs=1,
        local_bs=cfg.batch_size,
        local_lr=cfg.lr,
        local_momentum=0.9,
        client_type=client_type,
    )
    model = _build_model(cfg)
    mapping = {"mapping": {5: 3}, "flip_ratio": 1.0} if cfg.scenario == "label_flipping" else None
    attack_type = cfg.attack_type

    def train(global_state, context: ClientContext):
        model.load_state_dict(global_state)
        model.to(cfg.device)
        Client._performed_attacks = 0
        update, _grad, _local_model, loss, attacked, _t = client.participant_update(
            global_epoch=context.round - 1,
            model=model,
            attack_type=attack_type,
            malicious_behavior_rate=1.0 if client_type == "attacker" and cfg.scenario == "label_flipping" else 0.0,
            source_class=5 if cfg.scenario == "label_flipping" else None,
            target_class=3 if cfg.scenario == "label_flipping" else None,
            dataset_name=cfg.dataset_name,
            untarget=False,
            label_flip_mapping=mapping,
        )
        return update, {
            "loss": float(loss),
            "attacked": int(attacked),
            "client_type": client_type,
            "round": context.round,
        }

    return train


def _run_client(cfg: SmokeConfig) -> None:
    _apply_env(cfg)
    run_dir = cfg.workdir
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("INFO", str(run_dir / f"client_{cfg.client_id}.module.log"))
    if cfg.port <= 0:
        raise ValueError("client role requires a positive --port")
    partition = load_partition(str(cfg.partition_path), cfg.client_id)
    train_fn = _make_train_fn(cfg)
    FLClient(
        client_id=str(cfg.client_id),
        server_host=cfg.server_host,
        server_port=cfg.port,
        train_fn=train_fn,
        num_samples=len(partition),
        connect_retries=120,
        retry_interval_s=1.0,
        timeout_s=max(1200.0, cfg.timeout_s),
        device=cfg.device,
    ).run()


# ---------------------------------------------------------------------------
# 服务端适配器
# ---------------------------------------------------------------------------

def _make_server_hooks(cfg: SmokeConfig):
    from federated_learning.attack_alg.server_attacks import apply_server_attack
    from federated_learning.fl_algorithm import DPFLA, average_weights, multi_krum, trimmed_mean
    from federated_learning.fl_core import FL

    model = _build_model(cfg)
    initial_state = model.state_dict()
    eval_model = _build_model(cfg)
    test_loader = _build_testloader(cfg)
    criterion = nn.CrossEntropyLoss()
    malicious = set(cfg.malicious_ids)
    defense = cfg.defense.lower()

    dpfla = DPFLA() if defense == "dpfla" else None
    dpfla_models = [_build_model(cfg).cpu() for _ in range(cfg.num_clients + 1)] if dpfla else []

    def aggregate(updates: List[Dict[str, torch.Tensor]], meta: List[Dict[str, Any]],
                  global_state: Optional[Dict[str, torch.Tensor]]):
        client_ids = [int(item["client_id"]) for item in meta]
        clients_types = [
            "Bad update" if client_id in malicious else "Good update"
            for client_id in client_ids
        ]

        if cfg.scenario in SERVER_ATTACKS:
            if global_state is None:
                raise ValueError("server attack requires global_state")
            updates, _attacked = apply_server_attack(
                cfg.scenario,
                updates,
                clients_types,
                global_state,
                num_attackers=len(malicious),
                num_clients=len(updates),
            )

        if defense == "fedavg":
            return average_weights(updates, [1 for _ in updates], float16_floats=False)
        if defense == "dpfla":
            global_model = dpfla_models[0]
            global_model.load_state_dict(global_state)
            local_models = []
            for index, update in enumerate(updates):
                local_model = dpfla_models[index + 1]
                local_model.load_state_dict(update)
                local_models.append(local_model)
            scores = dpfla.score(
                global_model,
                local_models,
                clients_types,
                selected_clients=client_ids,
                p=128,
                w=10,
                model_name=cfg.model_name,
                val_loader=None,
                device=None,
                use_validation=False,
            )
            return average_weights(updates, scores, float16_floats=False)
        if defense == "mkrum":
            return multi_krum(updates, num_attackers=len(malicious))[0]
        if defense == "tmean":
            trim_ratio = (len(malicious) / len(updates)) if malicious else 0.0
            return trimmed_mean(updates, trim_ratio=trim_ratio)
        raise ValueError(f"unknown defense: {cfg.defense}")

    def evaluate(state: Dict[str, torch.Tensor], round_index: int) -> Dict[str, Any]:
        eval_model.load_state_dict(state)
        eval_model.to(cfg.device)
        dummy = SimpleNamespace(
            device=cfg.device,
            criterion=criterion,
            num_classes=10,
            labels_dict={index: str(index) for index in range(10)},
        )
        metric, loss = FL.test(dummy, eval_model, cfg.device, test_loader, dataset_name=cfg.dataset_name)
        return {"metric": float(metric), "loss": float(loss), "round": round_index}

    return initial_state, aggregate, evaluate


def _run_prepare(cfg: SmokeConfig) -> None:
    """只生成分区 JSON，供多终端手动模式拷贝到各客户端机器。"""
    _apply_env(cfg)
    if cfg.partition_path is None:
        raise ValueError("prepare role requires --partition")
    _create_partition(cfg, cfg.partition_path)
    print(f"partition written: {cfg.partition_path}")
    print(f"dataset={cfg.dataset} clients={cfg.num_clients} "
          f"max_samples={cfg.max_samples} seed={cfg.seed} scenario={cfg.scenario}")


def _run_server(cfg: SmokeConfig) -> None:
    _apply_env(cfg)
    if cfg.port <= 0:
        raise ValueError("server role requires a positive --port")
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    setup_logging("INFO", str(cfg.workdir / "server.module.log"))

    initial_state, aggregate, evaluate = _make_server_hooks(cfg)
    server = FLServer(
        initial_state=initial_state,
        num_clients=cfg.num_clients,
        rounds=cfg.rounds,
        aggregate_fn=aggregate,
        evaluate_fn=evaluate,
        host=cfg.bind_host,
        port=cfg.port,
        registration_timeout_s=max(120.0, cfg.timeout_s / 2),
        round_timeout_s=cfg.timeout_s,
        log_path=str(cfg.workdir / "rounds.jsonl"),
    )

    ready_path = cfg.workdir / "server.ready"

    def mark_ready() -> None:
        if server.wait_ready(120.0):
            ready_path.write_text(str(server.port), encoding="utf-8")

    threading.Thread(target=mark_ready, daemon=True).start()
    server.serve()


# ---------------------------------------------------------------------------
# 编排
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _wait_for_ready(path: Path, timeout_s: float, server_proc: subprocess.Popen) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if server_proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {server_proc.returncode}")
        if path.exists():
            return
        time.sleep(0.2)
    raise TimeoutError(f"server ready file {path} not found within {timeout_s}s")


def _child_env(cfg: SmokeConfig) -> Dict[str, str]:
    env = os.environ.copy()
    env.update({
        "YOLO_MODEL_SIZE": "n",
        "FL_IMG_SIZE": "320" if cfg.dataset == DATASET_VISDRONE else "32",
        "FL_AMP": "0",
        "FL_TRAIN_BATCH_SIZE": str(cfg.batch_size),
        "FL_TEST_BATCH_SIZE": "8" if cfg.dataset == DATASET_VISDRONE else "64",
        "FL_LOCAL_EPOCHS": "1",
        "FL_GLOBAL_ROUNDS": str(cfg.rounds),
        "FL_LOCAL_LR": str(cfg.lr),
        "FL_MOMENTUM": "0.9",
        "FL_GRADIENT_SCALE": "1",
        "FL_VISDRONE_LR_SCHEDULE": "constant",
        "VISDRONE_DATALOADER_WORKERS": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    })
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_one(cfg: SmokeConfig) -> Dict[str, Any]:
    _apply_env(cfg)
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    partition_path = cfg.workdir / "partitions.json"
    _create_partition(cfg, partition_path)
    cfg.partition_path = partition_path
    cfg.port = _find_free_port()
    ready_path = cfg.workdir / "server.ready"
    if ready_path.exists():
        ready_path.unlink()

    env = _child_env(cfg)
    common = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dataset", cfg.dataset,
        "--scenario", cfg.scenario,
        "--defense", cfg.defense,
        "--num-clients", str(cfg.num_clients),
        "--rounds", str(cfg.rounds),
        "--seed", str(cfg.seed),
        "--max-samples", str(cfg.max_samples),
        "--batch-size", str(cfg.batch_size),
        "--lr", str(cfg.lr),
        "--device", cfg.device,
        "--port", str(cfg.port),
        "--workdir", str(cfg.workdir),
        "--partition", str(partition_path),
        "--timeout", str(cfg.timeout_s),
        "--bind-host", cfg.bind_host,
        "--server-host", cfg.server_host,
        "--visdrone-root", cfg.visdrone_root,
    ]
    server_log_path = cfg.workdir / "server.log"
    server_log = open(server_log_path, "w", encoding="utf-8")
    server_proc = subprocess.Popen(
        common + ["--role", "server"],
        cwd=str(ROOT), env=env, stdout=server_log, stderr=subprocess.STDOUT,
    )

    client_procs: List[subprocess.Popen] = []
    client_logs = []
    try:
        _wait_for_ready(ready_path, min(180.0, cfg.timeout_s), server_proc)
        for client_id in range(cfg.num_clients):
            log_path = cfg.workdir / f"client_{client_id}.log"
            log_handle = open(log_path, "w", encoding="utf-8")
            client_logs.append((log_handle, log_path))
            client_procs.append(subprocess.Popen(
                common + ["--role", "client", "--client-id", str(client_id)],
                cwd=str(ROOT), env=env, stdout=log_handle, stderr=subprocess.STDOUT,
            ))

        deadline = time.time() + cfg.timeout_s
        for proc in client_procs:
            proc.wait(timeout=max(1.0, deadline - time.time()))
        server_proc.wait(timeout=max(1.0, deadline - time.time()))
    except Exception:
        for proc in client_procs:
            if proc.poll() is None:
                proc.kill()
        if server_proc.poll() is None:
            server_proc.kill()
        raise
    finally:
        server_log.close()
        for handle, _ in client_logs:
            handle.close()

    server_text = server_log_path.read_text(encoding="utf-8", errors="replace")
    client_texts = [
        path.read_text(encoding="utf-8", errors="replace")
        for _, path in client_logs
    ]
    rounds_path = cfg.workdir / "rounds.jsonl"
    rounds = []
    if rounds_path.exists():
        for line in rounds_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rounds.append(json.loads(line))

    metrics = [item.get("eval", {}).get("metric") for item in rounds if item.get("eval")]
    attack_seen = False
    if cfg.scenario == "label_flipping":
        attack_seen = any("Label flipping attack launched" in text for text in client_texts)
    elif cfg.scenario in SERVER_ATTACKS:
        attack_seen = "服务端注入攻击" in server_text and "已应用" in server_text

    defense_seen = cfg.defense == "dpfla" and "聚合权重" in server_text
    updates_seen = sum(text.count("update sent") for text in client_texts)

    client_codes = [proc.returncode for proc in client_procs]
    status = (
        server_proc.returncode == 0
        and all(code == 0 for code in client_codes)
        and len(rounds) == cfg.rounds
        and len(metrics) == cfg.rounds
        and all(value is not None and abs(float(value)) < 1e9 for value in metrics)
        and updates_seen >= cfg.num_clients * cfg.rounds
        and (attack_seen or cfg.scenario == "baseline")
        and (defense_seen or cfg.defense != "dpfla")
    )

    result = {
        "dataset": cfg.dataset,
        "scenario": cfg.scenario,
        "defense": cfg.defense,
        "num_clients": cfg.num_clients,
        "rounds": cfg.rounds,
        "max_samples": cfg.max_samples,
        "device": cfg.device,
        "status": "PASS" if status else "FAIL",
        "server_returncode": server_proc.returncode,
        "client_returncodes": client_codes,
        "round_metrics": metrics,
        "updates_seen": updates_seen,
        "attack_seen": attack_seen,
        "defense_seen": defense_seen,
        "workdir": str(cfg.workdir),
    }
    (cfg.workdir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _phase_a_runs(args: argparse.Namespace) -> List[SmokeConfig]:
    runs: List[SmokeConfig] = []
    for dataset in (DATASET_CIFAR, DATASET_VISDRONE):
        scenarios = ["baseline", "minmax"] if dataset == DATASET_VISDRONE else ["baseline", "label_flipping", "minmax"]
        if dataset == DATASET_VISDRONE:
            scenarios.append("ipm")
        for scenario in scenarios:
            defenses = ["fedavg", "dpfla"] if scenario != "baseline" else ["fedavg"]
            for defense in defenses:
                runs.append(_config_from_args(argparse.Namespace(
                    **{**vars(args), "dataset": dataset, "scenario": scenario, "defense": defense}
                )))
    return runs


def _phase_b_runs(args: argparse.Namespace) -> List[SmokeConfig]:
    runs: List[SmokeConfig] = []
    for dataset in (DATASET_CIFAR, DATASET_VISDRONE):
        scenarios = ["label_flipping", "minmax"] if dataset == DATASET_CIFAR else ["minmax", "ipm"]
        for scenario in scenarios:
            for defense in ("mkrum", "tmean"):
                runs.append(_config_from_args(argparse.Namespace(
                    **{**vars(args), "dataset": dataset, "scenario": scenario, "defense": defense}
                )))
    return runs


def _orchestrate(args: argparse.Namespace) -> int:
    if args.scenario or args.defense:
        runs = [_config_from_args(args)]
    elif args.phase == "a":
        runs = _phase_a_runs(args)
    elif args.phase == "b":
        runs = _phase_b_runs(args)
    else:
        runs = _phase_a_runs(args) + _phase_b_runs(args)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = Path(args.workdir).resolve() if args.workdir else (ROOT / "logs_dist_smoke").resolve()
    root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for index, cfg in enumerate(runs):
        cfg.workdir = root / cfg.dataset / cfg.scenario / cfg.defense / f"{stamp}_{index:02d}"
        cfg.timeout_s = args.timeout or (900.0 if cfg.dataset == DATASET_VISDRONE else 300.0)
        print(f"\n=== [{index + 1}/{len(runs)}] {cfg.dataset} | {cfg.scenario} | {cfg.defense} "
              f"| clients={cfg.num_clients} ===")
        started = time.time()
        try:
            result = _run_one(cfg)
            result["elapsed_s"] = round(time.time() - started, 1)
            results.append(result)
            print(f"  -> {result['status']} metrics={result['round_metrics']} "
                  f"updates={result['updates_seen']} elapsed={result['elapsed_s']}s")
        except Exception as exc:  # noqa: BLE001 - 编排层兜底，继续下一条
            results.append({
                "dataset": cfg.dataset,
                "scenario": cfg.scenario,
                "defense": cfg.defense,
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "workdir": str(cfg.workdir),
                "elapsed_s": round(time.time() - started, 1),
            })
            print(f"  -> FAIL {type(exc).__name__}: {exc}")

    summary_dir = root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"summary_{stamp}.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Distributed smoke results", "", f"Generated: {stamp}", "",
             "| dataset | scenario | defense | clients | status | round metrics | elapsed |",
             "|---|---|---|---|---|---|---|"]
    for item in results:
        lines.append(
            f"| {item.get('dataset')} | {item.get('scenario')} | {item.get('defense')} | "
            f"{item.get('num_clients', '-')} | {item.get('status')} | "
            f"{item.get('round_metrics', item.get('error', '-'))} | {item.get('elapsed_s', '-')}s |"
        )
    summary_md = summary_dir / f"summary_{stamp}.md"
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    passed = sum(1 for item in results if item.get("status") == "PASS")
    print(f"\n=== summary: {passed}/{len(results)} PASS ===")
    print(f"JSON: {summary_path}")
    print(f"MD:   {summary_md}")
    return 0 if passed == len(results) else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="distributed smoke tests for DPFLA")
    parser.add_argument("--role", default="orchestrator", choices=("orchestrator", "prepare", "server", "client"))
    parser.add_argument("--dataset", default=DATASET_CIFAR, choices=(DATASET_CIFAR, DATASET_VISDRONE))
    parser.add_argument("--scenario", default=None, choices=SCENARIOS)
    parser.add_argument("--defense", default=None, choices=DEFENSES)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--bind-host", default="127.0.0.1", help="server bind host; multi-terminal use 0.0.0.0")
    parser.add_argument("--server-host", default="127.0.0.1", help="client connects to this server host")
    parser.add_argument("--visdrone-root", default=str(VISDRONE_ROOT))
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--client-id", default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--phase", default="a", choices=("a", "b", "all"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = _config_from_args(args)
    if cfg.role == "prepare":
        _run_prepare(cfg)
        return 0
    if cfg.role == "server":
        _run_server(cfg)
        return 0
    if cfg.role == "client":
        _run_client(cfg)
        return 0
    return _orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
