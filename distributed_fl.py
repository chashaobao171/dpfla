"""distributed_fl.py — 单文件联邦学习传输层（裸 TCP，可信局域网）。

设计边界
--------
- 只依赖 PyTorch + Python 标准库，**不 import 本项目任何模块**。
- 传输整份 ``state_dict``（DPFLA 需要完整模型，不是 delta）。
- 聚合 / 本地训练 / 服务端评估全部通过可调用对象注入，便于 DPFLA、
  Multi-Krum、Trimmed-Mean 等复用。
- 消息带 ``round``，过期/重复更新会被丢弃。
- 分区 JSON 固定 seed，落盘可复现。
- 无 TLS / 无鉴权 / 无断线重连：仅用于可信局域网或本机多进程实验。

最小用法
--------
    from distributed_fl import make_partitions, FLServer, FLClient

    make_partitions(50000, 10, seed=42, path="partitions.json")

    server = FLServer(initial_state=model.state_dict(), num_clients=10, rounds=15,
                      aggregate_fn=my_aggregate, evaluate_fn=my_evaluate)
    server.serve()

    client = FLClient("3", "127.0.0.1", 5000, train_fn=my_train,
                      partition_path="partitions.json")
    client.run()

多终端时：server 端 ``FLServer(host="0.0.0.0")``，client 端把 ``"127.0.0.1"``
换成 server 的局域网 IP；各终端需持有同一份分区 JSON 与数据集。

``train_fn(global_state, ctx) -> (new_state, metrics)``，其中 ``ctx`` 是
``ClientContext``；``aggregate_fn(updates, meta, global_state) -> state_dict``。

冒烟测试
--------
    python distributed_fl.py selftest

CLI
---
    python distributed_fl.py server --init-state init.pt --num-clients 10 --rounds 15
    python distributed_fl.py client --server 127.0.0.1:5000 --client-id 3 \
        --train my_module:my_train --partition partitions.json
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import logging
import queue
import random
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

__all__ = [
    "ClientContext",
    "FLClient",
    "FLServer",
    "fedavg_state_dicts",
    "load_partition",
    "make_partitions",
    "partition_sizes",
    "run_client",
    "run_server",
    "setup_logging",
]

logger = logging.getLogger("distributed_fl")

_MAX_HEADER_BYTES = 64 * 1024
_MAX_PAYLOAD_BYTES = 4 * 1024 ** 3

StateDict = Dict[str, torch.Tensor]
TrainFn = Callable[[StateDict, "ClientContext"], Tuple[StateDict, Dict[str, Any]]]
AggregateFn = Callable[[List[StateDict], List[Dict[str, Any]], Optional[StateDict]], StateDict]
EvaluateFn = Callable[[StateDict, int], Dict[str, Any]]


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO", log_path: Optional[str] = None) -> None:
    """配置模块级 logger（不覆盖项目自己的日志）。"""
    logger.handlers.clear()
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_path:
        path = Path(log_path)
        if str(path.parent) not in ("", "."):
            path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    logger.propagate = False


# ---------------------------------------------------------------------------
# 帧协议：4B header_len | JSON header | payload
# ---------------------------------------------------------------------------

class ProtocolError(RuntimeError):
    pass


class RoundTimeoutError(RuntimeError):
    pass


class ClientConnectionError(RuntimeError):
    pass


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = sock.recv(min(size - len(buf), 1024 * 1024))
        if not chunk:
            raise ConnectionError("peer closed connection")
        buf.extend(chunk)
    return bytes(buf)


def _send_frame(sock: socket.socket, header: Dict[str, Any], payload: bytes = b"") -> None:
    header = dict(header)
    header["payload_len"] = len(payload)
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    if len(header_bytes) > _MAX_HEADER_BYTES:
        raise ProtocolError(f"header too large: {len(header_bytes)} bytes")
    sock.sendall(struct.pack("!I", len(header_bytes)))
    sock.sendall(header_bytes)
    if payload:
        sock.sendall(payload)


def _recv_frame(sock: socket.socket) -> Tuple[Dict[str, Any], bytes]:
    header_len = struct.unpack("!I", _recv_exact(sock, 4))[0]
    if header_len == 0 or header_len > _MAX_HEADER_BYTES:
        raise ProtocolError(f"invalid header length: {header_len}")
    header = json.loads(_recv_exact(sock, header_len).decode("utf-8"))
    payload_len = int(header.get("payload_len", 0))
    if payload_len < 0 or payload_len > _MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"invalid payload length: {payload_len}")
    payload = _recv_exact(sock, payload_len) if payload_len else b""
    return header, payload


# ---------------------------------------------------------------------------
# state_dict 序列化 / 校验
# ---------------------------------------------------------------------------

def _to_cpu_state(state: StateDict) -> StateDict:
    if not isinstance(state, dict):
        raise TypeError("state_dict must be a dict")
    return {
        key: (value.detach().cpu().clone() if torch.is_tensor(value) else value)
        for key, value in state.items()
    }


def _dump_state(state: StateDict) -> bytes:
    buffer = io.BytesIO()
    torch.save(_to_cpu_state(state), buffer)
    return buffer.getvalue()


def _load_state(payload: bytes) -> StateDict:
    if not payload:
        raise ProtocolError("empty state payload")
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("payload is not a state_dict")
    return state


def _validate_state(state: StateDict, reference: StateDict) -> StateDict:
    if not isinstance(state, dict):
        raise TypeError("aggregate_fn / train_fn must return a dict")
    if set(state) != set(reference):
        missing = sorted(set(reference) - set(state))
        extra = sorted(set(state) - set(reference))
        raise ValueError(f"state_dict keys mismatch, missing={missing[:5]}, extra={extra[:5]}")

    result: StateDict = {}
    for key, value in state.items():
        ref = reference[key]
        if torch.is_tensor(value) and torch.is_tensor(ref):
            if tuple(value.shape) != tuple(ref.shape):
                raise ValueError(f"shape mismatch at {key}: {tuple(value.shape)} vs {tuple(ref.shape)}")
            result[key] = value.detach().cpu().clone()
        else:
            result[key] = value
    return result


def _client_sort_key(client_id: str) -> Tuple[int, Any]:
    """数字 ID 按数值排，非数字按字符串排，避免 "10" 排在 "2" 前。"""
    try:
        return 0, int(client_id)
    except (TypeError, ValueError):
        return 1, str(client_id)


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if torch.is_tensor(obj):
        tensor = obj.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:  # noqa: BLE001 - 兼容 numpy 标量
            pass
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return obj.tolist()
        except Exception:  # noqa: BLE001 - 兼容 numpy 数组
            pass
    return str(obj)


# ---------------------------------------------------------------------------
# 默认 FedAvg：浮点参数加权平均，整数/bool buffer 取第一个客户端
# ---------------------------------------------------------------------------

def fedavg_state_dicts(
    updates: Sequence[StateDict],
    meta: Optional[Sequence[Dict[str, Any]]] = None,
    global_state: Optional[StateDict] = None,
) -> StateDict:
    """按 ``meta[i]['num_samples']`` 加权平均；缺省等权。

    ``global_state`` 仅为兼容服务端注入攻击的聚合器签名（本函数不使用）。
    非浮点 buffer（如 ``num_batches_tracked``）不参与平均，直接取第一份，
    避免被转成 float 后破坏 dtype。
    """
    if not updates:
        raise ValueError("updates is empty")

    reference = updates[0]
    if not isinstance(reference, dict):
        raise TypeError("each update must be a state_dict")
    for update in updates[1:]:
        if set(update) != set(reference):
            raise ValueError("all updates must have identical state_dict keys")

    weights: List[float] = []
    for index, _ in enumerate(updates):
        if meta is not None and index < len(meta):
            weights.append(float(meta[index].get("num_samples", 1) or 1))
        else:
            weights.append(1.0)
    total = sum(weights)
    if total <= 0:
        weights = [1.0] * len(updates)
        total = float(len(updates))

    result: StateDict = {}
    for key, ref_value in reference.items():
        values = [update[key] for update in updates]
        if not torch.is_tensor(ref_value):
            result[key] = ref_value
            continue
        if not ref_value.is_floating_point():
            result[key] = ref_value.clone()
            continue

        acc_dtype = torch.float64 if ref_value.dtype == torch.float64 else torch.float32
        acc = torch.zeros_like(ref_value, dtype=acc_dtype)
        for value, weight in zip(values, weights):
            if not torch.is_tensor(value):
                raise TypeError(f"non-tensor value at {key}")
            if tuple(value.shape) != tuple(ref_value.shape):
                raise ValueError(f"shape mismatch at {key}")
            acc += value.to(device=ref_value.device, dtype=acc_dtype) * (weight / total)
        result[key] = acc.to(ref_value.dtype)
    return result


# ---------------------------------------------------------------------------
# 数据分区：固定 seed，落盘 JSON
# ---------------------------------------------------------------------------

def _dirichlet(rng: random.Random, alpha: float, size: int) -> List[float]:
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    draws = [rng.gammavariate(alpha, 1.0) for _ in range(size)]
    total = sum(draws)
    if total <= 0:
        return [1.0 / size] * size
    return [draw / total for draw in draws]


def make_partitions(
    num_samples: int,
    num_clients: int,
    seed: int = 42,
    mode: str = "iid",
    alpha: float = 0.5,
    labels: Optional[Sequence[int]] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """生成客户端分区索引。

    mode="iid"      : 全局 shuffle 后均分；
    mode="dirichlet": 按类别做 Dirichlet 分配（需提供 labels），得到标签 Non-IID。

    返回并可选写入::

        {"version": 1, "seed": ..., "num_samples": ..., "num_clients": ...,
         "mode": ..., "alpha": ..., "partitions": {"0": [...], ...}}
    """
    if num_samples <= 0 or num_clients <= 0:
        raise ValueError("num_samples and num_clients must be positive")
    rng = random.Random(seed)
    mode = mode.lower()

    if mode == "iid":
        indices = list(range(num_samples))
        rng.shuffle(indices)
        base, remainder = divmod(num_samples, num_clients)
        partitions: List[List[int]] = []
        start = 0
        for client in range(num_clients):
            size = base + (1 if client < remainder else 0)
            partitions.append(indices[start:start + size])
            start += size
    elif mode == "dirichlet":
        if labels is None or len(labels) != num_samples:
            raise ValueError("dirichlet mode requires labels with length num_samples")
        partitions = [[] for _ in range(num_clients)]
        by_class: Dict[int, List[int]] = {}
        for index, label in enumerate(labels):
            by_class.setdefault(int(label), []).append(index)
        for label in sorted(by_class):
            class_indices = by_class[label]
            rng.shuffle(class_indices)
            props = _dirichlet(rng, alpha, num_clients)
            counts = [int(prop * len(class_indices)) for prop in props]
            remainder = len(class_indices) - sum(counts)
            fractions = sorted(
                range(num_clients),
                key=lambda i: props[i] * len(class_indices) - counts[i],
                reverse=True,
            )
            for client in fractions[:remainder]:
                counts[client] += 1
            offset = 0
            for client, count in enumerate(counts):
                partitions[client].extend(class_indices[offset:offset + count])
                offset += count
        for part in partitions:
            rng.shuffle(part)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    data: Dict[str, Any] = {
        "version": 1,
        "seed": seed,
        "num_samples": num_samples,
        "num_clients": num_clients,
        "mode": mode,
        "alpha": alpha if mode == "dirichlet" else None,
        "partitions": {str(client): part for client, part in enumerate(partitions)},
    }
    if path:
        out = Path(path)
        if str(out.parent) not in ("", "."):
            out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def load_partition(path: str, client_id: Any) -> List[int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    key = str(client_id)
    if key not in data["partitions"]:
        raise KeyError(f"client {key} not found in {path}")
    return [int(index) for index in data["partitions"][key]]


def partition_sizes(path: str) -> Dict[str, int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key: len(value) for key, value in data["partitions"].items()}


# ---------------------------------------------------------------------------
# Client / Server
# ---------------------------------------------------------------------------

@dataclass
class ClientContext:
    client_id: str
    round: int
    num_samples: int
    device: str = "cpu"
    partition: Optional[List[int]] = None


class FLServer:
    """按轮广播 / 收集 / 聚合的 FL 服务端。"""

    def __init__(
        self,
        initial_state: StateDict,
        num_clients: int,
        rounds: int,
        aggregate_fn: AggregateFn = fedavg_state_dicts,
        evaluate_fn: Optional[EvaluateFn] = None,
        host: str = "0.0.0.0",
        port: int = 5000,
        registration_timeout_s: float = 600.0,
        round_timeout_s: float = 1800.0,
        log_path: Optional[str] = None,
    ) -> None:
        if num_clients <= 0 or rounds <= 0:
            raise ValueError("num_clients and rounds must be positive")
        self.initial_state = _to_cpu_state(initial_state)
        self.num_clients = int(num_clients)
        self.rounds = int(rounds)
        self.aggregate_fn = aggregate_fn
        self.evaluate_fn = evaluate_fn
        self.host = host
        self.port = int(port)
        self.registration_timeout_s = float(registration_timeout_s)
        self.round_timeout_s = float(round_timeout_s)
        self.log_path = log_path

        self._global_state = _to_cpu_state(initial_state)
        self._clients: Dict[str, Dict[str, Any]] = {}
        self._events: "queue.Queue[Tuple[Any, ...]]" = queue.Queue()
        self._listener: Optional[socket.socket] = None
        self._closing = False
        self.ready_event = threading.Event()

    @property
    def final_state(self) -> StateDict:
        return _to_cpu_state(self._global_state)

    def wait_ready(self, timeout: float = 10.0) -> bool:
        return self.ready_event.wait(timeout)

    # -- 生命周期 ---------------------------------------------------------

    def serve(self) -> StateDict:
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((self.host, self.port))
        self._listener.listen(max(1, self.num_clients))
        self._listener.settimeout(self.registration_timeout_s)
        self.port = self._listener.getsockname()[1]
        self.ready_event.set()
        logger.info("server listening on %s:%s, expected %s clients", self.host, self.port, self.num_clients)

        try:
            self._accept_clients()
            for round_index in range(1, self.rounds + 1):
                self._run_round(round_index)
        finally:
            self._shutdown()
        return self.final_state

    # -- 注册 -------------------------------------------------------------

    def _accept_clients(self) -> None:
        assert self._listener is not None
        while len(self._clients) < self.num_clients:
            try:
                conn, address = self._listener.accept()
            except socket.timeout as exc:
                raise TimeoutError(
                    f"registration timeout: {len(self._clients)}/{self.num_clients} clients connected"
                ) from exc

            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(self.registration_timeout_s)
            threading.Thread(
                target=self._handle_client,
                args=(conn, address),
                name=f"fl-client-{address[0]}:{address[1]}",
                daemon=True,
            ).start()
            self._wait_for_connection(conn)

        logger.info("all %s clients registered: %s", len(self._clients), sorted(self._clients))

    def _wait_for_connection(self, conn: socket.socket) -> None:
        deadline = time.time() + self.registration_timeout_s
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("registration timeout")
            try:
                event = self._events.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError("registration timeout") from exc

            kind = event[0]
            if kind == "register":
                (
                    _,
                    client_id,
                    event_conn,
                    num_samples,
                    dataset_hash,
                    ack_event,
                    ack_box,
                    ack_sent,
                ) = event
                self._register_client(client_id, event_conn, num_samples, dataset_hash, ack_event, ack_box)
                if event_conn is conn:
                    # 等 handler 把 registered_ack 写完，避免与随后的 global_model 并发写同一 socket
                    if not ack_sent.wait(self.registration_timeout_s):
                        raise TimeoutError("waiting registered_ack to be sent")
                    return
            elif kind == "disconnect":
                _, client_id, event_conn = event
                self._remove_client(client_id)
                if event_conn is conn:
                    return
            elif kind == "update":
                logger.warning("dropping pre-training update from %s", event[1])
            else:
                logger.warning("unexpected event during registration: %s", kind)

    def _register_client(
        self,
        client_id: str,
        conn: socket.socket,
        num_samples: int,
        dataset_hash: Optional[str],
        ack_event: threading.Event,
        ack_box: Dict[str, Any],
    ) -> None:
        if client_id in self._clients:
            ack_box["ok"] = False
            ack_box["reason"] = f"duplicate client_id: {client_id}"
            ack_event.set()
            logger.warning("reject duplicate client_id=%s", client_id)
            return

        self._clients[client_id] = {
            "conn": conn,
            "num_samples": max(1, int(num_samples)),
            "dataset_hash": dataset_hash,
        }
        ack_box["ok"] = True
        ack_event.set()
        logger.info(
            "client registered: id=%s samples=%s dataset_hash=%s",
            client_id, max(1, int(num_samples)), dataset_hash,
        )

    def _remove_client(self, client_id: Optional[str]) -> None:
        if client_id is None:
            return
        info = self._clients.pop(client_id, None)
        if info is not None:
            try:
                info["conn"].close()
            except OSError:
                pass
            logger.warning("client removed: %s", client_id)

    # -- 客户端线程 -------------------------------------------------------

    def _handle_client(self, conn: socket.socket, address: Tuple[str, int]) -> None:
        client_id: Optional[str] = None
        try:
            header, _ = _recv_frame(conn)
            if header.get("type") != "register":
                raise ProtocolError("first message must be register")
            client_id = str(header.get("client_id", "")).strip()
            if not client_id:
                raise ProtocolError("register message missing client_id")

            ack_event = threading.Event()
            ack_box: Dict[str, Any] = {}
            ack_sent = threading.Event()
            self._events.put((
                "register",
                client_id,
                conn,
                int(header.get("num_samples", 1) or 1),
                header.get("dataset_hash"),
                ack_event,
                ack_box,
                ack_sent,
            ))
            if not ack_event.wait(self.registration_timeout_s):
                raise TimeoutError("waiting registration decision")
            if not ack_box.get("ok"):
                _send_frame(conn, {"type": "error", "message": ack_box.get("reason", "rejected")})
                ack_sent.set()
                return

            _send_frame(conn, {
                "type": "registered_ack",
                "client_id": client_id,
                "num_clients": self.num_clients,
                "rounds": self.rounds,
            })
            ack_sent.set()
            conn.settimeout(self.round_timeout_s)

            while True:
                message, payload = _recv_frame(conn)
                message_type = message.get("type")
                if message_type == "model_update":
                    self._events.put(("update", client_id, message, payload))
                elif message_type == "leave":
                    break
                else:
                    logger.warning("unknown message from %s: %s", client_id, message_type)
        except Exception as exc:  # noqa: BLE001 - 连接线程必须兜底
            if self._closing:
                logger.debug("client %s connection closed during shutdown", client_id)
            else:
                logger.warning("client %s (%s:%s) disconnected: %s", client_id, address[0], address[1], exc)
        finally:
            self._events.put(("disconnect", client_id, conn))
            try:
                conn.close()
            except OSError:
                pass

    # -- 每轮 -------------------------------------------------------------

    def _run_round(self, round_index: int) -> None:
        logger.info("round %s/%s: broadcasting global model", round_index, self.rounds)
        self._drain_disconnects()

        payload = _dump_state(self._global_state)
        for client_id, info in list(self._clients.items()):
            try:
                _send_frame(info["conn"], {"type": "global_model", "round": round_index}, payload)
            except OSError as exc:
                raise ClientConnectionError(f"failed to send global model to {client_id}: {exc}") from exc

        updates = self._collect_updates(round_index)
        states = [update["state"] for update in updates]
        meta = [update["meta"] for update in updates]

        new_state = self.aggregate_fn(states, meta, self._global_state)
        self._global_state = _validate_state(new_state, self._global_state)

        metrics: Optional[Dict[str, Any]] = None
        if self.evaluate_fn is not None:
            metrics = _jsonable(self.evaluate_fn(self._global_state, round_index))
        self._write_log(round_index, meta, metrics)
        logger.info(
            "round %s/%s done: %s updates, eval=%s",
            round_index, self.rounds, len(updates), metrics if metrics is not None else "n/a",
        )

    def _collect_updates(self, round_index: int) -> List[Dict[str, Any]]:
        expected = len(self._clients)
        collected: Dict[str, Dict[str, Any]] = {}
        deadline = time.time() + self.round_timeout_s

        while len(collected) < expected:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                event = self._events.get(timeout=min(1.0, remaining))
            except queue.Empty:
                continue

            kind = event[0]
            if kind == "update":
                _, client_id, header, payload = event
                if client_id not in self._clients or client_id in collected:
                    continue
                if int(header.get("round", -1)) != round_index:
                    logger.warning(
                        "drop stale update: client=%s update_round=%s current_round=%s",
                        client_id, header.get("round"), round_index,
                    )
                    continue
                collected[client_id] = {
                    "state": _load_state(payload),
                    "meta": {
                        "client_id": client_id,
                        "num_samples": int(header.get("num_samples", 1) or 1),
                        "metrics": _jsonable(header.get("metrics", {})),
                    },
                }
            elif kind == "disconnect":
                client_id = event[1]
                if client_id in self._clients:
                    self._remove_client(client_id)
                    raise ClientConnectionError(
                        f"client {client_id} disconnected during round {round_index}"
                    )
            elif kind == "register":
                logger.warning("ignoring late registration during round %s", round_index)

        if len(collected) < expected:
            raise RoundTimeoutError(
                f"round {round_index}: received {len(collected)}/{expected} updates "
                f"within {self.round_timeout_s}s"
            )
        return [collected[client_id] for client_id in sorted(collected, key=_client_sort_key)]

    def _drain_disconnects(self) -> None:
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return
            if event[0] == "disconnect":
                self._remove_client(event[1])

    def _write_log(
        self,
        round_index: int,
        meta: Sequence[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]],
    ) -> None:
        if not self.log_path:
            return
        record = {
            "round": round_index,
            "clients": [
                {
                    "client_id": item["client_id"],
                    "num_samples": item["num_samples"],
                    "metrics": item["metrics"],
                }
                for item in meta
            ],
            "eval": metrics,
        }
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _shutdown(self) -> None:
        self._closing = True
        for client_id, info in list(self._clients.items()):
            try:
                _send_frame(info["conn"], {"type": "stop"})
            except OSError:
                pass
            try:
                info["conn"].close()
            except OSError:
                pass
        self._clients.clear()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None
        logger.info("server stopped")


class FLClient:
    """连接服务器、按轮本地训练并回传整份 state_dict 的客户端。"""

    def __init__(
        self,
        client_id: str,
        server_host: str,
        server_port: int,
        train_fn: TrainFn,
        partition_path: Optional[str] = None,
        num_samples: Optional[int] = None,
        dataset_hash: Optional[str] = None,
        connect_retries: int = 20,
        retry_interval_s: float = 3.0,
        timeout_s: float = 3600.0,
        device: str = "cpu",
    ) -> None:
        self.client_id = str(client_id)
        self.server_host = server_host
        self.server_port = int(server_port)
        self.train_fn = train_fn
        self.partition = load_partition(partition_path, self.client_id) if partition_path else None
        if num_samples is not None:
            self.num_samples = int(num_samples)
        elif self.partition is not None:
            self.num_samples = len(self.partition)
        else:
            self.num_samples = 1
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.dataset_hash = dataset_hash
        self.connect_retries = int(connect_retries)
        self.retry_interval_s = float(retry_interval_s)
        self.timeout_s = float(timeout_s)
        self.device = device

    def _connect(self) -> socket.socket:
        last_error: Optional[BaseException] = None
        for attempt in range(1, self.connect_retries + 1):
            try:
                conn = socket.create_connection((self.server_host, self.server_port), timeout=10.0)
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.settimeout(self.timeout_s)
                logger.info("connected to %s:%s", self.server_host, self.server_port)
                return conn
            except OSError as exc:
                last_error = exc
                logger.warning(
                    "connect attempt %s/%s failed: %s",
                    attempt, self.connect_retries, exc,
                )
                if attempt < self.connect_retries:
                    time.sleep(self.retry_interval_s)
        raise ConnectionError(f"cannot connect to {self.server_host}:{self.server_port}: {last_error}")

    def run(self) -> None:
        conn = self._connect()
        try:
            _send_frame(conn, {
                "type": "register",
                "client_id": self.client_id,
                "num_samples": self.num_samples,
                "dataset_hash": self.dataset_hash,
            })
            header, _ = _recv_frame(conn)
            if header.get("type") == "error":
                raise ConnectionError(f"server rejected registration: {header.get('message')}")
            if header.get("type") != "registered_ack":
                raise ProtocolError(f"unexpected registration response: {header.get('type')}")

            logger.info(
                "registered: id=%s samples=%s rounds=%s",
                self.client_id, self.num_samples, header.get("rounds"),
            )
            conn.settimeout(self.timeout_s)

            while True:
                header, payload = _recv_frame(conn)
                message_type = header.get("type")
                if message_type == "global_model":
                    round_index = int(header["round"])
                    global_state = _load_state(payload)
                    context = ClientContext(
                        client_id=self.client_id,
                        round=round_index,
                        num_samples=self.num_samples,
                        device=self.device,
                        partition=self.partition,
                    )
                    new_state, metrics = self.train_fn(global_state, context)
                    new_state = _validate_state(new_state, global_state)
                    _send_frame(
                        conn,
                        {
                            "type": "model_update",
                            "round": round_index,
                            "client_id": self.client_id,
                            "num_samples": self.num_samples,
                            "metrics": _jsonable(metrics),
                        },
                        _dump_state(new_state),
                    )
                    logger.info("round %s: update sent", round_index)
                elif message_type == "stop":
                    logger.info("received stop, exiting")
                    break
                elif message_type == "error":
                    raise RuntimeError(f"server error: {header.get('message')}")
                else:
                    logger.warning("unknown message from server: %s", message_type)
        finally:
            try:
                conn.close()
            except OSError:
                pass


def run_server(initial_state: StateDict, num_clients: int, rounds: int, **kwargs: Any) -> StateDict:
    server = FLServer(initial_state, num_clients, rounds, **kwargs)
    return server.serve()


def run_client(client_id: str, server_host: str, server_port: int, train_fn: TrainFn, **kwargs: Any) -> None:
    FLClient(client_id, server_host, server_port, train_fn, **kwargs).run()


# ---------------------------------------------------------------------------
# selftest / CLI
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.bn = nn.BatchNorm1d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.fc(x))


def _selftest() -> None:
    setup_logging("WARNING")

    print("[1/3] partition determinism ...")
    first = make_partitions(101, 4, seed=7, mode="iid")
    second = make_partitions(101, 4, seed=7, mode="iid")
    assert first == second, "partition is not deterministic"
    indices = sorted(index for part in first["partitions"].values() for index in part)
    assert indices == list(range(101)), "partition does not cover all samples"

    print("[2/3] framing with a large payload ...")
    left, right = socket.socketpair()
    try:
        tensor = torch.randn(256, 64)
        payload = _dump_state({"w": tensor})
        _send_frame(left, {"type": "blob"}, payload)
        header, received = _recv_frame(right)
        assert header["payload_len"] == len(payload)
        assert torch.equal(_load_state(received)["w"], tensor), "payload mismatch"
    finally:
        left.close()
        right.close()

    print("[3/3] 2 clients x 2 rounds ...")
    model = _TinyModel()
    initial = _to_cpu_state(model.state_dict())

    def train_fn(global_state: StateDict, context: ClientContext) -> Tuple[StateDict, Dict[str, Any]]:
        state = _to_cpu_state(global_state)
        delta = 0.01 * (int(context.client_id) + 1) * context.round
        state["fc.weight"] = state["fc.weight"] + delta
        return state, {"delta": delta}

    server = FLServer(
        initial_state=initial,
        num_clients=2,
        rounds=2,
        host="127.0.0.1",
        port=0,
        registration_timeout_s=10.0,
        round_timeout_s=10.0,
    )
    server_errors: List[BaseException] = []
    client_errors: List[BaseException] = []

    def serve() -> None:
        try:
            server.serve()
        except BaseException as exc:  # noqa: BLE001 - 测试线程兜底
            server_errors.append(exc)

    def client_main(client_id: int) -> None:
        try:
            FLClient(
                str(client_id),
                "127.0.0.1",
                server.port,
                train_fn,
                num_samples=1,
                connect_retries=50,
                retry_interval_s=0.05,
                timeout_s=15.0,
            ).run()
        except BaseException as exc:  # noqa: BLE001 - 测试线程兜底
            client_errors.append(exc)

    server_thread = threading.Thread(target=serve, daemon=True)
    server_thread.start()
    assert server.wait_ready(10.0), "server did not start"

    client_threads = [threading.Thread(target=client_main, args=(index,)) for index in range(2)]
    for thread in client_threads:
        thread.start()
    for thread in client_threads:
        thread.join(20.0)
    server_thread.join(20.0)

    assert not server_errors, f"server error: {server_errors[0]!r}"
    assert not client_errors, f"client error: {client_errors[0]!r}"

    expected = initial["fc.weight"].clone()
    for round_index in (1, 2):
        deltas = [0.01 * (index + 1) * round_index for index in range(2)]
        expected = expected + sum(deltas) / 2.0

    final = server.final_state
    assert torch.allclose(final["fc.weight"], expected, atol=1e-6), "aggregation math mismatch"
    assert torch.equal(final["bn.num_batches_tracked"], initial["bn.num_batches_tracked"]), (
        "integer buffer dtype/value changed"
    )
    print("SELFTEST OK")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="single-file federated learning transport")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("selftest", help="run CPU smoke test")

    server_parser = subparsers.add_parser("server", help="run FL server")
    server_parser.add_argument("--host", default="0.0.0.0")
    server_parser.add_argument("--port", type=int, default=5000)
    server_parser.add_argument("--num-clients", type=int, required=True)
    server_parser.add_argument("--rounds", type=int, required=True)
    server_parser.add_argument("--init-state", required=True, help="torch.save(state_dict) file")
    server_parser.add_argument("--aggregate", default=None, help="module:function, default FedAvg")
    server_parser.add_argument("--evaluate", default=None, help="module:function")
    server_parser.add_argument("--registration-timeout", type=float, default=600.0)
    server_parser.add_argument("--round-timeout", type=float, default=1800.0)
    server_parser.add_argument("--log", default=None, help="per-round JSONL path")

    client_parser = subparsers.add_parser("client", help="run FL client")
    client_parser.add_argument("--server", required=True, help="host:port")
    client_parser.add_argument("--client-id", required=True)
    client_parser.add_argument("--train", required=True, help="module:function")
    client_parser.add_argument("--partition", default=None)
    client_parser.add_argument("--num-samples", type=int, default=None)
    client_parser.add_argument("--timeout", type=float, default=3600.0)
    client_parser.add_argument("--device", default="cpu")

    return parser


def _load_callable(spec: Optional[str]) -> Optional[Callable[..., Any]]:
    if not spec:
        return None
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"callable must be 'module:function', got {spec!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    setup_logging("INFO")

    if args.command == "selftest":
        _selftest()
        return

    if args.command == "server":
        state = torch.load(args.init_state, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        FLServer(
            initial_state=state,
            num_clients=args.num_clients,
            rounds=args.rounds,
            aggregate_fn=_load_callable(args.aggregate) or fedavg_state_dicts,
            evaluate_fn=_load_callable(args.evaluate),
            host=args.host,
            port=args.port,
            registration_timeout_s=args.registration_timeout,
            round_timeout_s=args.round_timeout,
            log_path=args.log,
        ).serve()
        return

    host, port_text = args.server.rsplit(":", 1)
    FLClient(
        client_id=args.client_id,
        server_host=host,
        server_port=int(port_text),
        train_fn=_load_callable(args.train),  # type: ignore[arg-type]
        partition_path=args.partition,
        num_samples=args.num_samples,
        timeout_s=args.timeout,
        device=args.device,
    ).run()


if __name__ == "__main__":
    main()
