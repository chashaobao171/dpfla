# 多机分布式操作指南（1 台 server + 2 台 client）

目标：在局域网内用 3 台机器验证真正的分布式训练。server 只做广播/聚合/评估，两个 client 各自在本机训练并上传完整模型参数。

> 2 客户端时 DPFLA 的 KMeans 会退化成每簇 1 个样本，聚合权重通常相同；**只验证分布式链路，不验证防御效果**。需要防御对比请用 ≥3 客户端；Multi-Krum 需要 ≥5 客户端（代码会在 `--num-clients 2 --defense mkrum` 时直接报错）。

## 0. 前置条件

- 三台机器在同一局域网，互相能 `ping`；server 放行 TCP 端口（下文用 `6000`）。
- 三台机器都有同一份项目代码，且 `python` 能 import `torch`、`ultralytics`。
- 数据集：
  - CIFAR10：每台机器本地有数据；`get_cifar10()` 会自动找 `/root/autodl-tmp/data` 等路径，找不到会尝试下载。
  - VisDrone：每台机器要有同一份数据；用 `--visdrone-root` 指定路径；server 还需要 val 548 张用于评估。
- YOLO 权重：`yolov8n.pt` 可自动下载，或提前放到各机器项目根目录。

## 1. 准备分区（只在 server 执行一次）

```bash
cd /path/to/DPFLA-master

python run_distributed_smoke.py --role prepare \
  --dataset cifar10 --num-clients 2 --scenario label_flipping \
  --partition ./dist_run/partitions.json \
  --max-samples 32 --seed 42 --device cpu
```

把生成的 `partitions.json` 拷到两台 client 的可访问路径。**客户端数量变了就必须重跑 prepare**，因为 JSON 是静态的按 `num_clients` 切好的索引。

## 2. 启动 server（server 机器）

先查本机局域网 IP（不要用 Docker 的 `172.17.x.x`，用 `192.168.x.x` 或实际网卡地址）：

```bash
hostname -I
ip addr show
```

```bash
python run_distributed_smoke.py --role server \
  --dataset cifar10 --scenario label_flipping --defense dpfla \
  --num-clients 2 --rounds 1 \
  --bind-host 0.0.0.0 --port 6000 \
  --partition ./dist_run/partitions.json \
  --workdir ./dist_run/server \
  --device cpu --timeout 1800
```

`--rounds 1` 表示只做 1 次聚合；本轮结束后 server 会自己给两个 client 发 `stop` 并退出 0，**不需要 Ctrl-C**。

## 3. 启动两个 client（各自机器）

client 0：

```bash
python run_distributed_smoke.py --role client \
  --dataset cifar10 --scenario label_flipping --defense dpfla \
  --num-clients 2 --rounds 1 \
  --server-host <SERVER_LAN_IP> --port 6000 \
  --client-id 0 \
  --partition /path/to/partitions.json \
  --workdir ./dist_run/client0 \
  --device cpu --timeout 1800
```

client 1：只改两处：

```bash
--client-id 1
--workdir ./dist_run/client1
```

## 4. 判断是否真的分布式跑通

server 目录：

- `server.module.log` 出现 `all 2 clients registered`
- `rounds.jsonl` 恰好 1 行，含 2 个 client 的 `num_samples` / metrics 和 1 个 `eval`
- 控制台出现 `server stopped`

每个 client：

- `client_<id>.module.log` 出现 1 次 `update sent`（1 轮对应 1 次；若 `--rounds 2` 则 2 次）
- 日志出现 `received stop, exiting`
- 进程退出码 0

补充检查（项目自身的 loguru 日志默认打在控制台；只有把控制台重定向到 `console.log` 后才会进文件）：

```bash
# 若 client0 的控制台已重定向到 dist_run/client0/console.log
grep "Label flipping attack launched" dist_run/client0/console.log

# DPFLA 的聚合权重同样在 server 控制台（或重定向后的 console.log）
grep "聚合权重" dist_run/server/console.log
```

CIFAR 标签翻转时应看到恶意端 `client 0` 的攻击日志；DPFLA 会打印两端权重（2 客户端时通常相同）。

## 5. VisDrone 版本（可选）

两台 client 用 `--scenario ipm`，不要用 `minmax`（2 客户端时 MinMax 只有 1 个良性端，构造会退化）：

```bash
# server 机器准备（三台机器数据路径要一致）
python run_distributed_smoke.py --role prepare \
  --dataset visdrone --num-clients 2 --scenario ipm \
  --partition ./dist_run/visdrone_partitions.json \
  --max-samples 4 --seed 42 --device cuda \
  --visdrone-root /path/to/visdrone

# server
python run_distributed_smoke.py --role server \
  --dataset visdrone --scenario ipm --defense dpfla \
  --num-clients 2 --rounds 1 \
  --bind-host 0.0.0.0 --port 6000 \
  --partition ./dist_run/visdrone_partitions.json \
  --workdir ./dist_run/visdrone_server \
  --device cuda --timeout 1800 \
  --visdrone-root /path/to/visdrone

# client 0（client 1 改 --client-id 1 / --workdir）
python run_distributed_smoke.py --role client \
  --dataset visdrone --scenario ipm --defense dpfla \
  --num-clients 2 --rounds 1 \
  --server-host <SERVER_LAN_IP> --port 6000 \
  --client-id 0 \
  --partition /path/to/visdrone_partitions.json \
  --workdir ./dist_run/visdrone_client0 \
  --device cuda --timeout 1800 \
  --visdrone-root /path/to/visdrone
```

VisDrone 的 server 评估会读取 `FL_VISDRONE_ROOT`，`run_distributed_smoke.py` 已自动把它设为 `--visdrone-root`，所以三台机器的数据路径可以不同。

## 6. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `connection refused` | server 未启动、IP/端口错、防火墙未放行 |
| server 一直等 | client 数量不是 `--num-clients`，或 client 没启动/连错 IP |
| `client X not found in partitions.json` | `prepare --num-clients` 与实际客户端数不一致，重跑 prepare |
| 数据路径不一致 | CIFAR 各机数据根要能找到；VisDrone 用 `--visdrone-root` 并三台一致 |
| 1 轮就退出 | 正常：`--rounds 1` 就是 1 次聚合后自然结束 |
| 2 客户端 DPFLA 权重相同 | 正常退化，不代表防御失效；只用于链路验证 |
| MKrum 报错 | 2 客户端不满足 `n >= 2f+3`；用 FedAvg/DPFLA，或扩到 5 客户端 |

## 7. 安全与日志

- 传输无 TLS / 无鉴权 / 无断线重连，只在可信局域网使用。
- 日志在各自 `--workdir`：`server.module.log`、`rounds.jsonl`、`client_<id>.module.log`；控制台日志需自行重定向。
- 分布式不写 TensorBoard；`dist_run/` 和 `logs_dist_smoke/` 均被 `.gitignore` 忽略。
