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

## 6. 对齐单机正式实验参数（10 客户端）

单机正式口径是 **10 个模拟客户端**、30% 恶意、IID、全量训练集。分布式要真正同参，需要启动 **10 个 client 进程**（可分散在多台机器），而不是 2 个。2 客户端只能做链路验证，不能同时满足 10 客户端和 30% 恶意。

| 参数 | CIFAR10 正式 | VisDrone 正式 |
|---|---|---|
| 客户端 | 10 | 10 |
| 轮数 | 15 | 15 |
| local epochs | 1 | 3 |
| batch size | 64 | 16 |
| test batch | 1000 | 64 |
| lr | 0.01 | 0.0002 |
| 模型/分辨率 | ResNet18CIFAR | yolov8m @1280 |
| AMP | 0 | 1 |
| 恶意率 | 0.3 | 0.3 |
| 梯度放大 | 1（不放大） | 1（不放大） |
| 数据 | 全量训练集 + 全量测试集 | 全量训练集 + 全量 val |

CIFAR 命令骨架：

```bash
# 只执行一次：全量分区
python run_distributed_smoke.py --role prepare \
  --dataset cifar10 --num-clients 10 --scenario label_flipping \
  --partition ./dist_run/parity/cifar10/partitions.json \
  --full-data --seed 7 --device cuda

# server
python run_distributed_smoke.py --role server \
  --dataset cifar10 --scenario label_flipping --defense dpfla \
  --num-clients 10 --rounds 15 \
  --batch-size 64 --test-batch-size 1000 --test-samples 0 \
  --lr 0.01 --local-epochs 1 --seed 7 --malicious-rate 0.3 --full-data \
  --bind-host 0.0.0.0 --port 6000 \
  --partition ./dist_run/parity/cifar10/partitions.json \
  --workdir ./dist_run/parity/cifar10/server --device cuda --timeout 7200

# 10 个 client（--client-id 0..9，每个终端/进程一个）
python run_distributed_smoke.py --role client \
  --dataset cifar10 --scenario label_flipping --defense dpfla \
  --num-clients 10 --rounds 15 \
  --batch-size 64 --test-batch-size 1000 --test-samples 0 \
  --lr 0.01 --local-epochs 1 --seed 7 --malicious-rate 0.3 --full-data \
  --server-host <SERVER_LAN_IP> --port 6000 --client-id 0 \
  --partition ./dist_run/parity/cifar10/partitions.json \
  --workdir ./dist_run/parity/cifar10/client0 --device cuda --timeout 7200
```

VisDrone 命令骨架（把 `--scenario minmax` 可换成 `--scenario ipm`）：

```bash
python run_distributed_smoke.py --role prepare \
  --dataset visdrone --num-clients 10 --scenario minmax \
  --partition ./dist_run/parity/visdrone/partitions.json \
  --full-data --seed 7 --device cuda --visdrone-root /path/to/visdrone

python run_distributed_smoke.py --role server \
  --dataset visdrone --scenario minmax --defense dpfla \
  --num-clients 10 --rounds 15 \
  --batch-size 16 --test-batch-size 64 \
  --lr 0.0002 --local-epochs 3 \
  --model-size m --image-size 1280 --amp 1 \
  --seed 7 --malicious-rate 0.3 --full-data \
  --bind-host 0.0.0.0 --port 6000 \
  --partition ./dist_run/parity/visdrone/partitions.json \
  --workdir ./dist_run/parity/visdrone/server --device cuda --timeout 43200 \
  --visdrone-root /path/to/visdrone

# 10 个 client，除 --client-id / --workdir 外与 server 参数一致
python run_distributed_smoke.py --role client \
  --dataset visdrone --scenario minmax --defense dpfla \
  --num-clients 10 --rounds 15 \
  --batch-size 16 --test-batch-size 64 \
  --lr 0.0002 --local-epochs 3 \
  --model-size m --image-size 1280 --amp 1 \
  --seed 7 --malicious-rate 0.3 --full-data \
  --server-host <SERVER_LAN_IP> --port 6000 --client-id 0 \
  --partition ./dist_run/parity/visdrone/partitions.json \
  --workdir ./dist_run/parity/visdrone/client0 --device cuda --timeout 43200 \
  --visdrone-root /path/to/visdrone
```

注意：

- 10 个 client 进程可分布在多台机器；`--num-clients 10` 是进程数，不是机器数。
- VisDrone `yolov8m.pt` 需在三台机器都存在；1280 分辨率必须开 AMP，否则 32GB 显存可能 OOM。
- 如果实际只有 1 台 server + 2 台 client 且每台只跑 1 个 client 进程，就只能用 `--num-clients 2`；此时恶意率无法同时保持 30%，请用 `--malicious-rate 0.5`（1/2）并只做链路验证。
- 单机正式 CIFAR 的 `--test-samples 0` 表示全量 10k 测试集；默认 256 只用于冒烟。
- `--full-data` 生成的是全量索引；客户端数量变化后必须重新 `prepare`。
- VisDrone `label_flipping` 的分布式映射目前是 `{5→3}`；单机演示的 car→truck 是 `{3→5}`。正式对比请用 `minmax` 或 `ipm`，避免标签映射口径混淆。

## 7. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `connection refused` | server 未启动、IP/端口错、防火墙未放行 |
| server 一直等 | client 数量不是 `--num-clients`，或 client 没启动/连错 IP |
| `client X not found in partitions.json` | `prepare --num-clients` 与实际客户端数不一致，重跑 prepare |
| 数据路径不一致 | CIFAR 各机数据根要能找到；VisDrone 用 `--visdrone-root` 并三台一致 |
| 1 轮就退出 | 正常：`--rounds 1` 就是 1 次聚合后自然结束 |
| 2 客户端 DPFLA 权重相同 | 正常退化，不代表防御失效；只用于链路验证 |
| MKrum 报错 | 2 客户端不满足 `n >= 2f+3`；用 FedAvg/DPFLA，或扩到 5 客户端 |

## 8. 安全与日志

- 传输无 TLS / 无鉴权 / 无断线重连，只在可信局域网使用。
- 日志在各自 `--workdir`：`server.module.log`、`rounds.jsonl`、`client_<id>.module.log`；控制台日志需自行重定向。
- 分布式不写 TensorBoard；`dist_run/` 和 `logs_dist_smoke/` 均被 `.gitignore` 忽略。
