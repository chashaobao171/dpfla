#!/usr/bin/env bash
# CIFAR10 + ResNet18CIFAR，MinMax 攻击 + DPFLA，10 个客户端进程（3 恶意），50 轮全量分布式。
# 用法：bash scripts/run_distributed_full.sh smoke|full
# 日志只写入 logs_dist_full/：smoke_2r 为链路/显存体检，full50 为正式实验。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-full}"
if [[ "$MODE" == "smoke" ]]; then
  ROUNDS=2
  OUT="$ROOT/logs_dist_full/smoke_2r"
  TIMEOUT=900
else
  ROUNDS=50
  OUT="$ROOT/logs_dist_full/full50"
  TIMEOUT=7200
fi

mkdir -p "$OUT"

# 正式实验前把同目录的旧 full50 产物移出，避免覆盖；不删除。
if [[ "$MODE" == "full" && -d "$OUT/cifar10" ]]; then
  ARCH="/root/dpfla_cleanup_20261101/dist_full_previous_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$ARCH"
  mv "$OUT"/* "$ARCH"/ 2>/dev/null || true
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python run_distributed_smoke.py \
  --role orchestrator \
  --dataset cifar10 \
  --scenario minmax \
  --defense dpfla \
  --num-clients 10 \
  --rounds "$ROUNDS" \
  --batch-size 64 \
  --test-batch-size 1000 \
  --test-samples 0 \
  --lr 0.01 \
  --local-epochs 1 \
  --seed 7 \
  --malicious-rate 0.3 \
  --full-data \
  --device cuda \
  --workdir "$OUT" \
  --timeout "$TIMEOUT" \
  --bind-host 127.0.0.1 \
  --server-host 127.0.0.1
