#!/usr/bin/env bash
# MinMax + DPFLA，10 个客户端进程（3 恶意），server + 10 client 共 11 条 TB 曲线。
# 用法：bash scripts/run_distributed_full.sh smoke|full [cifar10|visdrone]
# 日志只写入 logs_dist_full/；TB 同时写入 tb/ 与云平台 /root/tf-logs。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-full}"
DATASET="${2:-cifar10}"

if [[ "$MODE" == "smoke" ]]; then
  ROUNDS=2
  OUT="$ROOT/logs_dist_full/smoke_2r"
else
  ROUNDS=50
  OUT="$ROOT/logs_dist_full/full50"
fi

if [[ "$DATASET" == "visdrone" ]]; then
  BATCH=16
  TEST_BATCH=64
  LR=0.0002
  MODEL_FAMILY=${YOLO_MODEL_FAMILY:-26}
  MODEL_SIZE=${YOLO_MODEL_SIZE:-s}
  IMAGE_SIZE=1280
  AMP=1
  # 50 轮全量实测约 14h；18h 总超时兜底，避免再次被 12h 截断。
  if [[ "$MODE" == "smoke" ]]; then TIMEOUT=7200; CLIENT_EVAL=64; else TIMEOUT=64800; CLIENT_EVAL=0; fi
  # 与正式 VisDrone 50 轮实验一致：每客户端每轮 12 step，避免把总训练量放大 3.4 倍。
  export FL_LOCAL_STEPS=12
  EXTRA=(--visdrone-root /root/autodl-tmp/data/visdrone)
elif [[ "$DATASET" == "cifar10" ]]; then
  BATCH=64
  TEST_BATCH=1000
  LR=0.01
  MODEL_FAMILY=8
  MODEL_SIZE=n
  IMAGE_SIZE=32
  AMP=0
  unset FL_LOCAL_STEPS || true
  if [[ "$MODE" == "smoke" ]]; then TIMEOUT=1800; CLIENT_EVAL=1000; else TIMEOUT=7200; CLIENT_EVAL=0; fi
  EXTRA=()
else
  echo "unknown dataset: $DATASET" >&2
  exit 2
fi

mkdir -p "$OUT"

# 同数据集旧正式产物移出归档，不删除；另一数据集的结果保留。
if [[ "$MODE" == "full" && -d "$OUT/$DATASET" ]]; then
  ARCH="/root/dpfla_cleanup_20261101/dist_full_previous_${DATASET}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$ARCH"
  mv "$OUT/$DATASET" "$ARCH"/
fi
if [[ "$MODE" == "smoke" ]]; then
  rm -rf "$OUT/$DATASET"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python run_distributed_smoke.py \
  --role orchestrator \
  --dataset "$DATASET" \
  --scenario minmax \
  --defense dpfla \
  --num-clients 10 \
  --rounds "$ROUNDS" \
  --batch-size "$BATCH" \
  --test-batch-size "$TEST_BATCH" \
  --test-samples 0 \
  --lr "$LR" \
  --local-epochs 1 \
  --seed 7 \
  --malicious-rate 0.3 \
  --full-data \
  --model-size "$MODEL_SIZE" \
  --model-family "$MODEL_FAMILY" \
  --image-size "$IMAGE_SIZE" \
  --amp "$AMP" \
  --device cuda \
  --workdir "$OUT" \
  --timeout "$TIMEOUT" \
  --bind-host 127.0.0.1 \
  --server-host 127.0.0.1 \
  --tb11 \
  --client-eval-samples "$CLIENT_EVAL" \
  "${EXTRA[@]}"
