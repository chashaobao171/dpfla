#!/usr/bin/env bash
# 启动 50 轮全量自动队列（体检 → 18 条 50 轮 → 自动覆盖 TB）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p queue

if [ -f queue/DONE ]; then
  echo "queue/DONE 已存在。若要重跑，先删除 queue/ 下的 DONE/state.json 等文件。"
  exit 1
fi

# 要求 GPU 可见（必须在非沙箱环境执行本脚本）
python - <<'EOF'
import sys
import torch
if not torch.cuda.is_available():
    print("CUDA 不可用，终止（请在非沙箱环境执行）")
    sys.exit(1)
free, total = torch.cuda.mem_get_info()
print(f"GPU: {torch.cuda.get_device_name(0)} | 空闲 {free/1024**3:.1f}G / 共 {total/1024**3:.1f}G")
EOF

# 已有训练进程则拒绝启动
if ps -eo cmd | grep -E "python.*run-test/" | grep -v grep >/dev/null 2>&1; then
  echo "已有训练进程在跑，拒绝启动队列："
  ps -eo pid,etime,cmd | grep -E "python.*run-test/" | grep -v grep || true
  exit 1
fi

tmux kill-session -t full50 2>/dev/null || true
tmux kill-session -t full50wd 2>/dev/null || true

tmux new-session -d -s full50 -c "$ROOT" \
  "python scripts/full50_supervisor.py >> queue/supervisor.console.log 2>&1"
sleep 2
tmux new-session -d -s full50wd -c "$ROOT" "bash scripts/full50_watchdog.sh"

echo "已启动："
echo "  队列:   tmux attach -t full50   （状态: queue/status.json）"
echo "  守护:   tmux attach -t full50wd （日志: queue/watchdog.log）"
echo "  汇总:   queue/auto_results.md / queue/results.csv"
