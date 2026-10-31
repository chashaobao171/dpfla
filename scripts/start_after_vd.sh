#!/usr/bin/env bash
# 启动 VisDrone 结束后的续跑队列（等待 full50 DONE → 剩余 VisDrone → CIFAR 多映射筛选/全重跑）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p queue_after_vd

if [ -f queue_after_vd/DONE ]; then
  echo "queue_after_vd/DONE 已存在，续跑已完成。"
  exit 0
fi

tmux kill-session -t aftervd 2>/dev/null || true
tmux kill-session -t aftervdwd 2>/dev/null || true

tmux new-session -d -s aftervd -c "$ROOT" \
  "python scripts/after_vd_supervisor.py >> queue_after_vd/console.log 2>&1"
sleep 2
tmux new-session -d -s aftervdwd -c "$ROOT" "bash scripts/after_vd_watchdog.sh"

echo "已启动续跑队列："
echo "  等待 full50 结束后自动开始"
echo "  队列: tmux attach -t aftervd"
echo "  守护: tmux attach -t aftervdwd"
echo "  状态: queue_after_vd/status.json"
echo "  汇总: queue_after_vd/auto_results.md"
echo "  主机续费/重启后，重新运行 bash scripts/start_after_vd.sh 即可断点续跑"
