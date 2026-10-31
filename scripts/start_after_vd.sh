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

# 已在运行就不要重复拉起（避免打断正在跑的 job）
if [ -f queue_after_vd/pid ]; then
  pid="$(cat queue_after_vd/pid 2>/dev/null || true)"
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    echo "aftervd 已在运行（pid $pid），无需重启。"
    exit 0
  fi
fi

# 主机续费/重启后：如果 full50 还没完成且已经不在运行，先把 full50 拉起来
if [ ! -f queue/DONE ]; then
  if ! tmux has-session -t full50 2>/dev/null; then
    echo "full50 未完成且不在运行，先启动 full50 队列..."
    bash scripts/start_full50.sh || echo "⚠️ full50 启动失败（可能 GPU 尚未就绪），aftervd 会继续等待"
  fi
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
