#!/usr/bin/env bash
# 50 轮队列守护：supervisor 不在且没有训练子进程时自动重启（断点续跑由 supervisor 负责）。
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p queue

while true; do
  if [ -f queue/DONE ]; then
    echo "$(date '+%F %T') DONE 已存在，watchdog 退出" >> queue/watchdog.log
    exit 0
  fi

  if [ -f queue/supervisor.pid ]; then
    pid="$(cat queue/supervisor.pid 2>/dev/null || true)"
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      sleep 300
      continue
    fi
  fi

  # supervisor 不在，但训练子进程仍在：等它结束，避免双开
  if ps -eo cmd | grep -E "python.*run-test/" | grep -v grep >/dev/null 2>&1; then
    sleep 300
    continue
  fi

  echo "$(date '+%F %T') supervisor 不在，自动重启" >> queue/watchdog.log
  tmux kill-session -t full50 2>/dev/null || true
  tmux new-session -d -s full50 -c "$ROOT" \
    "python scripts/full50_supervisor.py >> queue/supervisor.console.log 2>&1"
  sleep 300
done
