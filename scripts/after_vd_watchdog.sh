#!/usr/bin/env bash
# VisDrone 结束后续跑队列的守护：runner 不在且没有训练进程时自动重启。
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p queue_after_vd

while true; do
  if [ -f queue_after_vd/DONE ]; then
    echo "$(date '+%F %T') DONE 已存在，watchdog 退出" >> queue_after_vd/watchdog.log
    exit 0
  fi
  if [ -f queue_after_vd/pid ]; then
    pid="$(cat queue_after_vd/pid 2>/dev/null || true)"
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      sleep 300
      continue
    fi
  fi
  if ps -eo cmd | grep -E "python.*run-test/" | grep -v grep >/dev/null 2>&1; then
    sleep 300
    continue
  fi
  echo "$(date '+%F %T') after_vd runner 不在，自动重启" >> queue_after_vd/watchdog.log
  tmux kill-session -t aftervd 2>/dev/null || true
  tmux new-session -d -s aftervd -c "$ROOT" \
    "python scripts/after_vd_supervisor.py >> queue_after_vd/console.log 2>&1"
  sleep 300
done
