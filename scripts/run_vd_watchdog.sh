#!/bin/bash
# VisDrone 夜间 watchdog：每 10 分钟检查一次，supervisor 不在且没有训练进程时自动拉起。
# 启动：setsid nohup bash scripts/run_vd_watchdog.sh >/dev/null 2>&1 < /dev/null &

set -u
cd /root/chashaobao/DPFLA-master || exit 1

LOG=/tmp/flrun/vd_watchdog.out
STATE=/tmp/vd_state.txt

while :; do
  st=$(cat "$STATE" 2>/dev/null || echo "")
  if [ -n "$st" ] && [ "$st" != "DONE" ]; then
    if ! pgrep -af "^bash scripts/run_vd_sign_supervisor.sh" >/dev/null 2>&1 \
       && ! pgrep -af "^python -u run-test/visdrone/run_screen_server_attack.py" >/dev/null 2>&1; then
      echo "[wd] restart supervisor $(date '+%F %T') state=$st" >> "$LOG"
      setsid nohup bash scripts/run_vd_sign_supervisor.sh >/dev/null 2>&1 < /dev/null &
    fi
  fi
  sleep 600
done
