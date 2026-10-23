#!/bin/bash
# IPM 夜间/日间 watchdog：每 10 分钟检查，supervisor 不在且无训练进程时自动拉起。
set -u
cd /root/chashaobao/DPFLA-master || exit 1

LOG=/tmp/flrun/vd_ipm_watchdog.out
STATE=/tmp/vd_ipm_state.txt

while :; do
  st=$(cat "$STATE" 2>/dev/null || echo "")
  if [ -n "$st" ] && [ "$st" != "DONE" ]; then
    if ! pgrep -af "^bash scripts/run_vd_ipm_supervisor.sh" >/dev/null 2>&1 \
       && ! pgrep -af "^python -u run-test/visdrone/run_screen_server_attack.py" >/dev/null 2>&1; then
      echo "[ipm-wd] restart supervisor $(date '+%F %T') state=$st" >> "$LOG"
      setsid nohup bash scripts/run_vd_ipm_supervisor.sh >/dev/null 2>&1 < /dev/null &
    fi
  fi
  sleep 600
done
