#!/bin/bash
# 探针 watchdog：每 10 分钟检查，supervisor 不在且无训练进程时拉起。
set -u
cd /root/chashaobao/DPFLA-master || exit 1

LOG=/tmp/flrun/vd_probe_watchdog.out
[ -f /tmp/vd_probe_done ] && exit 0

while :; do
  if [ ! -f /tmp/vd_probe_done ]; then
    if ! pgrep -af "^bash scripts/run_vd_probe.sh" >/dev/null 2>&1 \
       && ! pgrep -af "^python -u run-test/visdrone/run_screen_server_attack.py" >/dev/null 2>&1; then
      echo "[probe-wd] restart supervisor $(date '+%F %T') idx=$(cat /tmp/vd_probe_state.txt 2>/dev/null)" >> "$LOG"
      setsid nohup bash scripts/run_vd_probe.sh >/dev/null 2>&1 < /dev/null &
    fi
  fi
  sleep 600
done
