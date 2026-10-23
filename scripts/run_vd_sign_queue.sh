#!/bin/bash
# VisDrone SignFlipping 剩余四臂自动串行（2026-10-23 夜间）
#
# 设计要点：
# - 必须由**普通 Shell**启动（`setsid nohup bash scripts/run_vd_sign_queue.sh &`），
#   不能放在 run_in_background 的沙箱任务里，否则 CUDA 不可用。
# - 每个实验用 FL_ATTACHED=1 + FL_DETACHED=1：不脱壳，$! 就是真实 python PID，便于等待。
# - 每条结束后把 5 轮结果写入状态文件，并自动 git add/commit/push（仅 logs_3 与 tb）。
# - 03:30 之后不再启动新实验，给 04:00 关机留余量。

set -u
cd /root/chashaobao/DPFLA-master || exit 1

LOG=/tmp/flrun/vd_sign_queue.out
RUN_LOG_DIR=/tmp/flrun
CURRENT_PID=${1:-474672}

mkdir -p "$RUN_LOG_DIR"
echo "[queue] start $(date '+%F %T'), wait pid $CURRENT_PID" >> "$LOG"

wait_pid() {
  local p=$1
  local i=0
  while kill -0 "$p" 2>/dev/null; do
    sleep 20
    i=$((i + 1))
    [ $i -gt 330 ] && return 1
  done
  return 0
}

run_one() {
  local attack=$1
  local rule=$2
  local hhmm
  hhmm=$(date +%H%M)
  if [ "$hhmm" -ge 0330 ]; then
    echo "[queue] skip $attack/$rule (after 03:30)" >> "$LOG"
    return 0
  fi

  echo "[queue] launch $attack/$rule $(date '+%T')" >> "$LOG"
  FL_ATTACHED=1 FL_DETACHED=1 FL_GLOBAL_ROUNDS=5 \
  FL_SERVER_ATTACK="$attack" FL_RULE="$rule" \
    nohup python -u run-test/visdrone/run_screen_server_attack.py \
    > "$RUN_LOG_DIR/vd_${attack}_${rule}_fix.log" 2>&1 &
  local pid=$!

  local i=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    i=$((i + 1))
    if [ $i -gt 240 ]; then
      echo "[queue] timeout $attack/$rule, kill $pid" >> "$LOG"
      kill "$pid" 2>/dev/null
      break
    fi
  done
  wait "$pid" 2>/dev/null

  local f
  f=$(ls -t logs_3/visdrone/run_screen_${attack}_${rule}_*.log 2>/dev/null | head -1)
  echo "[queue] done $attack/$rule $(date '+%T')" >> "$LOG"
  if [ -n "$f" ]; then
    echo "[queue] rounds: $f" >> "$LOG"
    grep -E "Round [0-9]/5 - " "$f" | tail -5 >> "$LOG"
  fi

  # 自动归档推送（只提交原始日志与 TB，避免误提交其他改动）
  git add logs_3/visdrone tb >/dev/null 2>&1
  git commit -m "exp(visdrone): ${attack} ${rule} 5 轮自动归档" >/dev/null 2>&1
  git push origin dev >/dev/null 2>&1
}

# 等当前 FedAvg 跑完（最多约 110 分钟）
wait_pid "$CURRENT_PID"
echo "[queue] current run finished $(date '+%T')" >> "$LOG"
sleep 30

run_one signflipping DPFLA
run_one signflipping tmean
run_one signflipping mkrum
run_one signflipping fedavg

echo "[queue] ALL DONE $(date '+%F %T')" >> "$LOG"
