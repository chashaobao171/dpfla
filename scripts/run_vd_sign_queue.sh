#!/bin/bash
# VisDrone SignFlipping 四臂自动串行（2026-10-23 夜间，v2）
#
# v1 的两个 bug 已修：
#  1) 时间闸门整数比较错误（22:49 被当成已过 03:30）→ 改为只判断 03:30–05:00；
#  2) 用 `kill -0 $!` 判断实验结束不可靠（会把刚启动的 run 误判为完成）→ 改为**日志完成标记**驱动：
#     新日志出现 + 出现 "Round 5/5 - "（或 "测试失败"）才算结束。
# 另外用 mkdir 锁避免工具重试导致的双队列。
#
# 启动方式（必须普通 Shell，不能在 run_in_background 沙箱任务里）：
#   setsid nohup bash scripts/run_vd_sign_queue.sh > /tmp/flrun/vd_sign_queue.nohup 2>&1 < /dev/null &

set -u
cd /root/chashaobao/DPFLA-master || exit 1

LOCK=/tmp/vd_sign_queue.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "[queue] another runner holds $LOCK, exit $(date '+%F %T')" >> /tmp/flrun/vd_sign_queue.out
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

LOG=/tmp/flrun/vd_sign_queue.out
RUN_LOG_DIR=/tmp/flrun
mkdir -p "$RUN_LOG_DIR"
echo "[queue] v2 start $(date '+%F %T')" >> "$LOG"

wait_no_training() {
  local i=0
  # 必须锚定 ^python：`pgrep -f` 会匹配到工具 wrapper 的 bash -c 命令行
  #（其中含脚本路径），导致队列误以为实验一直在跑（2026-10-23 实际踩坑）。
  while pgrep -af "^python -u run-test/visdrone/run_screen_server_attack.py" >/dev/null 2>&1; do
    sleep 15
    i=$((i + 1))
    [ $i -gt 80 ] && return 1
  done
  return 0
}

run_one() {
  local attack=$1
  local rule=$2
  local hhmm
  hhmm=$(date +%H%M)
  # 只在凌晨 03:30–05:00 跳过（给 04:00 关机留余量）
  if [ "$hhmm" -ge 330 ] && [ "$hhmm" -lt 500 ]; then
    echo "[queue] skip $attack/$rule (after 03:30)" >> "$LOG"
    return 0
  fi

  wait_no_training
  echo "[queue] launch $attack/$rule $(date '+%T')" >> "$LOG"

  local marker
  marker=$(mktemp /tmp/vd_marker.XXXXXX)
  touch "$marker"
  sleep 2

  FL_ATTACHED=1 FL_DETACHED=1 FL_GLOBAL_ROUNDS=5 \
  FL_SERVER_ATTACK="$attack" FL_RULE="$rule" \
    nohup python -u run-test/visdrone/run_screen_server_attack.py \
    > "$RUN_LOG_DIR/vd_${attack}_${rule}_fix.log" 2>&1 &

  # 等新日志文件出现（最多 120s）
  local f=""
  local i=0
  while [ $i -lt 40 ]; do
    f=$(find logs_3/visdrone -maxdepth 1 -name "run_screen_${attack}_${rule}_*.log" -newer "$marker" 2>/dev/null | head -1)
    [ -n "$f" ] && break
    sleep 3
    i=$((i + 1))
  done
  rm -f "$marker"
  if [ -z "$f" ]; then
    echo "[queue] ERROR: no log file for $attack/$rule $(date '+%T')" >> "$LOG"
    return 1
  fi
  echo "[queue] log: $f" >> "$LOG"

  # 完成判定：Round 5/5 或 测试失败；最多 100 分钟
  i=0
  local status="timeout"
  while [ $i -lt 200 ]; do
    if grep -q "Round 5/5 - " "$f"; then status="done"; break; fi
    if grep -q "测试失败" "$f"; then status="failed"; break; fi
    sleep 30
    i=$((i + 1))
  done
  echo "[queue] $status $attack/$rule $(date '+%T')" >> "$LOG"
  grep -E "Round [0-9]/5 - " "$f" | tail -5 >> "$LOG"

  wait_no_training

  # 自动归档推送（只提交原始日志与 TB）
  git add logs_3/visdrone tb >/dev/null 2>&1
  git commit -m "exp(visdrone): ${attack} ${rule} 5 轮自动归档" >/dev/null 2>&1
  git push origin dev >/dev/null 2>&1
}

run_one signflipping DPFLA
run_one signflipping tmean
run_one signflipping mkrum
run_one signflipping fedavg

echo "[queue] ALL DONE $(date '+%F %T')" >> "$LOG"
