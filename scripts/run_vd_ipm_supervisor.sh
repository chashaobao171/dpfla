#!/bin/bash
# VisDrone IPM 四臂自动串行 supervisor（2026-10-24）
# 复用 SignFlipping supervisor 的可靠机制：
# - 状态文件断点续跑；日志 Round 5/5 完成标记；锚定 ^python；
# - mkdir 锁；heartbeat；每项自动记录 auto_results.md 并 git push。

set -u
cd /root/chashaobao/DPFLA-master || exit 1

ATTACK=ipm
LOG=/tmp/flrun/vd_ipm_supervisor.out
HEART=/tmp/flrun/vd_ipm_supervisor.heartbeat
STATE=/tmp/vd_ipm_state.txt
RESULTS=log/visdrone/auto_results.md
LOCK=/tmp/vd_ipm_supervisor.lock

mkdir -p /tmp/flrun
[ -f "$STATE" ] || echo "$ATTACK DPFLA" > "$STATE"
[ -f "$RESULTS" ] || : > "$RESULTS"

if ! mkdir "$LOCK" 2>/dev/null; then
  if [ -f "$LOCK/pid" ] && kill -0 "$(cat "$LOCK/pid")" 2>/dev/null; then
    exit 0
  fi
  rm -rf "$LOCK"
  mkdir "$LOCK" || exit 1
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT

echo "[ipm-sup] start $(date '+%F %T') state=$(cat "$STATE")" >> "$LOG"
heartbeat() { date '+%F %T' > "$HEART"; }
has_python() { pgrep -af "^python -u run-test/visdrone/run_screen_server_attack.py" >/dev/null 2>&1; }

wait_no_python() {
  local i=0
  while has_python; do
    sleep 20
    heartbeat
    i=$((i + 1))
    [ $i -gt 300 ] && return 1
  done
  return 0
}

record_and_push() {
  local f=$1 rule=$2
  if [ -n "$f" ] && ! grep -q "$f" "$RESULTS"; then
    {
      echo "## ${ATTACK} ${rule} $(date '+%F %T')"
      echo "\`${f}\`"
      grep -E "Round [0-9]/5 - " "$f"
      echo
    } >> "$RESULTS"
  fi
  git add logs_3/visdrone tb "$RESULTS" >/dev/null 2>&1
  git commit -m "exp(visdrone): ${ATTACK} ${rule} 5 轮自动归档" >/dev/null 2>&1
  git push origin dev >/dev/null 2>&1
}

run_item() {
  local rule=$1
  local old
  old=$(ls -t logs_3/visdrone/run_screen_${ATTACK}_${rule}_*.log 2>/dev/null | head -1)
  if [ -n "$old" ] && grep -q "Round 5/5 - " "$old" \
     && [ "$(date -d "$(stat -c %y "$old")" +%Y%m%d)" = "$(date +%Y%m%d)" ]; then
    echo "[ipm-sup] already done $rule, record only" >> "$LOG"
    record_and_push "$old" "$rule"
    return 0
  fi

  wait_no_python
  echo "[ipm-sup] launch $ATTACK/$rule $(date '+%T')" >> "$LOG"
  local marker
  marker=$(mktemp /tmp/vd_ipm_marker.XXXXXX)
  touch "$marker"
  sleep 2

  FL_ATTACHED=1 FL_DETACHED=1 FL_GLOBAL_ROUNDS=5 \
  FL_SERVER_ATTACK="$ATTACK" FL_RULE="$rule" \
    nohup python -u run-test/visdrone/run_screen_server_attack.py \
    > "/tmp/flrun/vd_${ATTACK}_${rule}_sup.log" 2>&1 &

  local f=""
  local i=0
  while [ $i -lt 40 ]; do
    f=$(find logs_3/visdrone -maxdepth 1 -name "run_screen_${ATTACK}_${rule}_*.log" -newer "$marker" 2>/dev/null | head -1)
    [ -n "$f" ] && break
    sleep 3
    i=$((i + 1))
  done
  rm -f "$marker"
  if [ -z "$f" ]; then
    echo "[ipm-sup] ERROR no log $rule $(date '+%T')" >> "$LOG"
    return 1
  fi
  echo "[ipm-sup] log $f" >> "$LOG"

  i=0
  local status="timeout"
  while [ $i -lt 200 ]; do
    if grep -q "Round 5/5 - " "$f"; then status="done"; break; fi
    if grep -q "测试失败" "$f"; then status="failed"; break; fi
    sleep 30
    heartbeat
    i=$((i + 1))
  done
  echo "[ipm-sup] $status $ATTACK/$rule $(date '+%T')" >> "$LOG"
  record_and_push "$f" "$rule"
  wait_no_python
}

while :; do
  read -r attack rule < "$STATE" || true
  if [ "${attack:-}" = "DONE" ] || [ -z "${attack:-}" ]; then
    echo "[ipm-sup] finished $(date '+%T')" >> "$LOG"
    break
  fi
  run_item "$rule"
  case "$rule" in
    DPFLA) echo "$ATTACK tmean" > "$STATE" ;;
    tmean) echo "$ATTACK mkrum" > "$STATE" ;;
    mkrum) echo "$ATTACK fedavg" > "$STATE" ;;
    fedavg) echo "DONE" > "$STATE" ;;
    *) echo "DONE" > "$STATE" ;;
  esac
  heartbeat
done

echo "[ipm-sup] ALL DONE $(date '+%F %T')" >> "$LOG"
