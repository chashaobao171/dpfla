#!/bin/bash
# VisDrone SignFlipping 夜间监督进程（v3，2026-10-23）
#
# 设计：
# - 用状态文件 /tmp/vd_state.txt 记录"下一项"，重启后从断点继续；
# - 用日志完成标记（Round 5/5 或 测试失败）判断实验结束，不用 PID；
# - 只有当前没有任何训练 python 时才启动下一项（锚定 ^python，避免匹配工具 wrapper）；
# - 每完成一项：追加结果到 log/visdrone/auto_results.md 并 git commit/push；
# - mkdir 锁避免重复实例；heartbeat 文件供 watchdog/人工检查；
# - 03:30–05:00 不再启动新实验。
#
# 启动（必须普通 Shell，不能放 run_in_background 沙箱任务）：
#   setsid nohup bash scripts/run_vd_sign_supervisor.sh >/dev/null 2>&1 < /dev/null &

set -u
cd /root/chashaobao/DPFLA-master || exit 1

LOG=/tmp/flrun/vd_supervisor.out
HEART=/tmp/flrun/vd_supervisor.heartbeat
STATE=/tmp/vd_state.txt
RESULTS=log/visdrone/auto_results.md
LOCK=/tmp/vd_supervisor.lock

mkdir -p /tmp/flrun
[ -f "$STATE" ] || echo "signflipping tmean" > "$STATE"
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

echo "[sup] start $(date '+%F %T') state=$(cat "$STATE")" >> "$LOG"

heartbeat() { date '+%F %T' > "$HEART"; }

has_python() {
  pgrep -af "^python -u run-test/visdrone/run_screen_server_attack.py" >/dev/null 2>&1
}

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
  local f=$1 attack=$2 rule=$3
  if [ -n "$f" ] && ! grep -q "$f" "$RESULTS"; then
    {
      echo "## ${attack} ${rule} $(date '+%F %T')"
      echo "\`${f}\`"
      grep -E "Round [0-9]/5 - " "$f"
      echo
    } >> "$RESULTS"
  fi
  git add logs_3/visdrone tb "$RESULTS" >/dev/null 2>&1
  git commit -m "exp(visdrone): ${attack} ${rule} 5 轮自动归档" >/dev/null 2>&1
  git push origin dev >/dev/null 2>&1
}

# 1) 接手当前已在跑的 SignFlipping + DPFLA（不重复启动）
initial_dpfla() {
  [ -f /tmp/vd_initial_done ] && return 0
  local f
  f=$(ls -t logs_3/visdrone/run_screen_signflipping_DPFLA_*.log 2>/dev/null | head -1)
  if [ -z "$f" ]; then
    touch /tmp/vd_initial_done
    return 0
  fi
  echo "[sup] wait current DPFLA: $f $(date '+%T')" >> "$LOG"
  while ! grep -q "Round 5/5 - " "$f"; do
    has_python || break
    sleep 30
    heartbeat
  done
  record_and_push "$f" signflipping DPFLA
  touch /tmp/vd_initial_done
  echo "[sup] DPFLA finished/recorded $(date '+%T')" >> "$LOG"
}

run_item() {
  local attack=$1 rule=$2
  local hhmm
  hhmm=$(date +%H%M)
  if [ "$hhmm" -ge 330 ] && [ "$hhmm" -lt 500 ]; then
    echo "[sup] skip $attack/$rule (after 03:30)" >> "$LOG"
    return 0
  fi

  # 重启恢复：若今天该实验已完成，只补记录，不重复跑
  local old
  old=$(ls -t logs_3/visdrone/run_screen_${attack}_${rule}_*.log 2>/dev/null | head -1)
  if [ -n "$old" ] && grep -q "Round 5/5 - " "$old" \
     && [ "$(date -d "$(stat -c %y "$old")" +%Y%m%d)" = "$(date +%Y%m%d)" ]; then
    echo "[sup] already done $attack/$rule, record only" >> "$LOG"
    record_and_push "$old" "$attack" "$rule"
    return 0
  fi

  wait_no_python
  echo "[sup] launch $attack/$rule $(date '+%T')" >> "$LOG"

  local marker
  marker=$(mktemp /tmp/vd_marker.XXXXXX)
  touch "$marker"
  sleep 2

  FL_ATTACHED=1 FL_DETACHED=1 FL_GLOBAL_ROUNDS=5 \
  FL_SERVER_ATTACK="$attack" FL_RULE="$rule" \
    nohup python -u run-test/visdrone/run_screen_server_attack.py \
    > "/tmp/flrun/vd_${attack}_${rule}_sup.log" 2>&1 &

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
    echo "[sup] ERROR no log $attack/$rule $(date '+%T')" >> "$LOG"
    return 1
  fi
  echo "[sup] log $f" >> "$LOG"

  i=0
  local status="timeout"
  while [ $i -lt 200 ]; do
    if grep -q "Round 5/5 - " "$f"; then status="done"; break; fi
    if grep -q "测试失败" "$f"; then status="failed"; break; fi
    sleep 30
    heartbeat
    i=$((i + 1))
  done
  echo "[sup] $status $attack/$rule $(date '+%T')" >> "$LOG"
  record_and_push "$f" "$attack" "$rule"
  wait_no_python
}

initial_dpfla

while :; do
  read -r attack rule < "$STATE" || true
  if [ "${attack:-}" = "DONE" ] || [ -z "${attack:-}" ]; then
    echo "[sup] finished $(date '+%T')" >> "$LOG"
    break
  fi
  run_item "$attack" "$rule"
  case "$attack $rule" in
    "signflipping tmean") echo "signflipping mkrum" > "$STATE" ;;
    "signflipping mkrum") echo "signflipping fedavg" > "$STATE" ;;
    "signflipping fedavg") echo "DONE" > "$STATE" ;;
    *) echo "DONE" > "$STATE" ;;
  esac
  heartbeat
done

echo "[sup] ALL DONE $(date '+%F %T')" >> "$LOG"
