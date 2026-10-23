#!/bin/bash
# ALIE vs IPM 1 轮探针（2026-10-24）
# 目的：在花 4×5 轮之前，用 R1 判断哪种攻击更能同时打伤 TMean 与 Krum。
# 顺序：alie tmean → alie mkrum → ipm tmean → ipm mkrum
# 先等待当前 ALIE+DPFLA 5 轮跑完并归档。

set -u
cd /root/chashaobao/DPFLA-master || exit 1

LOG=/tmp/flrun/vd_probe.out
HEART=/tmp/flrun/vd_probe.heartbeat
STATE=/tmp/vd_probe_state.txt
RESULTS=log/visdrone/auto_results.md
LOCK=/tmp/vd_probe.lock

mkdir -p /tmp/flrun
[ -f "$STATE" ] || echo "0" > "$STATE"

if ! mkdir "$LOCK" 2>/dev/null; then
  if [ -f "$LOCK/pid" ] && kill -0 "$(cat "$LOCK/pid")" 2>/dev/null; then
    exit 0
  fi
  rm -rf "$LOCK"
  mkdir "$LOCK" || exit 1
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT

echo "[probe] start $(date '+%F %T') idx=$(cat "$STATE")" >> "$LOG"
heartbeat() { date '+%F %T' > "$HEART"; }
has_python() { pgrep -af "^python -u run-test/visdrone/run_screen_server_attack.py" >/dev/null 2>&1; }
wait_no_python() { local i=0; while has_python; do sleep 20; heartbeat; i=$((i+1)); [ $i -gt 300 ] && return 1; done; return 0; }

record_and_push() {
  local f=$1 attack=$2 rule=$3 tag=$4
  if [ -n "$f" ] && ! grep -q "$f" "$RESULTS"; then
    {
      echo "## ${attack} ${rule} ${tag} $(date '+%F %T')"
      echo "\`${f}\`"
      grep -E "Round [0-9]+/[0-9]+ - " "$f"
      echo
    } >> "$RESULTS"
  fi
  git add logs_3/visdrone tb "$RESULTS" >/dev/null 2>&1
  git commit -m "exp(visdrone): ${attack} ${rule} ${tag}自动归档" >/dev/null 2>&1
  git push origin dev >/dev/null 2>&1
}

# 1) 等当前 ALIE+DPFLA 5 轮结束
current=$(ls -t logs_3/visdrone/run_screen_alie_DPFLA_*.log 2>/dev/null | head -1)
if [ -n "$current" ] && ! grep -q "Round 5/5 - " "$current"; then
  echo "[probe] wait current ALIE+DPFLA: $current $(date '+%T')" >> "$LOG"
  while ! grep -q "Round 5/5 - " "$current"; do
    has_python || break
    sleep 30
    heartbeat
  done
fi
[ -n "$current" ] && record_and_push "$current" alie DPFLA "5轮"
echo "[probe] current ALIE+DPFLA recorded $(date '+%T')" >> "$LOG"

PROBES=("alie tmean" "alie mkrum" "ipm tmean" "ipm mkrum")

idx=$(cat "$STATE" 2>/dev/null || echo 0)
while [ "$idx" -lt "${#PROBES[@]}" ]; do
  read -r attack rule <<< "${PROBES[$idx]}"
  echo "[probe] launch probe $attack/$rule (idx=$idx) $(date '+%T')" >> "$LOG"

  wait_no_python
  marker=$(mktemp /tmp/vd_probe_marker.XXXXXX)
  touch "$marker"
  sleep 2

  FL_ATTACHED=1 FL_DETACHED=1 FL_GLOBAL_ROUNDS=1 \
  FL_SERVER_ATTACK="$attack" FL_RULE="$rule" \
    nohup python -u run-test/visdrone/run_screen_server_attack.py \
    > "/tmp/flrun/vd_${attack}_${rule}_probe.log" 2>&1 &

  f=""
  i=0
  while [ $i -lt 40 ]; do
    f=$(find logs_3/visdrone -maxdepth 1 -name "run_screen_${attack}_${rule}_*.log" -newer "$marker" 2>/dev/null | head -1)
    [ -n "$f" ] && break
    sleep 3
    i=$((i + 1))
  done
  rm -f "$marker"

  if [ -z "$f" ]; then
    echo "[probe] ERROR no log $attack/$rule" >> "$LOG"
  else
    i=0
    status="timeout"
    while [ $i -lt 60 ]; do
      if grep -q "Round 1/1 - " "$f"; then status="done"; break; fi
      if grep -q "测试失败" "$f"; then status="failed"; break; fi
      sleep 20
      heartbeat
      i=$((i + 1))
    done
    echo "[probe] $status $attack/$rule $(date '+%T')" >> "$LOG"
    record_and_push "$f" "$attack" "$rule" "探针1轮"
  fi

  wait_no_python
  idx=$((idx + 1))
  echo "$idx" > "$STATE"
  heartbeat
done

touch /tmp/vd_probe_done
echo "[probe] ALL DONE $(date '+%F %T')" >> "$LOG"
