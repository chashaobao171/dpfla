#!/usr/bin/env bash
# ---------------------------------------------------------------
# 在 tmux 会话中启动 VisDrone 联邦实验
#   - 实时日志：终端直接看刷屏输出
#   - 断网不影响：进程跑在云主机的 tmux 会话里，本地断线/关 Trae 都不会杀进程
#
# 用法:
#   ./scripts/run_fl_tmux.sh <实验名> [环境变量覆盖...]
#
# 示例:
#   ./scripts/run_fl_tmux.sh run_no_attack_baseline                   # 基线(默认 640)
#   ./scripts/run_fl_tmux.sh run_no_attack_baseline FL_IMG_SIZE=1024  # 分辨率单变量实验
#   ./scripts/run_fl_tmux.sh run_dpfla_label_flipping FL_GLOBAL_ROUNDS=50
#
# 操作:
#   脱离会话(继续后台跑):  Ctrl-b 然后按 d
#   重新连接看日志:        tmux attach -t fl
#   列出会话:              tmux ls
#   结束实验:              连上后 Ctrl-C
#
# 日志文件: logs_3/visdrone/<实验名>_<北京时间>.log
#
# 注意: 训练需要 GPU，必须能访问 /dev/nvidia*。
#       用自己的终端直接跑不受沙箱影响；若通过 agent 跑需非沙箱模式。
# ---------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v tmux >/dev/null 2>&1; then
  echo "✗ 未安装 tmux。安装方式: apt-get install -y tmux   或   conda install -y tmux" >&2
  exit 1
fi

EXPERIMENT="${1:-run_no_attack_baseline}"
shift || true

SCRIPT_DIR="run-test/visdrone"
SCRIPT="${SCRIPT_DIR}/${EXPERIMENT}.py"
SESSION="${FL_TMUX_SESSION:-fl}"

if [ ! -f "$SCRIPT" ]; then
  echo "✗ 找不到实验脚本: $SCRIPT" >&2
  echo "  可选实验:" >&2
  ls "$SCRIPT_DIR"/run_*.py 2>/dev/null | sed 's#.*/##; s#\.py$##; s#^#    #' >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "✗ tmux 会话 '$SESSION' 已存在。" >&2
  echo "  查看:  tmux attach -t $SESSION" >&2
  echo "  换名:  FL_TMUX_SESSION=fl2 $0 $EXPERIMENT $*" >&2
  exit 1
fi

# 组装环境变量覆盖（例如 FL_IMG_SIZE=1024）
ENVSTR=""
for kv in "$@"; do
  ENVSTR="${ENVSTR} ${kv}"
done

echo "启动 tmux 会话: $SESSION"
echo "  实验: ${EXPERIMENT}"
echo "  覆盖:${ENVSTR:-（无）}"
echo "  日志: logs_3/visdrone/${EXPERIMENT}_*.log"
echo

# FL_ATTACHED=1 → 前台运行(实时刷屏)；由 tmux 保证本地断网不中断
tmux new-session -d -s "$SESSION" -c "$PWD" \
  "FL_ATTACHED=1${ENVSTR} python ${SCRIPT}; ec=\$?; echo; echo \"--- 实验结束(exit=\$ec) ---\"; exec bash"

echo "已启动。进入查看实时日志:  tmux attach -t $SESSION"
echo "脱离(后台继续跑):          Ctrl-b 然后按 d"
