#!/usr/bin/env bash
set -euo pipefail

# 一键启动 TensorBoard（适配远程服务器 + 端口转发场景）
# 用法：
#   bash scripts/start_tensorboard.sh
#
# 然后在本地 IDE（Cursor/VSCode）里把 6006 端口转发出来，访问：
#   http://localhost:6006

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${1:-${ROOT_DIR}/runs}"
PORT="${PORT:-6006}"
HOST="${HOST:-0.0.0.0}"

echo "[TensorBoard] logdir: ${LOGDIR}"
echo "[TensorBoard] host:   ${HOST}"
echo "[TensorBoard] port:   ${PORT}"
echo "[TensorBoard] open:   http://localhost:${PORT}"
echo

exec tensorboard --logdir "${LOGDIR}" --host "${HOST}" --port "${PORT}" --reload_interval 5

