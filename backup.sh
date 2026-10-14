#!/bin/bash
# 一键备份当前项目到 GitHub
#
# 用法: ./backup.sh "本次改动说明" [分支]
#   分支默认 = 当前分支（日常工作应在 dev 上，故默认推 dev）
#   要推 main 必须显式指定: ./backup.sh "说明" main
#
# 约定（2026-10-14 起）:
#   - 默认推送 dev 分支（云平台日常开发分支）
#   - main 只在"版本确认可用"时由人工手动推送，不做默认目标
cd "$(dirname "$0")" || exit 1

MSG="$1"
[ -z "$MSG" ] && MSG="backup: $(date '+%Y-%m-%d %H:%M:%S')"
BRANCH="${2:-$(git rev-parse --abbrev-ref HEAD)}"

if [ "$BRANCH" = "main" ]; then
  echo "[backup] 注意: 目标分支是 main（约定为人工手动发布分支）"
fi

git add -A
if git diff --cached --quiet; then
  echo "[backup] 没有需要提交的改动，工作区干净"
  exit 0
fi

git commit -m "$MSG" || exit 1

echo "[backup] 同步远程最新版本 (origin/$BRANCH)..."
if ! git pull --rebase origin "$BRANCH"; then
  echo "[backup] 变基失败（可能存在冲突），请手动处理: git rebase --abort 可撤销"
  exit 1
fi

if git push origin "$BRANCH"; then
  echo "[backup] 备份成功 -> origin/$BRANCH: $MSG"
  git --no-pager log -1 --oneline
else
  echo "[backup] 推送失败，请检查网络"
  exit 1
fi
