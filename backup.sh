#!/bin/bash
# 一键备份当前项目到 GitHub
# 用法: ./backup.sh "本次改动说明"   (不传说明则用时间戳)
cd "$(dirname "$0")" || exit 1

MSG="$1"
[ -z "$MSG" ] && MSG="backup: $(date '+%Y-%m-%d %H:%M:%S')"

git add -A
if git diff --cached --quiet; then
  echo "[backup] 没有需要提交的改动，工作区干净"
  exit 0
fi

git commit -m "$MSG" || exit 1

echo "[backup] 同步远程最新版本..."
if ! git pull --rebase origin main; then
  echo "[backup] 变基失败（可能存在冲突），请手动处理: git rebase --abort 可撤销"
  exit 1
fi

if git push origin main; then
  echo "[backup] 备份成功: $MSG"
  git --no-pager log -1 --oneline
else
  echo "[backup] 推送失败，请检查网络"
  exit 1
fi
