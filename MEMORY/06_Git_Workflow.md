# 06_Git_Workflow.md — Git 与备份流程

<!-- 最后更新：2026-10-10 -->

## §项目信息

| 项 | 值 |
|----|----|
| GitHub 仓库 | https://github.com/chashaobao171/dpfla |
| 远址 | `git@github.com:chashaobao171/dpfla.git` |
| 本地路径 | `/root/chashaobao/DPFLA-master` |
| 默认分支 | `main`（另有历史 `origin/master`） |

## §备份策略（重要）

- **GitHub = 最佳前版本备份**（只存"已确认效果可以"的版本）。
- **云平台 = 最新工作区**（改代码、调参、跑实验都在这里，允许脏）。
- **只有认为效果可以了，才一键覆盖 GitHub**，不要边调边推。

## §一键备份

```bash
cd /root/chashaobao/DPFLA-master
./backup.sh "本次改动说明"     # 不传说明则用时间戳
```

`backup.sh` 做的事：`git add -A` → 无改动则退出 → `git commit` → `git pull --rebase origin main` → `git push origin main`。
变基冲突时脚本会中止并提示：`git rebase --abort` 可撤销。

## §手动流程（需要更细粒度时）

```bash
git status                 # 看改动
git diff --stat            # 看改了哪些文件
git add -A
git commit -m "说明"
git pull --rebase origin main   # 同步远程
git push
```

## §.gitignore 规则

以下**不入库**：`*.pt`（权重）、`logs_*/`、`runs/`、`cache/`、`data/`、`*.log`、`visdrone_temp*.yaml`、`__pycache__/`、`.idea/`、`.kiro/`、`.vscode/`、`.ipynb_checkpoints/`。

> 注意：`*.yaml` 被忽略，但 `!visdrone.yaml` 例外（保留）。`orthogonal_matrices/` 与 `FedAIoT-main/` **目前仍在 git 跟踪中**。

## §安全须知

- **不要** `git push --force` 到 `main`。
- **不要** `git reset --hard` / `git checkout --` 丢弃未确认的本地改动。
- 提交前用 `git check-ignore <file>` 确认大文件确实被忽略。
- 删除任何数据前，确认不在 `05_Environment.md` §5 的数据白名单内。

## §常用命令速查

| 场景 | 命令 |
|------|------|
| 状态 | `git status` |
| 历史 | `git log --oneline -10` |
| 远程 | `git remote -v` / `git fetch origin` |
| 分支 | `git branch -a` |
| 撤销未暂存改动 | `git checkout -- <file>` |
