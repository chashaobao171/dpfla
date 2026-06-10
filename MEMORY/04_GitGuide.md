# 04_GitGuide.md — GitHub 日常更新流程

<!-- 建立时间：2026-06-10 -->
<!-- 项目已关联 https://github.com/chashaobao171/dpfla -->

---

## 项目信息

| 项目 | 地址 |
|------|------|
| GitHub 仓库 | https://github.com/chashaobao171/dpfla |
| 本地路径 | `/root/chashaobao/DPFLA-master` |
| 默认分支 | `main` |
| 认证方式 | GitHub Personal Access Token（credential helper 缓存） |

---

## 日常更新流程（三步走）

### 1. 查看改动

```bash
cd /root/chashaobao/DPFLA-master
git status
```

会看到类似这样的输出：

```
Changes not staged for commit:
  modified:   federated_learning/fl_core.py
  modified:   run-test/visdrone/run_no_attack_baseline.py
Untracked files:
  logs_3/visdrone/run_20260610_1430.log
```

### 2. 提交改动

```bash
# 把所有改动暂存（-A 表示全部）
git add -A

# 提交（替换 "your message" 为这次改了什么）
git commit -m "your message"
```

**Commit message 规范**：

```bash
# 格式：用一句话描述改动
git commit -m "修复 fl_core.py val bug，改用 torch.save() 替代 yolo.save()"

# 多个改动可以分多次 commit
git commit -m "更新 visdrone_fed_hparams.py 超参数为 BS=64, EPOCH=10"
git commit -m "新增 FedLA 对照实验脚本"
```

### 3. 推送到 GitHub

```bash
git push
```

推送后 GitHub 会自动更新。

---

## 注意事项

### .gitignore 规则

以下文件不会进入 git，不会被提交：

| 文件/目录 | 原因 |
|-----------|------|
| `*.pt`（模型权重） | 体积大，从 ultralytics 重新下载 |
| `logs_*/`、`runs/` | 训练日志和运行结果 |
| `cache/` | pickle 缓存 |
| `data/` | 数据集，本地存放 |
| `*.yaml`（temp 版） | 临时配置文件 |
| `.idea/`、`.vscode/`、`.kiro/` | 编辑器配置 |
| `__pycache__/` | Python 缓存 |

### 提交前检查

```bash
# 查看即将提交的文件
git status

# 查看具体改动内容（diff）
git diff --stat

# 避免提交大文件
git check-ignore "path/to/large_file.pt"
```

### 推送失败处理

如果推送时提示 "failed to push"，说明 GitHub 上有新的提交（可能在网页上手动改了东西），需要先拉取：

```bash
git pull --rebase origin main
git push
```

---

## Token 认证（已配置）

认证信息已通过 GitHub Personal Access Token 配置到 git credential helper 中。
每次 `git push` 不需要手动输入密码，autodl 会自动使用缓存的凭证。

**Token 安全建议**：定期（如每 30 天）在 GitHub 上 revoke 旧 token 并重新生成，然后更新本地凭证：

```bash
# 1. 在 GitHub 上 revoke 旧 token
#    Settings → Developer settings → Personal access tokens → 删除旧 token

# 2. 生成新 token（勾选 repo 权限）

# 3. 更新本地凭证
git remote set-url origin https://[新TOKEN]@github.com/chashaobao171/dpfla.git

# 4. 测试
git push
```

---

## 常用 Git 命令速查

| 场景 | 命令 |
|------|------|
| 查看状态 | `git status` |
| 查看改动 | `git diff` |
| 提交所有改动 | `git add -A && git commit -m "message"` |
| 推送到 GitHub | `git push` |
| 查看提交历史 | `git log --oneline -10` |
| 撤销未提交的改动 | `git checkout -- .` |
| 查看远程地址 | `git remote -v` |
| 更新本地分支信息 | `git fetch origin` |
| 查看所有分支 | `git branch -a` |

---

## 分支管理

当前仓库只有 `main` 分支。后续如果需要做实验对比，可以在本地创建新分支：

```bash
# 创建并切换到新分支
git checkout -b experiment/fedla-compare

# 在新分支上工作、提交、推送
git add -A && git commit -m "fedla experiment" && git push -u origin experiment/fedla-compare

# 切回主分支
git checkout main
```
