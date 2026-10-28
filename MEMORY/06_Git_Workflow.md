# 06_Git_Workflow.md — Git 与备份流程

<!-- 最后更新：2026-10-27 -->

## §项目信息

| 项 | 值 |
|----|----|
| GitHub 仓库 | https://github.com/chashaobao171/dpfla |
| 远址 | `git@github.com:chashaobao171/dpfla.git` |
| 本地路径 | `/root/chashaobao/DPFLA-master` |
| `main` | **发布分支**：只在"版本确认可用"时，由**人工手动**推送 |
| `dev` | **日常工作分支**：改动 / 实验记录**默认推这里**（2026-10-14 起）|
| `master` | 历史遗留（= 初始提交 `8ef2a47`）|
| `archive/2026-10-27-full` | **只读存档（不再修改）**：完整快照，含 yolo 权重 / 数据缓存 / runs —— `dba6888`，403 文件 |
| `archive/2026-10-27-lean` | **只读存档（不再修改）**：仅 DPFLA 项目本体 —— `61119a0`，272 文件 |
| `socket` | **参考分支（不合并）**：用户早期自写的 socketFed（52 文件），P4 分布式协议参考；2026-10-28 已核验，见 `08_Roadmap.md` §5 |
| `FLPoison` | **参考分支（不合并）**：上游 `vio1etus/FLPoison` 整仓拷贝（92 文件），攻击/防御实现参考 |

## §备份策略（重要）

- **GitHub = 最佳前版本备份**（只存"已确认效果可以"的版本）。
- **云平台 = 最新工作区**（改代码、调参、跑实验都在这里，允许脏）。
- **只有认为效果可以了，才一键覆盖 GitHub**，不要边调边推。
- **分支约定（2026-10-14 起，用户明确）**：
  - **默认推送 `dev`** —— 日常改动、实验记录都走 dev。
  - **`main` 只在用户认为"版本可用"时由人工手动推送**，**不作为默认目标**。

## §存档分支（2026-10-27 建立，只读）

用户要求为"进入分布式改造前"的项目状态建立 **两个不再修改** 的存档分支（作为阶段小结）：

| 分支 | 提交 | 内容 | 相对 dev |
|------|------|------|----------|
| `archive/2026-10-27-full` | `dba6888` | 完整快照 | dev(1d06b97) **+** `yolov8{l,m,s,n}.pt`、`cache/`、`runs/`、`orthogonal_matrices/{387,512,579}.pkl`（后三类为 gitignore 项，用临时索引强制纳入）|
| `archive/2026-10-27-lean` | `61119a0` | 仅 DPFLA 项目本体 | dev **−** `FedAIoT-main/`（参考第三方项目）、`refs/`（参考论文）、`federated_learning/attack_alg/enhanced_label_flip_attack.py`（无引用死代码）**+** 上述 3 个 pkl（保证随机正交矩阵可复现）|

- 两者均以 **临时索引（`GIT_INDEX_FILE=/tmp/...`）** 构建，**未触碰 dev 工作区与索引**；构建方式可复用：
  `GIT_INDEX_FILE=/tmp/x git read-tree dev && git add -f <文件> && T=$(git write-tree) && git commit-tree $T -p dev -m ... && git branch <名> <commit>`
- ⚠️ **这两个分支从此不再追加任何提交**；后续工作一律走 `dev`。
- full 分支含 84M 的 `yolov8l.pt`（GitHub 单文件 <100M 限制内，已成功推送）；full 的 blob 已进本地 `.git`（约 +160M 对象）。

### 本地 zip 备份（2026-10-27）

| 文件 | 大小 | 内容 |
|------|------|------|
| `/root/chashaobao/data.zip` | 2.2G | `data/`（= `/root/autodl-tmp/data`：MNIST / CIFAR10 / VisDrone 原始数据）|
| `/root/chashaobao/DPFLA-master.zip` | 366M | 整个项目目录（含 `.git`，含两条存档分支）|

## §一键备份

```bash
cd /root/chashaobao/DPFLA-master
./backup.sh "本次改动说明"          # 默认推「当前分支」（日常 = dev）
./backup.sh "本次改动说明" main     # 推 main 必须显式指定
```

`backup.sh` 做的事：`git add -A` → 无改动则退出 → `git commit` → `git pull --rebase origin <分支>` → `git push origin <分支>`；`<分支>` 默认取**当前分支**。
变基冲突时脚本会中止并提示：`git rebase --abort` 可撤销。

> 2026-10-14 修正：该脚本原先**写死 `origin main`**，会把日常改动推错到 main；现已改为**默认当前分支**。

## §手动流程（需要更细粒度时）

```bash
git status                 # 看改动
git diff --stat            # 看改了哪些文件
git add -A
git commit -m "说明"
git pull --rebase origin dev    # 同步远程（日常为 dev）
git push origin dev
```

## §.gitignore 规则

以下**不入库**：`*.pt`（权重）、`runs/`、`cache/`、`data/`、`visdrone_temp*.yaml`、`__pycache__/`、`.idea/`、`.kiro/`、`.vscode/`、`.ipynb_checkpoints/`，以及下面例外之外的 `logs_*/`、`*.log`。

> **入 git 的例外**：
> - `log/` 及其 `*.log` —— 归档的合格日志（跨主机同步 / 对外引用）。
> - **`logs_3/` 及其 `*.log` —— 原始日志，2026-10-16 用户明确要求"原始日志也一并推送"**，全部跨主机留存（体积约 9MB / 35 文件，可接受）。
> - `visdrone.yaml`（`*.yaml` 例外保留）；`tb/`（TensorBoard 事件，故意跟踪）。
>
> **TensorBoard 保留策略（2026-10-23 用户确认更新）**：`tb/`（仓库镜像）、`/root/tf-logs`（**TB 实际 logdir**）、`runs/` 三处只保留"**最终结论**"实验：
> ① 写进 `01_Facts.md` 的正式实验；② 与 `log/` 归档一一对应的；③ **探索性过程数据（σ 阶梯筛选、算法中间版本、1 轮冒烟、跑错口径的）一律删**（数值在记忆，`logs_3/` 原始日志不动）。
> - 现状（10-23 清理后）：`/root/tf-logs` **14**、`tb/` **13**、`runs/` **4**；`runs/detect/`（53 个 ultralytics 评估垃圾目录）已删。
> - 7 端无攻击对照（74.25%）的 TB 曲线已按用户要求删除，数值与原始日志保留。
> - ⚠️ 注意：**同分钟启动的多条实验会撞名写进同一个 TB 目录**（时间戳到分钟），导致曲线混杂、无法单独清理 → 批量跑实验时错开分钟启动。
> - `orthogonal_matrices/` 与 `FedAIoT-main/` **目前仍在 git 跟踪中**。

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
