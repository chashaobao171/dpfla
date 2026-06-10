# MEMORY 文件索引（必读顺序）

```
MEMORY/
├── 00_QuickStart.md          ← 先读这个！新会话第一页（≤5分钟上手）
├── 01_Workbench_Memory.md    ← 深度工作台记忆（实验记录/故障/冻结决策）
├── 02_Methods_Reference.md   ← 方法论参考（算法原理/代码骨架/设计决策）
├── 03_ThirdParty_Skeleton.md ← 第三方参考工程最小骨架（仅供借鉴）
├── 01_ChangeLog.md           ← 改动历史（每次重大改动追加记录）
└── 04_GitGuide.md            ← GitHub 日常更新流程（token / 命令速查）
```

---

## 各文件职责

| 文件 | 目标读者 | 更新频率 |
|------|---------|---------|
| `00_QuickStart.md` | 每次新会话第一个打开的文件 | 每次任务变更时更新 |
| `01_Workbench_Memory.md` | 需要理解历史实验/故障时查阅 | 追加式更新 |
| `02_Methods_Reference.md` | 理解算法原理/代码设计时查阅 | 结构性更新 |
| `03_ThirdParty_Skeleton.md` | 迁移第三方工程时查阅 | 一次性建立，后续微调 |
| `01_ChangeLog.md` | 查看所有历史改动和回退记录 | 追加式更新 |
| `04_GitGuide.md` | 日常 git 提交/推送参考 | 基本不变 |

---

## 快速定位提示

| 需要找什么 | 去哪个文件 |
|-----------|-----------|
| 上次做到哪了 | `00_QuickStart.md` → §当前状态 |
| 核心算法原理 | `02_Methods_Reference.md` → §DPFLA / §YOLO集成 |
| 实验脚本怎么跑 | `00_QuickStart.md` → §运行命令 |
| 遇到报错/故障 | `01_Workbench_Memory.md` → §故障字典 |
| 冻结红线（不能改什么） | `00_QuickStart.md` → §冻结红线 |
| 路径/数据/环境配置 | `01_Workbench_Memory.md` → §环境配置 |
| 改动历史/回退记录 | `01_ChangeLog.md` |
| Git 日常更新流程 | `04_GitGuide.md` |
| mAP 暴跌根因诊断（完整报告） | `01_Workbench_Memory.md` → §VisDrone mAP 暴跌根因诊断 |
