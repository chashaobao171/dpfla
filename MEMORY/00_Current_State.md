# 00_Current_State.md — 当前状态

<!-- 最后更新：2026-10-14 -->
<!-- ⚠️ 本文件会过期。与 01_Facts.md 冲突时，以 01_Facts.md 的实测数据为准 -->

## 一句话

**DPFLA**：联邦学习投毒防御项目，落地场景 **VisDrone + YOLOv8** 目标检测。

- 原目标（无攻击联邦基线 mAP@0.5 ≥ 32%）→ ✅ **已超额**：**v8m @1280 → R5 = 45.08%**（见 `01_Facts.md` §1c）。
- **当前阶段：进入研究本体——VisDrone 上的「攻击 vs 防御」实验。**

## 进度快照（2026-10-14）

| 项 | 状态 |
|---|---|
| VisDrone 基线 | ✅ **45.08%**（v8m @1280 / BS=16 / AMP，5 轮）|
| VisDrone 攻击+无防御 | 🟡 **已跑 3 次**：×15 两次**崩**（R1=0%）；**×1 几乎无效**（R4 43.24% vs 基线 43.82%）→ **攻击强度仍需调** |
| VisDrone 攻击+DPFLA | ⬜ **待跑**（待②的攻击强度定下来）|
| MNIST / CIFAR10 | ✅ 三份齐全，已归档（见 `log/README.md`）|
| 日志归档 | ✅ 新建 `log/`（只放合格日志；原始日志在 `logs_3/`）|

## 本阶段任务：VisDrone 攻击/防御三实验

**统一口径（必须与基线一致，否则不可比）**：`v8m` + `imgsz=1280` + `BS=16` + `AMP` + 10 客户端 + **攻击率 30%（=3 个恶意客户端）** + IID。

```bash
cd /root/chashaobao/DPFLA-master      # 确认在 dev 分支

COMMON="YOLO_MODEL_SIZE=m FL_IMG_SIZE=1280 FL_TRAIN_BATCH_SIZE=16 FL_AMP=1 FL_TEST_BATCH_SIZE=64 FL_DATALOADER_WORKERS=4"

# ① 基线（已完成，如需复现）
./scripts/run_fl_tmux.sh run_no_attack_baseline       $COMMON

# ② 有攻击 + 无防御（FedAvg）—— 预期 mAP 明显下降
./scripts/run_fl_tmux.sh run_no_defense_label_flipping $COMMON FL_MALICIOUS_RATE=0.3

# ③ 有攻击 + DPFLA 防御 —— 预期接近基线
./scripts/run_fl_tmux.sh run_dpfla_label_flipping      $COMMON FL_MALICIOUS_RATE=0.3
```

- 观测：`tmux attach -t fl`（Ctrl-b d 脱离）／`tail -f logs_3/visdrone/<新日志>`
- 攻击设置：**单标签反转 `1(people) → 9(motor)`**（固定映射 `{1:9}`，②③ 完全一致）；**行为率 100%**；30% 恶意客户端（3 个）
- ⚠️ **攻击强度主杠杆 = `FL_GRADIENT_SCALE`**（[client.py:522-543](file:///root/chashaobao/DPFLA-master/federated_learning/client.py#L522-L543)，**代码默认 15**：上传前把恶意客户端的权重 delta 放大 15×；**代码默认值不要改**，用环境变量覆盖）
- `FL_FLIP_RATIO` 只控制投毒**数量**（默认 1.0 = 全翻源类；<1 只翻一部分）
  - 2026-10-14 实测：默认 ×15 → **R1 即崩**（mAP 0.00%、测试损失 9412/10866，远超未训练的 902）；且把 `flip_ratio` 从 1.0 降到 0.1 **完全无效** → 见 `04_Retracted.md` R10 / R11
  - **实验②已跑结果**（详见 `01_Facts.md` §1d）：×15 → R1 崩到 0%；**×1 → R1 19.88%、R4 43.24%（基线 43.82%）→ 攻击几乎无效**
  - ⏭ **待办：在不放大的前提下加强攻击**。候选（按性价比）：① **换源类** `1(people 8.0%)` → **`0(pedestrian 23.2%)`**（污染面 ×2.9）；② 多类翻转 `{0:9,1:9,2:9}`（×4.3，但偏离"单标签"）；③ 放大倍数取中间值 `FL_GRADIENT_SCALE=3~5`
  - 已"满"的开关：**行为率 1.0** ✅、**`FL_FLIP_RATIO` 1.0** ✅（无法再加强）；未满：源类 / 翻转类别数 / 恶意客户端比例 / 放大倍数
- **产出后**：达标即复制进 `log/visdrone/02_...` 与 `03_...`，并更新 `log/README.md` 索引

## 关键上下文（新会话必读）

1. **环境**：两台 autodl 主机交替用（新机 4090 48G / 20 核 / **90G 内存**；老机 4080S 32G / 12 核 / **62G 内存**）。**跑 GPU 必须非沙箱**。详见 `05_Environment.md` §1。
2. **路径**：VisDrone 路径**唯一来源是 `server.py:98-100`**（恒为 `/root/autodl-tmp/data/visdrone`），与 `data/images|labels` 无关。
3. **代码基线口径**：FedAvgM / 梯度裁剪均 env 门控、**默认关**；`FL_IMG_SIZE` 默认 640。
4. **已完成里程碑**：32% → 35.92%（1024/v8s）→ **45.08%**（1280/v8m）。两次都**非单变量**，若要写进论文需补归因臂。
5. **"换聚合模型"已证伪**（`04_Retracted.md` R1）；**轮数增加已排除**（曲线已收敛）。

## 已知未决 / 风险

- ⚠️ **攻击强度未定（当前最大未决项）**：VisDrone 上"纯数据投毒（×1）打不动、×15 又直接崩"，**尚无"可控但有感"的设定** → 见上方 ⏭ 待办。
- ⚠️ **归因未做**：45.08% 同时改了 模型(s→m)+分辨率(1024→1280)+batch(32→16)，无法单独归因。
- ⚠️ **45.08% 尚无外部参照**（未跑集中式 oracle @1280）。多项内部核验已通过（无泄漏、初始 0%、类别映射一致），详见 `01_Facts.md` §1c。
- ⚠️ `fl_core.py:199-202` 评估权重的类别名**第 5/6 位写反**（truck/tricycle 对调）——**不影响 mAP 数值**，仅影响按类读数，待修。
- 训练循环仍为手写版：缺 **EMA / warmup / 原生增强**（AMP 已有）。

## 铁律

1. **单变量**：一次只改一个变量（`03_History.md` / `04_Retracted.md` 教训）。
2. **不手搓**：优先用原生/现成实现；大改前先调研并出清单。
3. **落库**：实验结论先写 `01_Facts.md` 再推 dev；`logs_3/` 不进 git。
4. 硬约束见 `05_Environment.md` §冻结红线（DPFLA=SVD+KMeans；指标=YOLO 原生 val mAP@0.5；聚合 float32）。
