# 11_LabelFlip_50R_Plan.md — 标签翻转 50 轮达标计划

<!-- 最后更新：2026-11-13 -->
<!-- 级别：执行计划。用户 2026-11-13 明确修改攻击规则；本计划优先于 10_YOLO26s_Plan 的 IPM 分支。 -->
<!-- 前置：YOLO26s 已接入；10×3 LR1e-3 基线 R10=36.02；MinMax+DPFLA 11×3 R11=35.69 已达标。 -->

## §0 用户新规则（2026-11-13 定稿）

> **2026-11-13 20:33 更新（优先）**：停止所有进程，**不再先跑无攻击基线**；直接做标签翻转，**多试几种部分标签反转**，每条上限 50 轮。先用 **5 轮 ×3 epoch** 筛策略，候选再上 **10–11 轮**，必要时才跑 50 轮；**若确认 DPFLA 达不到“mAP 高 ≥1%”目标，立即停止当前策略、换下一种再跑**。不再用 full_cycle 50×24 旧参数。
>
> **2026-11-14 06:25 更新（优先）**：用户要求后续实验改用**小步多轮**配方，便于展示长收敛曲线：**20 轮 × 1 epoch，LR 8e-4**（训练量 ≈ 旧 5×3 的 615 step），目标是 **R20 ≈ 旧 R5 效果（≈30.6）**；判定点从 R5 改为 **R20**，门槛不变（DPFLA ≥30 且 ≥TMean+1）。3、4 类筛查与后续展示矩阵均用此配方；旧 5×3 结果仅作历史对照，checkpoint 加 `_r20x1` 后缀区分。
>
> **2026-11-13 22:42 更新（优先）**：筛选顺序改为 **DPFLA → TMean → MKrum**。**若 DPFLA 对 TMean 不达标（DPFLA < TMean+1 或 DPFLA R5<30），直接跳过 MKrum，换下一个类数**；只有 TMean 过了才跑 MKrum，MKrum 也过才算该 pattern 通过。`FL_FLIP_RATIO` **固定 1.0 不再调**；翻转类数按 **1 → 2 → 3 → 4** 依次试（one_class / two_class / three_class / four_class）。四种都不行后，才考虑替换 TMean（换弱基线）。

1. **先跑无攻击 50 轮完整基线**，看 YOLO26s 最高能到多少（已启动：50 轮 ×24 step，LR 1e-3）。
2. **CIFAR10 和 VisDrone 都必须支持两类攻击，且标签翻转必须包含**。
3. **VisDrone 舍弃 IPM**；两类攻击改为 **标签翻转 + MinMax**。
4. 研究 VisDrone 标签翻转：可尝试**非全循环**翻转、多种翻转方式/翻转个数，目标是：
   - DPFLA **mAP ≥35%**；
   - DPFLA 比 Multi-Krum / Trimmed-Mean **高 ≥1%**。
5. 先跑 **50 轮完整标签翻转 + DPFLA**，看能否直接达标；不行再改翻转方式/强度。
6. 实在达不到时，才考虑“把两个基线中较好的那个换成差一些的”作为最后手段；必须记录原因，优先用合法调参/翻转策略解决。

## §1 当前状态（2026-11-13 20:28）

| 项 | 状态 |
|---|---|
| 无攻击 50 轮基线 | **运行中**：`v26s_50x24_base`，50 轮 ×24 step，LR1e-3，10 客户端，1280/BS16/AMP；日志 `logs_3/visdrone/run_no_attack_baseline_20261113_2028.log` |
| 已有基线 | 10×3 LR1e-3 R10=36.02（107.3 min） |
| 已有 MinMax+DPFLA | 11×3 LR1e-3 R10=35.23 / R11=35.69（已达标） |
| 标签翻转脚本 | 已参数化：`FL_FLIP_PATTERN` / `FL_FLIP_MAPPING` / `FL_FLIP_RATIO` |
| 四方法入口 | DPFLA / FedAvg / **新增 MKrum / TMean**（VisDrone label_flipping） |
| CIFAR | 已有 label_flipping 四方法；MinMax 已有；满足“两类攻击” |

## §2 标签翻转策略库（已实现）

`run-test/visdrone/visdrone_fed_hparams.py::resolve_label_flip_mapping()`：

- `FL_FLIP_PATTERN` 预设：
  - `full_cycle`：i→(i+1)%10（10 类全循环，默认）
  - `full_cycle_reverse`：i→(i-1)%10
  - `pair_swap`：0↔1、2↔3、…、8↔9（5 对，10 类）
  - `single_pair`：0→1（1 类）
  - `two_class`：0↔1（2 类）
  - `three_class`：0→1→2→0（3 类）
  - `person_pair`：0↔1
  - `car_van`：3↔4
  - `vehicle_cycle`：3→4→…→9→3（仅车辆 7 类）
  - `all_to_car`：非 car → car（9 类，强破坏）
  - `offset2` / `offset5`：整体偏移 2/5 类
- `FL_FLIP_MAPPING` 自定义：JSON `{"0":1,"1":2}` 或 `"0:1,1:2"`
- `FL_FLIP_RATIO`：被翻实例比例（1.0=全翻；0.75/0.5 调弱）
- `MALICIOUS_BEHAVIOR_RATE=1.0`：每轮必发动；`FL_GRADIENT_SCALE=1`：不放大更新

## §3 实验阶段与门槛

### Phase A — 无攻击 50 轮基线（进行中）

- 配置：50 轮 ×24 step，LR1e-3，10 客户端 IID，1280/BS16/AMP，30% 恶意占位但不攻击。
- 目的：得到 YOLO26s 50 轮最高基线；作为标签翻转的对照。
- **G-A**：正常完成 50 轮；记录 R50 / 末5 / best；无 NaN/OOM。
- 预计耗时：约 2h。

### Phase B — 50 轮全循环标签翻转 + DPFLA（首选）

- 配置：与 Phase A 完全相同，仅 `attack_type=label_flipping`、`FL_FLIP_PATTERN=full_cycle`、`FL_FLIP_RATIO=1.0`、`rule=DPFLA`。
- 入口：`run-test/visdrone/run_dpfla_label_flipping.py`。
- **G-B1**：DPFLA R50 / 末5 **≥35%**。
- **G-B2**：同口径再跑 MKrum / TMean，DPFLA 比两者都 **≥1%**。
- 若 G-B1/G-B2 通过 → 标签翻转主结果成立，进入 CIFAR 回归 + 归档。
- 若失败 → Phase C。

### Phase C — 翻转方式/强度筛选（短轮次）

- **顺序**：每个 pattern 按 **DPFLA → TMean → MKrum**。
  - 先跑 DPFLA、TMean；
  - 若 **DPFLA R20 < 30.0** 或 **DPFLA R20 < TMean R20 + 1.0** → **跳过 MKrum**，直接换下一个类数；
  - 若 TMean 已过 → 再跑 MKrum；MKrum 也满足 DPFLA ≥ MKrum+1 才算该 pattern 通过。
- **固定**：`FL_FLIP_RATIO=1.0`（不再调）；**2026-11-14 起 `FL_GLOBAL_ROUNDS=20`、`FL_LOCAL_EPOCHS=1`、LR 8e-4**（旧 5×3 / LR1e-3 仅用于已完成的 one_class/two_class 对照）、30% 恶意。
- **翻转类数 1→4**：
  - 1 类：`one_class`（0→1）
  - 2 类：`two_class`（0↔1）
  - 3 类：`three_class`（0→1→2→0）
  - 4 类：`four_class`（0→1→2→3→0）
- **G-C（短轮次门槛，2026-11-14 起判定点改为 R20）**：
  - DPFLA R20 **≥30.0**；
  - DPFLA R20 **≥ TMean R20 + 1.0**（TMean 是强基线；不过就直接换类数）；
  - TMean 过线后再补 **DPFLA ≥ MKrum R20 + 1.0**。
- 通过 → 用 checkpoint 把 DPFLA/TMean/MKrum 续到 **50 轮**（1 epoch/轮），验证绝对 mAP ≥35 与领先 ≥1。
- 不通过 → 立即停，换成下一个类数；**四种都不行后**才考虑替换 TMean。

### Phase D — 50 轮最终矩阵（标签翻转）

- 3–4 条：`{DPFLA, MKrum, TMean}`（必要时加 FedAvg），50 轮 ×24 step，LR1e-3，选定翻转策略。
- **G-D**：DPFLA ≥35%，且 > max(MKrum,TMean)+1%（R50/末5）。
- 若仍失败：
  1. 再增加训练量（50×41 step 或 25×3 epoch）；
  2. 调整翻转比例/类别，优先保持攻击合法；
  3. 最后才考虑用户提到的“换弱基线”，并在报告中明确标注。

### Phase E — VisDrone MinMax 50 轮复核（并行/后续）

- 已有 11×3 MinMax+DPFLA 35.69；为与标签翻转同口径，补 50×24 MinMax × {DPFLA, MKrum, TMean}（约 6h）。
- **G-E**：DPFLA ≥35，领先 ≥1%。

### Phase F — CIFAR10 两类攻击回归

- CIFAR 已有 `label_flipping` 四方法与 MinMax；跑：
  - 标签翻转：DPFLA / MKrum / TMean（15 轮，沿用 11-01 口径）；
  - MinMax：DPFLA / MKrum / TMean（15 轮）。
- **G-F**：DPFLA ≥65%，领先 ≥2%；两类攻击都覆盖。

### Phase G — 归档

- 更新 `01_Facts`、`00`、`03`、`08`、`10`；日志/TB 入库；必要时建 `archive/` 快照。

## §4 运行队列（建议顺序）

```
A. 50×24 无攻击基线（运行中，~2h）
B. 50×24 full_cycle + DPFLA（~2h）
C. 50×24 full_cycle + MKrum（~2h）
D. 50×24 full_cycle + TMean（~2h）
   ↓ 若 DPFLA 未达 35 或未领先
E. 5×3 策略筛选（pair_swap / two_class / three_class / vehicle_cycle / car_van / person_pair；每条 ~55min）
F. 选定策略 50×24 × {DPFLA, MKrum, TMean}（~6h）
G. 50×24 MinMax × {DPFLA, MKrum, TMean}（~6h）
H. CIFAR 两类攻击回归（~2h）
```

**自动执行规则**：上一条自然结束（日志出现 `测试完成` / R50）→ 解析结果 → 若达门槛进入下一条；若失败按 Phase C 策略筛选；每条结束立即回写记忆并提交。

## §5 监控与记录

- 每条运行使用独立 tmux 会话与日志/TB 目录。
- 每轮检查：mAP、loss、恶意端 aux/权重、是否有 NaN/OOM、是否误杀诚实端。
- 结果记录到 `01_Facts.md`，过程记录到 `03_History.md`；代码/日志/TB 提交到 `dev`。
- 关键指标：R50、末5、best、DPFLA−max(MKrum,TMean)。

## §6 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 全循环翻转太强/太弱 | DPFLA 不达标或基线也崩 | Phase C 调 pattern / flip_ratio |
| 标签翻转更新与诚实更新相似 | DPFLA 无法分离 | 试 `all_to_car`/`offset5` 等强异常；检查 aux/权重 |
| DPFLA 误杀诚实端 | mAP 下降 | 监控 aux，必要时重标定 0.60/0.75；记录依据 |
| 50×24 仍欠拟合 | 基线/DPFLA <35 | 提到 50×41 step 或 25×3 epoch |
| MKrum/TMean 太强 | DPFLA 领先 <1 | 先合法调参/换翻转策略；“换弱基线”仅最后手段并标注 |
| 时间超长 | 队列无法一次跑完 | 按优先级 A→B→C→D；失败再进 C |

## §7 当前执行

- 已停止无攻击 50×24 基线与 full_cycle 50×24（用户 2026-11-13 20:33 指令）。
- pair_swap（10 类，5 对）DPFLA R5=30.34、MKrum R5=23.53；TMean 在 R1=13.44 后按新规则停止（10 类不在 1–4 范围，保留作参考）。
- **one_class 已判定不通过（2026-11-14 00:34）**：DPFLA R1–R5 = 14.10 / 21.55 / 25.80 / 28.23 / 30.63；TMean = 13.59 / 21.75 / 25.85 / 28.82 / 30.84 → DPFLA R5 **−0.21**，未达 TMean+1 → **跳过 MKrum**。
- **two_class 已判定不通过（2026-11-14 02:25）**：DPFLA R1–R5 = 14.23 / 22.03 / 25.76 / 28.48 / 30.58；TMean = 13.99 / 21.60 / 25.71 / 28.48 / 30.78 → R5 DPFLA **−0.20**，未达 TMean+1 → **跳过 MKrum**。
- **当前：three_class → four_class 自动筛选队列（20×1 / LR8e-4，R20 判定）**（`scripts/label_flip_screen_supervisor.py`，tmux `lf_screen_3_4`，2026-11-14 06:26 启动）。每个 pattern：DPFLA 20×1 →（R20≥30 才跑）TMean 20×1 →（满足 +1 才跑）MKrum 20×1；日志 `logs_screen/<pattern>_<rule>_*.log`，汇总 `logs_screen/label_flip_summary.{json,md}`；checkpoint 带 `_r20x1` 后缀。
- 旧 5×3 结果（one_class / two_class）保留为历史对照；three_class 旧 5×3 只跑了约 3 分钟即被新配方替换，无有效结果。
- 之后按 `two_class` → `three_class` → `four_class` 依次筛；每种都是 DPFLA → TMean →（条件）MKrum，`FL_FLIP_RATIO=1.0`。

## §8 Checkpoint / 断点续跑（2026-11-13 22:30 已实现）

- `fl_core.run_experiment` 每轮保存轻量 checkpoint 到 `FL_CKPT_PATH`：`state_dict` + 轮次 + 指标历史 + DPFLA streak/cooldown；不再保存巨大的 `global_model/local_models`。
- **所有走 `server.run_exp` 的实验自动开启**：`server.py` 会按 dataset/model/rule/attack/replace_method/pattern/ratio/超参生成配置哈希，写入 `checkpoints/..._{hash}.pt`；`FL_RESUME=1` 从下一轮续跑，`FL_SAVE_CKPT=0` 可关闭。
- 入口脚本也可显式设路径：标签翻转脚本用 `checkpoints/label_flip_{pattern}_{rule}.pt`（DPFLA / mkrum / tmean / fedavg 各自独立）。
- 续跑：同一条命令加 `FL_RESUME=1`，并把 `FL_GLOBAL_ROUNDS` 改成目标轮数（如 11）；从 checkpoint 的下一轮继续，DPFLA 状态也恢复。
- 分布式 `run_distributed_smoke.py` 走独立 server/client 路径，暂未接入 checkpoint；下次分布式实验前补。
- ⚠️ 本次 pair_swap 的 5 轮 DPFLA/MKrum/TMean 在功能加入前已完成，**没有保存 checkpoint，不能回溯续跑**；从下一种 pattern 开始生效。pair_swap 若通过 TMean 对比，需从 0 跑 11 轮（或重跑 5 轮带 checkpoint 再续）。

## §9 FL_FLIP_RATIO 语义（2026-11-13）

- `FL_FLIP_PATTERN` / `FL_FLIP_MAPPING` 决定**翻哪些类、翻到哪个类**（映射覆盖范围）。
- `FL_FLIP_RATIO` 决定**在映射命中的实例中翻多少**：
  - **VisDrone（检测）**：按**边界框实例**计算；每个匹配的 box 以该比例翻转，`1.0`=全翻，`0.5`=约一半 box。种子由样本 index 固定，跨 epoch 可复现。
  - **CIFAR10（分类）**：按**样本**计算；`1.0`=所有命中类样本都翻，`0.5`=约一半样本翻。种子由样本 index 固定，跨 epoch 可复现（2026-11-13 修复，之前分类分支忽略该参数）。
- 攻击强度 ≈ 映射覆盖类数 × `FL_FLIP_RATIO`；两者要分开记录，避免把“翻得少”误判为“翻得弱”。
