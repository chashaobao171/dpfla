# 03_History.md — 时间线（改动与回退）

<!-- 最后更新：2026-10-23 -->
<!-- 追加式：新事件加在最上方。回退/证伪的事件保留，不删除。 -->
<!-- 结论性判断请以 01_Facts.md 为准；本文件只记"做了什么、结果如何"。 -->

## 2026-10-23

- **【17:41 起】VisDrone 换 MinMax（服务端注入）**：
  - 新增 `run-test/visdrone/run_screen_server_attack.py`（FL_SERVER_ATTACK / FL_RULE 控制），口径 v8m@1280 / BS16 / **test_bs=64** / AMP / 10 端 / 30% 恶意；脚本内置 CUDA 重试，避免瞬时不可用回落 CPU。
  - 踩坑：hparams 默认 `TEST_BATCH_SIZE=256` 在 1280 下初始评估 OOM（单进程已占 33GB）；与既有 VisDrone 口径一致改用 64。
  - **DPFLA R1 = 21.17%**（无攻击基线 R1 21.21%，高斯线 R1 21.16%）→ MinMax 被防住；MinMax λ*=2.8869，3 恶意端相对距离 0.00，aux=0.844 → 权重 0.020。
  - 异常：另有 **1 个诚实端 aux=0.908 → 权重 0.020**（VisDrone 首次出现诚实端高 aux；R1 未见性能损失）。
  - **R2 崩溃根因（19:10 定位）**：`YOLOWrapper.load_state_dict()` 先 `pop('fc2.weight'/'fc2.bias')`，**修改了调用方传入的 state_dict**；服务端注入后同步 `local_models` 时传的是 `local_weights` 原字典 → 3 个恶意端 fc2 被删。R1 恰好 0 号客户端是恶意端（w[0] 同样被删）不报错；R2 0 号变诚实端 → `average_weights` 抛 `KeyError: 'fc2.weight'`。**修复：load_state_dict 内先 `dict(state_dict)` 浅拷贝再 pop。**
  - 期间误判为显存问题，另加的"评估前 `model.to('cpu')` + `empty_cache`"保留（1280 下评估子进程确实需要独立显存），但**不是崩溃主因**。
  - **DPFLA 5 轮完成（20:02）**：21.17 / 35.93 / 40.67 / 42.52 / **43.85**（基线 21.21/37.05/41.44/43.82/45.08）→ **≥35 目标达成**，R5 与高斯+DPLFA（43.79）持平。
  - **收尾 deepcopy 异常（R5 后 Last-10 汇总）已修**：`average_weights` 不再 `copy.deepcopy(w[0])`，改 `v.detach().clone()`；5 轮数据不受影响。
  - **TB 整理（20:06）**：VisDrone 只留 5 条 —— 无攻击基线(1421)、高斯 FedAvg(1232)、高斯 DPFLA(1450)、**MinMax DPFLA(1911)**、MinMax TMean(2003，运行中)；失败/中断的 1741/1841/1838/1827/1802 移出到 `/tmp/pruned_visdrone_20261023/`；CIFAR 仍各 9 条（tf-logs 与 tb/ 同步）。
  - **用户口径修正（20:15）**：FedAvg（MinMax 无防御）**保留**；VisDrone 与 CIFAR 对齐 —— **2 攻击（标签反转 + MinMax）× 3 防御（DPFLA/MKrum/TMean）**，FedAvg 为攻击对照臂，**DPFLA mAP 领先两防御 1%** 即达标；**15 轮暂缓，先跑完所有 5 轮**提交老师；研究"能否舍弃高斯、两数据集统一用 MinMax + 标签反转"。
  - **第二攻击改为 SignFlipping（用户 2026-10-23 20:45 决定）**：VisDrone 不用标签反转（太弱，三防御挤一起）；CIFAR 保持"标签反转 + MinMax"不变。
  - **TMean（MinMax）5 轮完成（20:53）**：5.44 / 31.32 / 36.33 / 38.90 / **40.17**；R5 DPFLA（43.85）领先 **+3.68**（≥1 达标）。
  - **MKrum（MinMax）5 轮完成（21:48）**：0.06 / 7.85 / 20.32 / 26.00 / **29.30**；R5 DPFLA 领先 **+14.55**。
  - **FedAvg（MinMax 对照）5 轮完成（22:49）**：3.27 / 31.74 / 36.97 / 39.85 / **41.17**；低于 DPFLA 43.85，但**高于 TMean 40.17 / MKrum 29.30**。
  - **夜间自动队列三次踩坑（已修）**：① `run_in_background` 沙箱无 GPU → 实验必须普通 Shell 启动；② 时间闸门 `2249 >= 330` 误判已过 03:30 → 改为仅在 03:30–05:00 跳过；③ `pgrep -f` 匹配到工具 wrapper 自身、`kill -0 $!` 误判完成 → 改用锚定 `^python` + 日志 `Round 5/5` 完成标记。v2 队列 22:59 起稳定运行。
  - **SignFlipping + DPFLA 5 轮完成（23:49）**：20.26 / 35.25 / 40.68 / 42.78 / **43.37**；supervisor 已自动记录并推送，随后自动启动 TMean（23:49:58）。
  - 末尾第二次 eval（39.30）为 `Last 10 updates results` 诊断项（最近 N 轮权重快照等权平均），**不是 R5 指标**；5 轮时会混入 R1（20.26），必然偏低。
  - 计划：SignFlipping TMean → MKrum → FedAvg（supervisor 自动串行）。
- **【16:40–17:03】换攻击范式：MinMax 服务端注入，P3 CIFAR 达标（+2% 三口径全满足）**：
  - 新增 `federated_learning/attack_alg/server_attacks.py`：`minmax`（FLPoison `min.py` 二分 λ）/ `alie`（norm.ppf 推 z_max）/ `signflipping`；`fl_core.py` 聚合前注入并把被替换的恶意端**同步写回 `local_models`**（否则 DPFLA 读到未攻击更新，实测掉到 48.15）。
  - `DPFLA.py` 新增**协同重复检测**：到最近其他端相对距离 <0.01 判重复 → risk=0.98、权重 0.020（MinMax 三个恶意端向量完全相同，相对距离 0.00）。
  - **5 轮筛选**：MinMax DPFLA 55.29 / TMean 46.22 / MKrum 29.07；ALIE 54.62 / 48.22；SignFlipping 54.55 / 48.48 → 选定 MinMax（15 轮余量最大）。
  - **15 轮正式对比**：DPFLA **69.25**（末5 67.75）/ MKrum **29.19**（29.86）/ TMean **57.01**（54.32）/ FedAvg 60.69（58.78）→ **末5 领先 +37.90 / +13.43**，R15 +40.06 / +12.24，三口径均 ≥2%；DPFLA ≥65。
  - 机理：恶意端 aux=0.835 ≥0.75 → 权重 0.020（14/14 轮），诚实端 1.000；协同检测同向命中，属额外证据链。
  - **归档裁剪（用户要求）**：`log/cifar10` 与 `tb/` 只留 9 条（基线 / 标签翻转四方法 / MinMax 四方法）；`logs_3/` 不动；移出的旧产物暂存 `/tmp/pruned_cifar_20261023/`。
  - 试跑插曲：6 条实验同时启动触发 CUDA Error 304 → 回落到 CPU 空转，已全部停止；GPU 实验并发维持 4 条以内。
- **【有卡窗口 12:55–15:30】CIFAR 全套实验完成，P1 达标（≥65）** —— 完整数据 `01_Facts.md` §5c：
  - 环境：RTX 4090 48G / 208 核 / 1TB 内存，容器内存 90G。单条 15 轮 ≈ 3.5–4 分钟。
  - **A1 基线**（ResNet18CIFAR 10 端 15 轮）：R15 **71.47%**（R5 56.11 → R10 65.23）。
  - **7 端对照**：**74.25%**（高于 10 端）→ DPFLA 完美剔除后的理论上限。
  - **标签线**：FedAvg 65.68 / **DPFLA 70.00（v5）** / MKrum 67.00 / TMean 66.83 → DPFLA 领先 **+3.00**（P3 标签线达标）。
  - **高斯线四档（v5 最终版）**：σ=0.03 → **69.50**；**σ=0.04 → 69.26**；σ=0.05 → 68.78；σ=0.06 → 67.99。MKrum 64.70/65.70/64.91/67.02；TMean 68.22/**66.98·67.02**/68.34/67.86。
    - ~~**σ=0.04 达标 P3**：DPFLA 69.26 vs TMean 66.98/67.02 → **+2.24（≥2 ✓）**~~ → ⚠️ **该结论已被证伪**（用户 2026-10-23 质疑）：+2.24 仅是 R15 单点，末5 / 最佳轮仅 +0.5~0.9。详见 `04_Retracted.md` R13；P3 由后续 MinMax 线达成。
  - **DPFLA 算法改进（四处，全部由实测分布标定）**：
    1. 辅助强否决（aux≥0.60 视作强异常）—— 修 KMeans 漏杀（高斯线曾 15/45 漏杀）；
    2. aux≥0.75 → risk=0.98 → 权重首轮即 0.02 —— 修"权重上限锁死 0.086"（σ=0.03: **59.90 → 68.72**）；
    3. KMeans 少数簇但 aux<0.48 → 权重下限 0.50 —— 修误杀（VisDrone 曾把诚实端压到 0.137）；
    4. **强证据时诚实端满权重 1.0**（优先于灰区降权）—— 修"归一化后恶意端相对权重被抬高 ~12%"（σ=0.04：69.04 → **69.26**，诚实端实测 1.000）。
    - 标签线回归 69.69 → 70.00（未退化）；改进后恶意端权重全程 0.020。
  - 高斯 σ 阶梯（5 轮筛）：0.002/0.005/0.01 无效、0.02 弱、**0.03 有感**、0.04 强、0.06 极强未崩。CIFAR 的 σ 阈值比 VisDrone 高一个量级。
- **VisDrone ③ 探针（正确口径 v8m@1280/BS16/AMP，5 轮）**：
  - 首次误用 640/v8s 口径跑（已停，作废）；改正后：
  - **修复前 R1 = 3.72%**：3 个恶意端 aux=0.707 未触发 0.75 → KMeans 判多数簇拿 0.797；2 个诚实端被误杀到 0.137。
  - **修复后 R1 = 21.16%、R2 = 35.97%**（基线 21.21 / 37.05；② 无防御 16.25 / 24.59）→ **基本追平无攻击基线**；恶意端 aux=0.908 → 权重 **0.020**，诚实端 0.79–0.84。
  - 结论：**DPFLA 在 VisDrone 高斯线有效**，待 15 轮完整对比。
- **踩坑记录**：
  - 4 条 CIFAR 实验 + VisDrone（39GB）同跑 → CUDA OOM，挂掉 2 条（`g04_run_no_defense/tmean` 已补跑）→ **VisDrone 与 CIFAR 实验不并行**，CIFAR 最多 4 条同跑（各 ~5GB）。
  - 同分钟启动的多条实验会写**同名日志**（时间戳到分钟）→ 已知隐患，引用时以记忆表数值为准。
  - **VisDrone 必须显式带 `YOLO_MODEL_SIZE=m FL_IMG_SIZE=1280 FL_TRAIN_BATCH_SIZE=16 FL_AMP=1`**，否则默认回落 v8s@640（R1≈3.3%，与 1280 基线不可比）。
  - **等待循环 `while pgrep -f "<脚本名>"` 会匹配自身命令行 → 永久空转**（2026-10-23 残留 2 个，16:07 已清理；仅 sleep 不占 GPU）。今后改用日志完成标记 + 有限次循环。
- **可运行性核验 + 补缺口**（用户要求"确定全都可用"）：
  - 新增 `run-test/visdrone/run_dpfla_gaussian.py`，补齐 VisDrone 高斯线 {FedAvg, DPFLA, Multi-Krum, Trimmed-Mean} 四方法对比。
  - `visdrone_fed_hparams.MALICIOUS_RATE` 默认 0.1 → **0.3**（落实固定 30%，避免漏设 env 时只有 1 个恶意端）。
  - `scripts/run_fl_tmux.sh` 支持 `FL_DATASET`（默认 visdrone）→ CIFAR 可后台跑：`FL_DATASET=cifar10 ./scripts/run_fl_tmux.sh <实验名>`。
  - 静态核验通过：`compileall` 全绿；`fl_algorithm` 导出与 `fl_core` 引用名全部可解析；无 `Krum/m_krum/t_mean` 残留；CIFAR 9 个 / VisDrone 7 个入口脚本齐全，rule 名称与 `fl_core` 分支一一对应；`bash -n` 通过。
  - 环境限制：本机无卡且容器内存 2GB（当前已用 ~1.7GB），`import fl_core`（含 ultralytics）会被 OOM 杀 → **所有训练/完整导入只能在有卡主机验证**。
- **接入 Multi-Krum / Trimmed-Mean 两个防御（P3 准备）**：
  - 旧 `m_krum.py` / `t_mean.py` 删除，新增 `fl_algorithm/multi_krum.py`（state_dict 版，score=最近 n-f-2 距离之和，m=n-f-2，`n<2f+3` 回退 FedAvg）与 `trimmed_mean.py`（逐坐标裁剪，默认 trim_ratio=恶意率 0.3，非浮点 buffer 沿用 FedAvg 语义）。
  - `fl_algorithm/__init__.py` 统一导出；`fl_core.py` 的 `mkrum` / `tmean` 分支改为调用新接口，支持 `FL_MKRUM_NEIGHBORS` / `FL_MKRUM_M` / `FL_TRIM_RATIO`。
  - 新增入口：CIFAR10 `run_{mkrum,tmean}_{label_flipping,gaussian}.py`（4 个）、VisDrone `run_{mkrum,tmean}_gaussian.py`（2 个）。
  - VisDrone 共享超参 `GLOBAL_ROUNDS` 默认 50 → **15**（落实全局上限）。
  - CPU 合成数据验证：3 恶意端（+100 偏移）下两方法均正确剔除；回退与 trim_ratio=0 分支正常。
- **CIFAR 代码改造完成（T1–T4）**，推 dev：
  - 新增 `federated_learning/models/cifar_resnet.py`：CIFAR 专用 ResNet（3×3/s1 stem、无 maxpool、`linear` 头、11.17M），模型名 `ResNet18CIFAR`，支持 `return_features`。
  - `models/model.py` 注册；`fl_core.py` DPFLA 白名单加 `ResNet18CIFAR / ResNet18`；`DPFLA.py` 新增 `_detect_classifier_weight_key()`（`fc2 → fc → linear`）并清除三处硬编码。
  - 新增 `run-test/cifar10/cifar_fed_hparams.py` + 5 个实验脚本（基线 / 标签+FedAvg / 标签+DPFLA / 高斯+FedAvg / 高斯+DPFLA），统一 10 客户端 / 15 轮 / 1 epoch / BS64 / lr0.01 / IID / 固定映射 `{5→3}` / `FL_GRADIENT_SCALE=1`。
  - `sampling.py` 修正 CIFAR 候选路径首项被跳过的问题 → 数据根解析为 `/root/autodl-tmp/data`。
  - CPU 验证：模型前向、DPFLA 两条键路径、5 脚本参数块均通过；**未跑训练**（无卡 + 2GB 容器上限）。
- **用户定稿高层路线**（新增 `08_Roadmap.md`，高级别）：CIFAR10 有攻防 ≥65%（ResNet18 / 高斯+标签 / 不放大）、轮数上限改 15、DPFLA vs Multi-Krum / Trimmed-Mean（+2% / +1%）、后续分布式。
- **删除 CIFAR10 旧攻击日志与 TensorBoard**（用户要求："log 里 cifar 除基线外都删、tsboard 也删"）：
  - `log/cifar10/02_attack_nodefense__..._1214.log`、`03_attack_dpfla__..._1209.log` → 删除；**只留 `01_baseline__..._1128.log`**（60.74%）。
  - TB：`runs/CIFAR10_CNNCifar10_{DPFLA,fedavg}_attack-label_flipping_mr-0.3`、`/root/tf-logs/` 同名两目录 → 删除；**保留** `CIFAR10_CNNCifar10_fedavg_attack-no_attack_mr-0`（两处）。
  - 理由：②③ 均为 **×15 梯度放大**口径，与本轮"不放大"新要求不可比 → 作废重跑。数值保留在 `01_Facts.md` §5b 与 `log/README.md` 索引（已标注"日志已删"）。
- **CIFAR 计划落库**（`08_Roadmap.md` §2，T1–T9）：确认三条关键事实 —— ① torchvision `resnet18` 是 ImageNet stem，**不适合 32×32**；② FLPoison 分支自带 **CIFAR 专用 ResNet**（3×3/s1 stem、无 maxpool、末层 `linear`），可借用；③ CIFAR 当前用 FedProx（`mu=0.001`）而非 SGD，且 `Cifar10CNN` 有双 softmax。

## 2026-10-21

- **25 轮基线跑满**（用户要求：原 5 轮太少，配置与之前一致）：R1–R5 **逐位复现**旧日志（21.21/37.05/41.44/43.82/45.08），继续爬升到 **R13 峰值 47.64%**，R14–R22 平台 ≈47.1，R25 45.73%。
  - 结论：**45.08% 不是收敛值，真实平台 ≈47%**；② 的对比基准应改用同期 R10=47.22%。
  - 出处：`logs_3/visdrone/run_no_attack_baseline_20261021_1421.log`，已归档 `log/visdrone/01_baseline__...`（替换旧 5 轮归档）。

- **高斯 σ=0.02 跑满 10 轮（14:06 完成）**：R1 16.25 → R2 24.59 → R3 27.69 → R4 30.04 → R5 31.24 → R6 30.19 → R7 27.43 → R8 31.14 → R9 31.51 → **R10 32.37%**（基线 45.08%）。**平台 ≈27–32%、稳定不回升** → 定为 ② 的可用攻击档；日志已归档 `log/visdrone/02_attack_nodefense__run_no_defense_gaussian_20261021_1231.log`。
- **用户确认硬约束**：恶意客户端比例**固定 30%**（项目任务要求），不再作为可调杠杆（2026-10-21）。
- **TensorBoard 真正 logdir 排查**（用户仍能看到未清理项）：确认 AutoPanel 的 TB 进程为 `--logdir /root/tf-logs`，而此前只清了仓库内 `tb/`。已清理 `/root/tf-logs`（29 → **6**，只留刚跑的 20261021_1140 + `log/` 对应的 5 个）、删除 `/root/tf-logs-archive`；全盘 tfevents 现为 6(tf-logs) + 7(runs) + 1(tb 镜像) = 14。
- **高斯 σ=0.05 复验**（用户要求）：R1 0.00%（损失 408.8596，与 10-20 完全一致 → 可复现），R2 0.00%（损失 366.6）→ 判定为"权重破坏"型崩档，按判据 R2 后停止；日志留在 `logs_3/visdrone/run_no_defense_gaussian_20261021_1140.log`，tb 按新策略删除。

- **日志/张量板清理**（用户要求）：`tb/` 11 个目录**全部删除**——按规则"只保留与 `log/` 归档实验对应的 tensorboard"，而现存 11 个 tb 无一条对应 `log/`（VisDrone 基线早于 `tb/` 启用）。
- `logs_3/` 从 42 个文件删到 **13 个**（9.7M → 3.2M）：删除旧基线（20260608~20261010 共 16 条，含量为 0 或已被 45.08% 取代）、5 条 MNIST 重复运行、`demo_out.log`、`visdrone_all.log`、`oracle_1ep`、`proof_dynamic`、以及崩/废弃高斯 4 条（σ=0.5 / 0.05 / 0.05修复后 / 0.01 无完整轮次）。
- **保留**：`log/` 对应三数据集的 7 份、VisDrone 35.92% 里程碑（`..._20261010_1813.log`）、标签翻转 4 条（1949/2022/2057/0959）、高斯 σ=0.005（2021）。
- 被删的高斯崩溃日志**数值已固化在 `01_Facts.md` §1d(d)**，不再依赖原始文件。
- **`runs/` 旧 TensorBoard 也清理**（用户指出"tb 没清干净"）：`runs/` 37 个 tfevents 删掉 30 个（含 2026-06-10 ~ 2026-10-10 的 16 次旧 VisDrone 基线、MNIST/CIFAR 的重复运行），**仅保留 7 个**——每个恰好对应 `log/` 中一份最终归档（VisDrone 20261013、MNIST 05-15 两份 + 09-02 基线、CIFAR10 09-02 三份）。

## 2026-10-20（后续·高斯攻击线）

- 本机有卡（RTX 4090 48G / 208 核 / 1007G）；跑 `run_no_defense_gaussian`（新增脚本）共 5 次，口径 = 30% 恶意 / 行为率 100% / v8m@1280 / 不放大。
- **诊断并修复真实 bug**：`gaussian_attack` 给 BN `running_var` 加噪 → 变负 → eval 模式 sqrt(负数) → NaN → mAP 0.00%（测试损失却正常，因它在 `train()` 后用 batch 统计量算）。修复 = 跳过 BN 统计量。详见 `04_Retracted.md` R12。
- **结果阶梯**（详见 `01_Facts.md` §1d(d)）：σ=0.5 崩、σ=0.05 崩（修复后仍崩，真实权重破坏）、σ=0.005 **无效**（≈基线 −0.45）。
- 用户判定：**σ=0.005 的 R2=36.61% 太高 → 打不动，停止高斯线**。
- **新增长期规则**（用户 2026-10-20）：做防御测试时若同时开了 **DP 与高斯攻击**开关，必须先核实两者**不是同一段代码路径**。

## 2026-10-20（本会话）

- 从 GitHub `dev` 拉取（快进 3 个提交 → `d7d6658`）。**当前主机为无卡模式**（无 `/dev/nvidia*`、`cuda.is_available()=False`、1 CPU 核）→ 未跑训练。
- **删除两条崩溃日志**（用户要求）：`run_no_defense_label_flipping_20261014_1908.log`、`..._1929.log` —— 均为 `FL_GRADIENT_SCALE=15`、R1=0.00% 的退化运行。**数值保留在 `01_Facts.md` §1d(b)**，原始日志不再保留。
- **记忆整理（全量校验 80 条"出处"引用）**：
  - 修正错文件名：`02_Methods.md` 的 `fl_avg.py` → **`fl_algorithm/fed_avg.py`**。
  - `04_Retracted.md` 顶部补记**"出处文件可核验性"**说明：旧记忆文件（`00_QuickStart.md` 等）与本次删除的崩溃日志均已不存在，引用仅用于指认来历。
  - 其余引用（`logs_3/`、`log/`、代码文件）全部可解析；`visdrone_temp.yaml` 为运行时生成且被 gitignore，非缺失。

## 2026-10-16 ~ 10-17（上一会话）

- 从 GitHub **`dev` 分支**拉取覆盖本地工作区；本机 **4090 48G / torch 2.5.1+cu124**。
- 代码改动（②③ 脚本同步，保持口径一致）：
  - `run_no_defense_label_flipping.py` / `run_dpfla_label_flipping.py`：把单一 `source_class/target_class` 改为 **`FLIP_MAPPING` 多标签映射**；默认 `FL_FLIP_RATIO=1.0`。
  - 映射三次递进：`{1:9}` → `{0,1,2}→9` → `{0,1,2}→9,{3,4}→8` → **`{i:(i+1)%10}` 全类循环**。
- **实验②（攻击+无防御）追加 3 次运行**（均 `FL_GRADIENT_SCALE=1` 不放大 / 恶意率 30% / 行为率 100%）：

| 映射 | 覆盖 | R1 | R2 | R3 | 出处 |
|---|---|---|---|---|---|
| `{0,1,2}→9` | 34.3% | 17.53% | 34.82% | 39.08% | `..._20261016_2022.log` |
| `{0,1,2}→9,{3,4}→8` | 83.5% | 16.47% | 31.75% | — | `..._20261016_2057.log` |
| `{i:(i+1)%10}` | **100%** | **15.09%** | **27.23%** | 35.45%（R4 37.51 / R5 39.11）| `..._20261017_0959.log` |

- **结论**：纯数据投毒**仅靠加强标签映射已饱和**（覆盖 ×2.9，R1 仅 −2.44 点；全类循环 R2 单轮 <30% 后 R3 回弹）→ 用户据此**停止映射实验**，转"方案清单"评估。数据见 `01_Facts.md` §1d(c)，清单见 `chatlog/2026-10-16_多标签翻转攻击强度迭代.md`。
- **Git 规则**：`.gitignore` 放开 `logs_3/` 及 `*.log`（用户要求**原始日志一并推送**），`06_Git_Workflow.md` 登记该例外。

## 2026-10-14（上一会话）

- 从 GitHub **`dev` 分支**拉取覆盖本地工作区（切到分支 `dev`；另留安全分支 `backup-before-dev` 指向原 `9c5e458`）。本机 **4090 48G / 20 vCPU / 96G**。
- 代码改动：
  - 新增 **`FL_FLIP_RATIO`**：标签翻转的**实例比例**（改动 `poisoned_dataset.py`、`label_flip_attack.py`、`client.py` 与两个 VisDrone 实验脚本）
  - TensorBoard 目录由 `runs/` 改为 **`tb/`**（**纳入 git → 跨主机留存**），目录名加入 **模型大小 + 分辨率 + 运行时间戳**
- 启动 **实验②（攻击 + 无防御）**，三次尝试：

| # | 配置 | R1 | 结论 |
|---|---|---|---|
| 1 | ×15 + flip_ratio=1.0 | 0.00%（损失 9412.6）| 崩 |
| 2 | ×15 + flip_ratio=0.1 | 0.00%（损失 10866.9）| 崩；**投毒量降 90% 完全无效** |
| 3 | **×1 + flip_ratio=1.0** | **19.88%**（损失 297.8）| **攻击几乎无效**（R4 43.24% vs 基线 43.82%）|

- **根因确认**：`client.py:522-543` 的内联**更新外推**（`FL_GRADIENT_SCALE`，**默认 15**）才是崩溃来源。agent 曾误判"本项目没有梯度放大"（只查了未被调用的 `EnhancedLabelFlipAttack`）→ 见 `04_Retracted.md` **R10 / R11**。
- 结论：**纯数据投毒（×1）在此数据/模型上打不动模型**。"可控但有感"的攻击需另找杠杆（源类 / 翻转类别数 / 放大倍数取中间值）—— **待办**。

## 2026-10-10（上一会话）

- 盘点项目文件结构；**保守清理**：删除全部 `__pycache__`、`.ipynb_checkpoints`×3、`.idea/`、`.vscode/`、`cache/`(32M)、`visdrone_temp*.yaml`、空 `data/`。
- **重构 MEMORY** 为分层事实记忆包（本文件即新版之一）；修正 FedAIoT / 子进程 val 的错误结论（见 `04_Retracted.md`）。
- 环境：autodl 无卡模式（无 GPU、1 CPU 核）→ 未跑训练。

## 2026-09 系列

| 日期 | 事件 | 结果 | 证据 |
|------|------|------|------|
| 2026-09-16 | 50 轮基线运行 | R1=3.35%，日志在 R2 中断 | `logs_3/visdrone/run_no_attack_baseline_20260916_1437.log` |
| 2026-09-15 | 基线运行 | ❌ val 子进程被 OOM 杀（`code=-9`）| `..._20260915_2144.log` |
| 2026-09-07 | proof_dynamic 单轮验证 | 训练前冷启动 mAP=0%；R1=3.24% | `logs_3/visdrone/proof_dynamic_20260907_1520.log` |
| 2026-09-04 | **回退训练管线到纯 FedAvg**（9b3169d 口径）| FedAvgM/梯度裁剪改环境变量门控、默认关 | commit 见下 |
| 2026-09-04 | 全入口统一默认 YOLOv8s | `model.py` / `yolo_wrapper.py` 默认 `n`→`s` | ChangeLog |
| 2026-09-02 | 9/2 demo（FedAvgM 0.9 恒开 + clip 10.0）| 10 轮仅 2.55%，loss 705→1010 | `logs_3/demo_out.log` |

09-04 回退内容（`fl_core.py` / `client.py`）：
- FedAvgM 门控：`FL_SERVER_MOMENTUM` 默认 `0`（纯 FedAvg）
- 梯度裁剪门控：`FL_GRAD_CLIP` 默认 `0`（关闭）
- `drop_last` True→False（对齐 9b3169d）

## 2026-06-10（FedAvgM 引入 → 崩塌 → 修复 → 回退）

| 时间 | 事件 | 结果 |
|------|------|------|
| 上午 | YOLOv8s→l + conf=0.001 尝试 | 训练前冷启动 mAP=0%，被**误判**为"val bug"（见 04_Retracted） |
| 下午 | **Bug1+2+3 修复**（commit 9b3169d） | val 子进程检测头 / 梯度平均 / BN `num_batches_tracked` → 100 轮基线 R26-33≈23.7% ✅ |
| 傍晚 | 引入 **FedAvgM（服务端动量 0.9）+ 梯度裁剪**（commit 87e3f7a，同一提交还改了 LR 2e-4→0.01、EPOCH 10→1）| Round 2-3 mAP 崩塌 |
| 晚 | 修 FedAvgM：只对可学习参数做动量（8b196f0）、设备统一（d93c551）| mAP 仍 0（LR=0.01 未回退）|
| 深夜 | 追加两份分析（`round3_collapse_*.md`，观点互相矛盾）| — |
| 深夜 | f2741ff：BN 统计量重置 + 超参回退（LR=2e-4, EPOCH=3）| — |
| 深夜 | 8554d4d：**撤销 BN reset（实测有害）** + 修 `drop_last=False` | — |

> ⚠️ 87e3f7a 一次提交改了 4 个变量（聚合 + 裁剪 + LR + EPOCH），**导致无法归因**——这是本项目最重要的流程教训。

## 2026-06-09 及之前

| 日期 | 改动 | 文件 | 结果 |
|------|------|------|------|
| 2026-06-09 | Mosaic + MixUp 实现（50%/10%）| `visdrone_dataset.py` | ✅ 合入，效果未单变量验证 |
| 2026-06-09 | 分层 LR backbone×0.5 / head×2.0 | `client.py` | ✅ 合入 |
| 2026-06-09 | AMP 混合精度（旧记"已完成"）| `client.py` | ⚠️ **与实现不符**：2026-10-10 全仓核查无 `autocast`/`GradScaler`（见 04_Retracted R9）|
| 2026-06-09 | val conf 0.001 → 0.25（子进程 eval 仍用 0.001）| `fl_core.py` | ✅ 合入 |
| 2026-06-09 | 差配置诊断：BS16+EPOCH3+cosine → 14.68% | `visdrone_fed_hparams.py` | 已回退 |
| 2026-06-08 | FedLA 实现完成（`average_weights_fedla()`）| `fl_core.py` | R8=22.49% |
| 2026-06-07 | FedBN 实现完成 | `fl_core.py` | R7 峰值 21.55%，全程 ≤ FedAvg |
| 2026-06-06 | local epochs = 10 | hparams | ✅ |
| 2026-05-15 | MNIST：DPFLA 95% vs FedAvg 4.57% | — | ✅ 冻结 |
| 2026-05-14 | 环境迁移到 autodl，路径统一 | 多处 | ✅ |

## 关键 Git 提交

```
b9159b1 feat: 新增一键备份脚本 backup.sh
98c8e72 merge: 合并本地同步分支 fc7b29c
45bd3c9 sync: 云平台工作区同步
fc7b29c feat: 本地工作区同步 - run_demo + fl_core/client/model 修复 + MEMORY 整理
8554d4d revert: 撤销 BN reset（有害）+ 修复 drop_last=False
f2741ff fix: BN 统计量重置 + 超参回退（LR=2e-4, EPOCHS=3）
51bd31b docs: 追加 R3 mAP 仍为 0 的深度分析报告
d93c551 fix: FedAvgM 设备统一
8b196f0 fix: FedAvgM 只对可学习参数做动量平滑
87e3f7a feat: FedAvgM 服务端动量平滑 + 梯度裁剪
255e4c2 docs: 新增 FedAIoT 参考项目 + 深度借鉴报告
9b3169d fix: val 子进程检测头 + 梯度平均 + BN 聚合 (Bug 1+2+3)
8ef2a47 Initial commit: DPFLA federated learning project
```
