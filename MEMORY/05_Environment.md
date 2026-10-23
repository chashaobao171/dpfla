# 05_Environment.md — 环境 / 路径 / 数据 / 冻结红线

<!-- 最后更新：2026-10-14 -->
<!-- 本文件是路径、数据、命令的唯一事实源；其它记忆文件只引用，不重复定义。 -->

## §1 两台主机（以 autodl 控制台为准，2026-10-13 核实）

| 项 | 老主机 | **新主机（当前）** |
|----|--------|------------------|
| GPU | vGPU **32GB**（实测 RTX 4080 SUPER，driver 595.71.05，cap 8.9）| vGPU **48GB**（实测 RTX 4090，47.4GB 可用）|
| CPU | 12 vCPU（Xeon 8375C @2.9G）| **20 vCPU**（Xeon 8470Q）|
| **容器内存** | **62 GB** | **90 GB** |
| 磁盘 | `/root/autodl-tmp` 50G | `/root/autodl-tmp` 50G |
| torch / CUDA | 2.5.1+cu124 / CUDA 12.4 | 同 |

> ⚠️ **易错点（曾记错）**：`free -g` 显示的是**物理宿主机**内存（503GB / 1007GB），**不是容器配额**。
> 容器真实上限看控制台：**老机 62GB / 新机 90GB**。老机那次 `SIGKILL(137)` 正是撞到 **62GB** 容器上限。
>
> ⚠️ **无卡 vs 有卡差异**：无卡模式为 无 GPU / 1 核；切换后需重新确认。
>
> ⚠️ **沙箱拦截**：默认沙箱会拦截 `/dev/nvidia*`，导致 `nvidia-smi` 误报 "No devices were found"、`cuda.is_available()=False`。
> **跑训练/GPU 命令必须用非沙箱模式**，否则 GPU 不可见。

**环境验证命令**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望：2.5.1+cu124  True  NVIDIA GeForce RTX 4080 SUPER
python -c "from ultralytics import YOLO; print('YOLO OK')"
nvidia-smi
```

## §2 路径映射（实测）

| 资源 | 路径 |
|------|------|
| 项目根 | `/root/chashaobao/DPFLA-master` |
| VisDrone 原始数据根 | `/root/autodl-tmp/data/visdrone` |
| 训练图 / 标签 | `.../visdrone/VisDrone2019-DET-train/VisDrone2019-DET-train/{images,labels,annotations}` |
| 验证图 / 标签 | `.../visdrone/VisDrone2019-DET-val/VisDrone2019-DET-val/{images,labels,annotations}` |
| YOLO 模型权重 | 项目根 `yolov8{s,n,m,l}.pt` |
| MNIST | `/root/autodl-tmp/data/MNIST` |
| CIFAR10 | `/root/autodl-tmp/data/cifar-10-batches-py` |
| 文档 / 对比图 | 项目根 `docs/`（`visdrone_fl_research_report.md`、`cifar_compare.png`、`mnist_compare.png`） |
| 历史日志 | 项目根 `logs_3/`（含 `demo_out.log`、`visdrone_all.log`、`oracle_1ep_*.log`） |
| **TensorBoard 曲线** | 项目根 **`tb/`**（**受 git 跟踪 → 跨主机留存**）；本机镜像 `/root/tf-logs/`（仓库外，不跨机）|
| **TensorBoard 服务实际 logdir** | **`/root/tf-logs`**（AutoPanel 后台进程 PID 907，端口 6007，`--reload=5`）——**界面上能看到的是这个目录，不是仓库 `tb/`**；清理时必须两边都删 |
| **当前保留（2026-10-23，用户确认后清理）** | **`/root/tf-logs` 14 个 / `tb/` 13 个 / `runs/` 4 个** —— 只留"最终结论"实验：CIFAR 10（基线·标签四方法·**高斯仅 σ=0.04** 四方法+复现）+ VisDrone 3（基线·②·③探针）+ 旧归档 1（MNIST 基线）；`runs/` 只留与 `log/` 对应 4 个 |
| 已删的诊断曲线 | CIFAR **7 端无攻击对照（74.25%）TB 曲线已删**；数值保留在 `01_Facts.md` §5c(a)，原始日志 `logs_3/cifar10/run_no_attack_baseline_20261023_1417.log` 保留 |
| 保留原则 | ① 最终结论实验（进 `01_Facts.md` 的）；② 与 `log/` 归档一一对应；③ **探索性数据（σ 阶梯筛选、算法中间版本、冒烟）一律不留**（数值在记忆里，`logs_3/` 原始日志不动） |

**实测数量**：train 图 6471 张 / val 图 548 张（均含 10 类原始标注）。

**代码中的路径优先级**（`visdrone_dataset.py` / `generate_visdrone_yaml.py`）：

```
VisDrone:  /root/autodl-tmp/data/visdrone  >  /home/featurize/data/visdrone  >  D:/Pycharmworkplace/visdrone
```

**评估用 yaml**：运行时由 `generate_visdrone_yaml.py` 生成 `visdrone_temp.yaml`：

```yaml
path: /root/autodl-tmp/data/visdrone
val:  VisDrone2019-DET-val/VisDrone2019-DET-val/images
nc: 10
```

**注意区分**：`/root/autodl-tmp/data/{images,labels}`（含 trainA~D / validation / test，train 1763 张）是**第三方工程的数据结构**，主项目**不使用**。

## §3 依赖安装（仅首次或依赖异常）

```bash
pip install -r requirements.txt
pip uninstall opencv-python -y
pip install --force-reinstall --no-deps numpy==1.26.4 opencv-python-headless==4.8.1.78
# ultralytics 会自动升级 numpy，需手动回退
```

## §4 运行命令

```bash
cd /root/chashaobao/DPFLA-master

# 无攻击基线（纯 FedAvg）
FL_ATTACHED=1 python run-test/visdrone/run_no_attack_baseline.py

# 无防御对照（FedAvg + 标签翻转）
python run-test/visdrone/run_no_defense_label_flipping.py

# DPFLA 防御（DPFLA + 标签翻转）—— 核心实验
python run-test/visdrone/run_dpfla_label_flipping.py

# 防御对比（高斯攻击线，2026-10-23 新增）
python run-test/visdrone/run_dpfla_gaussian.py
python run-test/visdrone/run_mkrum_gaussian.py
python run-test/visdrone/run_tmean_gaussian.py

# ---- CIFAR10（ResNet18CIFAR，2026-10-23 新增；15 轮 / 不放大梯度）----
python run-test/cifar10/run_no_attack_baseline.py          # A1 基线
python run-test/cifar10/run_no_defense_label_flipping.py   # 标签 + FedAvg
python run-test/cifar10/run_dpfla_label_flipping.py        # 标签 + DPFLA
python run-test/cifar10/run_mkrum_label_flipping.py        # 标签 + Multi-Krum
python run-test/cifar10/run_tmean_label_flipping.py        # 标签 + Trimmed-Mean
python run-test/cifar10/run_no_defense_gaussian.py         # 高斯 + FedAvg
python run-test/cifar10/run_dpfla_gaussian.py              # 高斯 + DPFLA
python run-test/cifar10/run_mkrum_gaussian.py              # 高斯 + Multi-Krum
python run-test/cifar10/run_tmean_gaussian.py              # 高斯 + Trimmed-Mean

# TensorBoard（曲线目录 tb/，受 git 跟踪，可跨主机）
tensorboard --logdir tb/ --port 6006
```

- 默认 `nohup` 脱壳（关终端仍跑）；前台实时加 `FL_ATTACHED=1`；**tmux 启动脚本**：`./scripts/run_fl_tmux.sh <实验名> [ENV=VAL...]`（实时日志 + 断网不影响）。
  - CIFAR：`FL_DATASET=cifar10 ./scripts/run_fl_tmux.sh run_no_attack_baseline`（2026-10-23 起支持，默认仍是 visdrone）
  - VisDrone 超参默认值 2026-10-23 调整：`MALICIOUS_RATE=0.3`（固定 30%）、`GLOBAL_ROUNDS=15`（上限）
- 日志：`logs_3/visdrone/<tag>_<北京时间>.log`；**曲线：`tb/<dataset>_<model>_<rule>_attack-<attack>_mr-<rate>_yolov8<size>_img<res>_<运行时间戳>/`**
  - 命名规则（`fl_core.py:636-652`，2026-10-14 起生效）：含 **模型大小 + 分辨率 + 时间戳** → **每次运行独立目录**，TensorBoard 中互不混淆；目录写在 **`tb/`**（纳 git）与 `/root/tf-logs/`（本机镜像）
  - ⚠️ 旧命名（`runs/VisDrone_YOLO_<rule>_attack-<attack>_mr-<rate>/`）**不含分辨率/模型/时间戳**，故 10-10 与 10-13 两次运行**同名混在一起**；旧 `runs/` 仍被 gitignore，仅本机可见

**常用环境变量覆盖**：`YOLO_MODEL_SIZE`、`FL_IMG_SIZE`(默认640，训练+评估口径)、`FL_GLOBAL_ROUNDS`、`FL_LOCAL_EPOCHS`、`FL_TRAIN_BATCH_SIZE`、`FL_NUM_WORKERS`、`FL_LOCAL_LR`、`FL_SERVER_MOMENTUM`(默认0)、`FL_GRAD_CLIP`(默认0)、`FL_FLIP_RATIO`(默认0.3，标签翻转的实例比例；1.0=全翻)。
- 防御相关（2026-10-23 新增）：`FL_MKRUM_NEIGHBORS`（Multi-Krum 邻居数，默认 n-f-2）、`FL_MKRUM_M`（选中客户端数，默认 n-f-2）、`FL_TRIM_RATIO`（Trimmed-Mean 裁剪比例，默认=恶意率 0.3）。
- CIFAR 专用：`CIFAR_MODEL`（默认 `ResNet18CIFAR`）、`FL_GAUSSIAN_STD`（默认 0.05）、`FL_GRADIENT_SCALE`（脚本 `setdefault` 为 **1=不放大**）。

## §5 冻结红线（禁止擅改）

1. **DPFLA 主路径必须走 SVD+KMeans**：`rule=='DPFLA'` 时 `use_validation=False`，不用 loss 打分版作主结论。
2. **评估指标唯一**：主指标 = YOLO 原生 `val()` 的 `mAP@0.5`，**禁用 fallback 自定义 mAP**。
3. **聚合精度**：YOLO 场景统一 `float16_floats=False`（float32）。
4. **数据白名单**：`logs_3/mnist/`、`runs/MNIST_*/` 为实验数据保留目录，删除前需确认不在此列。
5. **基线配置（以当前代码为准，见 01_Facts §6）**：`TRAIN_BATCH_SIZE=64`、`LOCAL_EPOCHS=3`、`LOCAL_LR=2e-4`、`LR_SCHEDULE=constant`。禁用 cosine。
   > 历史记录里曾出现 `LOCAL_EPOCHS=10` 的"好配置"表述，但实测达到 R26-33≈23.7% 的 100 轮基线用的是 **EPOCH=3**；两者不冲突（BS=64 下 EPOCH=3 即可收敛）。
6. **单变量原则**：任何实验一次只改一个变量（87e3f7a 的教训）。
7. **数据分布固定 IID**：脚本不设 `dd_type`，走默认 `IID`（`arguments.py:58`）；**不考虑 Non-IID**（2026-10-10 用户确认）。

> 注：旧版"红线第 7 条：fl_core.py val bug"已删除——该结论已证伪，见 `04_Retracted.md` R2。

## §6 故障字典

| 现象 | 原因 | 处理 |
|------|------|------|
| `libGL.so.1` 加载失败 | ultralytics 装了 GUI 版 opencv | 卸载 opencv-python，装 opencv-python-headless |
| val 子进程 `code=-9` | 内存不足被 OOM 杀 | 降低并发/显存占用后重试（**非代码 bug**） |
| `pickle data was truncated` | 缓存损坏 | `rm cache/cache_yolo_val_*.pkl`（缓存可重建） |
| Round 0/1 mAP=0 | 换头后检测头随机初始化（冷启动） | 正常；R1 起 >0 即链路正常 |
| 训练看起来慢，GPU 未利用 | 前期在 CPU 做标注缓存 + Dirichlet 划分 | 等缓存生成完再观察 |
| 换聚合/改超参后 mAP 更差 | 一次改了多个变量，无法归因 | 回退，改单变量重跑 |
| IDE 里出现"永不结束的命令行" | 等待脚本用 `while pgrep -f "<实验名>"` 监控，**pgrep 会匹配到脚本自身命令行** → 死循环（仅 sleep，不占 GPU） | 改用**日志完成标记** + 有限次 `seq` 循环；已残留的用 `ps -eo pid,ppid,etime,cmd` 定位后 `kill` |
